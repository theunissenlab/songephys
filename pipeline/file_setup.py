import os
import shutil

import numpy as np
import luigi
from neo.io import RHDIO, NixIO

from format import gdrive
from format.io import load_yaml, save_yaml
from format.lazy import create_lazy_signals_from_nix
from pipeline_utils import (
    associate_pyoperant_csvs_to_rhds,
    get_session_sites,
    parse_filelist,
    session_basename,
)


class SiteFileList(luigi.Task):
    """Checks if a file list file exists

    sites/
        SITE/
            filelist [output]
    """
    site_dir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "filelist"))

    def run(self):
        raise Exception("The filelist file must be manually created")


class SiteConfig(luigi.Task):
    """Creates a yaml config file for the site defining its sessions

    The yaml can be made manually, but this can be slightly easier because it
    is sometimes confusing when one playback csv spans multiple RHD files,
    and vice-versa.

    REQUIRES google drive authentication (needs to download csvs and read them
    to find trial timestamps)

    sites/
        SITE/
            filelist [input]
            session.yaml [output]

    Example filelist:
          seewiesen
          ZF4F_trialdata_20190612134340.csv
          ZF4F_2t_190612_134437.rhd
          ZF4F_2t_190612_135747.rhd

    Converted into sessions.yaml:
          lab: seewiesen
          sessions:
            - id: ZF4F_2t_190612_134437
              pyoperants:
              - ZF4F_trialdata_20190612134340
              t: 0.0
            - id: ZF4F_2t_190612_135747
              t: 790.0
    """
    site_dir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "sessions.yaml"))

    # Let the run() function capture this to spit out the correct error message
    def requires(self):
        return SiteFileList(self.site_dir)

    def run(self):
        if not os.path.join(self.input().path, "filelist"):
            raise Exception("""

        Error, either

        - sessions.yaml must be created manually
        - a plain text file named 'filelist' must be created
          that includes the lab code name (i.e. "theunissen" or "seewiesen")
          on the first line, and csvs and rhds on other lines.

          Example filelist contents

          seewiesen
          ZF4F_trialdata_20190612134340.csv
          ZF4F_2t_190612_134437.rhd
          ZF4F_2t_190612_135747.rhd

          This will automatically create the session.yaml file

              """)

        site_files = parse_filelist(self.input().path)
        session_dicts = associate_pyoperant_csvs_to_rhds(
            self.site_dir,
            site_files["csvs"],
            site_files["rhds"],
            site_files["lab"]
        )

        save_yaml(self.output().path, {"lab": site_files["lab"], "sessions": session_dicts})


class DownloadSession(luigi.Task):
    """
    Download a session and pyoperant data for that session from google drive.

    Looks in google drive to create a directory

    /birds/<BIRDNAME>/sessions/<session_dir>/raw/

    and puts in an rhd file and 0+ pyoperant files
    """

    session_dir = luigi.Parameter()

    def output(self):
        self.session_name = session_basename(self.session_dir)
        output_directory = os.path.join(self.session_dir, "raw")

        self.site_dirs, self.pyoperants = get_session_sites(self.session_dir)

        return {
            "rhd": luigi.LocalTarget(os.path.join(
                output_directory, "{}.rhd".format(self.session_name)
            )),
            "pyoperants": [
                luigi.LocalTarget(os.path.join(
                    output_directory, "{}.csv".format(filename)
                )) for filename in self.pyoperants
            ]
        }

    def run(self):
        """Use the yaml file to download RHDs and pyoperant csvs for this session
        """
        if gdrive.root is None:
            raise Exception("Not authenticated with Google Drive; "
                            "cannot download raw data files")

        if not len(self.site_dirs):
            raise Exception("No sites were found with sessions.yaml containing {}".format(self.session_dir))

        # Get the lab name from the first site
        # This is needed for the download function to adjust
        # for slightly different directory structure on google drive
        session_config = load_yaml(os.path.join(self.site_dirs[0], "sessions.yaml"))
        LAB_NAME = session_config.get("lab", "theunissen")

        gdrive.downloaders[LAB_NAME].download_session(
            self.session_name,
            save_to=self.session_dir,
            pyoperant_files=self.pyoperants,
        )


class DownloadSite(luigi.Task):
    site_dir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget("dummy")

    def requires(self):
        self.site_config = load_yaml(os.path.join(self.site_dir, "sessions.yaml"))

        for session in self.site_config["sessions"]:
            session_dir = os.path.join(self.site_dir, "..", "..", "sessions", session["id"])
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
            yield DownloadSession(os.path.join(self.site_dir, "..", "..", "sessions", session["id"]))

    def run(self):
        pass


class Rhd2Nix(luigi.Task):
    """Converts an RHD file to a NIX file, and clean up the RHD file

    Inputs
    ------
    session_dir : str
        Path to session

    Outputs a nix file as segmentation/{session_name}.nix
    """

    session_dir = luigi.Parameter()

    def requires(self):
        return DownloadSession(self.session_dir)

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        if not os.path.exists(segmentation_dir):
            os.makedirs(segmentation_dir)

        nix_filename = "{}.nix".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(segmentation_dir, nix_filename))

    def run(self):
        inputs = self.input()

        site_dirs, _ = get_session_sites(self.session_dir)
        session_info = load_yaml(os.path.join(site_dirs[0], "sessions.yaml"))

        rhd_file = inputs["rhd"].path

        # Delete the digital input form the rhd file if from Seewiesen.
        # These are not used for stim id but to sync with bat logger
        if (session_info["lab"] == "seewiesen"):
            print("Seewiesen Data: digital events in rhd will be deleted")
            storeDIG = False
        else:
            storeDIG = True

        rhd = RHDIO(rhd_file)
        blk = rhd.read_block(storeDIG=storeDIG)
        nix = NixIO(self.output().path)
        nix.write_block(blk)
        nix.close()

        # Remove the RHD file now that we have a NIX file
        os.remove(inputs["rhd"].path)


class GenerateLazyFiles(luigi.Task):
    """Create memmapped array on disk for lazy loading

    Inputs
    ------
    session_dir : str
        Path to session

    Outputs memmapped files for the microphone and electrode data arrays
    """

    session_dir = luigi.Parameter()

    def requires(self):
        return Rhd2Nix(self.session_dir)

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        mic_path = os.path.join(raw_dir, "mic.memmap")
        elec_path = os.path.join(raw_dir, "raw.memmap")
        lookup_path = os.path.join(self.session_dir, "lazy.npy")

        return {
            "mic_memmap": luigi.LocalTarget(mic_path),
            "raw_memmap": luigi.LocalTarget(elec_path),
            "lazy_signals": luigi.LocalTarget(lookup_path)
        }

    def run(self):
        outputs = self.output()

        nix_file = self.input().path

        new_sigs = create_lazy_signals_from_nix(
            nix_file,
            outputs["mic_memmap"].path,
            outputs["raw_memmap"].path,
            batch_size=30000
        )

        np.save(outputs["lazy_signals"].path, new_sigs)
