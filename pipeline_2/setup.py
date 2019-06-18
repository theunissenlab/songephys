import glob
import os

import numpy as np
import luigi
from neo.io import RHDIO, NixIO

from format.gdrive_access import download
from format.io import load_yaml
from format.lazy import (
    LazySignal,
    LazySite,
    create_lazy_signals_from_nix,
)
from utils import session_basename


class SiteConfig(luigi.Task):
    site_dir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "sessions.yaml"))

    def run(self):
        raise Exception("The sessions.yaml file must be manually created")


class DownloadSession(luigi.Task):
    """
    Download a session and pyoperant data for that session from google drive.

    Looks in google drive to create a directory

    /birds/<BIRDNAME>/sessions/<session_dir>/raw/

    and puts in an rhd file and 0+ pyoperant files
    """

    session_dir = luigi.Parameter()
    pyoperants = luigi.ListParameter(default=[])

    def output(self):
        self.session_name = session_basename(self.session_dir)
        output_directory = os.path.join(self.session_dir, "raw")
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
        if download.drive is None:
            raise Exception("Not authenticated with Google Drive; "
                            "cannot download raw data files")

        # First, a hacky way to check if there are supposed to be pyoperants
        session_name = os.path.basename(self.session_dir)
        prefix = session_name.split("_")[0]
        site_dirs = glob.glob(os.path.join(
            self.session_dir,
            "..",
            "..",
            "sites",
            "{}*".format(prefix),
            "sessions.yaml"))
        pyoperants = []
        for site_dir in site_dirs:
            session_config = load_yaml(site_dir)
            for session in session_config["sessions"]:
                if session["id"] == session_name and "playbacks" in session:
                    pyoperants += session["playbacks"]
        self.pyoperants = pyoperants
        # end hack

        download.download_session(
            self.session_name,
            save_to=self.session_dir,
            pyoperant_files=self.pyoperants)



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

        rhd_file = inputs["rhd"].path
        rhd = RHDIO(rhd_file)
        nix = NixIO(self.output().path)
        nix.write_block(rhd.read_block())
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
