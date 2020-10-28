import glob
import os
import shutil

import pickle

import numpy as np
import luigi
from neo.io import RHDIO, NixIO
import pandas as pd

from format.gdrive_access import download
from format.io import load_yaml, save_yaml
from format.intan_utils import timestamp_from_intan_filename
from format.lazy import (
    LazySignal,
    create_lazy_signals_from_nix,
)
from format.gdrive_access.download import download_csvs
from utils import session_basename


class SiteFileList(luigi.Task):
    """Checks if a file list file exists
    """
    site_dir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "filelist"))

    def run(self):
        raise Exception("The filelist file must be manually created")


class SiteConfig(luigi.Task):
    site_dir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "sessions.yaml"))

    def run(self):
        """Creates a yaml config file for the session

        The yaml can be made manually, but this can be slightly easier
        because it is sometimes confusing when one playback csv spans
        multiple RHD files, and vice-versa.

        REQUIRES google drive authentication (needs to download csvs
        and read them to find trial timestamps)

        Example file contents:

              seewiesen
              ZF4F_trialdata_20190612134340.csv
              ZF4F_2t_190612_134437.rhd
              ZF4F_2t_190612_135747.rhd
        """
        if not os.path.join(self.site_dir, "filelist"):
            raise Exception("""
    Error, either

        - sessions.yaml must be created manually
        - a plain text file named 'filelist' must be created
          that includes the lab code name (i.e. "theunissen" or "seewiesen")
          on the first line, and csvs and rhds on other lines.

          This will automatically create the session.yaml file
              """)

        rhds = []
        csvs = []
        name = None

        with open(os.path.join(self.site_dir, "filelist"), "r") as filelist:
            for line in filelist.readlines():
                line = line.strip()
                if name is None:
                    if not line in ["theunissen", "seewiesen"]:
                        raise ValueError("The first line of filelist must be either 'theunissen' or 'seewiesen'")
                    name = line
                    continue

                if line.endswith(".rhd"):
                    rhds.append(line)
                elif line.endswith(".csv"):
                    csvs.append(line)

        # Remove duplicates if they exist from filelist
        rhds = list(set(rhds))
        csvs = list(set(csvs))

        # Get the timestamps from rhd filenames
        rhd_times = [
            (timestamp_from_intan_filename(rhd), rhd)
            for rhd in rhds
        ]
        rhd_times = sorted(rhd_times)

        t0 = rhd_times[0][0]

        rhd_dicts = [
            {
                "id": os.path.splitext(rhd_name)[0],
                "t": (rhd_time - t0).seconds,
            } for rhd_time, rhd_name in rhd_times
        ]

        # Download the csvs in a temporary directory
        csv_start_stop = []

        csv_temp_dir = os.path.join(self.site_dir, "tmp")
        if not os.path.exists(csv_temp_dir):
            os.makedirs(csv_temp_dir)

        site_name = session_basename(self.site_dir)
        download_csvs(site_name, csvs, csv_temp_dir, lab=name)

        # Get the first and last trial time for each csv file
        for csv in csvs:
            csv_data = pd.read_csv(
                    os.path.join(csv_temp_dir, csv),
                    parse_dates=True,
                    converters={"time": pd.to_datetime})
            first = csv_data["time"].iloc[0]
            last = csv_data["time"].iloc[-1]

            csv_start_stop.append((csv, first, last))

        # Associate csvs to RHDs using timestamp overlap conditions that
        # are highly sophisticated and complex
        for csv, start, stop in csv_start_stop:
            for i in range(len(rhd_times) - 1):
                if (
                        (rhd_times[i][0] <= start < rhd_times[i + 1][0]) or
                        (rhd_times[i][0] <= stop < rhd_times[i + 1][0]) or
                        (rhd_times[i][0] > start and rhd_times[i + 1][0] < stop) or
                        (start > rhd_times[i][0] and stop < rhd_times[i + 1][0])
                        ):
                    if "playbacks" not in rhd_dicts[i]:
                        rhd_dicts[i]["playbacks"] = [os.path.splitext(csv)[0]]
                    else:
                        rhd_dicts[i]["playbacks"].append(os.path.splitext(csv)[0])

            if rhd_times[-1][0] <= start or rhd_times[-1][0] <= stop:
                if "playbacks" not in rhd_dicts[-1]:
                    rhd_dicts[-1]["playbacks"] = [os.path.splitext(csv)[0]]
                else:
                    rhd_dicts[-1]["playbacks"].append(os.path.splitext(csv)[0])

        save_yaml(self.output().path, {"lab": name, "sessions": rhd_dicts})

        # Remove the temporary directory
        print("Removing temp dir {}".format(csv_temp_dir))
        shutil.rmtree(csv_temp_dir)


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
        """Use the yaml file to download RHDs and pyoperant csvs for this session
        """
        if download.drive is None:
            raise Exception("Not authenticated with Google Drive; "
                            "cannot download raw data files")

        # First, check if there are supposed to by pyoperant csv files
        # by loading the yaml file and looking.
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

            # The yaml file may specify a particular "lab".
            # This is needed for the download function to adjust
            # for slightly different directory structure on google drive
            LAB_NAME = session_config.get("lab", "theunissen")

        self.pyoperants = pyoperants

        download.download_session(
            self.session_name,
            save_to=self.session_dir,
            pyoperant_files=self.pyoperants,
            lab=LAB_NAME)



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
        
        # Read the yaml file to have information about the session
        site_dir = self.session_dir.split(sep='..')[0]
        self.session_info = load_yaml(
            os.path.join(site_dir, "sessions.yaml")
        )

        rhd_file = inputs["rhd"].path
 
        # Delete the digital input form the rhd file if from Seewiesen.  
        # These are not used for stim id but to sync with bat logger
       
        if (self.session_info["lab"] == 'seewiesen' ) :
            print('Seewiesen Data: digital events in rhd will be deleted')
            storeDIG=False
        else:
            storeDIG=True
            
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
