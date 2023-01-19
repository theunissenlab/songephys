"""Logic for downloading or copying data files

Data files need to be located and downloaded/copied so that they can be processed
by our data pipeline.

This module is for classes and functions that describe the logic for locating
and downloading such files.

To work in the pipeline, the downloader object should implement two methods:

    def download_csvs(self, experiment_name, csv_list, save_to=None):
        ...

    def download_session(self, experiment_name, session_id, save_to=None, pyoperant_files=tuple(), event_files=tuple())
        ...

It is up to the internals of the downloader object to figure out how to do those tasks
"""


from __future__ import print_function
import logging
import glob
import os


import zipfile

# For gdrive downloads
try:
    from gdrive_access import GDriveCommands
    from gdrive_access.errors import CredentialsNotFound
except:
    gdrive_access = None

# For owncloud downloads
from .io import load_yaml
try:
    import owncloud
except:
    owncloud = None

# The config file is in code and has a bnuch of globa variables - mostly for spike sorting

import numpy as np


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class BaseDownloader(object):
    """Base class of download logic

    Subclasses must implement:

        * download_file(rhd_file: T, download_to: str)
        * download_folder(folder_object: T, download_to: str)

        * locate_rhd_file(experiment_name: str, session_id: str) -> T
        * locate_pyoperant_file(experiment_name: str, pyoperant_file: str) -> T
        * list_directories(experiment_name: str) -> List[T]
        * get_stimuli_foldername(stimuli_folder: T) -> str

    T refers to the relevant type for that downloader. For gdrive these would be
    pydrive2.GoogleDriveFile objects, and for owncloud/local downloading these
    would just be strings.
    """
    def __init__(self,cfg):
        print("BASE")
        self.cfg = cfg
    
    def connect(self):
        raise NotImplementedError

    def close(self):
        pass

    def download_file(self, rhd_file, download_to):
        """Download the file to the given location on disk, depending on what type of Downloader this is"""
        raise NotImplementedError

    def download_folder(self, folder_object, download_to):
        """Download an entire folder to the given location"""
        raise NotImplementedError

    def locate_rhd_file(self, experiment_name, session_id):
        """Return the RHD file object (type depends on which type of Downloader is being used)"""
        raise NotImplementedError

    def locate_pyoperant_file(self, experiment_name, pyoperant_file):
        """Return the file object for the given pyoperant file for this Downloader"""
        raise NotImplementedError

    def list_stimuli_directories(self, experiment_name):
        """Return a list of file objects referencing the downloadable stimulus directories"""
        raise NotImplementedError

    def get_stimuli_foldername(self, stimuli_folder):
        """Get the string name of a stimuli_folder object (the basename for a string, or name of the google drive folder)"""
        raise NotImplementedError

    def download_csvs(self, experiment_name, csv_list, save_to=None):
        """Locates all pyoperant files in csv_list and downloads them to save_to
        """
        self.connect()

        downloaded = []
        for pyoperant_file in csv_list:
            try:
                csv_file = self.locate_pyoperant_file(experiment_name, pyoperant_file)
            except:
                logger.warning("File {} not found! "
                               "Skipping...".format(pyoperant_file))
                continue

            download_dest = os.path.join(save_to, pyoperant_file)
            self.download_file(csv_file, download_dest)
            downloaded.append(download_dest)

        logger.info("Downloaded: {}".format(downloaded))
        self.close()
        return downloaded

    def download_session(
                self,
                experiment_name,
                session_id,
                save_to=None,
                pyoperant_files=tuple(),
                event_files=tuple()
            ):
        """Download a recording session data into the proper location

        Creates directories in this structure if they don't exist yet
        `DATA_DIR/BIRD_NAME/SESSION_NAME/raw`
        and save rhd and pyoperant files here

        Params
            session_id (str): Recording session id in normal format
                (e.g. "GreYel3594M___170914_121518")
            save_to (str): Path of directory to download session to
            pyoperant_files (list of str):
                Each element is a name of a pyoperant file
                (e.g. "GreYel3594M_operant_20170912164335_trialdata.csv")
            event_files (list of str):
                Each element is a name of a event file
                (e.g. "GreYel3594M_operant_20170912164335_events.log")
        """
        self.connect()

        if session_id.endswith(".rhd"):
            session_id = os.path.splitext(session_id)[0]

        if save_to is None:
            bird_dir = os.path.join(self.cfg['EXPERIMENT']['DATADIR'], "birds", experiment_name)
            session_dir = os.path.join(bird_dir, session_id)
        else:
            session_dir = save_to

        raw_dir = os.path.join(session_dir, "raw")
        stimuli_dir = os.path.join(session_dir, "stimuli")

        # This will raise an exception if the rhd file is not found
        rhd_file = self.locate_rhd_file(experiment_name, session_id)

        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        if not os.path.exists(stimuli_dir):
            os.makedirs(stimuli_dir)

        save_rhd_to = os.path.join(raw_dir, "{}.rhd".format(session_id))

        if os.path.exists(save_rhd_to):
            logger.info("{} already downloaded. Not downloading...".format(save_rhd_to))
        else:
            self.download_file(rhd_file, save_rhd_to)

        # Download all the pyoperant csv files with playback trial information
        # Also collect all the stimulus directory names of files used in those pyoperant playbacks

        for pyoperant_file in pyoperant_files:
            if not pyoperant_file.endswith(".csv"):
                pyoperant_file = "{}.csv".format(pyoperant_file)

            try:
                csv_file = self.locate_pyoperant_file(experiment_name, pyoperant_file)
            except:
                logger.warning("File {} not found! "
                               "Skipping...".format(pyoperant_file))
                continue

            self.download_file(csv_file, os.path.join(raw_dir, pyoperant_file))

        if pyoperant_files:
            self.download_stim_dirs(pyoperant_files, experiment_name, raw_dir, stimuli_dir)
        
        for event_file in event_files:
            if not event_file.endswith(".log"):
                event_file = "{}.log".format(event_file)

            try:
                log_file = self.locate_pyoperant_file(experiment_name, event_file)
            except:
                logger.warning("File {} not found! "
                               "Skipping...".format(event_file))
                continue

            self.download_file(log_file, os.path.join(raw_dir, event_file))

        self.close()


    def download_stim_dirs(self, pyoperant_files, experiment_name, raw_dir, stimuli_dir):
        # Using the stimulus directories concatenated into a string (csv_string) from loading the trial data above
        # search through all the stim directories available and
        # NOTE: If we are symlinking, we can actually just symlink the outer stimulus folder... so overwrite this
        csv_string = ""
        stim_dir_list = []
        for pyoperant_file in pyoperant_files:
            with open(os.path.join(raw_dir, pyoperant_file), "r") as csvfile:
                csvlines = csvfile.readlines()
                for csvline in csvlines:
                    csvtoken = csvline.split(',')
                    if (csvtoken[0] == 'session'):
                        continue
                    stim_dir_path = csvtoken[3].replace('\\','/')
                    stim_dir_list.append(stim_dir_path.split('/')[-2])
                    csv_string += stim_dir_path

            stim_dir_list = list(np.unique(stim_dir_list))

        if pyoperant_files:
            stim_foldernames = []
            stimuli_folders = self.list_stimuli_directories(experiment_name)
            for stimuli_folder in stimuli_folders:
                stimuli_foldername = self.get_stimuli_foldername(stimuli_folder)
                stim_foldernames.append(stimuli_foldername)
                if stimuli_foldername in csv_string:
                    # If this is True, then at least one trial used the stimuli in this folder...
                    # ...so we should download it.
                    self.download_folder(stimuli_folder, stimuli_dir)

            # Check to see of all files were found
            if not set(stim_dir_list).issubset(set(stim_foldernames)) :
                logger.info("Stim download error: missing stim folders {} in {}".format(stim_dir_list, stim_foldernames))


class BaseGdriveDownloader(BaseDownloader):

    def connect(self):
        try:
            self.g = GDriveCommands(os.path.join(self.cfg["DOWNLOADS"]["GDRIVE_CREDENTIALS_DIR"], "settings.yaml"))
        except CredentialsNotFound:
            logger.warning("Google Drive authentication failed. Need to set up gdrive authentication:\n"
                       "python -m gdrive_access.setup_credentials --dir {}".format(self.cfg["DOWNLOADS"]["GDRIVE_CREDENTIALS_DIR"]))
            raise
        else:
            self.root = self.g.get_root("Intan Ephys Data")

    def download_file(self, file_object, download_to):
        logger.info("Downloading {} to {}".format(file_object["title"], download_to))
        self.g.download_file(file_object, download_to)

    def download_folder(self, folder_object, download_to):
        logger.info("Downloading {} to {}".format(folder_object["title"], download_to))
        self.g.download_folder(folder_object, download_to)

    def locate_rhd_file(self, experiment_name, session_id):
        return self.g.find(self.rhd_drive(experiment_name), "{}.rhd".format(session_id))

    def locate_pyoperant_file(self, experiment_name, pyoperant_file):
        return self.g.find(self.pyoperant_drive(experiment_name), pyoperant_file)

    def list_stimuli_directories(self, experiment_name):
        return self.g.ls(self.stimuli_drive(experiment_name))

    def get_stimuli_foldername(self, stimuli_folder):
        return stimuli_folder["title"]


class BaseOwncloudDownloader(BaseDownloader):

    def connect(self):
        self.g = owncloud.Client(self.cfg['DOWNLOADS']['OWNCLOUD_URL'])
        loginData = load_yaml(self.cfg['DOWNLOADS']['OWNCLOUD_CREDENTIALS_FILE'])

        try:
            self.g.login(loginData['user'], loginData['key'])
        except:
            logger.warning("Could not log on to Owncloud drive")
            raise
        else:
            self.root = '/'

    def close(self):
        self.g.logout()

    def download_file(self, file_object, download_to):
        logger.info("Downloading {} to {}".format(file_object, download_to))
        self.g.get_file(file_object, download_to)

    def download_folder(self, folder_object, download_to):
        download_zip_to = os.path.join(download_to + ".zip")
        logger.info("Downloading {} to {}".format(folder_object.get_name(), download_zip_to))
        self.g.get_directory_as_zip(stimuli_folder.get_path(), download_zip_to)
        # Unzip
        with zipfile.ZipFile(download_zip_to, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(download_to))

            # TODO: remove zip file when done?

    def locate_rhd_file(self, experiment_name, session_id):
        rhd_file = self.rhd_drive(experiment_name) + "{}.rhd".format(session_id)
        # Checks that the file exists
        self.g.file_info(rhd_file)
        return rhd_file

    def locate_pyoperant_file(self, experiment_name, pyoperant_file):
        csv_file = self.pyoperant_drive(experiment_name) + pyoperant_file
        self.g.file_info(csv_file)
        return csv_file

    def list_stimuli_directories(self, experiment_name):
        return self.g.list(self.stimuli_drive(experiment_name))

    def get_stimuli_foldername(self, stimuli_folder):
        return stimuli_folder.get_name()


class BaseLocalSymlinker(BaseDownloader):

    def connect(self):
        pass

    def download_stim_dirs(self, pyoperant_files, experiment_name, raw_dir, stimuli_dir):
        # If we are symlinking, we can actually just symlink the outer stimulus folder
        for item in glob.glob(os.path.join(self.stimuli_drive(experiment_name), "*")):
            self.download_folder(item, stimuli_dir)

    def download_file(self, file_object, download_to):
        logger.info("Symlinking {} to {}".format(file_object, download_to))
        if os.path.exists(download_to):
            os.remove(download_to)
        os.symlink(file_object, download_to)

    def download_folder(self, folder_object, download_to):
        target = os.path.join(download_to, os.path.basename(folder_object))
        logger.info("Symlinking {} to {}".format(folder_object, target))
        os.symlink(folder_object, target)

    def locate_rhd_file(self, experiment_name, session_id):
        rhd_file = os.path.join(self.rhd_drive(experiment_name), "{}.rhd".format(session_id))
        if not os.path.exists(rhd_file):
            raise IOError
        return rhd_file

    def locate_pyoperant_file(self, experiment_name, pyoperant_file):
        csv_file = os.path.join(self.pyoperant_drive(experiment_name), pyoperant_file)
        if not os.path.exists(csv_file):
            raise IOError
        return csv_file

    def list_stimuli_directories(self, experiment_name):
        return glob.glob(os.path.join(self.stimuli_drive(experiment_name), "*"))

    def get_stimuli_foldername(self, stimuli_folder):
        return os.path.basename(stimuli_folder)


##########################
### Actual Downloaders ###
##########################


class TheunissenGdriveDownloader(BaseGdriveDownloader):

    def rhd_drive(self, experiment_name):
        return self.g.find(self.root, experiment_name, 'RHD')

    def pyoperant_drive(self, experiment_name):
        return self.g.find(self.root, experiment_name, "pyoperant")

    def stimuli_drive(self, experiment_name):
        return self.g.find(self.root, experiment_name, "stimuli")


class SeewiesenGdriveDownloader(BaseGdriveDownloader):

    def rhd_drive(self, experiment_name):
        return self.g.find(self.root, "Seewiesen", experiment_name)

    def pyoperant_drive(self, experiment_name):
        return self.g.find(self.root, "Seewiesen", experiment_name)

    def stimuli_drive(self, experiment_name):
        return self.g.find(self.root, "Seewiesen", "Stimuli", "Bird1")


class Seewiesen2GdriveDownloader(BaseGdriveDownloader):

    def rhd_drive(self, experiment_name):
        return self.g.find(self.root, "Seewiesen2", experiment_name)

    def pyoperant_drive(self, experiment_name):
        return self.g.find(self.root, "Seewiesen2", experiment_name)

    def stimuli_drive(self, experiment_name):
        return self.g.find(self.root, "Seewiesen2", experiment_name, "Stimuli")


class PepeCanaryDownloader(BaseGdriveDownloader):

    def rhd_drive(self, experiment_name):
        return self.g.find(self.root, "PepeCanary", experiment_name)

    def pyoperant_drive(self, experiment_name):
        return self.g.find(self.root, "PepeCanary", experiment_name, "pyoperant")

    def stimuli_drive(self, experiment_name):
        return self.g.find(self.root, "PepeCanary", experiment_name, "Stimuli")


class SeewiesenOwncloudDownloader(BaseOwncloudDownloader):

    def rhd_drive(self, experiment_name):
        return "/Seewiesen/" + experiment_name + "/"

    def pyoperant_drive(self, experiment_name):
        return "/Seewiesen/" + experiment_name + "/"

    def stimuli_drive(self, experiment_name):
        return "/Seewiesen" + "/Stimuli" + "/Bird1/"


class LocalSymlinker(BaseLocalSymlinker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not all([
                self.cfg['LOCAL_RAW_FILES']["rhd_drive"],
                self.cfg['LOCAL_RAW_FILES']["pyoperant_drive"],
                self.cfg['LOCAL_RAW_FILES']["stimuli_drive"]
                ]):
            raise ValueError("LOCAL_RAW_FILES in songephys/code/config.py must have keys set: rhd_drive, pyoperant_drive, and stimuli_drive")

    def rhd_drive(self, experiment_name):
        if "{experiment_name}" in self.cfg['LOCAL_RAW_FILES']["rhd_drive"]:
            return self.cfg['LOCAL_RAW_FILES']["rhd_drive"].format(experiment_name=experiment_name)
        else:
            return self.cfg['LOCAL_RAW_FILES']["rhd_drive"]

    def pyoperant_drive(self, experiment_name):
        if "{experiment_name}" in self.cfg['LOCAL_RAW_FILES']["pyoperant_drive"]:
            return self.cfg['LOCAL_RAW_FILES']["pyoperant_drive"].format(experiment_name=experiment_name)
        else:
            return self.cfg['LOCAL_RAW_FILES']["pyoperant_drive"]

    def stimuli_drive(self, experiment_name):
        if "{experiment_name}" in self.cfg['LOCAL_RAW_FILES']["stimuli_drive"]:
            return self.cfg['LOCAL_RAW_FILES']["stimuli_drive"].format(experiment_name=experiment_name)
        else:
            return self.cfg['LOCAL_RAW_FILES']["stimuli_drive"]


downloaders = {
    "seewiesen": SeewiesenGdriveDownloader,
    "seewiesen2": Seewiesen2GdriveDownloader,
    "seewiesenCloud" : SeewiesenOwncloudDownloader,
    "theunissen": TheunissenGdriveDownloader,
    "pepe": PepeCanaryDownloader,
    "local": LocalSymlinker,
}
