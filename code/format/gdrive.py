"""Module with helper functions to access our google drive

Intan Ephys Data/
    <bird name>/
        RHD/
            ...
        pyoperant/
            ...
        Stimuli/
            TutFamBos
            ...
"""

from __future__ import print_function
import logging
import os

# For gdrive downloads
from gdrive_access import GDriveCommands
from gdrive_access.errors import CredentialsNotFound

# For owncloud downloads
from format.io import load_yaml
import owncloud

# The config file is in code and has a bnuch of globa variables - mostly for spike sorting
import config
import zipfile


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class EphysSiteDownloader(object):
    """
    Base class to define download functions that must change between labs/projects

    For example, to change the location of where to look for files, to rename
    files as they are downloaded, etc, inherit from this class and override/define
    the needed functions
    """
    
    def __init__(self, lab):
        self.lab = lab
        
    def connect(self):
        
        if (self.lab != 'seewiesenCloud'):
            self.cloud = 'gdrive'
            try:
                self.g = GDriveCommands(os.path.join(config.GDRIVE_CREDENTIALS_DIR, "settings.yaml"))
            except CredentialsNotFound:
                self.root = None
                logger.warning("Google Drive authentication failed. Need to set up gdrive authentication:\n"
                           "python -m gdrive_access.setup_credentials --dir {}".format(config.GDRIVE_CREDENTIALS_DIR))
            else:
                self.root = self.g.get_root("Intan Ephys Data")
        else:
            self.cloud = 'owncloud'
            self.g = owncloud.Client('https://owncloud.gwdg.de')
            loginData = load_yaml(os.path.join(os.environ.get('PROJECT_ROOT'), 'owncloud.yaml'))  
            
            try:
                self.g.login(loginData['user'], loginData['key'])
            except:
                self.root = None
                logger.warning("Could not log on to Owncloud drive")
            else:
                self.root = '/'
            
    def close(self):
        if (self.cloud == 'owncloud'):
            self.g.logout()
            
    def bird(self, session_id):
        return session_id.split("_")[0]

    def pyoperant_drive(self, bird):
        raise NotImplementedError

    def bird_drive(self, bird):
        raise NotImplementedError

    def stimuli_drive(self, bird):
        raise NotImplementedError

    def download_csvs(self, site_id, csv_list, save_to=None):
        """Locates all pyoperant files in csv_list and downloads them to save_to

        Uses site_id to find the files on google drive
        """
        bird = self.bird(site_id)
        self.connect()
        
        if (self.root == None):
            return

        downloaded = []
        for pyoperant_file in csv_list:
            print(self.pyoperant_drive(bird))
            try:
                if (self.cloud == 'gdrive'):
                    csv_file = self.g.find(self.pyoperant_drive(bird), pyoperant_file)
                elif (self.cloud == 'owncloud'):
                    csv_file = self.pyoperant_drive(bird)+ pyoperant_file
                    self.g.file_info(csv_file)
                                       
            except:
                logger.warning("File {} not found! "
                               "Skipping...".format(pyoperant_file))
                continue

            download_dest = os.path.join(save_to, pyoperant_file)
            if (self.cloud == 'gdrive'):
                self.g.download_file(csv_file, download_dest)
            elif (self.cloud == 'owncloud'):
                self.g.get_file(csv_file, download_dest)
            downloaded.append(download_dest)

        logger.info("Downloaded: {}".format(downloaded))
        self.close()
        return downloaded


    def download_session(
                self,
                session_id,
                save_to=None,
                pyoperant_files=tuple(),
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
                (e.g. "GreYel3594M_trialdata_20170912164335.csv")
        """
        bird = self.bird(session_id)
        self.connect()
        if (self.root == None) :
            return

        if session_id.endswith(".rhd"):
            session_id = os.path.splitext(session_id)[0]

        if save_to is None:
            bird_dir = os.path.join(config.DATADIR, "birds", self.bird)
            session_dir = os.path.join(bird_dir, session_id)
        else:
            session_dir = save_to

        raw_dir = os.path.join(session_dir, "raw")
        stimuli_dir = os.path.join(session_dir, "stimuli")
        try:
            if (self.cloud == 'gdrive'):
                rhd_file = self.g.find(self.rhd_drive(bird), "{}.rhd".format(session_id))
            elif (self.cloud == 'owncloud'):
                rhd_file = self.rhd_drive(bird) + "{}.rhd".format(session_id)
                self.g.file_info(rhd_file)
                
        except:
            raise Exception("Rhd file not found")

        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        if not os.path.exists(stimuli_dir):
            os.makedirs(stimuli_dir)

        if os.path.exists(os.path.join(raw_dir, "{}.rhd".format(session_id))):
            logger.info("{}.rhd already downloaded. "
                        "Not downloading...".format(session_id))
        else:
            logger.info("Downloading {}.rhd to {}".format(session_id, raw_dir))

            if (self.cloud == 'gdrive'):
                self.g.download_file(
                        rhd_file,
                        os.path.join(raw_dir, rhd_file["title"])
                        )
            elif (self.cloud == 'owncloud'):
                self.g.get_file(
                        rhd_file,
                        os.path.join(raw_dir, "{}.rhd".format(session_id))
                        )
                

        csv_string = ""

        for pyoperant_file in pyoperant_files:
            if not pyoperant_file.endswith(".csv"):
                pyoperant_file = "{}.csv".format(pyoperant_file)

            try:
                if (self.cloud == 'gdrive'):
                    csv_file = self.g.find(self.pyoperant_drive(bird), pyoperant_file)
                elif (self.cloud == 'owncloud'):
                    csv_file = self.pyoperant_drive(bird) + pyoperant_file
                    self.g.file_info(csv_file)
                    
            except:
                logger.warning("File {} not found! "
                               "Skipping...".format(pyoperant_file))
                continue

            logger.info("Downloading {} to {}".format(pyoperant_file, raw_dir))
            
            if (self.cloud == 'gdrive'):
                self.g.download_file(csv_file, os.path.join(raw_dir, pyoperant_file) )
            elif (self.cloud == 'owncloud'):
                self.g.get_file(csv_file, os.path.join(raw_dir, pyoperant_file) )

            with open(os.path.join(raw_dir, pyoperant_file), "rb") as csvfile:
                csv_string += csvfile.read().decode("utf-8")

        # We can load the csvs and check which stimulus directories to download
        if pyoperant_files:
            if (self.cloud == 'gdrive'):
                stimuli_folders = self.g.ls(self.stimuli_drive(bird))
                for stimuli_folder in stimuli_folders:
                    # TODO (check if this is needed)
                    if stimuli_folder["title"] in csv_string:
                        download_to = os.path.join(stimuli_dir, stimuli_folder["title"])
                        logger.info("Downloading {} to {}".format(
                                stimuli_folder["title"],
                                download_to
                                ))
                        self.g.download_folder(stimuli_folder, download_to)
            elif (self.cloud == 'owncloud'):
                stimuli_folders = self.g.list(self.stimuli_drive(bird))
                for stimuli_folder in stimuli_folders:
                    if stimuli_folder.get_name() in csv_string:
                        download_to = os.path.join(stimuli_dir, stimuli_folder.get_name()+'.zip')
                        logger.info("Downloading {} to {}".format(
                                stimuli_folder.get_name(),
                                download_to
                                ))
                        self.g.get_directory_as_zip(stimuli_folder.get_path(), download_to)
                        # Unzip
                        with zipfile.ZipFile(download_to, 'r') as zip_ref:
                            zip_ref.extractall(stimuli_dir)
        self.close()

class TheunissenDownloader(EphysSiteDownloader):

    def bird_drive(self, bird):
        return self.g.find(self.root, bird)
    
    def rhd_drive(self, bird):
        return self.g.find(self.root, bird, 'RHD')

    def pyoperant_drive(self, bird):
        return self.g.find(self.root, bird, "pyoperant")

    def stimuli_drive(self, bird):
        return self.g.find(self.root, bird, "stimuli")


class SeewiesenDownloader(EphysSiteDownloader):

    def bird_drive(self, bird):
        return self.g.find(self.root, "Seewiesen", bird)
    
    def rhd_drive(self, bird):
        return self.g.find(self.root, "Seewiesen", bird)

    def pyoperant_drive(self, bird):
        return self.g.find(self.root, "Seewiesen", bird)

    def stimuli_drive(self, bird):
        return self.g.find(self.root, "Seewiesen", "Stimuli", "Bird1")

class SeewiesenOwncloudDownloader(EphysSiteDownloader):

    def bird_drive(self, bird):
        return "/Seewiesen/" + bird + "/"
    
    def rhd_drive(self, bird):
        return "/Seewiesen/" + bird + "/"
    
    def pyoperant_drive(self, bird):
        return "/Seewiesen/" + bird + "/"

    def stimuli_drive(self, bird):
        return "/Seewiesen" + "/Stimuli" + "/Bird1/"



class TestDownloader(EphysSiteDownloader):
    """Set up a test folder of data and use this to point to it"""
    def __init__(self, lab):
        super().__init__(lab)
        # self.test_root = self.g.find(self.root, "SongephysTestPipelineData")
        # I broke this... it will have to be done in connect somehow...

    def bird_drive(self, bird):
        return self.g.find(self.test_root, bird)
    
    def rhd_drive(self, bird):
        return self.g.find(self.test_root, bird, "RHD")
    
    def pyoperant_drive(self, bird):
        return self.g.find(self.test_root, bird, "pyoperant")

    def stimuli_drive(self, bird):
        return self.g.find(self.test_root, bird, "stimuli")


downloaders = {
    "seewiesen": SeewiesenDownloader(lab ='seewiesen'),
    "seewiesenCloud" : SeewiesenOwncloudDownloader(lab='seewiesenCloud'),
    "theunissen": TheunissenDownloader(lab='theunissen'),
    "test": TestDownloader(lab='theunissen')
}
