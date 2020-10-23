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
import time

from pydrive.drive import GoogleDrive

from .auth import get_auth
from format.lab_utils import drive2local_by_lab

import config


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


try:
    drive = GoogleDrive(get_auth())
except:
    logger.warning("Google Drive authentication failed")
    drive = None
    gdrive_root_dir = None
    logger.info("Run the gdrive setup script")
else:
    gdrive_root_dir = drive.ListFile({
        "q": "title = 'Intan Ephys Data'"
    }).GetList()[0]


def download_file(gdrive_file, download_to_path):
    """Download a file from google drive

    Params
    gdrive_file (pydrive file): file to download
    download_to_path (str): location on local filesystem to download data
    """
    time.sleep(0.1)
    gdrive_file.GetContentFile(download_to_path)


def download_folder(gdrive_folder, download_to_path):
    if not os.path.exists(download_to_path):
        os.makedirs(download_to_path)

    for f in gdrive_listdir(gdrive_folder):
        download_file(f, os.path.join(download_to_path, f["title"]))


def gdrive_look_in(dir, filename):
    """Look for a filename in google drive directory

    Args:
        dir: a str name of the Google Drive directory to search in
        filename: the str name of the file or directory to search for

    Returns:
        PyDrive object representing the file being searched for
    """
    time.sleep(0.1)
    file_list = drive.ListFile({
        "q": "title = '{}' and '{}' in parents and "
             "trashed = false".format(filename, dir["id"])
    }).GetList()
    if not len(file_list):
        raise Exception("{}/{} not found".format(dir["title"], filename))
    return file_list[0]


def gdrive_listdir(dir):
    time.sleep(0.1)
    return drive.ListFile({
        "q": "'{}' in parents and trashed = false".format(dir["id"])
    }).GetList()


def gdrive_get_folder(*path):
    current_position = gdrive_root_dir
    for path_element in path:
        current_position = gdrive_look_in(current_position, path_element)

    return current_position


def download_csvs(
        site_id,
        csv_list,
        save_to=None,
        lab="theunissen"
    ):
    bird_name = site_id.split("_")[0]

    if lab == "theunissen":
        gdrive_bird_dir = gdrive_look_in(gdrive_root_dir, bird_name)
        gdrive_pyoperant_dir = gdrive_look_in(gdrive_bird_dir, "pyoperant")
    elif lab == "seewiesen":
        gdrive_seewiesen_dir = gdrive_look_in(gdrive_root_dir, "Seewiesen")
        gdrive_bird_dir = gdrive_look_in(gdrive_seewiesen_dir, bird_name)
        gdrive_pyoperant_dir = gdrive_bird_dir

    downloaded = []
    for pyoperant_file in csv_list:
        try:
            csv_file = gdrive_look_in(gdrive_pyoperant_dir, pyoperant_file)
        except:
            logger.warning("File {} not found! "
                           "Skipping...".format(pyoperant_file))
            continue

        download_dest = os.path.join(save_to, csv_file["title"])
        download_file(csv_file, download_dest)
        downloaded.append(download_dest)

    print(downloaded)

    return downloaded


def download_session(
        session_id,
        save_to=None,
        pyoperant_files=tuple(),
        lab="theunissen",
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
    if session_id.endswith(".rhd"):
        session_id = os.path.splitext(session_id)[0]

    bird_name = session_id.split("_")[0]

    if save_to is None:
        bird_dir = os.path.join(config.DATADIR, "birds", bird_name)
        session_dir = os.path.join(bird_dir, session_id)
    else:
        session_dir = save_to
    raw_dir = os.path.join(session_dir, "raw")
    stimuli_dir = os.path.join(session_dir, "stimuli")

    if lab == "theunissen":
        gdrive_bird_dir = gdrive_look_in(gdrive_root_dir, bird_name)
        gdrive_rhd_dir = gdrive_look_in(gdrive_bird_dir, "RHD")
        try:
            rhd_file = gdrive_look_in(gdrive_rhd_dir, "{}.rhd".format(session_id))
        except:
            raise Exception("Rhd file not found")
    elif lab == "seewiesen":
        gdrive_seewiesen_dir = gdrive_look_in(gdrive_root_dir, "Seewiesen")
        gdrive_bird_dir = gdrive_look_in(gdrive_seewiesen_dir, bird_name)
        gdrive_rhd_dir = gdrive_bird_dir
        try:
            rhd_file = gdrive_look_in(gdrive_rhd_dir, "{}.rhd".format(session_id))
        except:
            raise Exception("Rhd file not found")

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(stimuli_dir):
        os.makedirs(stimuli_dir)

    if os.path.exists(os.path.join(raw_dir, "{}.rhd".format(session_id))):
        logger.info("{} already downloaded. "
                    "Not downloading...".format(rhd_file["title"]))
    else:
        logger.info("Downloading {} to {}".format(rhd_file["title"], raw_dir))

        download_file(
            rhd_file,
            os.path.join(raw_dir, drive2local_by_lab[lab](rhd_file["title"]))
        )

    csv_string = ""
    if lab == "theunissen":
        gdrive_pyoperant_dir = gdrive_look_in(gdrive_bird_dir, "pyoperant")
    elif lab == "seewiesen":
        gdrive_pyoperant_dir = gdrive_bird_dir

    for pyoperant_file in pyoperant_files:
        if not pyoperant_file.endswith(".csv"):
            pyoperant_file = "{}.csv".format(pyoperant_file)

        try:
            csv_file = gdrive_look_in(gdrive_pyoperant_dir, pyoperant_file)
        except:
            logger.warning("File {} not found! "
                           "Skipping...".format(pyoperant_file))
            continue

        logger.info("Downloading {} to {}".format(csv_file["title"], raw_dir))
        download_file(csv_file, os.path.join(raw_dir, csv_file["title"]))
        with open(os.path.join(raw_dir, csv_file["title"]), "rb") as csvfile:
            csv_string += csvfile.read().decode("utf-8")

    # We can load the csvs and check which stimulus directories to download
    if pyoperant_files:
        if lab == "theunissen":
            gdrive_stimuli_dir = gdrive_look_in(gdrive_bird_dir, "stimuli")
        elif lab == "seewiesen":
            gdrive_stimuli_dir = gdrive_look_in(gdrive_seewiesen_dir, "Stimuli")
            gdrive_stimuli_dir = gdrive_look_in(gdrive_stimuli_dir, "Bird1")

        stimuli_folders = gdrive_listdir(gdrive_stimuli_dir)
        for stimuli_folder in stimuli_folders:
            # TODO (check if this is needed)
            if stimuli_folder["title"] in csv_string:
                download_to = os.path.join(
                        stimuli_dir,
                        stimuli_folder["title"])
                logger.info("Downloading {} to {}".format(
                    stimuli_folder["title"],
                    download_to
                ))
                download_folder(stimuli_folder, download_to)
