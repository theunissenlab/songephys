from __future__ import print_function
import os

from pydrive.drive import GoogleDrive

from auth import get_auth

import config


drive = GoogleDrive(get_auth())


def download_file(gdrive_file, download_to_path):
    """Download a file from google drive

    Params
    gdrive_file (pydrive file): file to download
    download_to_path (str): location on local filesystem to download data
    """
    gdrive_file.GetContentFile(download_to_path)


def search_for_file(search_term):
    """Get the file matching the given name"""
    file_list = drive.ListFile({
        "q": "title = '{}'".format(search_term)
    }).GetList()

    return file_list


def download_session(session_id, pyoperant_files=(,), stimulus_dir=None):
    """Download a recording session data into the proper location

    Creates directories in this structure if they don't exist yet
    `DATA_DIR/BIRD_NAME/SESSION_NAME/raw`
    and save rhd and pyoperant files here

    Params
    session_id (str): Recording session id in normal format (e.g. "GreYel3594M___170914_121518")
    pyoperant_files (list of str):
        Each element is a name of a pyoperant file (e.g. "GreYel3594M_trialdata_20170912164335.csv")
    """
    bird_name = session_id.split("_")[0]

    bird_dir = os.path.join(config.DATADIR, "birds", bird_name)
    session_dir = os.path.join(bird_dir, session_id)
    raw_dir = os.path.join(session_dir, "raw")

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    rhd_file = search_for_file("{}.rhd".format(session_id))

    if not len(rhd_file):
        raise Exception("Rhd file not found")
    elif os.path.exists(os.path.join(raw_dir, "{}.rhd".format(session_id))):
        print("Warning, rhd file already downloaded. Skipping...")
    else:
        rhd_file = rhd_file[0]
        print("Downloading {} to {}".format(rhd_file["title"], raw_dir))
        download_file(rhd_file, os.path.join(raw_dir, rhd_file["title"]))

    for pyoperant_file in pyoperant_files:
        if not pyoperant_file.endswith(".csv"):
            pyoperant_file = "{}.csv".format(pyoperant_file)

        csv_file = search_for_file(pyoperant_file)
        if not len(csv_file):
            print("Warning: file {} not found, skipping...".format(csv_file))
            continue

        csv_file = csv_file[0]
        print("Downloading {} to {}".format(csv_file["title"], raw_dir))
        download_file(csv_file, os.path.join(raw_dir, csv_file["title"]))

    # We can load the csvs and check which stimulus directories to download
    if not stimulus_dir:
        pass
    else:
        pass
        # TODO: download the stimulus data (wav file)

