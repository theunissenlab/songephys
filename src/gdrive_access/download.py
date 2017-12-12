from __future__ import print_function
import os
import time

from pydrive.drive import GoogleDrive

from auth import get_auth

import config


drive = GoogleDrive(get_auth())

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
    time.sleep(0.1)
    file_list = drive.ListFile({
        "q": "title = '{}' and '{}' in parents and trashed = false".format(filename, dir["id"])
    }).GetList()
    if not len(file_list):
        raise Exception("{}/{} not found".format(dir["title"], filename))
    return file_list[0]


def gdrive_listdir(dir):
    time.sleep(0.1)
    return drive.ListFile({
        "q": "'{}' in parents and trashed = false".format(dir["id"])
    }).GetList()


def download_session(session_id, pyoperant_files=tuple()):
    """Download a recording session data into the proper location

    Creates directories in this structure if they don't exist yet
    `DATA_DIR/BIRD_NAME/SESSION_NAME/raw`
    and save rhd and pyoperant files here

    Params
    session_id (str): Recording session id in normal format (e.g. "GreYel3594M___170914_121518")
    pyoperant_files (list of str):
        Each element is a name of a pyoperant file (e.g. "GreYel3594M_trialdata_20170912164335.csv")
    """
    if session_id.endswith(".rhd"):
        session_id = os.path.splitext(session_id)[0]

    bird_name = session_id.split("_")[0]

    bird_dir = os.path.join(config.DATADIR, "birds", bird_name)
    session_dir = os.path.join(bird_dir, session_id)
    raw_dir = os.path.join(session_dir, "raw")
    stimuli_dir = os.path.join(session_dir, "stimuli")

    gdrive_bird_dir = gdrive_look_in(gdrive_root_dir, bird_name)
    gdrive_rhd_dir = gdrive_look_in(gdrive_bird_dir, "RHD")
    try:
        rhd_file = gdrive_look_in(gdrive_rhd_dir, "{}.rhd".format(session_id))
    except:
        raise Exception("Rhd file not found")

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(stimuli_dir):
        os.makedirs(stimuli_dir)

    if os.path.exists(os.path.join(raw_dir, "{}.rhd".format(session_id))):
        print("Warning, {} already downloaded. Not downloading...".format(rhd_file["title"]))
    else:
        print("Downloading {} to {}".format(rhd_file["title"], raw_dir))
        download_file(rhd_file, os.path.join(raw_dir, rhd_file["title"]))

    csv_string = ""
    gdrive_pyoperant_dir = gdrive_look_in(gdrive_bird_dir, "pyoperant")
    for pyoperant_file in pyoperant_files:
        if not pyoperant_file.endswith(".csv"):
            pyoperant_file = "{}.csv".format(pyoperant_file)

        try:
            csv_file = gdrive_look_in(gdrive_pyoperant_dir, pyoperant_file)
        except:
            print("Warning: file {} not found, skipping...".format(csv_file))
            continue

        print("Downloading {} to {}".format(csv_file["title"], raw_dir))
        download_file(csv_file, os.path.join(raw_dir, csv_file["title"]))
        with open(os.path.join(raw_dir, csv_file["title"]), "rb") as csvfile:
            csv_string += csvfile.read()

    # We can load the csvs and check which stimulus directories to download
    if pyoperant_files:
        gdrive_stimuli_dir = gdrive_look_in(gdrive_bird_dir, "stimuli")
        stimuli_folders = gdrive_listdir(gdrive_stimuli_dir)
        for stimuli_folder in stimuli_folders:
            # TODO (check if this is needed)
            if stimuli_folder["title"] in csv_string:
                download_to = os.path.join(stimuli_dir, stimuli_folder["title"])
                print("Downloading {} to {}".format(stimuli_folder["title"], download_to))
                download_folder(stimuli_folder, download_to)
