from __future__ import print_function
import os

from pydrive.drive import GoogleDrive

from auth import get_auth

import config


drive = GoogleDrive(get_auth())


def download_file(gdrive_file, download_to_path):
    # get file by file_id
    gdrive_file.GetContentFile(download_to_path)


def search_for_file(search_term):
    file_list = drive.ListFile({
        "q": "title = '{}'".format(search_term)
    }).GetList()

    for file1 in file_list:
        print('title: %s, id: %s' % (file1['title'], file1['id']))

    return file_list


def download_session(session_id, pyoperant_files):
    bird_name = session_id.split("_")[0]

    bird_dir = os.path.join(config.DATADIR, "birds", bird_name)
    session_dir = os.path.join(bird_dir, session_id)
    raw_dir = os.path.join(session_dir, "raw")

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    '''
    rhd_file = search_for_file("{}.rhd".format(session_id))

    if not len(rhd_file):
        raise Exception("Rhd file not found")
    elif os.path.exists(os.path.join(raw_dir, "{}.rhd".format(session_id))):
        print("Warning, rhd file already downloaded. Skipping...")
    else:
        rhd_file = rhd_file[0]
        print("Downloading {} to {}".format(rhd_file["title"], raw_dir))
        download_file(rhd_file, os.path.join(raw_dir, rhd_file["title"]))
    '''

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

