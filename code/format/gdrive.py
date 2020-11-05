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

from gdrive_access import GDriveCommands
from gdrive_access.errors import CredentialsNotFound, NotFoundError

from format.lab_utils import drive2local_by_lab

import config


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


try:
    g = GDriveCommands(os.path.join(config.GDRIVE_CREDENTIALS_DIR, "settings.yaml"))
except CredentialsNotFound:
    root = None
    logger.warning("Google Drive authentication failed. Need to set up gdrive authentication:\n"
        "python -m gdrive_access.setup_credentials --dir {}".format(config.GDRIVE_CREDENTIALS_DIR))
else:
    root = g.get_root("Intan Ephys Data")


def download_csvs(
        site_id,
        csv_list,
        save_to=None,
        lab="theunissen"
    ):
    bird_name = site_id.split("_")[0]

    if lab == "theunissen":
        gdrive_pyoperant_dir = g.find(root, bird_name, "pyoperant")
    elif lab == "seewiesen":
        gdrive_pyoperant_dir = g.find(root, "Seewiesen", bird_name, "pyoperant")

    downloaded = []
    for pyoperant_file in csv_list:
        if not g.exists(gdrive_pyoperant_dir, pyoperant_file):
            logger.warning("File {} not found! "
                           "Skipping...".format(pyoperant_file))
            continue

        download_dest = os.path.join(save_to, csv_file["title"])
        g.download_file(csv_file, download_dest)
        downloaded.append(download_dest)

    print("Downloaded:", downloaded)
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
        lab (str): either seewiesen or theunissen
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
        bird_root = g.find(root, bird_name)
        pyoperant_root = g.find(bird_root, "pyoperant")
        stimuli_root = g.find(bird_root, "stimuli")
    elif lab == "seewiesen":
        bird_root = g.find(root, "Seewiesen", bird_name)
        pyoperant_root = bird_root
        stimuli_root = g.find(root, "Seewiesen", "Stimuli", "Bird1")

    try:
        rhd_file = g.find(bird_root, "RHD", "{}.rhd".format(session_id))
    except NotFoundError:
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

        g.download_file(
            rhd_file,
            os.path.join(raw_dir, drive2local_by_lab[lab](rhd_file["title"]))
        )

    csv_string = ""

    for pyoperant_file in pyoperant_files:
        if not pyoperant_file.endswith(".csv"):
            pyoperant_file = "{}.csv".format(pyoperant_file)

        try: 
            csv_file = g.find(pyoperant_root, pyoperant_file)
        except NotFoundError:
            logger.warning("File {} not found! "
                           "Skipping...".format(pyoperant_file))
            continue

        logger.info("Downloading {} to {}".format(csv_file["title"], raw_dir))
        g.download_file(csv_file, raw_dir)

        with open(os.path.join(raw_dir, csv_file["title"]), "rb") as csvfile:
            csv_string += csvfile.read().decode("utf-8")

    # We can load the csvs and check which stimulus directories to download
    if pyoperant_files:
        stimuli_foldres = g.ls(stimuli_root)
        for stimuli_folder in stimuli_folders:
            # TODO (check if this is needed)
            if stimuli_folder["title"] in csv_string:
                logger.info("Downloading {} to {}".format(
                    stimuli_folder["title"],
                    stimuli_dir
                ))
                g.download_folder(stimuli_folder, stimuli_dir)
