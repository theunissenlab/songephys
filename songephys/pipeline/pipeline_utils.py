import glob
import os
import config
import pathlib
import tempfile

import pandas as pd
from luigi import Parameter

from songephys.format.download import downloaders
from songephys.format.intan_utils import timestamp_from_intan_filename
from songephys.format.io import load_yaml
from songephys.utils import intervals_overlap

import config


class SiteParameter(Parameter):
    def normalize(self, x):
        # Checks that the site directory is
        # 1. in a folder called 'sites'
        # 2. has no trailing slash
        # normpath removes the trailing slash
        x = os.path.normpath(x)
        containing_folder = os.path.dirname(x)
        if os.path.basename(containing_folder) != "sites":
            raise ValueError("The site parameter {} is not in a directory called sites".format(x))

        return x


class SessionParameter(Parameter):
    def normalize(self, x):
        # Checks that the session directory is
        # 1. in a folder called 'sessions'
        # 2. has no trailing slash
        # normpath removes the trailing slash
        x = os.path.normpath(x)
        containing_folder = os.path.dirname(x)
        if os.path.basename(containing_folder) != "sessions":
            raise ValueError("The session-dir parameter {} is not in a directory called sessions".format(x))

        return x


def group_raw_files(filelist):
    """Sorts the rhd files and csvs files from a single list

    The site filelist should be a plain text file with the first line being
    either "seewiesen" or "theunissen". This lab name will be used later to
    determine other parameters such as where to look for the download files.

    Params
    ======
    config (dict)
        path to the filelist. Should look like SITEPATH/filelist

    Returns
    =======
    A dictionary with the following keys:
        rhds (list):
            list of names of the rhd files
        csvs (list):
            list of string names of the csvs files
        lab (string):
            lab name found in the first line (either seewiesen or theunissen)
    """
    rhds = []
    csvs = []

    for line in filelist:
        line = line.strip()
        if line.endswith(".rhd"):
            rhds.append(line)
        elif line.endswith(".csv"):
            csvs.append(line)

    # Remove duplicates if they exist from filelist
    rhds = list(set(rhds))
    csvs = list(set(csvs))

    return {"rhds": rhds, "csvs": csvs}


def parse_filelist(filelist_path):
    """Extracts the rhd files, csvs files and lab name from a site filelist

    The site filelist should be a plain text file with the first line being
    either "seewiesen" or "theunissen". This lab name will be used later to
    determine other parameters such as where to look for the download files.

    Params
    ======
    filelist_path (string)
        path to the filelist. Should look like SITEPATH/filelist

    Returns
    =======
    A dictionary with the following keys:
        rhds (list):
            list of names of the rhd files
        csvs (list):
            list of string names of the csvs files
        lab (string):
            lab name found in the first line (either seewiesen or theunissen)
    """
    rhds = []
    csvs = []
    name = None

    # Read list of csvs, rhds, and the lab name from the first line
    with open(filelist_path, "r") as filelist:
        for line in filelist.readlines():
            line = line.strip()
            if name is None:
                valid_lines = ["theunissen", "seewiesen", "seewiesenCloud", "local", "test"]
                valid_lines += [name + "-2" for name in valid_lines]
                if line not in valid_lines:
                    raise ValueError("The first line of filelist must be "
                                     "either {}".format(", ".join(valid_lines)))
                name = line
                continue

            if line.endswith(".rhd"):
                rhds.append(line)
            elif line.endswith(".csv"):
                csvs.append(line)

    # Remove duplicates if they exist from filelist
    rhds = list(set(rhds))
    csvs = list(set(csvs))

    return {"rhds": rhds, "csvs": csvs, "lab": name}


def associate_pyoperant_csvs_to_rhds(csv_files, rhd_files, experiment_name, download_type):
    """Determine relative times of sessions from rhd_files and which csvs go with them

    Arguments
    ---------
    csv_files : List[str]
        List of csv files (basenames only). The downloader is responsible for locating these
        csv files based on the given ``experiment_name``.
    rhd_files : List[str]
        List of RHD files (basenames only). The downloader is responsible for locating these
        rhd files based on the given ``experiment_name``.
    experiment_name : str
        Name of experiment, used for organising data. This used to be the name of the subject
        until we started doing multi-subject recordings. Is used by the downloader to discover
        where data files are.
    download_type : str
        A key for a predefined procedures for locating and downloading our data files. The
        downloader should know where to find data based on the given ``experiment_name``,
        and valid downloader names can be found in songephys/code/format/download.py

    Returns
    -------
    sessions : List[dict]
        A list of dictionaries, representing sessions in chronological order. Each session dict
        has the following keys:
            id : str
                name of the session (name of the RHD without file extension)
            t : float
                time of session start relative to the first session in seconds
            playbacks : List[str]
                A list of csv files genearted by pyoperant that contain at least one
                trial found in the RHDs
    """
    # First remove duplicates
    rhd_files = list(set(rhd_files))
    csv_files = list(set(csv_files))

    # Get the timestamps from rhd filenames of each session
    rhd_times = [
        (timestamp_from_intan_filename(rhd), rhd)
        for rhd in rhd_files
    ]
    rhd_times = sorted(rhd_times)

    t0 = rhd_times[0][0]

    rhd_dicts = [
        {
            "id": os.path.splitext(rhd_name)[0],
            "t": (rhd_time - t0).total_seconds(),
        } for rhd_time, rhd_name in rhd_times
    ]

    # Download the csvs in a temporary directory.
    # These are used to determine what stimulus files must be downloaded.
    csv_start_stop = []

    with tempfile.TemporaryDirectory() as csv_temp_dir:
        downloaders[download_type].download_csvs(experiment_name, csv_files, csv_temp_dir)

        # Get the first and last trial time for each csv file
        for csv in csv_files:
            csv_data = pd.read_csv(
                    os.path.join(csv_temp_dir, csv),
                    parse_dates=True,
                    converters={"time": pd.to_datetime}
            )
            first = csv_data["time"].iloc[0]
            last = csv_data["time"].iloc[-1]

            csv_start_stop.append((csv, first, last))

        # Associate csvs to RHDs using timestamp overlap conditions that
        # are highly sophisticated and complex
        for csv, start, stop in csv_start_stop:
            for i in range(len(rhd_times) - 1):
                if intervals_overlap((start, stop), (rhd_times[i][0], rhd_times[i + 1][0])):
                    if "playbacks" not in rhd_dicts[i]:
                        rhd_dicts[i]["playbacks"] = [os.path.splitext(csv)[0]]
                    else:
                        rhd_dicts[i]["playbacks"].append(os.path.splitext(csv)[0])

            # Any playback csv files that start after the last rhd are associated
            # with the last rhd. I currently can't verify if those csvs actually have
            # trials that were played during the rhd because that would require checking
            # the duration of the RHD which is not yet downloaded.
            if rhd_times[-1][0] <= start or rhd_times[-1][0] <= stop:
                if "playbacks" not in rhd_dicts[-1]:
                    rhd_dicts[-1]["playbacks"] = [os.path.splitext(csv)[0]]
                else:
                    rhd_dicts[-1]["playbacks"].append(os.path.splitext(csv)[0])

    return rhd_dicts


def session_basename(filename):
    if filename.endswith("/"):
        filename = filename[:-1]
    return os.path.splitext(os.path.basename(filename))[0]


def get_session_sites(session_path):
    """Get all site directories containing the given session

    Params
    ======
    session_path (str):
        path to ephys session directory

    Returns
    =======
    Tuple of
    site_dirs (list):
        list of site paths whose configuration yaml files point to the given session
    pyoperants (list):
        list of pyoperant csv files that the session is associated with from the site configs
    """
    session_name = session_basename(session_path)
    prefix = session_name.split("_")[0]

    potential_site_dirs = glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(session_path)),
        "sites",
        "{}*".format(prefix)
    ))

    site_dirs = []
    pyoperants = []
    for site_dir in potential_site_dirs:
        try:
            session_config = load_yaml(os.path.join(site_dir, "site_config.yaml"))
        except:
            continue

        if "sessions" not in session_config:
            continue

        for session in session_config["sessions"]:
            if session["id"] == session_name:
                site_dirs.append(site_dir)
                pyoperants += session.get("playbacks", [])
                break

    return site_dirs, list(set(pyoperants))
