import os
import config
import tempfile

from format import gdrive

from utils import intervals_overlap


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
                if line not in ["theunissen", "seewiesen"]:
                    raise ValueError("The first line of filelist must be "
                                     "either 'theunissen' or 'seewiesen'")
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


def associate_pyoperant_csvs_to_rhds(site_dir, csvs, rhds, lab):
    """Maps pyoperant csvs to corresponding rhd files using trial times and rhd timestamp
    
    Will download csv files to a temporary directory so it can read in the trial times
    (lab parameter required to know where to download files from).
    
    Makes no assumption about the duration of each rhd file, just assumes that one rhd
    stops where the next one starts.
    
    Params
    ======
    site_dir (string):
        path of site being processed
    csvs (list):
        list of pyoperant csv file basenames
    rhds (list):
        list of rhd file basenames
    lab (string):
        name of lab (i.e seewiesen, theunissen)
    
    Returns
    =======
    List of dictionaries, each dictionary representing one session. Sorted in order
    of session time.
    
    Each dictionary has the following keys:
        id: name of session
        t: time of session start relative to first session
        playbacks: list of pyoperant csv files with trials occuring during this session
    """
    site_name = session_basename(site_dir)

    # Get the timestamps from rhd filenames of each session
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

    # Download the csvs in a temporary directory.
    # These are used to determine what stimulus files must be downloaded.
    csv_start_stop = []

    with tempfile.TemporaryDirectory() as csv_temp_dir:
        gdrive.download_csvs(site_name, csvs, csv_temp_dir, lab=lab)

        # Get the first and last trial time for each csv file
        for csv in csvs:
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

            # Any playback csv files that start after the last rhd should be associated
            # with the last rhd
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
        session_path,
        "..",
        "..",
        "sites",
        "{}*".format(prefix)
    ))
    
    site_dirs = []
    pyoperants = []
    lab_name = None
    for site_dir in potential_site_dirs:
        session_config = load_yaml(os.path.join(site_dir, "sessions.yaml"))
        for session in session_config["sessions"]:
            if session["id"] == session_name:
                lab_name = session_config.get("lab", "theunissen")
                site_dirs.append(site_dir)
                pyoperants += session.get("playbacks", [])
                break
    
    return site_dirs, pyoperants