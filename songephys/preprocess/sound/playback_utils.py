"""Module for functions to load and read pyoperant stimulus data
"""
import datetime
import os
import re

import numpy as np
import pandas as pd
import scipy.io.wavfile
import wavio
from pathlib import PureWindowsPath

def _parse_julie_filename_metadata(fname):
    """Er

    MAT - male adult tutor
    MAF - male adult familiar
    FAM - female adult mother
    MAU - male adult unfamiliar
    MJU - male juvenile unfamiliar
    MJB - male juvenile brother
    MJP - male juvenile pupil
    """
    familiarity_codes = {
        "F": "familiar",
        "T": "tutor",
        "M": "mother",
        "U": "unfamiliar",
        "P": "pupil",
        "B": "brother"
    }
    fname = os.path.basename(fname)
    metadata = dict()
    vals = fname.split("_")

    def _is_right_format(vals):
        try:
            l = len(tuple(vals[1]))
        except ValueError:
            return False
        else:
            return l == 3

    if len(vals) >= 3 and _is_right_format(vals):
        metadata["emitter"] = vals[0]
        sex, age, fam = tuple(vals[1])
        metadata["source"] = familiarity_codes.get(fam, "unfamiliar")
        metadata["source_sex"] = sex.lower()
        metadata["callerAge"] = "adult" if age == "A" else "juvenile"
        metadata["callid"] = vals[2]

        # TODO (kevin) these should not be hardcoded
        metadata["type"] = "song" if vals[2] == "So" else "call"
        metadata["stim_class"] = "Con"
    elif len(vals) == 4 and len(vals[1]) == 2 and vals[3].startswith("fs"):
        metadata["emitter"] = vals[0]
        metadata["source"] = "unknown"
        metadata["source_sex"] = None
        metadata["callerAge"] = None
        metadata["callid"] = vals[1]
        metadata["type"] = "song" if vals[1].lower() == "so" else "call"
        metadata["stim_class"] = "Con"
    else:
        # This is likely not a standard format
        metadata["emitter"] = vals[0]
        metadata["source"] = "unfamiliar"
        metadata["source_sex"] = None
        metadata["callerAge"] = None
        metadata["callid"] = None
        metadata["type"] = "song" if vals[0].startswith("stim") else "ripple" if vals[0].startswith("randripple") else None
        metadata["stim_class"] = "Con" if vals[0].startswith("stim") else "Noise" if vals[0].startswith("randripple") else None

    return metadata

def _parse_filename_bird_type_data(fname):
    # all we want to find is the CallType and the bird name
    fname = os.path.basename(fname)
    metadata = {"type":None,"source":None}
    vals = fname.split("_")
    # TODO fill this in
    calltypes = ["So","DC", "LT", "Te", "Tu", "Ag"]
    for val in vals:
        if val in calltypes:
            metadata['type'] = val
        if re.match('^[A-Za-z]{6}\d{4}$',val) is not None:
            metadata['source'] = val
    return metadata

def _parse_ladder_filename_data(fname):
    fname = os.path.basename(fname)
    fname = os.path.splitext(fname)[0]
    metadata = dict()
    vals = fname.split("_")

    if fname.startswith("randripple"):
        metadata["key"] = "{} {}".format(vals[0], "ripple")
        metadata["emitter"] = vals[0]
        metadata["callid"] = "ripple"
        metadata["type"] = "ripple"
        metadata["source_sex"] = None
        metadata["source"] = "unfamiliar"
        metadata["stim_class"] = "Noise"
    elif fname.endswith("Familiar"):
        metadata["key"] = "{} {}".format(vals[3], vals[0])
        metadata["emitter"] = vals[3]
        metadata["callid"] = vals[0]
        metadata["type"] = "song" if vals[0].lower() == "so" else "dc"
        metadata["source_sex"] = None
        metadata["source"] = "familiar"
        metadata["stim_class"] = "Call"
    elif fname.endswith("Unfamiliar"):
        metadata["key"] = "{} {}".format(vals[3], vals[0])
        metadata["emitter"] = vals[3]
        metadata["callid"] = vals[0]
        metadata["type"] = "song" if vals[0].lower() == "so" else "dc"
        metadata["source_sex"] = None
        metadata["source"] = "unfamiliar"
        metadata["stim_class"] = "Call"
    elif fname.endswith("SHUFFLED"):
        metadata["key"] = "{} shuffled_so".format(vals[3])
        metadata["emitter"] = vals[3]
        metadata["callid"] = vals[0]
        metadata["type"] = "shuffled_song"
        metadata["source_sex"] = None
        metadata["source"] = "test"
        metadata["stim_class"] = "Ladder"
    else:
        metadata["key"] = "{} {}".format(vals[3], vals[0])
        metadata["emitter"] = vals[3]
        metadata["callid"] = vals[0]
        metadata["type"] = "song" if vals[0].lower() == "so" else "dc"
        metadata["source_sex"] = None
        metadata["source"] = "test"
        metadata["stim_class"] = "Ladder"

    return metadata



def _transform_pyoperant_playback_csv(df, stim_loc):
    """Transform dataframe into correct datatypes and derive columns

    New columns:
    emitter, source, source_sex, callerAge, callid, type, stim_class, duration

    Params
    df (pd.DataFrame)
    stim_loc (str): path to stimulus directory
    """
    df = df.rename(columns={"index": "trial"})

    def _convert_datetime(t):
        try:
            return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
        except:
            return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

    df["time"] = df["time"].apply(_convert_datetime)
    df["stimulus_name"] = df["stimulus_name"].apply(
        lambda path: path.replace("\\", "/")
    )

    def locate_stimulus_directory(path):
        # Starts with the basename of the path, and goes up until it finds a file
        # that exists in stim_dir
        # TODO make sure this works on other projects
        path, relative_path_of_stim = os.path.split(path)
        path_of_stim = os.path.join(stim_loc,relative_path_of_stim)
        while not os.path.exists(path_of_stim) and path != '':
            path, basename = os.path.split(path)
            relative_path_of_stim = os.path.join(basename, relative_path_of_stim)
            path_of_stim = os.path.join(stim_loc,relative_path_of_stim)
        if path == '':
            raise IOError("Got to end of path without finding the file in stim_dir")
        return path_of_stim

    df["stimulus_name"] = df["stimulus_name"].apply(
        locate_stimulus_directory
    )

    # Fill in duration column for dataframe by reading wav files
    def _get_duration(wav_file):
        # rate, data = scipy.io.wavfile.read(wav_file)
        w = wavio.read(wav_file)
        rate, data = w.rate, w.data
        return data.shape[0] / np.float(rate)
    df["duration"] = df["stimulus_name"].apply(_get_duration)

    metadata = df["stimulus_name"].apply(_parse_julie_filename_metadata)
    metadata = pd.DataFrame.from_records(metadata)

    return df[["duration", "stimulus_name", "trial", "time"]].join(metadata)


def read_pyoperant_playback_csv(path, stim_loc):
    """Basically just load the csv"""
    df = pd.read_csv(path)
    return _transform_pyoperant_playback_csv(df, stim_loc=stim_loc)


def save_pyoperant_db(df, dest_path):
    """Pickle dataframe"""
    df.to_pickle(dest_path)


def group_digital_code_trials(digital_codes):
    """Group digital codes by the pyoperant session

    In one recording session, there may be multiple pyoperant sessions.
    We want to distinguish the digital codes sent during one pyoperant
    session from another. To do this, we can look at when the trial number
    communicated in the digital code resets

    Returns a list of arrays, the nth array contains the trial indices
    of the nth pyoperant session
    """
    result = []
    last_trial = np.inf
    for code in digital_codes:
        trial = code[1]["trial"]
        if trial < last_trial:
            result.append([])
        result[-1].append(trial)
        last_trial = trial

    return [np.array(r) for r in result]


def convert_pyoperant_playback_csvs(stimulus_codes, pyoperant_csvs, stim_dirs, dest_path):
    """Convert one or more pyoperant csvs to a pickled dataframe

    1. Combine csvs by time column, and reindex column
    2. save as a pickled dataframe

    Params
    pyoperant_csvs (list of str): list of pyoperant csv paths to combine
    stim_dirs (list of str): list of paths to directories containing
        the new wavs for each pyoperant csv
    """
    trial_set = group_digital_code_trials(stimulus_codes)

    pyoperant_dataframes = [
        read_pyoperant_playback_csv(filename, stim_dir)
        for filename, stim_dir in zip(pyoperant_csvs, stim_dirs)
    ]

    if not (len(pyoperant_dataframes) == len(trial_set)):
        raise Exception("Different number of pyoperant csvs ({}) than in digital signal ({})".format(len(pyoperant_dataframes), len(trial_set)))

    pyoperant_dataframes = [
        df.loc[np.isin(df["trial"], trials)]
        for df, trials in zip(pyoperant_dataframes, trial_set)
    ]

    pyoperant_lengths = [len(df) for df in pyoperant_dataframes]
    digital_lengths = [len(trials) for trials in trial_set]

    if not all([a == b for a, b in zip(pyoperant_lengths, digital_lengths)]):
        raise Exception("Wrong number of trials detected in pyoperants and digital signal: {} and {}".format(pyoperant_lengths, digital_lengths))

    new_df = pd.concat(pyoperant_dataframes)
    new_df = new_df.sort_values("time")
    new_df.index = np.arange(len(new_df))

    save_pyoperant_db(new_df, dest_path)


## Operant task 
def _transform_pyoperant_operant_csv(df, stim_loc):
    """Transform dataframe into correct datatypes and derive columns

    New columns:
    source, type, duration

    Params
    df (pd.DataFrame)
    stim_loc (str): path to stimulus directory
    """
    df = df.rename(columns={"index": "trial"})

    def _convert_datetime(t):
        try:
            return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
        except:
            return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

    df["time"] = df["time"].apply(_convert_datetime)
    df["stimulus_name"] = df["stimulus_name"].apply(
        lambda path: path.replace("\\", "/")
    )

    def locate_stimulus_directory(path):
        # Starts with the basename of the path, and goes up until it finds a file
        # that exists in stim_dir
        # TODO make sure this works on other projects
        path, relative_path_of_stim = os.path.split(path)
        path_of_stim = os.path.join(stim_loc,relative_path_of_stim)
        while not os.path.exists(path_of_stim) and path != '':
            path, basename = os.path.split(path)
            relative_path_of_stim = os.path.join(basename, relative_path_of_stim)
            path_of_stim = os.path.join(stim_loc,relative_path_of_stim)
        if path == '':
            raise IOError("Got to end of path without finding the file in stim_dir")
        return path_of_stim

    df["stimulus_name"] = df["stimulus_name"].apply(
        locate_stimulus_directory
    )
    # Fill in duration column for dataframe by reading wav files
    def conv_to_sec(x):
        return float(x.split(":")[-1])
    mask = df['rt'].isnull()
    df.loc[~mask,'rt'] = df.loc[~mask,'rt'].apply(conv_to_sec)
    # TODO check that this number is right... it could be a stim timeout
    df.loc[mask,'rt'] = df['max_wait'][mask]

    df['duration'] = df['rt']



    metadata = df["stimulus_name"].apply(_parse_filename_bird_type_data)
    metadata = pd.DataFrame.from_records(metadata)

    return df[["duration", "stimulus_name", "trial", "time", "condition_name", "response", "correct", "reward"]].join(metadata)


def read_pyoperant_operant_csv(path, stim_loc):
    """Basically just load the csv"""
    df = pd.read_csv(path)
    return _transform_pyoperant_operant_csv(df, stim_loc=stim_loc)

def convert_pyoperant_operant_csvs(stimulus_codes, pyoperant_csvs, stim_dirs, dest_path):
    """Convert one or more pyoperant csvs to a pickled dataframe

    1. Combine csvs by time column, and reindex column
    2. save as a pickled dataframe

    Params
    pyoperant_csvs (list of str): list of pyoperant csv paths to combine
    stim_dirs (list of str): list of paths to directories containing
        the new wavs for each pyoperant csv
    """
    trial_set = group_digital_code_trials(stimulus_codes)

    pyoperant_dataframes = [
        read_pyoperant_operant_csv(filename, stim_dir)
        for filename, stim_dir in zip(pyoperant_csvs, stim_dirs)
    ]

    if not (len(pyoperant_dataframes) == len(trial_set)):
        raise Exception("Different number of pyoperant csvs ({}) than in digital signal ({})".format(len(pyoperant_dataframes), len(trial_set)))

    pyoperant_dataframes = [
        df.loc[np.isin(df["trial"], trials)]
        for df, trials in zip(pyoperant_dataframes, trial_set)
    ]

    pyoperant_lengths = [len(df) for df in pyoperant_dataframes]
    digital_lengths = [len(trials) for trials in trial_set]

    if not all([a == b for a, b in zip(pyoperant_lengths, digital_lengths)]):
        raise Exception("Wrong number of trials detected in pyoperants and digital signal: {} and {}".format(pyoperant_lengths, digital_lengths))
    
    # TODO We may want to use the stim code time as the column here instead of the pyop time
    new_df = pd.concat(pyoperant_dataframes)
    new_df = new_df.sort_values("time")
    new_df.index = np.arange(len(new_df))

    save_pyoperant_db(new_df, dest_path)

def create_digital_events_df( digital_epochs, dest_path ):
    # time, duration, name
    times = []
    durations = []
    names = []
    def _convert_to_seconds(epoch_time):
        return epoch_time.item()
    for event_type in digital_epochs:
        events = digital_epochs[event_type]
        times.extend(list(map(_convert_to_seconds,events.times)))
        durations.extend(list(map(_convert_to_seconds,events.durations)))
        names.extend([event_type]*len(events))
    df = pd.DataFrame({"start_time": times, "duration": durations, "event_name": names})
    save_pyoperant_db(df, dest_path)

def fix_stimulus_codes(stim_codes, csv_files, stim_dirs, sample_rate, session_starttime, session_offset):
    # This is only used in the case that you dont output stimulus codes
    csv_starts = []
    csv_dfs = []
    trial_groups = []
    for csv, stim_dir in zip(csv_files, stim_dirs):
        start = csv.strip("_trialdata.csv").split('_')[-1]
        csv_start = datetime.datetime.strptime(start, "%Y%m%d%H%M%S")
        csv_starts.append(csv_start)
        csv_dfs.append(read_pyoperant_operant_csv(csv,stim_dir))
        trial_groups.append([])

    for ix, stim_code in enumerate(stim_codes):
        # save datetimes in stim codes for easy access later
        stim_code[1]['datetime'] = session_starttime + session_offset + datetime.timedelta(seconds=stim_code[0] / sample_rate)
        
        # get the smallest positive delta for stim_date - csv_start
        deltas = np.asarray([(stim_code[1]['datetime'] - st).total_seconds() for st in csv_starts])
        deltas[deltas < 0] = np.inf

        trial_groups[np.argmin([deltas])].append(ix)
    
    for csv_df,trial_group in zip(csv_dfs,trial_groups):
        # get approx first trial
        min_stim_code = stim_codes[trial_group[0]]
        min_ind = np.argmin(np.abs(csv_df['time'] - min_stim_code[1]['datetime']))
        min_trial = csv_df.loc[min_ind,'trial']

        # get all datetimes
        stim_datetimes = [stim_codes[sc][1]['datetime'] for sc in trial_group]

        def calc_errors(stim_datetimes, df_datetimes):
            errors = [np.abs(sd-dfd) for (sd,dfd) in zip(stim_datetimes,df_datetimes)]
            return np.sum(errors)
        

        possible_inds = [min_trial]
        if min_trial > 1:
            possible_inds.append(min_trial-1)
        if min_trial+len(trial_group) <= max(csv_df.trial):
            possible_inds.append(min_trial+1)

        if len(possible_inds) == 1:
            starting_ind = possible_inds[0]
        else:
            errors = []
            for start_ind in possible_inds:
                trial_range = range(start_ind,start_ind+len(trial_group))
                df_datetimes = [csv_df.loc[csv_df.trial == t,'time'].iloc[0] for t in trial_range]
                errors.append(calc_errors(stim_datetimes,df_datetimes))
            
            starting_ind = possible_inds[np.argmin(errors)]
        trial_range = range(starting_ind,starting_ind+len(trial_group))
        for t,final_trial in zip(trial_group,trial_range):
            stim_codes[t][1]['trial'] = final_trial

    return stim_codes

def calculate_event_timedelta(ttl_times, event_logs, session_starttime, ttl_pulse_width=10.0):
    # there is an offset between the timedeltas calculated from sampletime + file starttime
    # when compared to pyoperant computer time. This is probably from some small delay from filestart to actually starting 
    log_starts = []
    log_dfs = []
    for log in event_logs:
        start = log.strip("_events.log").split('_')[-1]
        log_start = datetime.datetime.strptime(start, "%Y%m%d%H%M%S")
        log_starts.append(log_start)
        log_df = pd.read_csv(log, delimiter='\t',names=['time','label','action','etc'])
        log_df['time'] = pd.to_datetime(log_df['time'], format="%Y-%m-%d %H:%M:%S.%f")
        log_df = log_df.loc[log_df['label'] == 'TTLMonitor']
        log_dfs.append(log_df)
    
    timedeltas = []
    for ttl_time in ttl_times:
        # save datetimes in stim codes for easy access later
        ttl_abstime = session_starttime + datetime.timedelta(seconds=ttl_time.item())

        # get the smallest positive delta for stim_date - csv_start
        deltas = np.asarray([(ttl_abstime - st).total_seconds() for st in log_starts])
        deltas[deltas < 0] = np.inf
        log_df = log_dfs[np.argmin([deltas])]

        # now get the smallest delta within that csv
        smallest_delta = log_df.iloc[np.argmin(np.abs(log_df['time'] - ttl_abstime))]['time'] - ttl_abstime
        if np.abs(smallest_delta.seconds) > ttl_pulse_width/2.:
            # If we cant find the TTL log code for this then we shouldnt record the time.
            # the TTL pulses go on during the break between operant tasks, so this is quite common
            continue
        else:
            timedeltas.append(smallest_delta)
    return np.mean(timedeltas)

# End Operant task

def binary_converter(code, bits_per_char):
    """Coinvert binary code into character strings
    """
    num_vals = len(code) // bits_per_char
    vals = np.zeros(num_vals)
    for ii in range(num_vals):
        binvec = code[ii * bits_per_char: (ii + 1) * bits_per_char]
        vals[ii] = np.sum((2 ** np.arange(bits_per_char - 1, -1, -1)) * binvec)

    return "".join(map(chr, vals.astype(np.uint8)))


def convert_digital_to_metadata(digital_signal, sample_rate, code_length=384, bits_per_char=8):
    """Detect periods of digital codes and decode stimulus metadata

    Copied from zeebeez because of a problem sometimes where the digital code starts
    with "on" state instead of "off" state
    """
    #sample_rate = digital_signal.annotations["sampling_rate"]
    on_times = np.array(np.round(digital_signal.times * sample_rate)).astype("int")
    # There is a certain bug where the pulses start in the on state
    # and the first pulse is missed.
    gaps = np.hstack([np.array(np.inf), np.diff(on_times)])
    starts = on_times[gaps > code_length]
    # If starts[0] is 0, then we probably have that bug.
    # Instead, start on the first time that we see a "off"
    if starts[0] == 0:
        starts[0] = int(np.argmax(np.diff(on_times) > 1.0))

    codes = list()
    for ii, start in enumerate(starts):
        metadata = dict()
        signal = np.zeros(code_length, dtype=np.int8)
        on_bits = on_times[((on_times - start) > 0) & ((on_times - start) < code_length)] - start - 1
        signal[on_bits] = 1
        code = binary_converter(signal, bits_per_char)
        metadata["code"] = code
        metadata["object"] = code[:4]
        metadata["action"] = code[4:8]
        try:
            metadata["repetition"] = int(code[8:12])
        except ValueError:
            #print("Repetition is not an integer! Setting to 0.")
            #raise Exception
            metadata["repetition"] = 0
        try:
            metadata["trial"] = int(code[12:16])
        except ValueError:
            #print("Trial is not an integer! Setting to 0.")
            metadata["trial"] = 0
        metadata["md5"] = code[16:]
        codes.append((start, metadata))

    return codes


def get_vocalizer_relation(annotations):
    familiarity = annotations.get("source")

    if familiarity == "tutor":
        return "tutor"
    elif familiarity == "pupil":
        return "subject"
    elif familiarity == "familiar":
        if annotations.get("source_sex") == "m":
            return "con_male_familiar"
        elif annotations.get("source_sex") == "f":
            return "con_female_familiar"
    elif familiarity == "unfamiliar":
        if annotations.get("source_sex") == "m":
            return "con_male_unfamiliar"
        elif annotations.get("source_sex") == "f":
            return "con_female_unfamiliar"
    elif familiarity == "mother":
        return "mother"
    elif familiarity == "brother":
        return "brother"
    else:
        return "unfamiliar"


__all__ = [
    "convert_pyoperant_playback_csvs",
    "convert_digital_to_metadata",
    "get_vocalizer_relation"
]
