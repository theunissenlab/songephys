"""Module for functions to load and read pyoperant stimulus data
"""
import datetime
import os

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


def _get_duration(wav_file):
    # rate, data = scipy.io.wavfile.read(wav_file)
    w = wavio.read(wav_file)
    rate, data = w.rate, w.data
    return data.shape[0] / np.float(rate)


def _transform_pyoperant_csv(df, stim_loc):
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
        filename = os.path.basename(path)
        dirname = os.path.basename(os.path.dirname(path))
        # print(os.path.join(stim_loc, "..", dirname, filename))
        return os.path.join(stim_loc, dirname, filename)

    df["stimulus_name"] = df["stimulus_name"].apply(
        locate_stimulus_directory
    )
    # Fill in duration column for dataframe by reading wav files
    df["duration"] = df["stimulus_name"].apply(_get_duration)

    metadata = df["stimulus_name"].apply(_parse_julie_filename_metadata)
    metadata = pd.DataFrame.from_records(metadata)

    return df[["duration", "stimulus_name", "trial", "time"]].join(metadata)


def read_pyoperant_csv(path, stim_loc):
    """Basically just load the csv"""
    df = pd.read_csv(path)
    print(path)
    print(len(df))
    return _transform_pyoperant_csv(df, stim_loc=stim_loc)


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


def convert_pyoperant_csvs(stimulus_codes, pyoperant_csvs, stim_dirs, dest_path):
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
        read_pyoperant_csv(filename, stim_dir)
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


def binary_converter(code, bits_per_char):
    """Coinvert binary code into character strings
    """
    num_vals = len(code) // bits_per_char
    vals = np.zeros(num_vals)
    for ii in range(num_vals):
        binvec = code[ii * bits_per_char: (ii + 1) * bits_per_char]
        vals[ii] = np.sum((2 ** np.arange(bits_per_char - 1, -1, -1)) * binvec)

    return "".join(map(chr, vals.astype(np.uint8)))


def convert_digital_to_metadata(digital_signal, code_length=384, bits_per_char=8):
    """Detect periods of digital codes and decode stimulus metadata

    Copied from zeebeez because of a problem sometimes where the digital code starts
    with "on" state instead of "off" state
    """
    sample_rate = digital_signal.annotations["sampling_rate"]
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
            print("Repetition is not an integer! Setting to 0.")
            raise Exception
            metadata["repetition"] = 0
        try:
            metadata["trial"] = int(code[12:16])
        except ValueError:
            print("Trial is not an integer! Setting to 0.")
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
    "convert_pyoperant_csvs",
    "convert_digital_to_metadata",
    "get_vocalizer_relation"
]
