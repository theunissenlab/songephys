"""Module for functions to load and read pyoperant stimulus data
"""
import datetime
import os

import numpy as np
import pandas as pd
import scipy.io.wavfile


def _parse_julie_filename_metadata(fname):
    familiarity_codes = {"F": "familiar", "T": "tutor"}
    fname = os.path.basename(fname)
    metadata = dict()
    vals = fname.split("_")

    if len(vals) >= 3:
        metadata["emitter"] = vals[0]
        sex, age, fam = tuple(vals[1])
        metadata["source"] = familiarity_codes.get(fam, "unfamiliar")
        metadata["source_sex"] = sex.lower()
        metadata["callerAge"] = "adult" if age == "A" else "juvenile"
        metadata["callid"] = vals[2]

        # TODO (kevin) these should not be hardcoded
        metadata["type"] = "song" if vals[2] == "So" else "call"
        metadata["stim_class"] = "Con"

    return metadata


def _get_duration(wav_file):
    rate, data = scipy.io.wavfile.read(wav_file)
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
    df["stimulus_name"] = df["stimulus_name"].apply(
        lambda path: os.path.join(stim_loc, os.path.basename(path))
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


__all__ = ["convert_pyoperant_csvs"]
