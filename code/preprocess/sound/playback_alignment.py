"""Functions to help determine stimulus alignment
"""
import os
import datetime
from collections import defaultdict

import numpy as np
import wavio
from scipy.signal import correlate

from soundsig.signal import resample_signal


def _find_alignment(signal, template):
    """Find best index in signal to align template waveform
    """
    correlated = correlate(signal, template, mode="same")
    peak = np.argmax(np.abs(correlated))
    return correlated, peak - len(template) // 2


def compute_realigned_trial_times(
        playback_df,
        intan_start_time,
        intan_lazy_mic,
        window=2
    ):
    """Compute the stimulus start times (relative to intan data) of trials

    The pyoperant software controls stimulus playbacks, and uses timestamps
    separately from the Intan RHD software which records microphone and ephys
    data. Normally, pyoperant sends a digital pulse to the intan board for
    alignment purposes. If these pulses are missing (due to error in configuration)
    or corrupted (other bugs), we must find the best alignment by matching
    the audio stimuli to the microphone signal recorded.

    For each trial, computes a best alignment to the Intan data, then sets the
    offset for that trial equal to that best alignment.

    Parameters
    ----------
    playback_df : pandas.DataFrame
        Dataframe generated from the pyoperant csv file(s), each row is one trial.
        Must have columns 'stimulus_name' (local path to stimulus file), 'time'
        (trial timestamp), 'duration' (duration of stimulus in seconds)
    intan_start_time : datetime.datetime
        Datetime estimate of the Intan start time. Used as a starting point
        for alignment (probably get this from the RHD filename if no better source)
    intan_lazy_mic : format.lazy.LazySignal
        Lazy loaded signal for microphone data for the session from the RHD
        file.
    window : float
        Half-width of window (in seconds) to look for alignment peak

    Returns
    -------
    An array of floating point numbers with same length as playback_df, where
    the nth value is the start time for the nth trial in seconds from start
    of Intan recording.
    """
    new_start_times = []
    shifts = []

    intan_sr = intan_lazy_mic.sampling_rate 

    _loaded_stim_cache = {}
    for row_idx, row in playback_df.iterrows():
        if row.stimulus_name in _loaded_stim_cache:
            wav_data = _loaded_stim_cache[row.stimulus_name]
        else:
            wav_object = wavio.read(row.stimulus_name)

            stim_sr = wav_object.rate
            wav_data = wav_object.data[:, 0]
            if stim_sr != intan_sr:
                _, wav_data  = resample_signal(wav_data, stim_sr, intan_sr)
            _loaded_stim_cache[row.stimulus_name] = wav_data

        estimated_start = (row.time - intan_start_time).total_seconds()
        if estimated_start < 0 or estimated_start > intan_lazy_mic.N / intan_lazy_mic.sampling_rate:
            new_start_times.append(estimated_start)
            continue

        recorded_data = intan_lazy_mic.time_slice(
            estimated_start - window,
            estimated_start + row.duration + window
        ).data

        recorded_data = recorded_data - np.mean(recorded_data, axis=0)

        best_score = 0
        best_estimate_index = None
        for i in np.arange(recorded_data.shape[1]):
            correlated, estimated_index = _find_alignment(recorded_data[:, i], wav_data)
            score = np.max(np.abs(correlated) / np.max(np.abs(recorded_data[:, i])))
            if score > best_score:
                best_estimate_index = estimated_index
        
        new_start_times.append((estimated_start - window) + best_estimate_index/ intan_sr)
        shifts.append(new_start_times[-1] - estimated_start)

    print("Offsetting playbacks by best correlation\n"
            "On average, shifted by {:.2f}".format(np.mean(shifts)))

    return new_start_times


def _calculate_playback_repetitions(playbacks_df):
    """Compute the repetition number for each stimulus

    Parameters
    ----------
    playbacks_df : pandas.DataFrame
        DataFrame with column 'stimulus_name' as a unique identifier (e.g.
        file path) for each trial

    Returns
    -------
    A numpy array of integers of the same length as playbacks_df, where
    the nth entry is the repetition number for the nth trial.
    """
    repetition = []

    counts = defaultdict(int)
    for i, row in playbacks_df.iterrows():
        repetition.append(counts[row.stimulus_name])
        counts[row.stimulus_name] += 1

    return np.array(repetition)


def create_replacement_codes(playbacks_df, trial_times, sampling_rate, max_time):
    """Generate an list of codes mimicing the digital codes sent by pyoperant

    Mimics the format output by
    preprocess.sound.playback_utils.convert_digital_to_metadata

    Parameters
    ----------
    playback_df : pandas.DataFrame
        Dataframe generated from the pyoperant csv file(s), each row is one trial.
        Must have columns 'stimulus_name' (local path to stimulus file),
        'duration' (stimulus length in seconds), and 'trial' (integer trial #)
    trial_times : array
        Array of floats for trial start times (probably produced by
        _calculate_playback_repetitions)
    sampling_rate : float
        Sampling rate of data
    max_time : float
        Maximum time (in seconds) of Intan data file. Function will exclude
        codes for trials starting before 0 or ending after max_time

    Returns
    -------
    A list of dictionaries, where each dictionary has the keys 'action',
    'object', 'repetition', and 'trial'
    """
    repetition = _calculate_playback_repetitions(playbacks_df)

    codes = []
    for i, row in playbacks_df.iterrows():
        if trial_times[i] < 0 or trial_times[i] + row.duration > max_time:
            continue

        codes.append((
            int(trial_times[i] * sampling_rate),
            {
                "action": "queu",
                "object": "Spea",
                "repetition": repetition[i],
                "trial": row.trial
            }
        ))

    return codes
