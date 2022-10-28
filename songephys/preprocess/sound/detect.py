import os

import numpy as np
import pandas as pd
import quantities as pq
import scipy
import umap
from hdbscan import HDBSCAN
from sklearn.neighbors import LocalOutlierFactor


from soundsig.sound import spectrogram
from soundsig.signal import highpass_filter, lowpass_filter, bandpass_filter

from preprocess.sound.playback_utils import (
    convert_digital_to_metadata,
    get_vocalizer_relation
)
from utils import intervals_overlap

import config


playback_columns = ["name", "description", "sound_type", "vocalization_type", "playback_file", "stim_id", "start_time", "duration"]
autodetected_columns = ["description", "name", "playback_file", "sound_type", "vocalization_type",
            "vocalizer", "start_time", "duration", "label"]


def get_amplitude_envelope(sound, fs=30000.0, lowpass=5000.0, highpass=2000.0):
    streams_filtered_env = highpass_filter(sound.T, fs, highpass).T

    # Rectify and lowpass filter
    streams_filtered_env = np.abs(lowpass_filter(streams_filtered_env.T, fs, lowpass).T)
    streams_filtered_env = lowpass_filter(streams_filtered_env.T, fs, 100).T
    return streams_filtered_env


def compute_smart_threshold(amp_env, sampling_rate=30000.0, multiple=3):
    """Compute a threshold by detecting baseline periods

    A chunk of audio may be mostly silence or may be mostly vocalizations
    with few or no moments of silence available. This makes it difficult to
    define a detection threshold that is robust across many windows.

    This function identifies a baseline that is robust to the overall statistics
    of sound in the window (should generally work when there is at least
    1/10th of a second of silence between vocalizations) by assuming that
    the 0.1s bin with the least variance in peak to peak distance is near
    the silence baseline.

    Parameters
    ==========
    amp_env : numpy array of floats > 0 representing the amplitude envelope of the signal
    sampling_rate (float): sampling rate of the amp env signal
    multiple : How many multiples of the std of the signal to add to the baseline
    """
    high_peaks = scipy.signal.find_peaks(amp_env)[0]
    low_peaks = scipy.signal.find_peaks(-amp_env)[0]
    
    if len(high_peaks) == len(low_peaks):
        peak_idx = high_peaks[:-1]
        peak2peak = np.abs(amp_env[high_peaks[:-1]] - amp_env[low_peaks[:-1]])
    elif len(high_peaks) > len(low_peaks):
        peak_idx = high_peaks[1:-1]
        peak2peak = np.abs(amp_env[high_peaks[1:-1]] - amp_env[low_peaks[1:]])
    elif len(low_peaks) > len(high_peaks):
        peak_idx = high_peaks[:-1]
        peak2peak = np.abs(amp_env[high_peaks[:-1]] - amp_env[low_peaks[1:-1]])
        
    dsamp = int(sampling_rate / 20.0)
    
    windows = []
    for i in range(0, len(amp_env), dsamp):
        windows.append((peak_idx >= i) & (peak_idx < i + dsamp))

    # If there is only one window... lets just return the mean
    if len(windows) == 1:
        return np.mean(amp_env) + multiple * np.std(amp_env)

    all_means = np.array([
        np.mean(peak2peak[window]) if len(peak2peak[window]) > 1 else np.inf
        for window in windows
    ])
    all_stds = np.array([
        np.std(peak2peak[window]) if len(peak2peak[window]) > 1 else np.inf
        for window in windows
    ])

    # Assume the baseline is in the bottom 5% of the data
    mean_cutoff = np.percentile(all_means, 5)

    _best = np.argmin(all_stds[all_means < mean_cutoff])
    best_window = np.where(all_means < mean_cutoff)[0][_best]
    
    all_good_windows = [
        i for i in range(len(windows)) if
        np.abs(all_means[i] - all_means[best_window]) < 2 * all_stds[best_window]
    ]
    
    if not len(all_good_windows):
        return np.mean(amp_env) + multiple * np.std(amp_env)
    
    dat = np.concatenate([amp_env[peak_idx[windows[i]]] for i in all_good_windows])
    
    thresh = np.mean(dat) + multiple * np.std(dat)
    
    return thresh


def check_if_any_interval_overlaps(test_interval, check_intervals):
    """Checks if test_interval (t1, t2) overlaps any interval in check_intervals

    NOTE: only is guaranteed to be correct for series of check_intervals that are sorted and non-overlapping themselves

    This function was written to check if a given interval for a
    potential live vocalization overlapped any known playback periods.

    Params
    ======
    test_interval (tuple): t1, t2 tuple of interval to check
    check_intervals (list of tuples):
        list of tuples of (t_i[0], t_i[1]) to check if interval overlaps.
        check_intervals should be sorted and nonoverlapping. Otherwise the
        correctness of this function is not guaranteed

    Returns
    =======
    Returns a boolean value True if test_interval is found to
    overlap with an interval in check_intervals, otherwise
    False
    """
    t1, t2 = test_interval
    check_intervals = np.array(check_intervals)
    if not len(check_intervals):
        return False

    idx = np.searchsorted(check_intervals[:, 0], t1)

    if idx != 0 and intervals_overlap(test_interval, check_intervals[idx - 1]):
        return True
    elif idx < len(check_intervals) and intervals_overlap(test_interval, check_intervals[idx]):
        return True
    else:
        return False


def find_live_periods(lazy_mic, default_threshold, default_threshold_std, window=5):
    """Detect all periods of sound, learning toward grouping up close together sounds

    Returns a list of tuples with (t_start, t_stop) where t_start and t_stop
    are floating point values in seconds.
    """
    last_interval_to_check = None
    all_intervals = []

    # Iterate over 5s time windows
    for t in np.arange(0, lazy_mic.N / lazy_mic.sampling_rate, window):
        end_time = min(t + window, lazy_mic.N, lazy_mic.sampling_rate)

        # Load the sound from this time window, filter it,
        # and make an amplitude envelope from the mean of the two channels
        sliced = lazy_mic.time_slice(t, end_time)
        t_arr = sliced.t
        sig = sliced.data
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, lazy_mic.sampling_rate, 1000, 8000).T
        amp_env = get_amplitude_envelope(
            sig,
            lazy_mic.sampling_rate,
            highpass=2000,
            lowpass=8000)
        amp_env = np.mean(amp_env, axis=1)

        # Select the threshold. If its too large relative to the default threshold
        # then use the default threshold instead
        threshold = compute_smart_threshold(
            amp_env,
            sampling_rate=lazy_mic.sampling_rate
        )

        if threshold > default_threshold + default_threshold_std:
            threshold = default_threshold

        # Locate intervals by array index where the sound power exceeds the threshold
        # using parameters optimized to group up multiple events in longer
        # chunks (defined by the fuse_duration as 0.5 seconds)
        intervals = threshold_events(
            amp_env,
            threshold,
            polarity=1,
            sampling_rate=lazy_mic.sampling_rate,
            ignore_width=0.05,
            min_size=0.05,
            fuse_duration=0.5
        )
        # Convert intervals into time in seconds, with logic to merge continuous
        # periods that might span the previous/next windows
        if last_interval_to_check is not None:
            if not len(intervals):
                all_intervals.append(last_interval_to_check)
                last_interval_to_check = None
            elif intervals[0][0] < (0.5 * lazy_mic.sampling_rate):
                if intervals[0][1] == len(sig):
                    last_interval_to_check = (
                        last_interval_to_check[0],
                        intervals[0][1] / lazy_mic.sampling_rate + t
                    )
                else:
                    all_intervals.append((
                        last_interval_to_check[0],
                        intervals[0][1] / lazy_mic.sampling_rate + t
                    ))
                    last_interval_to_check = None
                intervals = intervals[1:]
            else:
                all_intervals.append(last_interval_to_check)
                all_intervals.append((
                    intervals[0][0] / lazy_mic.sampling_rate + t,
                    intervals[0][1] / lazy_mic.sampling_rate + t
                ))
                intervals = intervals[1:]
                last_interval_to_check = None

        for i0, i1 in intervals:
            if i1 == len(sig):
                last_interval_to_check = (i0 / lazy_mic.sampling_rate + t, i1 / lazy_mic.sampling_rate + t)
                break
            all_intervals.append((i0 / lazy_mic.sampling_rate + t, i1 / lazy_mic.sampling_rate + t))


    return all_intervals


def split_live_syllables(
        lazy_mic,
        detected_sound_intervals,
        default_threshold,
        default_threshold_std
    ):
    """From predetermined sound periods, try to segment them into individual calls/syllables

    Tries to break up periods of sound into constituient parts; e.g. a song into syllables,
    by incrementally raising the amplitude threshold until the longest piece is shorter than 1s
    """

    all_syllable_intervals = []

    for t1, t2 in detected_sound_intervals:
        # Load the sound from this time window, filter it,
        # and make an amplitude envelope from the mean of the two channels
        sliced = lazy_mic.time_slice(t1, t2)
        t_arr = sliced.t
        sig = sliced.data
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, lazy_mic.sampling_rate, 1000, 8000).T
        amp_env = get_amplitude_envelope(
            sig,
            lazy_mic.sampling_rate,
            highpass=2000,
            lowpass=8000)
        amp_env = np.mean(amp_env, axis=1)

        # Since the interval is known to be mostly sound, it doesn't make sense
        # to compute the detection threshold from this interval alone. So we
        # load a period extending 1s on either end to compute a threshold from a
        # period that is presumed to include some silence.
        sliced_extended = lazy_mic.time_slice(t1 - 1.0, t2 + 1.0)
        sig_extended = sliced_extended.data
        sig_extended = sig_extended - np.mean(sig_extended, axis=0)
        sig_extended = bandpass_filter(sig_extended.T, lazy_mic.sampling_rate,
                1000, 8000).T
        amp_env_extended = get_amplitude_envelope(
                sig_extended,
                lazy_mic.sampling_rate,
                highpass=2000,
                lowpass=8000
        )
        amp_env_extended = np.mean(amp_env_extended, axis=1)

        threshold = compute_smart_threshold(amp_env_extended, lazy_mic.sampling_rate)
        if threshold > default_threshold + default_threshold_std:
            threshold = default_threshold + default_threshold_std

        # Compute intervals within this period, optimizing for separating sounds
        # that have more than 20ms of silence between them
        # Then gradually raise the threshold used until the longest period detected is
        # no greater than 1s.
        longest_interval_short_enough = False
        idx = 0
        while longest_interval_short_enough == False:
            intervals = threshold_events(
                    amp_env,
                    threshold,
                    polarity=1,
                    sampling_rate=lazy_mic.sampling_rate,
                    ignore_width=0.02,
                    min_size=0.01,
                    fuse_duration=0.02
            )
            lengths = [np.diff(x) / lazy_mic.sampling_rate for x in intervals]
            if len(lengths) and np.max(lengths) > 1:
                threshold *= 1.25
            else:
                longest_interval_short_enough = True
                longest_interval_short_enough

            # Make sure we don't get in an infinite loop. If we don't find a threshold
            # after 10 tries, just end it here.
            idx += 1
            if idx > 10:
                break

        if not len(intervals):
            # We didn't find any, just use this whole period as a call
            all_syllable_intervals.append([t1, t2])
        else:
            for interval in intervals:
                all_syllable_intervals.append((
                    t_arr[interval[0]],
                    t_arr[interval[1] - 1]
                ))

    return all_syllable_intervals


def cluster_syllables(lazy_mic, syllable_intervals, exclude_intervals=None, pad=0.05):
    """
    Applies UMAP + HDBSCAN labeling of the spectrogram data in given syllable intervals

    This function works for 1 or 2 channel audio data
    """
    centered_specs = []
    final_syllable_intervals = []

    for t1, t2 in syllable_intervals:
        # Excluded intervals are those that we don't want to detect calls in
        # e.g. playback periods
        if exclude_intervals is not None and check_if_any_interval_overlaps((t1, t2), exclude_intervals):
            continue
        else:
            final_syllable_intervals.append((t1, t2))

        # Load sound signal
        sliced = lazy_mic.time_slice(t1, t2)
        t_arr = sliced.t
        sig = sliced.data
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, lazy_mic.sampling_rate, 1000, 8000).T

        # Compute the center of mass (COM) of the power - this is to center the spectrogram on
        amp_env = get_amplitude_envelope(
            sig,
            lazy_mic.sampling_rate,
            highpass=2000,
            lowpass=8000)
        amp_env = np.mean(amp_env, axis=1)
        com = np.sum(amp_env * t_arr) / np.sum(amp_env)

        # Now make a new slice of fixed width centered on the COM
        sliced = lazy_mic.time_slice(com - pad, com + pad)
        t_arr = sliced.t
        sig = sliced.data
        sig = sig - np.mean(sig, axis=0)
        sig = bandpass_filter(sig.T, lazy_mic.sampling_rate, 1000, 8000).T

        # Compute spectrogram on all channels available
        spectrograms_by_channel = [
            spectrogram(sig[:, ch], 30000, 1000, 50, min_freq=250, max_freq=8000, cmplx=False)
            for ch in range(sig.shape[1])
        ]
        centered_specs.append(np.array([result[2] for result in spectrograms_by_channel]))

    centered_specs = np.array(centered_specs)
    if len(centered_specs) > 10:
        proj = umap.UMAP(n_components=6).fit_transform(
                centered_specs.reshape(len(centered_specs), -1)
        )
        labels = HDBSCAN(min_cluster_size=10).fit_predict(proj)
    else:
        labels = [0 for _ in range(len(centered_specs))]

    return final_syllable_intervals, centered_specs, labels


def detect_call_and_syllable_dicts(mic):
    """Automatically detect live calls from audio signal
    """
    raise NotImplementedError


def threshold_events(
        signal,
        threshold,
        polarity=1,
        sampling_rate=1,
        ignore_width=None,
        min_size=1,
        fuse_duration=0
    ):
    """Detect intervals crossing a threshold

    TODO: leave the possiblity of open ended intervals....?
    """
    if polarity not in (-1, 1):
        raise ValueError("Polarity must equal +/- 1")

    starts_on = (polarity * signal[0] >= polarity * threshold)

    crossings = np.diff((polarity * signal >= polarity * threshold).astype(np.int))
    interval_starts = np.where(crossings > 0)[0] + 1
    interval_stops = np.where(crossings < 0)[0] + 1

    if starts_on:
        interval_starts = np.concatenate([[0], interval_starts])

    if len(interval_stops) < len(interval_starts):
        interval_stops = np.concatenate([interval_stops, [len(signal)]])

    # Ignore events that are too short
    intervals = np.array([
        (i, j) for i, j in zip(interval_starts, interval_stops)
        if (not ignore_width or ((j - i) / sampling_rate) > ignore_width)
    ])
    if not len(intervals):
        return np.array([])

    gaps = (intervals[1:, 0] - intervals[:-1, 1]) / sampling_rate
    gaps = np.concatenate([gaps, [np.inf]])

    fused_intervals = []
    current_interval_start = None
    for (i, j), gap in zip(intervals, gaps):
        if current_interval_start is None:
            current_interval_start = i
        if gap > fuse_duration:
            fused_intervals.append((current_interval_start, j))
            current_interval_start = None

    return np.array([(i, j) for i, j in fused_intervals if ((j - i) / sampling_rate) >= min_size])


def generate_stim_dicts(digital_codes, sample_rate, playbacks_db):
    """Generate segment data from digital signal data and sound (stim) database

    Args:
    digital_signals : neo.core.Events
    playbacks_db : pandas.DataFrame
        dataframe of stimuli played

    Returns a list of dicts: List of segment information
        Each element is a dict defining segment start and end,
        and metadata to include in Segment annotations
        See docstring for audio_segment.segment() for required fields
    """

    if not all([code[1]["action"] == "queu" for code in digital_codes]):
        raise Exception("There are corrupted bits in the digital signals."
                "Should look into it before proceeding")
        # digital_codes = [
        #   code for code in digital_codes
        #   if code[1]["action"] == "queu"]

    # Some data files have the digital pulses missing; in this case, we want
    # to carefully align the stimuli to the envelope (based on timestamp)

    stim_dicts = []

    if len(playbacks_db) != len(digital_codes):
        raise Exception("Number of stimuli in database "
            "does not match number of digital pulses during sesssion")

    # Iterate through the digital codes received during this session.
    # The playbacks_db should have one row for each stimulus played,
    # IN THE SAME ORDER as the digital pulses are received.
    #  Note that the metadata contains additional information (trial no,
    #  stimulus hash) that can be cross referenced with the events file
    #  generated by pyoperant to verify that the digital codes are always
    #  being paired with the correct stimulus in playbacks_db...
    for stimulus_index, (start_time, metadata) in enumerate(digital_codes):
        start_time = float(start_time) / sample_rate

        # Query the database for the correct trial
        stim = playbacks_db[playbacks_db.index == stimulus_index]
        if len(stim) == 0:
            raise Exception("Trial not found in playback databaset file")
        else:
            # TODO (kevin): check what stim_id ends up being
            stim_id = stim.index[0]
            stim = dict(stim.iloc[0])

        vocalizer_relation = get_vocalizer_relation(stim)

        stim_dicts.append({
            "name": "playback_{}".format(stim["emitter"]),
            "description": "",
            "sound_type": "playback",
            "vocalization_type": stim.get("callid", "UV"),
            "vocalizer": vocalizer_relation,
            "playback_file": stim.get("stimulus_name"),
            "stim_id": stim_id,
            "start_time": start_time,
            "duration": stim["duration"],
        })

    return stim_dicts 


