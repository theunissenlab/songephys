"""
Filtering for data
"""
import logging
import time

import numpy as np
import tqdm
from soundsig.signal import highpass_filter, lowpass_filter

from preprocess.ephys.detect import (
    detect_spike_times,
    get_spike_snippets
)
from utils import convert_seconds_to_minutes, estimate_time_remaining

import config


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def common_average_filter(
        input_audiofile,
        output_audiofile,
        batch_size,
        channel_groups=None,
    ):
    """Filter ephys signals (in batches)

    Iterate over signal loader in batches of batch_size and subtract
    the common average from each channel. Add the ``channel_groups`` parameter to
    perform this operation over subsets of channels (e.g. if some channels
    have a shared noise source that independent of the others)

    The filtered data is written directly to the output_audiofile. The output_audiofile will
    be extended to the length of the input_audiofile.

    Args
    ----
    input_audiofile : format.wav.AudioFile
        Input signal to read unfiltered data from
    output_audiofile : format.wav.AudioFile
        Output signal to write filtered data to
    batch_size : int
        Size of windows to filter in samples
    channel_groups : Optional[List[ArrayLike]]
        A list of array-like indexes by which the input data should be grouped
        for filtering. The ArrayLike objects should be arrays of integers used to
        index into the 2nd dimension of the input_audiofile data (the channel
        dimension). For example, to seperately filter the first 16 channels from the
        last 15 channels, use ``channel_groups = [np.arange(16), np.arange(16, 32)]``
    """
    if channel_groups is None:
        channel_groups = [np.arange(input_audiofile.channels)]

    batch_starts = np.arange(0, input_audiofile.frames, batch_size)

    output_audiofile.open(
        readonly=False,
        channels=input_audiofile.channels,
        subtype="FLOAT",
        samplerate=input_audiofile.sampling_rate)

    all_noise_intervals = []
    for start in tqdm.tqdm(batch_starts):
        _, block = input_audiofile.read(start, start + batch_size)

        for channel_indices in channel_groups:
            chunk = block[:, channel_indices]
            # Highpass filter followed by mean subtraction
            chunk = highpass_filter(
                chunk.T,
                input_audiofile.sampling_rate,
                config.HIGHPASS_CUTOFF
            ).T
            chunk = common_average_reference(chunk)
            block[:, channel_indices] = chunk

        output_audiofile.write(start, block)

    output_audiofile.close()

    return output_audiofile


def common_average_reference(signal):
    """Subtract the common mean across channels"""
    return signal - np.mean(signal, axis=1)[:, None]



def find_correlated_events(signal, event_radius):
    """Detect threshold crossings that are correlated across channels

    Note: Not in use as of commit 4d2da7d53753279cc2b33fd494f2b5615b8c7cdc

    Return the indices of the events, the snippets of the signal during
    those events, and the correlation matrix across electrodes during
    those events.
    """
    all_spike_indices = set()
    for elec_idx, sig in enumerate(signal.swapaxes(0, 1)):
        spike_indices = detect_spike_times(
            sig,
            threshold_std=0.75 * config.SPIKE_THRESHOLD_STD,
            spike_radius=event_radius,
        )
        all_spike_indices = all_spike_indices.union(set(spike_indices))

    all_spike_indices = np.array(sorted(list(all_spike_indices)))

    snippets = get_spike_snippets(
        signal,
        all_spike_indices,
        event_radius)

    correlation_matrices = []
    for idx, snippet in zip(all_spike_indices, snippets):
        corr = np.corrcoef(snippet.T)
        np.fill_diagonal(corr, 0)
        correlation_matrices.append(corr)

    return all_spike_indices, snippets, np.array(correlation_matrices)


def _consolidate_intervals(intervals):
    """Combine overlapping intervals into non-overlapping intervals

    Ex. [(0, 4), (2, 7), (9, 11)] -> [(0, 7), (9, 11)]

    Args
    ----
    intervals : list
        List of tuples, where each tuple is a start and stop index
    """
    intervals = sorted(intervals)
    new_intervals = []
    last_start = None
    for idx, (i1, i2) in enumerate(intervals):
        if last_start is None:
            last_start = i1

        if idx + 1 >= len(intervals):
            new_intervals.append((last_start, i2))
            break
        elif intervals[idx + 1][0] > i2:
            new_intervals.append((last_start, i2))
            last_start = None

    return new_intervals


def create_noise_intervals(event_indices, event_radius):
    """Convert events into non-overlapping windows of specified width

    Args
    ----
    event_indices : np.ndarray (int)
        Array of integer indices of noise events
    event_radius : int
        Specify width of noise event on either side of the peak

    Returns an (N x 2) array where N is the number of noise events
    and the second axis is the start index and stop index of the
    corresponding intervals
    """
    noise_intervals = []
    for idx in event_indices:
        noise_intervals.append((
            int(max(0, idx - event_radius)),
            int(idx + event_radius)
        ))

    noise_intervals = _consolidate_intervals(noise_intervals)
    return np.array(noise_intervals).astype(np.int)


def filter_snippets(noise_snippets, filter_fn):
    """Filter in specified intervals
    """
    interval_sizes = np.array([snippet.shape[0] for snippet in noise_snippets])
    interval_endpoints = np.cumsum([interval_sizes])
    interval_startpoints = interval_endpoints - interval_sizes

    # Double the snippet array to eliminate edges
    noise_snippets = np.concatenate(noise_snippets)
    cut_size = len(noise_snippets)
    noise_snippets = np.concatenate([
        noise_snippets[-cut_size:],
        noise_snippets,
        noise_snippets[:cut_size]
    ])

    if len(noise_snippets) < config.FILTER_BLOCK + config.N_LAGS:
        return

    if noise_snippets.shape[0] % config.FILTER_BLOCK == config.N_LAGS:
        noise_snippets = np.vstack([noise_snippets, noise_snippets[:1]])

    # FILTER HERE
    clean_signal = filter_fn(noise_snippets)
    clean_signal = clean_signal[cut_size:-cut_size]

    result = []
    for i1, i2 in zip(interval_startpoints, interval_endpoints):
        result.append(clean_signal[i1:i2])

    return result
