import sys
import warnings
from collections import defaultdict

try:
    import concurrent.futures as concurrent_futures
except ImportError:
    concurrent_futures = None

import numpy as np

from ... import config


def detect_spike_times(signal, threshold_std=3, samples_pre=20, samples_post=20, min_spacing=1):
    """Get indices of peaks in signal crossing threshold

    Parameters
    ----------
    signal : np.ndarray
        1D signal
    threshold_std : int | float
        Threshold in stds to detect spikes
    samples_pre : int
        Number of samples to include per snippet preceding the peak index
    samples_post : int
        Number of samples to include per snippet after the peak index
    min_spacing : int
        Minimum number of samples separating two spikes

    Returns
    -------
    spike_times : np.ndarray[int]
        An array of indexes relative to the start of ``signal`` of spike peaks
        that cross the given threshold
    """
    threshold = threshold_std * np.std(signal)
    mask = np.where(np.abs(signal) >= threshold)[0]
    mask = np.concatenate([mask, [len(signal)]])

    edges = np.where(np.diff(mask) > min_spacing)[0]

    spikes_found = []
    start_ind = 0

    for stop_ind in edges:
        spike_ind = np.argmax(np.abs(signal[mask[start_ind]:mask[stop_ind] + 1]))
        spike_ind = mask[start_ind] + int(spike_ind)

        # Reject peaks that occur too close to the edge of the signal
        if (spike_ind - samples_pre < 0) or (spike_ind + samples_post + 1 > len(signal)):
            pass
        # Reject peaks that are larger than config.Z_SPIKE_LIMIT times the threshold
        elif np.abs(signal[spike_ind]) > (config.Z_SPIKE_LIMIT * threshold):
            pass
        else:
            spikes_found.append(spike_ind)
        start_ind = stop_ind + 1

    return np.array(spikes_found).astype(int)


def get_spike_snippets(signal, spike_indices, samples_pre: int, samples_post: int):
    """Get waveform snippets from raw signal based on peak locations

    Parameters
    ----------
    signal : np.ndarray
        A signal of length N
    spike_indices : np.ndarray[int]
        Array of integer index values where spikes peak
    samples_pre : int
        Number of samples to include per snippet preceding the peak index
    samples_post : int
        Number of samples to include per snippet after the peak index

    Returns
    -------
    snippets : np.ndarray
        A 2d (N x M) array where N is the number of spikes and M is the
        size of a waveform (samples_pre + samples_post + 1)
    """
    wfs = []
    for spike_idx in spike_indices:
        spike_wf = signal[spike_idx - samples_pre:spike_idx + samples_post + 1]
        wfs.append(spike_wf)

    return np.array([np.array(wf) for wf in wfs])


def get_unique_mask(waveforms):
    unique_waveforms, counts = np.unique([tuple(x) for x in waveforms], axis=0, return_counts=True)

    repeated_wfs = [tuple(wf) for wf, ct in zip(unique_waveforms, counts) if ct > 1]
    nonunique_selector = np.array([(tuple(wf) in repeated_wfs) for wf in waveforms]).astype(bool)

    return ~nonunique_selector


def extract_spikes(
        signal_file: 'format.wav.AudioFile',
        channel: int,
        batch_size: int,
        threshold_std: float,
        t_pre: float,
        t_post: float,
        min_time_between_spikes: float
    ):
    """Get spike times and spike waveforms from a signal

    Reads the data in chunks of ``batch_size``, detects spikes by
    threshold crossings, and extracts snippets with the requested duration.

    Parameters
    ----------
    signal_file : format.wav.AudioFile
        The ephys signal in AudioFile format (which allows for reading in chunks)
    channel : int
        Channel in signal_file to detect spikes on
    batch_size : int
        Number of samples per chunk
    threshold_std : float
        Number of standard deviations of the signal to set threshold
    t_pre : float
        Size of spike snippet before peak in seconds
    t_post : float
        Size of spike snippet after peak in seconds

    Returns
    -------
    spikes : dict
        A dictionary with the following keys "indicies" and "waveforms":

        indices : np.ndarray[int]
            A 1D array of indices showing spike peaks relative to the start of ``signal_file``
        waveforms : np.ndarray[float]
            An array of shape (N, M), where N is the number of spikes detected, and M
            is the size of a spike snippet, defined by ``t_pre`` and ``t_post``
    """
    elec_spike_indices = []
    elec_spike_wfs = []

    _, samples_pre, samples_post, _, _, peak_idx = calculate_snippet_dimensions(
            t_pre, t_post, signal_file.sampling_rate)
    min_spacing = int(round(min_time_between_spikes * signal_file.sampling_rate))

    for batch_idx, chunk in signal_file.batches(batch_size):
        sig = chunk[:, channel]
        indices = detect_spike_times(
            sig,
            threshold_std=threshold_std,
            samples_pre=samples_pre,
            samples_post=samples_post,
            min_spacing=min_spacing,
        )
        waveforms = get_spike_snippets(sig, indices, samples_pre, samples_post)

        if len(indices):
            elec_spike_indices.append(indices + batch_idx)
            elec_spike_wfs.append(waveforms)

    if len(elec_spike_indices):
        elec_spike_indices = np.concatenate(elec_spike_indices)
        elec_spike_wfs = np.concatenate(elec_spike_wfs)
    else:
        elec_spike_indices = np.zeros((0,))
        elec_spike_wfs = np.zeros((0, samples_pre + samples_post + 1))

    # Remove duplicate spikes - if duplicated spikes occur, remove them
    nonduplicate_selector = get_unique_mask(elec_spike_wfs)
    if np.any(~nonduplicate_selector):
        warnings.warn("{} identical spike shapes detected in {}".format(int(np.sum(~nonduplicate_selector)), signal_file))

    return {
        "peak_idx": peak_idx,
        "indices": elec_spike_indices[nonduplicate_selector],
        "waveforms": elec_spike_wfs[nonduplicate_selector]
    }


def calculate_snippet_dimensions(t_pre, t_post, sampling_rate):
    """Compute number of samples and time array for snipping waveforms

    Previous code used a snippet "radius" (equal time cut before and after peak)
    Instead, we want to be able to vary how much time before and after peak are
    included in the snippets for analysis.

    Args
    ----
    t_pre : float
        Time before peak in seconds
    t_post : float
        Time after peak in seconds
    sampling_rate : int
        Sampling rate of signal

    Returns a tuple of 5 elements:
        - time array (time array for cut snippets using the following params)
        - samples_pre (number of samples before peak)
        - samples_post (number of samples after peak)
        - t_pre (time before peak in seconds, discretized by sampling_rate steps)
        - t_post (time after peak in seconds, discretized by sampling_rate steps)
        - peak_idx (index of peak value)
    """
    expect_samples_pre = int(np.ceil(t_pre * sampling_rate))
    expect_samples_post = int(np.ceil(t_post * sampling_rate))

    t_arr = np.zeros(expect_samples_pre + expect_samples_post)
    t_arr[:expect_samples_pre] = -np.arange(expect_samples_pre, 0, -1) / sampling_rate
    t_arr[expect_samples_pre:] = np.arange(expect_samples_post) / sampling_rate

    t_pre_discrete = expect_samples_pre / sampling_rate
    t_post_discrete = expect_samples_post / sampling_rate
    peak_idx = expect_samples_pre

    return (
        t_arr,
        expect_samples_pre,
        expect_samples_post,
        t_pre_discrete,
        t_post_discrete,
        peak_idx
    )


def recut_waveforms(
        lazy_site,
        spike_times,
        elec_id,
        channel_ids=None,
        t_pre=0.0005,
        t_post=0.001,
        tqdm_fn=None,
    ):
    """Recuts sorted spike snippets using the given spike times
    """
    if channel_ids is not None:
        elec_idx = np.where(channel_ids == elec_id)[0][0]
    else:
        elec_idx = elec_id

    sampling_rate = lazy_site.get_signal_sampling_rate("filtered")

    t_arr, samples_pre, samples_post, t_pre_discrete, t_post_discrete, peak_idx = calculate_snippet_dimensions(
        t_pre,
        t_post,
        sampling_rate,
    )
    output_waveforms = np.zeros((len(spike_times), samples_pre + samples_post))

    iterator = enumerate(spike_times)
    if tqdm_fn is not None:
        pbar = tqdm_fn(total=len(spike_times))
    else:
        pbar = None

    def _parallel_slice(spike_idx, peak_time, t_pre_discrete, t_post_discrete):
        sliced = lazy_site.time_slice(
            peak_time - t_pre_discrete,
            peak_time + t_post_discrete,
            signal="filtered"
        )
        return spike_idx, sliced.data[:, elec_idx]

    if concurrent_futures:
        with concurrent_futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for spike_idx, peak_time in iterator:
                futures.append(executor.submit(_parallel_slice, spike_idx, peak_time, t_pre_discrete, t_post_discrete))

            for future in concurrent_futures.as_completed(futures):
                spike_idx, wf = future.result()
                if len(wf) != output_waveforms.shape[1]:
                    raise ValueError("Slice size {} does not match expected {}".format(
                        wf.shape[0], output_waveforms.shape[1]))
                output_waveforms[spike_idx] = wf
                if pbar:
                    pbar.update(1)
            pbar.close()
    else:
        for spike_idx, peak_time in iterator:
            spike_idx, wf = _parallel_slice(spike_idx, peak_time, t_pre_discrete, t_post_discrete)
            if len(wf) != output_waveforms.shape[1]:
                raise ValueError("Slice size {} does not match expected {}".format(
                    wf.shape[0], output_waveforms.shape[1]))
            output_waveforms[spike_idx] = wf
            if pbar:
                pbar.update(1)
        pbar.close()

    return t_arr, output_waveforms, peak_idx, sampling_rate
