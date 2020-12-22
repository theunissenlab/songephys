import sys
from collections import defaultdict

try:
    import concurrent.futures as concurrent_futures
except ImportError:
    concurrent_futures = None

import numpy as np

import config


def detect_spike_times(signal, threshold_std=3, spike_radius=20, min_spacing=1):
    """Get indices of peaks in signal crossing threshold

    Args
    ----
    signal : np.ndarray
        1D signal

    Kwargs
    ------
    threshold_std : int or float
        Threshold in stds to detect spikes
    spike_radius : int
        Radius of time window of spike detection in samples
    min_spacing : int
        Minimum number of samples separating two spikes

    Returns an array of indexes at which putative spike peaks in input
    signal are located
    """
    threshold = threshold_std * np.std(signal)
    mask = np.where(np.abs(signal) >= threshold)[0]
    mask = np.concatenate([mask, [len(signal)]])

    edges = np.where(np.diff(mask) > min_spacing)[0]

    spikes_found = []
    start_ind = 0

    for stop_ind in edges:
        spike_ind = np.argmax(np.abs(signal[mask[start_ind]:mask[stop_ind] + 1]))
        spike_ind = mask[start_ind] + np.int(spike_ind)

        if (spike_ind - spike_radius < 0) or (spike_ind + spike_radius > len(signal)):
            pass
        elif np.abs(signal[spike_ind]) > (config.Z_SPIKE_LIMIT * threshold):
            pass
        else:
            spikes_found.append(spike_ind)
        start_ind = stop_ind + 1

    return np.array(spikes_found).astype(np.int)


def get_spike_snippets(signal, spike_indices, spike_radius):
    """Get waveform snippets from raw signal based on peak locations

    Args
    ----
    signal : np.ndarray
        1D signal
    spike_indices : np.ndarray
        Array of index values where spikes are centered
    spike_width : int
        Radius of time window of spike detection in samples

    Returns a 2d (N x M) array where N is the number of spikes and M is the
    size of a waveform (spike_radius * 2)
    """
    wfs = []
    for spike_idx in spike_indices:
        spike_wf = signal[spike_idx - spike_radius:spike_idx + spike_radius]
        wfs.append(spike_wf)

    return np.array([np.array(wf) for wf in wfs])


def extract_spikes(
        signal_loader,
        batch_size,
        threshold_std,
        spike_radius,
        channel_ids,
    ):
    """Get spikes and times from signal on all channels
    """
    elec_spike_indices = defaultdict(list)
    elec_spike_wfs = defaultdict(list)

    for batch_idx, chunk in signal_loader.batches(batch_size):
        for elec_idx, sig in enumerate(chunk.swapaxes(0, 1)):
            indices = detect_spike_times(
                sig,
                threshold_std=threshold_std,
                spike_radius=spike_radius,
                min_spacing=config.MIN_SPIKE_THRESHOLD_SPACING
            )
            waveforms = get_spike_snippets(sig, indices, spike_radius)

            if len(indices):
                elec_spike_indices[channel_ids[elec_idx]].append(indices + batch_idx)
                elec_spike_wfs[channel_ids[elec_idx]].append(waveforms)
    
    results = {}
    for elec in channel_ids:
        if len(elec_spike_indices[elec]):
            results[elec] = {
                "indices": np.concatenate(elec_spike_indices[elec]),
                "waveforms": np.concatenate(elec_spike_wfs[elec])
            }
        else:
            results[elec] = {
                "indices": np.zeros((0,)),
                "waveforms": np.zeros((0, spike_radius * 2))
            }

    return results


def get_snippet_size(t_pre, t_post, sampling_rate):
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

    t_arr, samples_pre, samples_post, t_pre_discrete, t_post_discrete, peak_idx = get_snippet_size(
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
