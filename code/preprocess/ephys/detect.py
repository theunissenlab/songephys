from collections import defaultdict

import numpy as np


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
                min_spacing=2
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