"""These are probably the functions to use here
"""


import numpy as np
import scipy.sparse as sparse
from binary_pursuit import pursue_iter, make_V
from clustering import GaussianMixtureClustererPruned
from preprocess import binned_spikes, collect_waveforms


def sort(block, electrode_idx, threshold=0, binary_pursuit_iters=4):
    """Sort spikes of one electrode in block

    Uses a Gaussian Mixture model to initialize the clusters,
    and then applies Binary Pursuit (Pillow et al. 2013)

    Params
    block (neo.core.Block)
        Block of spike data. Thresholded (but unsorted) spikes should be
        included in a single neo.core.Unit per electrode channel. The electrode
        channels are selected by the ones that follow the template
        "Amp Elec {electrode index}" as the name.
    electrode_idx (int)
        Index (0-15) of the electrode to be sorted
    threshold (float)
        A custom threshold to use to be more strict than the threshold used
        in initial sorting (defaults to 0)
    binary_pursuit_iters (int)
        Number of times to iterate the Binary Pursuit algorithm.
        (Convergence seems to happen within 4 iterations)

    Returns
    t_arrs (list):
        A list of 1d arrays. Each element corresponds to one neo.core.Segment of the block,
        and is an array of times corresponding to the time bins used in each segment.
    X (np.ndarray, (n_timesteps, n_units))
        2d binary array, with the rows corresponding to individual time bins and columns
        corresponding to unit. X and V both include data over all segmenets, with their
        rows concatenated together 
    V (np.ndarray, (n_timesteps,))
        A single 1d array of the voltage trace of the electrode uesd in binary pursuit
    W (np.ndarray, (waveform_width, n_units))
        A 2d array of waveforms. corresponding to the waveform for each unit, fit
        by binary pursuit
    """
    unit = (
        channel_index.units[0]
        for channel_index in block.channel_indexes
        if channel_index.name == "Amp Elec {}".format(electrode_idx)
    ).next()

    # Assumes we want to slice the middle 32 time-points of the waveforms
    waveforms = collect_waveforms(unit.spiketrains, threshold=threshold)[0][:, 8:40]

    model = GaussianMixtureClustererPruned(10, 4, p_min=0.99, n_min=300, w_min=0.1)
    model.fit(waveforms)

    labeled, p = model.predict(waveforms)

    t_arrs, X = binned_spikes(unit, threshold, model)
    V = make_V(block, electrode_idx)

    if not X.shape[1]:
        return None

    for X, V, W in pursue_iter(binary_pursuit_iters, X, V):
        pass

    return t_arrs, X, V, W


def unravel_into_spiketimes(t_arr, X):
    """Turn a binary spike array and corresponding bin times into an array of spike arrival times

    Params
    t_arr (np.ndarray, (n_bins))
        Timestamp corresponding to each bin
    X (np.ndarray, (n_bins, n_units))
        Binary array with 1's representing spikes and rows representing bins

    Returns
    spike_times (list, (n_units))
        Spike arrival times for each unit
    """
    bin_idx, unit_idx, _ = sparse.find(X)
    return [t_arr[bin_idx[unit_idx == u]] for u in range(X.shape[1])]


def unravel(t_arrs, X, V):
    """Using list of t_arrs (corresponding to segments), split up X and V by segment
   
    Params
    t_arrs (list):
        A list of 1d arrays. Each element corresponds to one neo.core.Segment of the block,
        and is an array of times corresponding to the time bins used in each segment.
    X (np.ndarray, (n_timesteps, n_units))
        2d binary array, with the rows corresponding to individual time bins and columns
        corresponding to unit. X and V both include data over all segmenets, with their
        rows concatenated together 
    V (np.ndarray, (n_timesteps,))
        A single 1d array of the voltage trace of the electrode uesd in binary pursuit

    Returns
    t_arrs (list):
        Same list that came in
    spike_times_list (list)
        Spike times per unit, per segment
    X_list (list):
        Binary spike arrays per segment
    V_list (list):
        Voltage traces per segment
    """
    # TODO: could turn into a generic function to slice arbitrary arrays using t_arrs
    splits = np.cumsum([0] + [len(t) for t in t_arrs])

    spike_times_list = []
    X_list = []
    V_list = []

    for segment_idx, t_arr in enumerate(t_arrs):
        start, stop = splits[segment_idx], splits[segment_idx + 1]

        X_list.append(X[start:stop])
        V_list.append(V[start:stop])
        spike_times_list.append(unravel_into_spiketimes(t_arr, X_list[-1]))

    return t_arrs, spike_times_list, X_list, V_list
