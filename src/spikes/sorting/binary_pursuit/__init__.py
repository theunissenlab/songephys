"""Python implementation of binary pursuit (Pillow et al 2013)

Much of this code is transcribed closely from matlab to python
from the matlab implementation found at

https://github.com/pillowlab/BinaryPursuitSpikeSorting
"""


import numpy as np
import scipy.sparse as sparse
from scipy.linalg import sqrtm, inv, toeplitz
from numpy.fft import fft, ifft

import spikes.utils as utils
from estimate_waveforms import fit_waveforms

from spikes.binning import unravel_into_spiketimes


def make_V(block, electrode_idx):
    """Concatenate voltages of electrode across all segments in block

    Assumes analogsignals[-1] has the voltage traces
    """
    return np.vstack([
        # TODO: should make this more robust than taking analogsignals[-1]
        segment.analogsignals[-1].as_array()[:, electrode_idx:electrode_idx+1]
        for segment in block.segments
    ])


def apply_filter(X, filters, flip=False, valid=False):
    """Apply a filter"""
    n_samples, _ = X.shape
    filter_width, n_filters, _ = filters.shape

    if flip:
        filters = np.flip(filters, axis=0)

    output = np.zeros((n_samples + filter_width + 1, n_filters))

    for filt in range(n_filters):
        filtered = X.dot(filters[:, filt, :].T)
        for i in range(filter_width):
            output[i:n_samples + i, filt] += filtered[:, -i-1]

    if valid:
        return output[filter_width:-filter_width]

    return output[filter_width/2:-(filter_width+1)/2 - 1]


def periodic_corr(x, y, max_lag=None):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    corr = ifft(fft(x) * fft(y).conj()).real
    if max_lag:
        return np.concatenate([corr[-max_lag:], corr[:max_lag + 1]])
    else:
        return corr


def whiten_V(X, V, W, temporal_filter_width, spatial_filter_width):
    """
    Params
    X (n_samples, n_units): Binary array of binned spike arrivals
    V (n_samples, n_electrodes): Filtered electrode data
    W (bin_width, n_units, n_electrodes): Filtered electrode data

    Returns
    XY (n_samples, n_units, n_electrodes)
    """
    n_samples, n_units = X.shape
    _, n_electrodes = V.shape

    # predict electrode data based on spikes and W
    V_pred = predict_V(X, W)
    V_err = V - V_pred
    V_err_whitened = np.zeros_like(V_err)

    temporal_filters = temporal_whitening(V_err, temporal_filter_width)

    # temporal whitening of residuals
    for elec_idx in range(n_electrodes):
        V_err_whitened[:, elec_idx] = np.convolve(
             V_err[:, elec_idx],
             temporal_filters[:, elec_idx],
             "full")[temporal_filter_width/2 - 1:-temporal_filter_width/2]

    spatial_filters = spatial_whitening(V_err_whitened, spatial_filter_width)

    V_whitened = np.zeros_like(V)

    # temporal whitening of raw signal
    for elec_idx in range(n_electrodes):
        V_whitened[:, elec_idx] = np.convolve(
             V[:, elec_idx],
             temporal_filters[:, elec_idx],
             "full")[temporal_filter_width/2 - 1:-temporal_filter_width/2] 

    return V_whitened

    return apply_filter(V_whitened, spatial_filters, flip=True)


def temporal_whitening(V_err, filter_width):
    n_samples, n_electrodes = V_err.shape

    filters = np.zeros((filter_width, n_electrodes))
    for elec_idx in range(n_electrodes):
        cross_corr = periodic_corr(
                V_err[:, elec_idx],
                V_err[:, elec_idx],
                max_lag=filter_width - 1)
        cross_corr = np.flipud(cross_corr[:filter_width]) / n_samples

        M = sqrtm(inv(toeplitz(cross_corr)))
        t_filter = M[:, filter_width / 2 - 1]
        filters[:, elec_idx] = t_filter

    return filters


def spatial_whitening(V_err, filter_width):
    n_samples, n_electrodes = V_err.shape

    filters = np.zeros((filter_width, n_electrodes, n_electrodes))

    # spatial cross correlations between electrodes

    cross_corr = np.zeros((n_electrodes, n_electrodes, filter_width))

    for t_offset in range(filter_width):
        cross_corr[:, :, t_offset] = np.roll(V_err, t_offset, axis=0).T.dot(V_err)

    cross_corr = cross_corr / n_samples

    M = np.zeros((n_electrodes * filter_width, n_electrodes * filter_width))

    for elec_i in range(n_electrodes):
        for elec_j in range(n_electrodes):
            if elec_i == elec_j:
                idx_start = elec_i * filter_width
                idx_end = (elec_i + 1) * filter_width
                M[idx_start:idx_end, idx_start:idx_end] = toeplitz(cross_corr[elec_i, elec_j])
            else:
                range_i = slice(elec_i * filter_width, (elec_i + 1) * filter_width)
                range_j = slice(elec_j * filter_width, (elec_j + 1) * filter_width)
                filler = toeplitz(cross_corr[elec_i, elec_j], cross_corr[elec_j, elec_i])
                M[range_i, range_j] = filler
                M[range_j, range_i] = filler.T

    Q = sqrtm(inv(M)).real

    for elec_idx in range(n_electrodes):
        Q_idx = elec_idx * filter_width + filter_width / 2
        filters[:, elec_idx, :] = Q[:, Q_idx].reshape(n_electrodes, filter_width).T

    return filters


def predict_V(X, W):
    """Predict V from spike times and waveforms
    """
    n_samples, n_units = X.shape
    bin_width, _, n_electrodes = W.shape

    V_pred = np.zeros((n_samples + bin_width, n_electrodes))

    for unit_idx in range(n_units):
        spike_indices = sparse.find(X[:, unit_idx])[0]
        for spike_idx in spike_indices:
            V_pred[spike_idx:spike_idx+bin_width] += W[:, unit_idx]

    return V_pred[bin_width/2:-bin_width/2]


def w_proj(W):
    """Do time convolution between waveforms"""
    bin_width, n_units, _ = W.shape

    W_projected = np.zeros((2 * bin_width - 1, n_units, n_units))

    for i_unit in range(n_units):
        for j_unit in range(n_units):
            projected = W[:, i_unit, :].dot(W[:, j_unit, :].T)

            for t_offset in range(bin_width):
                W_projected[t_offset:t_offset + bin_width, j_unit, i_unit] += (
                        projected[:, -t_offset-1])

            if j_unit > i_unit:
                W_projected[:, i_unit, j_unit] = np.flipud(W_projected[:, j_unit, i_unit])
            
    return W_projected, np.diag(W_projected[bin_width-1])



def binary_pursuit(X, V, W, p_spike, w_projections, w_norm, minimum_isi):
    """Run binary pursuit

    Params
        X (n_samples, n_units): Binary array of binned spike arrivals
        V (n_samples, n_electrodes): Filtered electrode data
        W (bin_width, n_units, n_electrodes): Filtered electrode data
        p_spike (n_units): Firing rate per bin of each unit
        w_projections (2 * bin_width - 1, n_units, n_units):
            Waveforms convolved with each other
        w_norm (n_units): Squared L2 norm of each waveform
        minimum_isi (int): minimum number of bins between two spikes from same unit
    """
    n_samples, n_units = X.shape
    bin_width, _, n_electrodes = W.shape
    half_width = bin_width / 2
    waveform_range = np.arange(-bin_width + 1, bin_width)

    V_pred = predict_V(X, W)
    residuals = V - V_pred

    max_iter = 20000

    d_log_likelihood = compute_d_log_likelihood(X, W, residuals, p_spike, w_norm)

    best_d_logli_by_unit, max_bin_by_unit = utils.max(d_log_likelihood)
    max_unit_idx = np.argmax(best_d_logli_by_unit)
    max_bin_idx = max_bin_by_unit[max_unit_idx]

    for iter_count in range(max_iter):
        if best_d_logli_by_unit[max_unit_idx] <= 0:
            break

        X, residuals, d_log_likelihood, updated_bins = _binary_pursuit_iter(X, W, 
            residuals,
            d_log_likelihood,
            w_projections,
            max_unit_idx,
            max_bin_idx)

        # Quick way to find the maxes again without looking at all bins

        # first, find cells whose previous max was in the updated region
        # cuz we need to search all time points for these
        units_with_maxima_in_updated_window = np.where(
                (max_bin_by_unit >= updated_bins[0]) *
                (max_bin_by_unit <= updated_bins[-1])
        )[0]
        d_logli = d_log_likelihood[:, units_with_maxima_in_updated_window]
        best_d_logli_updated, updated_idxs = utils.max(d_logli)
        best_d_logli_by_unit[units_with_maxima_in_updated_window] = best_d_logli_updated
        max_bin_by_unit[units_with_maxima_in_updated_window] = updated_idxs

        # then see if new maxima arose in the updated region
        # to see the new maxima in this window
        d_log_likelihood_within_window = d_log_likelihood[updated_bins]
        best_d_logli_in_window, max_bin_by_unit_in_window = utils.max(d_log_likelihood_within_window)

        best_d_logli_by_unit, max_not_in_window = utils.max(np.vstack((
            best_d_logli_in_window,
            best_d_logli_by_unit
        )))

        max_bin_by_unit[max_not_in_window == 0] = updated_bins[
                max_bin_by_unit_in_window[max_not_in_window == 0]
        ]

        max_unit_idx = np.argmax(best_d_logli_by_unit)
        max_bin_idx = max_bin_by_unit[max_unit_idx]

    # Remove interspike interval violations
    removals = 0
    for unit_idx in range(n_units):
        MAX_IDX = d_log_likelihood.shape[0]
        isis = np.diff(sparse.find(X[:MAX_IDX, unit_idx])[0])

        while np.any(isis < minimum_isi):
            spike_indices = sparse.find(X[:MAX_IDX, unit_idx])[0]
            n_spikes = len(spike_indices)

            bad_spikes = sparse.find(isis < minimum_isi)[1]
            bad_spikes = np.union1d(bad_spikes, bad_spikes[bad_spikes < n_spikes] + 1)
            bad_indices = spike_indices[bad_spikes]
            least_likely_spike_bin = np.argmax(d_log_likelihood[bad_indices, unit_idx]);
            worst_bin = bad_indices[least_likely_spike_bin]

            removals += 1

            X[worst_bin, unit_idx] = 0

            d_logli = -d_log_likelihood[worst_bin, unit_idx]

            low = max(worst_bin - half_width, 0)
            high = min(worst_bin + half_width, n_samples)

            low_w = (bin_width - high) if high < bin_width else 0
            high_w = (n_samples - low) if (n_samples - low < bin_width) else bin_width
            residuals[low:high] += W[low_w:high_w, unit_idx, :]

            update_range = worst_bin + waveform_range
            selector = np.where((update_range >= 0) * (update_range < len(d_log_likelihood)))[0]
            d_log_likelihood[update_range[selector]] += w_projections[selector, :, unit_idx]
            d_log_likelihood[worst_bin, unit_idx] = d_logli
            isis = np.diff(sparse.find(X[:MAX_IDX, unit_idx])[0])

    return X


def compute_d_log_likelihood(X, W, residuals, p_spike, w_norm):
    n_samples, n_units = X.shape
    bin_width, _, n_electrodes = W.shape

    spike_thresh = -np.log(p_spike) + np.log(1 - p_spike)

    residual_convolved = apply_filter(residuals, W, valid=True)
    d_log_likelihood = np.array(residual_convolved - np.matlib.repmat(
            0.5 * w_norm.T + spike_thresh, n_samples - bin_width + 1, 1))

    for unit_idx in range(n_units):
        spike_indices = sparse.find(X[bin_width/2:-bin_width/2, unit_idx])[0]
        d_log_likelihood[spike_indices, unit_idx] = (
                -d_log_likelihood[spike_indices, unit_idx] - w_norm[unit_idx][0]
        )

    d_log_likelihood = np.vstack((-100 * np.ones((bin_width / 2 + 1, n_units)), d_log_likelihood))

    return d_log_likelihood


def _binary_pursuit_iter(X, W,
        residuals,
        d_log_likelihood,
        w_projections,
        max_unit_idx,
        max_bin_idx):
    n_samples, n_units = X.shape
    bin_width, _, n_electrodes = W.shape
    half_width = bin_width / 2

    waveform_range = np.arange(-bin_width + 1, bin_width)

    X[max_bin_idx, max_unit_idx] = 0 if X[max_bin_idx, max_unit_idx] else 1  # flip the bit
    sgn = 1 if X[max_bin_idx, max_unit_idx] == 1 else -1

    d_logli = -d_log_likelihood[max_bin_idx, max_unit_idx]

    # subtract waveform from residuals
    low = max(max_bin_idx - half_width, 0)
    high = min(max_bin_idx + half_width, n_samples)

    low_w = (bin_width - high) if high < bin_width else 0
    high_w = (n_samples - low) if (n_samples - low < bin_width) else bin_width
    residuals[low:high] = (
            residuals[low:high] - 
            (sgn * W[low_w:high_w, max_unit_idx, :])
    )

    # update dlogli for all bins within this waveform's affect
    update_range = max_bin_idx + waveform_range
    selector = np.where((update_range >= 0) * (update_range < len(d_log_likelihood)))[0]

    d_log_likelihood[update_range[selector]] = (
        d_log_likelihood[update_range[selector]] -
        (sgn * w_projections[selector, :, max_unit_idx])
    )

    d_log_likelihood[max_bin_idx, max_unit_idx] = d_logli

    return X, residuals, d_log_likelihood, update_range[selector]


def pursue(X, V):
    """Run one iteration of binary pursuit"""
    p_spike = np.mean(X.A, axis=0)
    W = fit_waveforms(X, V, bin_width=32)
    V = whiten_V(X, V, W, 16, 5)
    W = fit_waveforms(X, V, bin_width=32)
    W_proj, W_norm = w_proj(W)

    for start in range(0, X.shape[0], 10000):
        tru_end = min(start + 10000, X.shape[0])
        X[start:tru_end] = binary_pursuit(
                X[start:tru_end],
                V[start:tru_end],
                W,
                p_spike,
                W_proj,
                W_norm[:, None],
                25)

    return X, V, W


def pursue_iter(n_iter, X, V):
    """Run binary pursuit for n_iter iterations"""
    X = X.copy()
    for _ in range(n_iter):
        yield pursue(X, V)


def resegment(t_arrs, X, V):
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