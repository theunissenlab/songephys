"""Functions for step of fitting W in binary pursuit (waveform for each unit - electrode pair)
"""


import numpy as np
import scipy.linalg


def spike_covariance(X, bin_width=32):
    """Compute spike train covariances at all offsets in time window

    Params
    ======
    X (n_samples, n_units): Binary array of binned spike arrivals

    Returns
    =======
    XX (n_units, n_samples, n_samples)
    """
    n_samples, n_units = X.shape
    cov = np.zeros((bin_width, n_units, n_units))

    for t_offset in range(bin_width):
        if t_offset:
            cov[t_offset, :, :] = X[:-t_offset].T.dot(X[t_offset:]).todense()
        else:
            cov[t_offset, :, :] = X.T.dot(X).todense()

    return cov
    

def spike_voltage_cross_covariance(X, V, bin_width=32):
    """
    Params
    ======
    X (n_samples, n_units): Binary array of binned spike arrivals
    V (n_samples, n_electrodes): Filtered electrode data

    Returns
    =======
    XY (n_samples, n_units, n_electrodes)
    """
    n_samples, n_units = X.shape
    _, n_electrodes = V.shape

    cov = np.zeros((bin_width, n_units, n_electrodes))

    for t_offset in range(bin_width):
        dt = - bin_width / 2 + t_offset
        if dt < 0:
            x = X[-dt:]
            v = V[:dt]
        elif dt == 0:
            x = X
            v = V
        elif dt > 0:
            x = X[:-dt]
            v = V[dt:]

        cov[t_offset, :, :] = x.T.dot(v)

    return cov


def fit_waveforms(X, V, bin_width=32):
    spike_covariances = spike_covariance(X, bin_width=bin_width)
    XV = spike_voltage_cross_covariance(X, V, bin_width=bin_width)

    _, n_units, _ = spike_covariances.shape
    _, _, n_electrodes = XV.shape

    XX = np.zeros((n_units * bin_width, n_units * bin_width))

    for t_offset, cov in enumerate(spike_covariances):
        idx_start = t_offset * n_units
        idx_end = (t_offset + 1) * n_units
        filler = spike_covariances[:-t_offset or None]
        filler = filler.reshape(filler.shape[0] * filler.shape[1], filler.shape[2])
        XX[idx_start:, idx_start:idx_end] = filler
        XX[idx_start:idx_end, idx_start:] = filler.T

    W = scipy.linalg.solve(XX, XV.reshape(bin_width * n_units, n_electrodes))

    return W.reshape(bin_width, n_units, n_electrodes)
