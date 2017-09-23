"""
I keep having to do weird stuff around spike binning.

Let me just write two functions:
    - one general purpose binner
    - one general purpose unbinner

These should basically be inverses of each other up to time bin-resolution
"""
from scipy import sparse as sparse


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

