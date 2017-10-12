import numpy as np


def bin_spikes(
        spike_times,
        min_time=None,
        dt=None,
        n_bins=None,
        t_align="start",
        binary=True):
    """Convert arrays of spike times into a single binary array

    Organize multi-unit spike arrival times into bins.

    Params:
    spike_times (list, N_units)
        A list of arrays in which each array contains the spike times
        for a single unit
    min_time (float, default=None)
        Time (in seconds) of first bin. If None, will use the floor of the earliest
        and ceil of the latest spike over all rows
    dt (float, default=1.0)
        Set the bin width (in seconds), for example 1/sample_rate
    n_bins(int, default=None)
        Set the number of bins. The total array will represent (dt * n_bins) seconds
        of data.
    t_align (str, default="start", choices=["start", "mid", "end"])
        Where to align the timestamps for each bin (start of the bin,
        midpoint of the time bin, or end of time bin)
    binary (bool, default=True)
        Return a binary array to indicate presence of a spike per bin.
        Set to False to return number of spikes per bin.

    Returns:
    t_arr (np.ndarray, (N_timesteps,))
        1D array representing the start time (in seconds) of each time bin
    spikes (np.ndarray, (N_units, N_timesteps))
        2D array in which each row represents a single unit/trial and each column
        represents one timestep
    """
    n_rows = len(spike_times)

    if min_time is None:
        min_time = min(row[0] for row in spike_times if len(row))

    n_timesteps = n_bins
    max_time = min_time + (n_timesteps) * dt
    t_arr = np.linspace(min_time, max_time, n_timesteps)
    dt = (max_time - min_time) / n_timesteps

    # generate output array with correct dimensions
    spikes = np.zeros((n_timesteps, n_rows))

    for unit_idx, times in enumerate(spike_times):
        for time in times:
            # This is just a hacky way to check if its a pyquantities constant
            if hasattr(time, "item"):
                time = time.item()
            bin_idx = int(np.floor((time - min_time) / dt))
            spikes[bin_idx, unit_idx] += 1

    if binary:
        spikes = (spikes > 0) * 1

    return t_arr, spikes


def max(A):
    """Find the maximum value and index of each row of an array

    Corresponds to matlab's max function

    Params:
    A (np.ndarray, (N, M))

    Returns:
    max_values (np.ndarray, (N,))
        Maximum values of each row of A
    max_indices (np.ndarray, (N,))
        Index of maximum value for each row in A
    """
    max_indices = np.argmax(A, axis=0)
    return np.diag(A[max_indices]).copy(), max_indices
