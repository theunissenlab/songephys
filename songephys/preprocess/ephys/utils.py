import numpy as np

from .detect import detect_spike_times, get_spike_snippets
from ... import config


def find_correlated_events(signal, event_radius):
    """Detect threshold crossings that are correlated across channels

    Note: Not in use as of commit 4d2da7d53753279cc2b33fd494f2b5615b8c7cdc

    Return the indices of the spike events, the snippets of the signal during
    those events, and the correlation matrix across electrodes during
    those events.
    """
    all_spike_indices = set()
    for elec_idx, sig in enumerate(signal.swapaxes(0, 1)):
        spike_indices = detect_spike_times(
            sig,
            threshold_std=0.75 * config.SPIKE_THRESHOLD_STD,
            samples_pre=event_radius,
            samples_post=event_radius,
        )
        all_spike_indices = all_spike_indices.union(set(spike_indices))

    all_spike_indices = np.array(sorted(list(all_spike_indices)))

    snippets = get_spike_snippets(
        signal,
        all_spike_indices,
        samples_pre=event_radius,
        samples_post=event_radius
    )

    correlation_matrices = []
    for idx, snippet in zip(all_spike_indices, snippets):
        corr = np.corrcoef(snippet.T)
        np.fill_diagonal(corr, 0)
        correlation_matrices.append(corr)

    return all_spike_indices, snippets, np.array(correlation_matrices)


def consolidate_intervals(intervals):
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
