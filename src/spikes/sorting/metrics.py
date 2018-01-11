"""
Metrics for evaluating cluster quality and properties
"""
import numpy as np

import scipy.stats
from sklearn.neighbors import NearestNeighbors

### Single cluster metrics


def cluster_snr(waveforms):
    n = len(waveforms)
    wf_max = np.max(waveforms, axis=1)
    wf_min = np.min(waveforms, axis=1)
    noise = waveforms - np.mean(waveforms, axis=0)

    return np.mean((wf_max - wf_min) / (2 * np.std(noise, axis=1)))


def interspike_interval(times, range=(0.0, 0.01), bins=10):
    return np.histogram(np.diff(times), range=range, bins=bins, normed=True)


def firing_rate(times):
    duration = np.max(times) - np.min(times)
    return len(times) / duration


def consistency(label_votes):
    """labels is a list of label arrays which vote for the cluster.

    consistency for a spike is the max number that agree on one cluster
    divided by the total number of votes

    the overall consistency is the mean consistency over all waveforms
    """

    n_votes = len(label_votes)
    _, counts = scipy.stats.mode(label_votes)

    return np.mean(counts / float(n_votes))


def _compute_overlap(neighbors, labels, A, B):
    a_or_b = (labels == A) | (labels == B)

    size = np.sum(a_or_b)

    total = []

    sample_count = np.min([500, np.sum(labels==A), np.sum(labels==B)])
    if sample_count < 20:
        return 1.0
    for cluster in [A, B]:
        points = np.random.choice(np.arange(np.sum(labels == cluster)), size=sample_count, replace=False)

        for pt in points:
            k_nearest_labels = labels[neighbors[labels == cluster][pt]]
            total.append(np.mean(k_nearest_labels == cluster))

    return np.mean(total)


def isolation(waveforms, labels, k=10):
    unique_labels = np.unique(labels)
    overlaps = np.zeros((len(unique_labels), len(unique_labels)))

    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=8).fit(waveforms)
    _, neighbors = knn.kneighbors(waveforms)

    for i, A in enumerate(unique_labels):
        for j, B in enumerate(unique_labels):
            if i == j:
                overlaps[i, j] = 1.0
                continue
            overlaps[i, j] = _compute_overlap(neighbors, labels, A, B)

    mins = np.min(overlaps, axis=0)

    print overlaps

    result = {}
    for i, label in enumerate(unique_labels):
        result[label] = mins[i]

    return result







