import os

import hdbscan
import numpy as np
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

import config


class SortNode(object):
    def __init__(self, times, waveforms):
        self.times = times
        self.waveforms = waveforms

    @property
    def time(self):
        return np.mean(self.times)

    @property
    def waveform(self):
        return np.mean(self.waveforms, axis=0)


def merge_nodes(nodes):
    times = np.concatenate([n.times for n in nodes])
    _sorter = np.argsort(times)
    times = times[_sorter]
    waveforms = np.concatenate([n.waveforms for n in nodes])[_sorter]

    return SortNode(times, waveforms)


def normalize2d(data):
    data = scipy.stats.zscore(data, axis=0)
    data = scipy.stats.zscore(data, axis=1)
    return data


def link_thresholded_data(session_dirs, unit):
    files = [
        os.path.join(session_dir,
            "spikes",
            "spike_waveforms-e{}.npy").format(unit)
        for session_dir in session_dirs
    ]
    loaded_files = [np.load(f)[()] for f in files]
    times = np.concatenate([
        config.SESSION_DURATION * offset + f["times"]
        for offset, f in enumerate(loaded_files)
    ])
    waveforms = np.vstack([
        f["waveforms"] for f in loaded_files
    ])

    return SortNode(times, waveforms)


def within(t_arr, t_start, t_stop):
    return (t_arr >= t_start) & (t_arr < t_stop)


def labels_to_idx(labels):
    result = []
    for l in sorted(np.unique(labels)):
        result.append(np.where(labels == l)[0])

    return result


def cluster_over_time(sort_node,
        t_step=60.0, p_thresh=0.99, max_components=10, 
        weight_thresh=0.02, cluster_pcs=20):

    clusters = []

    for t_start in np.arange(0, np.max(sort_node.times), t_step):
        idx = within(sort_node.times, t_start, t_start + t_step)
        # TODO: do pca here on cluster_pcs
        data = sort_node.waveforms[idx]

        gmm = BayesianGaussianMixture(n_components=max_components,
                max_iter=500, tol=0.001, warm_start=True)
        gmm.fit(data)
        prediction = gmm.predict_proba(data)
        labels = np.argmax(prediction, axis=1)
        prob = np.max(prediction, axis=1)
        
        # indicies of points assigned to cluster
        to_delete = np.array([], dtype=np.int)
        for label in range(max_components):
            if gmm.weights_[label] > weight_thresh:
                selection = (labels == label) & (prob > p_thresh)
                clusters.append(
                    SortNode(
                        sort_node.times[idx][selection],
                        sort_node.waveforms[idx][selection],
                    )
                )

        leftover_node = SortNode(
                np.delete(sort_node.times, to_delete),
                np.delete(sort_node.waveforms, to_delete, axis=0)
        )

    return (
        clusters,
        leftover_node
    )


def cluster_nodes_over_time(
        nodes,
        t_step=1800.0
    ):

    times = np.array([node.time for node in nodes])
    _sorter = np.argsort(times)
    times = times[_sorter]
    waveforms = np.array([node.waveform for node in nodes])[_sorter]

    clusters_by_timestep = []

    for t_start in np.arange(0, np.max(times), t_step):
        clusters = []

        # will fit on a larger time window
        extended_idx = within(times,
                t_start - 0.5 * t_step,
                t_start + 1.5 * t_step)
        narrow_idx = within(times[extended_idx],
                t_start,
                t_start + t_step)
        # idx = extended_idx[_narrow_idx]

        hdb = hdbscan.HDBSCAN(min_cluster_size=20)
        labels = hdb.fit_predict(normalize2d(waveforms[extended_idx]))

        for label in sorted(np.unique(labels[narrow_idx])):
            new_node = merge_nodes(
                np.array(nodes)[extended_idx][narrow_idx][labels[narrow_idx] == label]
            )
            clusters.append(new_node)

        clusters_by_timestep.append(clusters)

    return clusters_by_timestep


def sort(times, waveforms):
    # initial clustering using a bayesian gaussian 
    #   create sortnodes from these
    # take the nonsorted points and do another passover
    #   create sortnodes for these
    #   note how many points are lost and create a new sortnode for that
    # hdbscan clustering of subclusters over 30 minute chunks
    # linking clusters over time
    top_node = SortNode(times, waveforms)

    clusters, leftover_node = cluster_over_time(top_node, t_step=60.0)
    clusters += cluster_over_time(leftover_node, t_step=600.0)[0]

    nodes = cluster_nodes_over_time(clusters)
    return nodes
