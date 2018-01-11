"""
Functions for chunking a (already thresholded) block into time
windows and doing spike clustering within and across time.

Key functions:

1. Time slicing into windows
2. Sorting within windows -> save npy of clusters found, computed metrics
3. Renumbering of clusters across time -> resave npy files? just for convenience
4. Generate figures for clusters by time (time range fmt: {recording_time}_{t_start}_{t_stop})
    - /figures/{time_range}/clusters.npy
    - /figures/{time_range}/cluster{idx}.png
    - /figures/{time_range}/scatter.png
5. Lay out all figures in a web browser and let you reassign clusters


Would be nice:
1. Loading the spike waveform data from multiple sessions at once and combining
"""

import time

import os
import itertools

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from scipy.stats import norm

import pywt
from spikes.sorting.autocluster import sort
from sklearn.decomposition import PCA
from sklearn.lda import LDA


cmap = plt.cm.gist_ncar_r


def load_file(npy_file):
    return np.load(npy_file)[()]


def within(x, start, end):
    return (x >= start) & (x < end)


def time_slices(data, step_size=600):
    """
    Return a list of {"waveforms", "times"} for each time slice
    """
    waveforms = data["waveforms"]
    times = data["times"]

    t_starts = np.arange(0, int(np.ceil(times[-1] / step_size) * step_size), step_size / 2)
    time_ranges = zip(t_starts, t_starts + step_size)

    result = [
        {
            "waveforms": waveforms[within(times, t_start, t_stop)],
            "times": times[within(times, t_start, t_stop)]
        }
        for t_start, t_stop in time_ranges
    ]

    return time_ranges, result


def relabel(labels, perm):
    """
    Relabel elements of labels using a particular permutation

    Permutation order references the sorted order of the original elements

    Example:
    >>> relabel(np.array([1, 2, 1, 4]), [0, 1, 4])
    np.array([0, 1, 0, 4])

    """
    old_label_set = sorted(np.unique(labels))
    new_label_set = perm

    new_labels = np.copy(labels)

    for old, new in zip(old_label_set, new_label_set):
        new_labels[labels == old] = new

    return new_labels


def make_labels_consistent(all_previous_labels, new_labels):
    """
    Relabel new labels to be consistent with a set of previous labels for the same data

    Try to choose a permutation of previous existing labels that match the most previous ones

    One problem is that this doesn't let the next set add a new label; only resuse existing ones

    Example:
    >>> make_labels_consistent([[1,1,2,1,2], [1,1,1,3,2]], [4,4,5,5,5])
    [1,1,2,2,2]
    """
    previous_labels_set = np.unique(np.concatenate(all_previous_labels))
    new_labels_set = sorted(np.unique(new_labels))
    
    ones_we_havent_used_yet = set(new_labels_set) - set(previous_labels_set)

    choices = set(previous_labels_set)
    diff = len(new_labels_set) - len(previous_labels_set)
    if diff > 0:
        choices |= set(np.random.choice(list(ones_we_havent_used_yet), size=diff, replace=False))

    perms = list(itertools.permutations(choices, r=len(new_labels_set)))
    scores = np.zeros(len(perms))

    for i, perm in enumerate(perms):
        match_sum = np.zeros(len(new_labels))  # Counters of how well this datapoint was matched
        for prev_labels in all_previous_labels:
            match_sum += (prev_labels == relabel(new_labels, perm)).astype(np.int)

        scores[i] = np.sum(match_sum)

    return (
            relabel(new_labels, perms[np.argmax(scores)]),
            new_labels_set,
            list(perms[np.argmax(scores)])
    )


def process_votes(proposed_clusters):
    """asdf"""
    previous_labels = []
    for proposal in proposed_clusters:
        if not previous_labels:
            previous_labels.append(proposal)
        else:
            new_labels = make_labels_consistent(previous_labels, proposal)[0]
            previous_labels.append(new_labels)

    return np.vstack(previous_labels)


def relabel_across_time(time_ranges, data_blocks, label_blocks, threshold=0.8):
    used_labels = set(label_blocks[0])
    relabeled_labels = []
    relabeled_labels.append(label_blocks[0])

    for i in range(len(time_ranges) - 1):
        print "used_bales", used_labels
        t_start_1, t_stop_1 = time_ranges[i]
        t_start_2, t_stop_2 = time_ranges[i + 1]

        labels1 = relabeled_labels[i]
        labels2 = label_blocks[i + 1]

        time1 = data_blocks[i]["times"]
        time2 = data_blocks[i + 1]["times"]

        if not len(labels2):
            relabeled_labels.append(np.copy(labels2))
            continue

        new_proposed_labels, old_label_set, new_label_set = make_labels_consistent(
            [labels1[within(time1, t_start_2, t_stop_1)]],
            labels2[within(time2, t_start_2, t_stop_1)]
        )

        label_was_equal = (labels1[within(time1, t_start_2, t_stop_1)] == new_proposed_labels).astype(np.int)

        for j, new_label in enumerate(new_label_set):
            match_rating = np.mean(label_was_equal[new_proposed_labels == new_label])

            print new_label, old_label_set[j], match_rating

            if match_rating < threshold and used_labels:
                new_label_set[j] = max(used_labels) + 1
                used_labels.add(max(used_labels) + 1)

        relabeled2 = np.copy(labels2)

        for old, new in zip(old_label_set, new_label_set):
            print "labeling {} as {}".format(old, new)
            relabeled2[labels2 == old] = new
        print relabeled2

        relabeled_labels.append(relabeled2)

    return relabeled_labels


def gen_figures(time_range, data, labels, metrics):
    """generate figures for a single time block
    """
    figures = []
    print len(np.unique(labels))

    print data["waveforms"].shape

    if len(np.unique(labels)) > 2:
        pca = LDA(n_components=2).fit_transform(data["waveforms"], labels)
    else:
        pca = PCA(n_components=2).fit_transform(data["waveforms"])
    figures.append(plt.figure(figsize=(4, 4)))

    colors = dict((c, cmap((((c * 7) % 20) + 1) / 20.0)) for c in np.unique(labels))

    for cluster in sorted(np.unique(labels)):
        if cluster == -1:
            continue
        plt.scatter(*pca[labels == cluster].T, s=4.0, color=colors[cluster])

    if pca.shape[1] > 1:
        xmean, ymean = np.mean(pca, axis=0)
        xstd, ystd = np.std(pca, axis=0)
        plt.xlim(xmean - 3 * xstd, xmean + 3 * xstd)
        plt.ylim(ymean - 3 * ystd, ymean + 3 * ystd)
    else:
        pass

    for cluster in np.unique(labels):
        figures.append(plt.figure(figsize=(6, 4)))
        plt.plot(data["waveforms"][labels == cluster].T, color=colors[cluster], alpha=0.2)
        plt.ylim(-300, 150)
        plt.title("{}, n={}".format(cluster, np.sum(labels == cluster)))

        plt.text(plt.xlim()[0] + 1, 0, """snr:{snr:.2f}\nfr:{fr:.2f} Hz\ncon:{consistency:.2f}\niso:{isolation:.2f}""".format(**metrics[cluster]), fontsize=16, weight="bold" if metrics[cluster]["snr"] > 4 and metrics[cluster]["isolation"] > 0.8 and np.sum(labels == cluster) > 10 else "normal")
        plt.tight_layout()

        ax = plt.gca().twinx().twiny()
        bins, bin_edges = metrics[cluster]["isi"]
        ax.plot(bin_edges[:-1] + np.diff(bin_edges) / 2.0, bins, marker=".")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, np.max(bins) * 5)
        ax.set_xlim(0, 0.02)
        ax.vlines(0.001, 0, ax.get_ylim()[1] / 5, linestyle="--", color="Red")

    return figures

