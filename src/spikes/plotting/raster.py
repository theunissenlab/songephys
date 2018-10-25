import itertools

import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot(
        spike_times,
        min_time=0.0,
        max_time=1.0,
        color="k",
        linewidth=0.5,
        alpha=1.0,
        ax=None,
        cmap="Greys",
        mode=("scatter", "vlines", "pcolormesh")
    ):
    """Raster plot based on spike arrival times

    Inputs:
    spike_times (list (length N_units) of arrays of spike arrival times)
    """
    if not isinstance(mode, basestring):
        mode = "scatter"

    ax = ax or plt.gca()

    for i, spikes in enumerate(spike_times):
        if mode == "scatter":
            ax.scatter(spikes, i * np.ones(len(spikes)), s=linewidth, marker="s", color=color, alpha=alpha)
        elif mode == "vlines":
            ax.vlines(
                    spikes,
                    i + 0.2,
                    i + 0.8,
                    linewidth=linewidth,
                    color=color,
                    alpha=alpha
            )
        elif mode == "pcolormesh":
            ax.pcolormesh(spikes, cmap=cmap)

    ax.set_ylim(0.0, len(spike_times))
    ax.set_xlim(min_time, max_time)


def plot_epochs(epochs, elec_idx=0, color="Red", ax=None):
    ax = ax or plt.gca()

    for trial_idx, epoch in enumerate(epochs):
        for t, dur, label in zip(epoch.times, epoch.durations, epoch.labels):
            if label != elec_idx:
                continue
            ax.hlines([trial_idx + 0.5], t - epoch.segment.t_start, t + dur - epoch.segment.t_start, color=color)


def plot_probs(spike_times, cluster_probs, min_time=0.0, max_time=1.0, color="k", alpha=1.0):
    """Raster plot based on spike arrival times

    Inputs:
    spike_times (list (length N_units) of arrays of spike arrival times)
    """

    for i, (spikes, probs) in enumerate(zip(spike_times, cluster_probs)):
        plt.vlines(spikes, i, i + probs, color=color, alpha=alpha)

    if min_time < 0.0:
        plt.vlines([0.0], 0.0, len(spike_times), color="r", linestyle="--")

    plt.ylim(0.0, len(spike_times))
    plt.xlim(min_time, max_time)
