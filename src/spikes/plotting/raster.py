import itertools

import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot(spike_times, min_time=0.0, max_time=1.0, color="k", alpha=1.0, ax=None):
    """Raster plot based on spike arrival times

    Inputs:
    spike_times (list (length N_units) of arrays of spike arrival times)
    """
    ax = ax or plt.gca()

    for i, spikes in enumerate(spike_times):
        ax.vlines(spikes, i + 0.2, i + 0.8, linewidth=1, color=color, alpha=alpha)

    if min_time < 0.0:
        ax.vlines([0.0], 0.0, len(spike_times), color="r", linestyle="--")

    ax.set_ylim(0.0, len(spike_times))
    ax.set_xlim(min_time, max_time)


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
