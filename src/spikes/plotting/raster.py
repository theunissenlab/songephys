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
        ax.vlines(spikes, i + 0.2, i + 0.8, color=color, alpha=alpha)

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


def plot_spikes_with_timeseries(t, spike_times, y, t_start, t_stop, ax, spike_alpha=0.5, y_alpha=None):
    """
    y can be a 2d array of multiple timeseries to plot
    """
    axes = [ax, ax.twinx()]

    if y.ndim > 1:
        if y_alpha is None: y_alpha = np.ones(y.ndim) * 0.5
        for i, row in enumerate(y):
            axes[0].plot(t, row, alpha=y_alpha[i])
    else:
        if y_alpha is None: y_alpha = 0.5
        axes[0].plot(t, y, alpha=y_alpha)

    axes[0].set_xlim(t_start, t_stop)
    plot(spike_times, min_time=t_start, max_time=t_stop, alpha=spike_alpha, ax=axes[1])

