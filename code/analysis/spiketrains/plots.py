import matplotlib.pyplot as plt
import numpy as np

from analysis.spiketrains.psth import conv, gaussian_convolver


def bin_spikes(spike_times, min_time, max_time, dt):
    t_arr = np.arange(min_time, max_time, dt)
    result = np.zeros((len(spike_times), len(t_arr)))

    for i, row in enumerate(spike_times):
        for j, t in enumerate(t_arr):
            idx1 = np.searchsorted(row, t)
            idx2 = np.searchsorted(row, t + dt)
            result[i, j] += len(row[idx1:idx2])

    return t_arr, result.sum(axis=0)


def raster_plot(
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
    if not isinstance(mode, str):
        mode = "scatter"

    ax = ax or plt.gca()

    for i, spikes in enumerate(spike_times):
        if (len(spikes) > 0 ):
            if mode == "scatter":
                ax.scatter(spikes, i * np.ones(len(spikes)), s=linewidth, marker="|", color=color, alpha=alpha)
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


def psth_plot(        
        spike_times,
        min_time=0.0,
        max_time=1.0,
        bins=30,
        color="k",
        alpha=1.0,
        ax=None,
        mode=("hist", "gaussian"),
        conv_width=10,
        linewidth=2
    ):
    if not isinstance(mode, str):
        mode = "hist"

    if mode == "hist":
        frq, edges = np.histogram(
            np.concatenate(spike_times),
            range=(min_time, max_time),
            bins=bins
        )
        frq = frq / len(spike_times)
        y = frq / (edges[1] - edges[0])

        ax.bar(
            edges[:-1],
            y, 
            width=np.diff(edges),
            alpha=alpha,
            color=color,
            align="edge")

    elif mode == "gaussian":
        t_arr, binned = bin_spikes(spike_times, min_time, max_time, dt=0.001)
        convolved = conv([binned], gaussian_convolver, conv_width)[0]
        ax.plot(t_arr, convolved, color=color, alpha=alpha, linewidth=linewidth)
        
    ax.set_xlim(min_time, max_time)
