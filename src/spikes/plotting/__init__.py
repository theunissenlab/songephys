import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import raster


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
    raster.plot(spike_times, min_time=t_start, max_time=t_stop, alpha=spike_alpha, ax=axes[1])


def plot_waveform_overlay(waveforms, labels=None, colors=None, light_colors=None, alpha=1.0):
    """Plot spike waveforms. Label with colors if provided

    Params
    ======
    waveforms (np.ndarray, (N_spikes, width))
        Array of spike waveforms
    labels (np.ndarray, (N_spikes), default=None)
        Array of integers labeling the spikes by cluster
    colors (list, default=None)
        If labels is provided, this is the list of color labels to plot each cluster in
    light_colors (list, default=colors)
        Corresponding lighter color to plot individual spikes in to contrast with
        the mean line
    alpha (float)
        Transparency of the individual trial lines (to make the mean stand out)
    """

    if labels is not None:
        if not colors or len(colors) < len(np.unique(labels)):
            raise Exception("Not enough colors provided")
        light_colors = light_colors or colors

        clusters = np.unique(labels)
        for idx, cluster in enumerate(clusters):
            plt.plot(waveforms[labels == cluster].T, alpha=alpha,
                    color=light_colors[idx], linewidth=1)

        for idx, cluster in enumerate(clusters):
            plt.plot(np.mean(waveforms[labels == cluster].T, axis=1), color=colors[idx],
                    linewidth=3)
    else:
        plt.plot(waveforms.T, alpha=alpha, color="Gray", linewidth=1)
        plt.plot(np.mean(waveforms.T, axis=1), color="Black", linewidth=2)


