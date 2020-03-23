import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.neighbors import KernelDensity

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
        mode=("scatter", "vlines", "pcolormesh"),
        fix_raster_trials=None,
    ):
    """Raster plot based on spike arrival times

    Inputs:
    spike_times (list (length N_units) of arrays of spike arrival times)
    """
    if not isinstance(mode, str):
        mode = "scatter"

    ax = ax or plt.gca()

    for i, spikes in enumerate(spike_times):
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

    if fix_raster_trials is None:
        ax.set_ylim(0.0, len(spike_times))
    else:
        ax.set_ylim(len(spike_times) - fix_raster_trials, len(spike_times))
        ax.fill_between(
                [min_time, max_time],
                len(spike_times) - fix_raster_trials,
                0,
                color="Grey",
                alpha=0.1
                )

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
        linewidth=2,
        show_rate=False,
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
        frq = convolved / len(spike_times)
        y = frq / 0.001
        ax.plot(t_arr, y, color=color, alpha=alpha, linewidth=linewidth)

    elif mode == "kde":
        # do this just to get t_arr
        t_arr, binned = bin_spikes(spike_times, min_time, max_time, dt=0.001)

        model = KernelDensity(
            bandwidth=0.1, kernel="linear"
        ).fit(np.concatenate(spike_times)[:, None])

        ax.plot(
            t_arr,
            np.exp(model.score_samples(t_arr[:, None])),
            color=color,
            alpha=alpha,
        )


def make_sound_raster_psth_subplot(
        sound_idx=0,
        raster_idx=1,
        psth_idx=2,
        outer_grid_spec=None,
        height_ratios=(0.5, 1, 1),
    ):

    all_idx = np.array([sound_idx, raster_idx, psth_idx])

    orderer = -np.ones(3, dtype=np.int)  # sound_idx, raster_idx, psth_idx
    using = np.where(np.array([x is not None for x in all_idx]))

    orderer[using] = np.argsort(all_idx[using])

    sound_idx, raster_idx, psth_idx = orderer

    if outer_grid_spec is None:
        gs = gridspec.GridSpec(
            len(using[0]),
            1,
            height_ratios=height_ratios[-2:],
        )
    else:
        gs = gridspec.GridSpecFromSubplotSpec(
            len(using[0]),
            1,
            subplot_spec=outer_grid_spec,
            wspace=0.0,
            hspace=0.0,
            height_ratios=height_ratios,
        )

    if sound_idx != -1:
        sound_ax = gs[sound_idx, 0]
    else:
        sound_ax = None

    if raster_idx != -1:
        raster_ax = gs[raster_idx, 0]
    else:
        raster_ax = None

    if psth_idx != -1:
        psth_ax = gs[psth_idx, 0]
    else:
        psth_ax = None

    return {
        "sound_ax": sound_ax,
        "raster_ax": raster_ax,
        "psth_ax": psth_ax
    }

