"""
Plotting functions to give a summary of unit data in a session

Plots waveforms for each unit and psths+rasters to playback
"""
import matplotlib.pyplot as plt
import numpy as np

from segment_analysis import MultiSegment


def get_square_dims(n):
    root = np.sqrt(n)
    if root == np.floor(root):
        return int(root), int(root)
    else:
        a = int(np.ceil(root))
        b = (n + a) / a
        return a, b

def clear_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')


def plot_waveforms(waveforms, ax=None):
    """Plot waveforms for all units
    
    Params
    waveforms (np.ndarray, (n_spikes, n_timebins)):
    """

    if ax is None:
        ax = plt.gca()
    xmin = 0
    xmax = 32 / 25000.0 * 1000
    ax.plot(np.linspace(0, xmax, 32), waveforms.T, alpha=0.1, color="Gray")
    ax.plot(np.linspace(0, xmax, 32), np.mean(waveforms, axis=0), color="Blue")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("mV")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top='off', bottom='on', left='off', right='off', labelleft='on', labelbottom='on')


def plot_all_waveforms(multiseg):
    w, h = get_square_dims(len(multiseg.units))
    fig, axes = plt.subplots(nrows=w, ncols=h, figsize=(4 * w, 3 * h))

    for unit_idx, wfs in enumerate(multiseg.waveforms()):
        ax = axes.reshape(-1)[unit_idx]
        plot_waveforms(wfs, ax=ax)
        ax.set_title(unit_idx)

    for ax in axes.reshape(-1)[len(multiseg.units):]:
        clear_spines(ax)

    plt.tight_layout()


def plot_playback_responses(block):
    playback_files = filter(None, set([seg.annotations.get("playback_file") for seg in block.segments]))
    main_multiseg = MultiSegment(filter(
        lambda seg: seg.annotations["sound_type"] == "playback", block.segments
    ))

    w, h = get_square_dims(len(playback_files))

    figures = []

    for unit_idx in main_multiseg.units:
        fig, axes = plt.subplots(nrows=w, ncols=h, figsize=(6 * w, 3 * h))
        for i, playback_file in enumerate(playback_files):
            multiseg = main_multiseg.filter(lambda seg: seg.annotations["playback_file"] == playback_file)
            ax = axes.reshape(-1)[i]

            multiseg.plotter.plot_spectrogram(multiseg.segments[0],
                signal="playback",
                channel=None,
                ax=ax)
            
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_ylim(-ax.get_ylim()[1], ax.get_ylim()[1])
            ax.set_xticks([])
            ax.set_yticks([])
            clear_spines(ax)
            
            ax = ax.twinx()
            multiseg.plotter.plot_psth(unit_idx, ax=ax, color="Red", alpha=0.1, mode="gaussian", convolve_params=(4,))

            ax.set_xlim(0.0, 3.0)
            ax.set_ylim(-0.25, 0.25)
            ax.text(0.2, 0.15, "{vocalizer}\n{vocalization_type}".format(**multiseg.annotations))
            ax.set_xticks([])
            ax.set_yticks([])

            clear_spines(ax)

            ax_raster = ax.twinx()
            multiseg.plotter.plot_raster(unit_idx, ax=ax_raster)
            ax_raster.set_ylim(len(multiseg.segments), -len(multiseg.segments))
            ax_raster.set_xticks([])
            ax_raster.set_yticks([])

            clear_spines(ax_raster)
            
        for ax in axes.reshape(-1):
            clear_spines(ax)

        plt.tight_layout()
        figures.append(fig)

    return figures
