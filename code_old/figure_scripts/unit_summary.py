"""
Plotting functions to give a summary of unit data in a session

Plots waveforms for each unit and psths+rasters to playback
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

import config

from playbacks_analysis import PlaybacksManager


def get_square_dims(n):
    root = np.sqrt(n)
    if root == np.floor(root):
        return int(root), int(root)
    else:
        a = int(np.ceil(root))
        b = (n + a) // a
        return a, b


def _compute_grid(x, y, width, height, dim_x, dim_y, pad=0):
    """Compute coordinates of evenly laid out grid
    """
    dx = width / dim_x
    dy = height / dim_y

    box_width = dx - pad
    box_height = dy - pad

    boxes = []
    for i in range(dim_x):
        for j in range(dim_y):
            boxes.append([
                x + dx * i,  # x
                y + dy * j,  # y
                box_width,  # width
                box_height  # height
            ])

    return boxes


def _get_unit_idx(block, elec_idx, unit_idx):
    idx = 0
    for ch_idx in block.channel_indexes:
        if not ch_idx.name.startswith("Amp Elec"):
            continue

        for i, unit in enumerate(ch_idx.units):
            if ch_idx.index == elec_idx and i == unit_idx:
                return idx
            else:
                idx += 1


def plot_unit_playback_responses(block, elec_idx, unit_idx, fig, plot_location, width=5):
    playbacks_manager = PlaybacksManager(block.segments)
    playback_multisegs = playbacks_manager.group_playback_responses()

    w, h = get_square_dims(len(playbacks_manager.stim_ids))
    plot_location = plot_location + [w, h]

    figures = []

    unit_idx = _get_unit_idx(block, elec_idx, unit_idx)

    for ax_loc, stim_id in zip(_compute_grid(*plot_location, pad=0.01), playbacks_manager.stim_ids):
        multiseg = playback_multisegs[stim_id]
        ax = fig.add_axes(ax_loc)

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
        multiseg.plotter.plot_psth(unit_idx, ax=ax, color="Red", alpha=0.0, mode="gaussian", convolve_params=(width,))

        playback_labels = {}
        playback_labels["vocalizer"] = multiseg.annotations.get("vocalizer")

        playback_labels["vocalizer"] = " ".join(os.path.basename(multiseg.annotations.get("playback_file")).split("_")[1:3])
        playback_labels["vocalization_type"] = multiseg.annotations.get("vocalization_type")

        if not multiseg.annotations.get("vocalizer"):
            playback_labels["vocalizer"] = "tutor" + playback_labels["vocalizer"]
        if "MJP" in multiseg.annotations["playback_file"]:
            playback_labels["vocalizer"] = "self" + playback_labels["vocalizer"]

        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(-0.20, 0.20)
        ax.text(0.2, 0.15, "{vocalizer}\n{vocalization_type}".format(**playback_labels))
        ax.set_xticks([])
        ax.set_yticks([])

        clear_spines(ax)

        ax_raster = ax.twinx()
        multiseg.plotter.plot_raster(unit_idx, ax=ax_raster)
        ax_raster.set_ylim(len(multiseg.segments), -len(multiseg.segments))
        ax_raster.set_xticks([])
        ax_raster.set_yticks([])

        clear_spines(ax_raster)

    # return figures


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
    xmax = config.WF_SIZE / config.ELEC_SAMPLE_RATE * 1000

    ax.plot(np.linspace(0, xmax, config.WF_SIZE), waveforms.T, alpha=0.1, color="Gray")
    ax.plot(np.linspace(0, xmax, config.WF_SIZE), np.mean(waveforms, axis=0), color="Blue")
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
        if len(wfs):
            ax = axes.reshape(-1)[unit_idx]
            plot_waveforms(wfs, ax=ax)
            ax.set_title(unit_idx)

    for ax in axes.reshape(-1)[len(multiseg.units):]:
        clear_spines(ax)

    plt.tight_layout()

    return fig, axes


def plot_playback_responses(block, width=5):
    playbacks_manager = PlaybacksManager(block.segments)
    playback_multisegs = playbacks_manager.group_playback_responses()

    w, h = get_square_dims(len(playbacks_manager.stim_ids))

    figures = []

    for unit_idx in playbacks_manager.full_multiseg.units:
        fig, axes = plt.subplots(nrows=w, ncols=h, figsize=(6 * w, 3 * h))
        for i, stim_id in enumerate(playbacks_manager.stim_ids):
            multiseg = playback_multisegs[stim_id]
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
            multiseg.plotter.plot_psth(unit_idx, ax=ax, color="Red", alpha=0.0, mode="gaussian", convolve_params=(width,))

            playback_labels = {}
            playback_labels["vocalizer"] = multiseg.annotations.get("vocalizer")

            playback_labels["vocalizer"] = " ".join(os.path.basename(multiseg.annotations.get("playback_file")).split("_")[1:3])
            playback_labels["vocalization_type"] = multiseg.annotations.get("vocalization_type")

            if not multiseg.annotations.get("vocalizer"):
                playback_labels["vocalizer"] = "tutor" + playback_labels["vocalizer"]
            if "MJP" in multiseg.annotations["playback_file"]:
                playback_labels["vocalizer"] = "self" + playback_labels["vocalizer"]

            ax.set_xlim(0.0, 3.0)
            ax.set_ylim(-0.20, 0.20)
            ax.text(0.2, 0.15, "{vocalizer}\n{vocalization_type}".format(**playback_labels))
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


def _get_channel_index(block, electrode_number):
    return next(
        channel_index
        for channel_index in block.channel_indexes
        if channel_index.name.startswith("e{}".format(electrode_number))
    )


def _get_unit_waveforms(block, electrode_idx, unit_idx):
    try:
        ch_idx = _get_channel_index(block, electrode_idx)
        unit = ch_idx.units[unit_idx]
    except:
        raise Exception("Electrode {}, Unit {} might not exist".format(
            electrode_idx, unit_idx))

    return {
            "times": np.concatenate([st.times for st in unit.spiketrains]),
            "waveforms": np.concatenate([st.waveforms for st in unit.spiketrains])
    }


def _get_electrode_waveforms(session_dir, electrode_idx):
    """Get all thresholded waveforms from electrode
   
    This data can no longer be found in the block; need to load it from npy file
    """
    return np.load(os.path.join(
        session_dir, "spikes", "spike_waveforms-e{:d}.npy".format(electrode_idx)
    ))[()]


def plot_waveforms(data, ax, color="#aaaaff", mean_color="#0000ff"):
    n = len(data["waveforms"]) // 1000 + 1
    ax.plot(data["waveforms"][::n].T, color=color, alpha=0.1)
    ax.plot(np.mean(data["waveforms"], axis=0), color=mean_color, linewidth=3)


def plot_isi(data, ax):
    ax.hist(np.diff(data["times"]), bins=100, range=(0, 0.05))
    ax.vlines(0.001, *ax.get_ylim(), color="Red", linestyle="--")


def plot_1d_by_time(unit_data, elec_data, ax, plot_unit_data=True, alpha=0.1):
    labels = np.array([
        1 if t_spike in unit_data["times"] else 0
        for t_spike in elec_data["times"]
    ])
    if len(np.unique(labels)) == 2:
        lda = LDA(n_components=2).fit(elec_data["waveforms"], labels)
    else:
        lda = PCA(n_components=1).fit(elec_data["waveforms"])

    projection = lda.transform(elec_data["waveforms"])

    ax.scatter(unit_data["times"], lda.transform(unit_data["waveforms"]),
            s=2 if plot_unit_data else 4,
            color="Blue" if plot_unit_data else "Gray",
            alpha=10 * alpha if plot_unit_data else alpha)
    ax.scatter(elec_data["times"][(1 - labels).astype(np.bool)], projection[(1 - labels).astype(np.bool)], s=4, color="Gray", alpha=alpha)


def plot_2d(unit_data, elec_data, ax):
    pca = PCA(n_components=2).fit(elec_data["waveforms"])
    projection = pca.transform(elec_data["waveforms"])

    ax.scatter(*projection.T, s=5, color="Gray", alpha=0.01)
    ax.scatter(*pca.transform(unit_data["waveforms"]).T, s=5, color="Orange", alpha=0.1)


def unit_summary(block, electrode_idx, unit_idx):
    """Generate a unit summary"""
    # plot waveforms
    return

