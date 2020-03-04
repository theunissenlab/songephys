import os
import time
import suss.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import quantities as pq
from IPython.display import clear_output

from sklearn.neighbors import KernelDensity

from preprocess.sound.playback_utils import _parse_julie_filename_metadata
from util import border, get_square_dims

from .adaptive_kde import ssvkernel


def kde2(data, x):
    # TODO (kevin) not sure what units this is... but its a probability density
    return (
        np.exp(KernelDensity(bandwidth=0.1, kernel="exponential").fit(data[:, None]).score_samples(x[:, None]))
    ), None


def full_responses(EVENTS, file_parser=_parse_julie_filename_metadata):
    # Columns playback file name, vocalizer, vocalization_type, start time, duration
    rows = []
    for event in EVENTS:
        if event.get("sound_type") == "playback":
            metadata = file_parser(os.path.basename(event["playback_file"]))
            relation = None
            if metadata["source_sex"]:
                relation = os.path.basename(event["playback_file"]).split("_")[1]
            else:
                relation = metadata["emitter"]
            
            rows.append((
                event["playback_file"],
                os.path.basename(event["playback_file"]),
                relation,
                event["vocalizer"],
                metadata["type"][:2].upper() if not metadata.get("callid") else metadata["callid"][:2].upper(),
                event["start_time"].item() + 0.5,
                event["duration"].item() - 1.0,
                event["start_time"].item() + event["duration"].item(),
            ))

        elif event.get("sound_type") == "live":
            rows.append((
                None,
                None,
                None,
                None,
                None,
                event["start_time"].item() + 0.5,
                event["duration"].item() - 1.0,
                event["start_time"].item() + event["duration"].item(),
            ))


    return pd.DataFrame(data=rows,
        columns=["file_path", "file", "relation_short", "relation", "call_type", "start_time", "duration", "stop_time"])


def _get_layout(n, max_columns=None):
    w = 1
    h = 1

    increment_width = False
    while w * h < n:
        if increment_width and (not max_columns or w < max_columns):
            w += 1
        else:
            h += 1
        increment_width = not increment_width

    return w, h


def fig_grid(n, ax_size=(3, 2), max_columns=None):
    # dimension
    w, h = _get_layout(n, max_columns=max_columns)

    fig = plt.figure(figsize=ax_size)
    axes = []

    curr_x = 0
    curr_y = 0
    for i in range(n):
        axes.append(fig.add_axes([curr_x + 0.1, curr_y + 0.1, 0.8, 0.8]))
        curr_x += 1
        if curr_x == w:
            curr_x = 0
            curr_y += 1

    return fig, axes
   

def inject_spikes(df, spikes, buffer=20.0, key=None):
    if key is None:
        key = "spikes"

    spike_times = []
    
    for t_start, t_stop in zip(df["start_time"], df["stop_time"]):
        idx0 = np.searchsorted(spikes, t_start - buffer)
        idx1 = np.searchsorted(spikes, t_stop + buffer)

        spike_times.append(spikes[idx0:idx1])
    
    df[key] = pd.Series(spike_times, index=df.index)
    return df 


def filter_dataframe(df, selector):
    subset = df.where(selector & df["spikes"].apply(len) > 0).dropna(how="all")
    return subset


def get_spikes_in_window(df, selector):
    subset = filter_dataframe(df, selector)
    spike_times = []
    for i, row in subset.iterrows():
        spike_times.append(row["spikes"] - row["start_time"])
    return spike_times


def plot_raw(
        events,
        loader,
        channels,
        dt=0.2,
        t_start=-0.5,
        t_stop=0.5,
        ylim=None,
        mic=False,
        randomize=False,
        ax_size=(7, 3),
        max_columns=3,
        max_channels=16,
    ):

    print("Make sure you are using channel number by 8-23, not 0-16")
    try:
        iter(channels)
    except TypeError:
        channels = [channels]

    try:
        if randomize:
            iterator = events.sample(frac=1).iterrows()
        else:
            iterator = events.iterrows()

        for _, event in iterator:
            fig, axes = fig_grid(len(channels), ax_size=ax_size, max_columns=max_columns)

            for idx, channel in enumerate(channels):
                ax = axes[idx]
                if mic:
                    sig = loader.time_slice(event["start_time"] + t_start, event["start_time"] + t_stop, signal="mic")
                    ax.plot(sig.t, sig.data[:, channel], linewidth=1)
                else:
                    # We configured intan to record on channels 8-23
                    # when using 16 channels
                    if max_channels == 32:
                        channel_idx = channel
                    elif max_channels == 16:
                        channel_idx = channel - 8
                    sig = loader.time_slice(event["start_time"] + t_start, event["start_time"] + t_stop, signal="filtered")
                    ax.plot(sig.t, sig.data[:, channel_idx], linewidth=1)
                if ylim is not None:
                    ax.set_ylim(*ylim)

                ax.set_xlim(sig.t[0], sig.t[-1])
                ax.text(
                    0.01,
                    0.99,
                    "Ch {}".format(channel),
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=14,
                    transform=ax.transAxes)

                ax.text(
                    0.99,
                    0.99,
                    "{}".format(event["call_type"]),
                    horizontalalignment="right",
                    verticalalignment="top",
                    fontsize=14,
                    transform=ax.transAxes)

                hours = int(event["start_time"] // 3600)
                mins = int((event["start_time"] - (hours * 3600)) // 60)
                secs = int(event["start_time"] - (hours * 3600) - (mins * 60))
                ax.text(
                    0.01,
                    0.01,
                    "{:1d}h{:1d}m{:1d}s".format(hours, mins, secs),
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    fontsize=14,
                    transform=ax.transAxes)

                ax.vlines([event["start_time"]], *ax.get_ylim(), linestyle="--")

            clear_output(wait=True)
            plt.show()
            plt.close(fig)
            time.sleep(dt)
    except KeyboardInterrupt:
        plt.close(fig)
    
    return


def view_spike_waveforms_on_channel(SITE, channel, suffix=""):
    print("Use channel numbers 8-23")
    SPIKES = suss.io.read_pickle(os.path.join(SITE, "manually_curated", "curated-e{}{}.pkl".format(channel, suffix)))

    step = 500

    try:
        for i in range(step):
            fig, axes = fig_grid(len(SPIKES.labeled_nodes), (4, 3))
            for node_idx, (label, node) in enumerate(SPIKES.labeled_nodes):
                ax = axes[node_idx]
                ax.plot(node.flatten().waveforms[i::step].T, color="Blue", alpha=0.2)
                ax.plot(np.mean(node.flatten().waveforms, axis=0), linewidth=4)
                ax.set_ylim(-200, 100)
                ax.text(
                    0.01,
                    0.99,
                    "Unit {} (index {})".format(label, node_idx),
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=14,
                    transform=ax.transAxes)
                ax.grid(linestyle=":")

            clear_output(wait=True)
            plt.show()
            plt.close(fig)
    except KeyboardInterrupt:
        plt.close(fig)


def view_spike_isi_on_channel(SITE, channel, suffix="-kevin"):
    print("Use channel numbers 8-23")
    SPIKES = suss.io.read_pickle(os.path.join(SITE, "manually_curated", "curated-e{}{}.pkl".format(channel, suffix)))
    try:
        fig, axes = fig_grid(len(SPIKES.labeled_nodes), (4, 3))
        for node_idx, (label, node) in enumerate(SPIKES.labeled_nodes):
            ax = axes[node_idx]
            dt = np.diff(node.flatten().times)
            ax.hist(dt, bins=80, range=(0, 0.04), color="Black")
            ax.text(
                0.01,
                0.99,
                "Unit {} (index {})".format(label, node_idx),
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=14,
                transform=ax.transAxes)
            ax.text(
                0.99,
                0.99,
                "{:.1f}% < 1ms".format(100.0 * np.sum(dt < 0.001) / len(dt)),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=14,
                transform=ax.transAxes)
            ax.vlines([0.001], *ax.get_ylim(), color="Red", linestyle="--")

        clear_output(wait=True)
        plt.show()
        plt.close(fig)
    except KeyboardInterrupt:
        plt.close(fig)


def kde(data, x):
    y, t, optw, gs, C, confb95, yb = ssvkernel(data, tin=x)
    return (
        y
    ), confb95
       

def generate_full_comparison(SITE, df, channel, unit, adaptive_kde=False):
    print("Use channel numbers 8-23")

    print("Loading channel {} and unit {}".format(channel, unit))

    SPIKES = suss.io.read_pickle(os.path.join(SITE, "manually_curated", "curated-e{}.pkl".format(channel)))

    _node_lookup = dict(SPIKES.labeled_nodes)

    inject_spikes(df, _node_lookup[unit].flatten().times, buffer=2.0)

    call_types = df["call_type"].unique()

    x = np.linspace(-2.0, 2.0, 1000)
    mask = (x >= -1.5) & (x < 1.5)

    call_types_with_trials = []
    for call in call_types:
        filtered = get_spikes_in_window(
            df, (df["call_type"] == call)
        )
        if len(filtered) <= 2:
            pass
        else:
            call_types_with_trials.append((call, filtered))

    fig, axes = fig_grid(len(call_types_with_trials), (4, 3))

    all_psths = []
    for (call, filtered_df), ax in zip(call_types_with_trials, axes):
        if adaptive_kde:
            y, yerr = kde(np.concatenate(filtered_df), x)
            dt = x[1] - x[0]
            scale = dt * len(np.concatenate(filtered_df)) / len(filtered_df)
            y = y * scale
            yerr = yerr * scale

            # now y is in spikes per time bin.
            # to get to spikes per second (Hz)...

            y = y / dt
            yerr = yerr / dt

            ax.fill_between(x[mask], yerr[0, mask], yerr[1, mask], color="Grey", alpha=0.3)
            ax.set_xlabel("Hz")
        else:
            y, _ = kde2(np.concatenate(filtered_df), x)

        all_psths.append(y)

        ax.plot(x[mask], y[mask], linewidth=1)

        ax.text(
            0.95, 0.95,
            "n={}".format(len(filtered_df)), 
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=16)

        ax.text(
            0.05, 0.95,
            call,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=16)

    ylim = (np.min(all_psths) / 1.1, np.max(all_psths) * 1.1)
    for (call, filtered_df), ax in zip(call_types_with_trials, axes):
        ax.set_ylim(*ylim)

    plt.show()
    plt.close(fig)


def compare_responses(
        dfs,
        df_labels=None,
        show_individual_stims=False,
        min_time=-1.0,
        max_time=6.0,
        height_ratios=(0.5, 1, 1),
        plot_sound_fn=None,
        group_column="Key",
        figsize=(8, 6),
        bin_size=0.01,
        show_details=False,
        psth_ylim=None,
        psth_color=None,
        spike_time_key="spikes"):
    """Compare responses in dataframes of trials

    mostly copied from analysis.playbacks.categories.make_stim_rasters
    """
    cols = len(dfs)
    n_stims_per_df = [len(df["Key"].unique()) for df in dfs]
    max_stims = np.max(n_stims_per_df)

    if show_individual_stims:
        rows = int(max_stims)
    else:
        rows = 1

    fig = plt.figure(
        figsize=(cols * figsize[0], rows * figsize[1] + (figsize[1] / 2 if plot_sound_fn is not None else 0))
    )
    gs = gridspec.GridSpec(
        rows,
        cols,
        wspace=0.3,
        hspace=0.5
    )

    max_ylim = 0
    psth_axes = []
    from analysis.spiketrains.plots import make_sound_raster_psth_subplot, psth_plot, raster_plot
    from analysis.playbacks.categories import relative_spike_times
    for col_i, df in enumerate(dfs):
        filtered = relative_spike_times(
            df.sort_values(group_column),
            np.ones(len(df)).astype(np.bool),
            spike_time_key=spike_time_key,
            exclude_empty=True,
        )

        sub_axes = make_sound_raster_psth_subplot(
            sound_idx=None,
            raster_idx=1,
            psth_idx=0,
            outer_grid_spec=gs[0, col_i],
            height_ratios=height_ratios[-2:],
        )

        raster_ax = fig.add_subplot(sub_axes["raster_ax"])
        raster_plot(
                filtered,
                min_time=min_time,
                max_time=6.0,
                color="k",
                linewidth=1,
                ax=raster_ax,
                mode="vlines",
                fix_raster_trials=len(filtered),

        )
        raster_ax.set_yticks([])
        raster_ax.set_xlim(min_time, max_time)

        psth_ax = fig.add_subplot(sub_axes["psth_ax"])
        psth_plot(
                filtered,
                min_time=min_time,
                max_time=max_time,
                bins=int((max_time - min_time) / bin_size),
                color=psth_color or "k",
                linewidth=2.0,
                ax=psth_ax,
                mode="gaussian",
                conv_width=20,
        )
        psth_ax.set_xlim(min_time, max_time)
        psth_ax.set_xticks([])
        max_ylim = max(max_ylim, psth_ax.get_ylim()[1])
        psth_axes.append(psth_ax)

        if psth_ylim is not None:
            psth_ax.set_ylim(*psth_ylim)
        text_ax = psth_ax
        text_ax.text(
            0.05,
            1.05,
            df_labels[col_i],
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=plt.gca().transAxes
        )


        if min_time < 0:
            raster_ax.vlines([0], *raster_ax.get_ylim(),
                    linestyle=":", color="red")

        border(raster_ax, 0, 0, 0, 1)
        border(psth_ax, 1, 0, 0, 0)

    for ax in psth_axes:
        ax.set_ylim(0, max_ylim)
        ax.vlines([0], *ax.get_ylim(),
            linestyle=":", color="red")

    return fig




