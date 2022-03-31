"""
High-level process:

    * Create a dataframe with all playback events and info
    * Annotate that dataframe with spike times for unit of interest
    * For each filter (i.e. call type + vocalizer combination)
        filter the dataframe down to those rows
    * Generate comparison plots / analysis
"""
import os
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import quantities as pq
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity

from IPython.display import display

from preprocess.sound.playback_utils import _parse_julie_filename_metadata, _parse_ladder_filename_data
from plot_utils import border, get_square_dims

import suss.io


def create_stimulus_dataframe(events, parser=_parse_julie_filename_metadata):
    """Create dataframe from list of event dictionaries

    Args
    ----
    events : list
        List of dictionaries with the following keys filled in -
        'playback_file', 'start_time', 'duration'

        The code assumes that there is a 0.5 second padding
        added to either end of the interval.

        The dataframe will be annotated with stimulus information
        read from the playback file name.

    Returns a pandas dataframe with each row a stimulus. The dataframe
    will have the following columns -
    ["file", "relation_short", "call_type", "start_time",
        "duration", "stop_time"]
    """
    rows = []

    for event in events:
        stim_set = os.path.basename(os.path.dirname(event["playback_file"]))
        metadata = parser(event["playback_file"])
        if parser == _parse_julie_filename_metadata:
            rows.append((
                os.path.basename(event["playback_file"]),
                "{}".format(metadata["emitter"][:4]),
                metadata["callid"],
                event["start_time"].item() + 0.5,
                event["duration"].item() - 1.0,
                event["start_time"].item() + event["duration"].item(),
                event["playback_file"]
            ))
        elif parser == _parse_ladder_filename_data:
            rows.append((
                os.path.basename(event["playback_file"]),
                metadata["source"],
                metadata["type"],
                event["start_time"].item() + 0.5,  # This has to be hardcoded cuz i was dumb
                event["duration"].item() - 1.0,    # a long time ago
                event["start_time"].item() + event["duration"].item(),
                event["playback_file"]
            ))


    return pd.DataFrame(data=rows,
        columns=["file", "relation_short", "call_type",
            "start_time", "duration", "stop_time", "file_path"])


def inject_spikes(df, spike_times, t_buffer=1.0):
    """Add a column "spikes" to dataframe with array of spike times

    Args
    ----
    df : pandas.DataFrame
        Pandas dataframe with events
    spike_times : np.ndarray
        Numpy array of spike times in seconds
    t_buffer : time before and after the stimulus to include spikes

    Returns a new dataframe with the added column
    """
    spike_times = np.array(sorted(spike_times))
    spike_arrays = []

    for t_start, t_stop in zip(df["start_time"], df["stop_time"]):
        idx0 = np.searchsorted(spike_times, t_start - t_buffer)
        idx1 = np.searchsorted(spike_times, t_stop + t_buffer)

        spike_arrays.append(spike_times[idx0:idx1])

    df = df.copy()
    df["spikes"] = pd.Series(spike_arrays, index=df.index)

    return df

def get_unit_info(df, spike_times, spike_waveforms, t_buffer=1.0):
    """Get firing rate in each window and also all spike waveforms
    """
    _sorter = np.argsort(spike_times)
    spike_times = np.array(spike_times)[_sorter]
    spike_waveforms = np.array(spike_waveforms)[_sorter]
    spike_rates = []
    all_waveforms = None

    for t_start, t_stop in zip(df["start_time"], df["stop_time"]):
        idx0 = np.searchsorted(spike_times, t_start - t_buffer)
        idx1 = np.searchsorted(spike_times, t_stop + t_buffer)

        spike_rates.append(
            (
                t_start,
                len(spike_times[idx0:idx1]) / (t_stop - t_start + 2 * t_buffer)
            )
        )

        new_wfs = spike_waveforms[idx0:idx1]
        if len(new_wfs) and all_waveforms is None:
            all_waveforms = new_wfs
        elif len(new_wfs):
            all_waveforms = np.concatenate([all_waveforms, spike_waveforms[idx0:idx1]])

    return spike_rates, all_waveforms



def select_with_spikes(df, selector, spike_time_key="spikes", exclude_empty=True):
    """Select subset of dataframe, dropping rows that do not fit
    the selection criteria and have no spikes
    """
    if exclude_empty:
        subset = df[selector & (df[spike_time_key].apply(len) > 0)].dropna(how="all")
    else:
        subset = df[selector].dropna(how="all")

    return subset


def relative_spike_times(df, selector, spike_time_key="spikes", exclude_empty=True):
    """Get spike times relative to stimulus onset for selected subset of data
    """
    subset = select_with_spikes(
        df,
        selector,
        spike_time_key=spike_time_key,
        exclude_empty=exclude_empty
    )

    spike_times = []
    for i, row in subset.iterrows():
        spike_times.append(row[spike_time_key] - row["start_time"])

    return spike_times


def kde(data, x):
    #model = KernelDensity(
    #         bandwidth=0.1, kernel="linear"
    # ).fit(data[:, None])
    bins = np.linspace(x[0], x[-1] + (x[1] - x[0]), 30)
    halfwidth = (bins[1] - bins[0]) / 2
    bin_edges = np.concatenate([bins - halfwidth, [bins[-1] + halfwidth]])
    y = np.histogram(data, bins=bin_edges)[0]
    print(bin_edges)
    print(y)

    return interp1d(bins, y, kind="cubic")(x)
    # return np.histogram(data, bins=bin_edges)[0]

    # return (
    #     np.exp(model.score_samples(x[:, None]))
    #     # (x[1] - x[0])
    # )

def kde(data, x):
    model = KernelDensity(
             bandwidth=0.1, kernel="linear"
    ).fit(data[:, None])

    return np.exp(model.score_samples(x[:, None]))


def make_comparison_plot(df):
    relations = df["relation_short"].unique()
    call_types = df["call_type"].unique()
    categories = df.groupby([
        'relation_short',
        'call_type'
    ]).size().reset_index().rename(columns={0:'count'})

    x = np.linspace(-1.0, 1.0, 1000)
    baseline_matrix = np.zeros((len(relations), len(call_types)))
    sound_matrix = np.zeros((len(relations), len(call_types)))

    all_data = []

    fig, axes = plt.subplots(
            nrows=len(relations) + 3,
            ncols=len(call_types) + 1,
            figsize=((len(call_types) + 1) * 3, (len(relations) + 3) * 2))

    for row, rel in enumerate(relations):
        for col, cal in enumerate(call_types):
            filtered = relative_spike_times(
                df,
                (df["call_type"] == cal) & (df["relation_short"] == rel)
            )

            if not len(filtered):
                continue

            y = kde(np.concatenate(filtered), x)
            # y = y / len(filtered)
            # all_data.append(y)

            # axes[row][col].plot(x, y, linewidth=3)
            # bins = np.linspace(x[0], x[-1] + (x[1] - x[0]), 30)
            # halfwidth = (bins[1] - bins[0]) / 2
            # bin_edges = np.concatenate([bins - halfwidth, [bins[-1] + halfwidth]])

            frq, edges = np.histogram(np.concatenate(filtered), range=(x[0], x[-1]), bins=30)
            frq = frq / len(filtered)
            y = frq / (edges[1] - edges[0])
            new_x = (edges[:-1] + edges[1:]) / 2
            all_data.append(y)

            axes[row][col].bar(
                edges[:-1],
                y,
                width=np.diff(edges),
                # ec="k",
                align="edge")

            # axes[row][col].hist(
            #     np.concatenate(filtered),
            #     range=(x[0], x[-1]),
            #     bins=30
            # )

            baseline = (-1.0 < new_x) & (new_x < -0.5)
            baseline = np.mean(y[baseline])
            response = (0 < new_x) & (new_x < 0.2)
            response = np.mean(y[response])

            axes[row][col].plot([-1.0, -0.5], [baseline, baseline], linewidth=1, color="Grey")
            axes[row][col].plot([0, 0.2], [response, response], linewidth=1, color="Blue")

            ax = axes[row][col].twinx()
            ax.axis("off")
            ax.set_ylim(0, 1)
            ax.text(x[0] / 1.1, 1 / 1.1, len(filtered),
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=16)

            baseline_rates = [
                    np.sum((-1.0 < spikes) & (spikes < -0.5)) / 0.5
                    for spikes in filtered
            ]
            sound_rates = [
                    np.sum((0 < spikes) & (spikes < 0.2)) / 0.2
                    for spikes in filtered
            ]
            baseline_matrix[row, col] =baseline # np.mean(baseline_rates)
            sound_matrix[row, col] = response # np.mean(sound_rates)

    # Summarize for all in that column
    for col, cal in enumerate(call_types):
            filtered = relative_spike_times(
                df,
                (df["call_type"] == cal)
            )

            if not len(filtered):
                continue

            y = kde(np.concatenate(filtered), x)
            frq, edges = np.histogram(np.concatenate(filtered), range=(x[0], x[-1]), bins=30)
            frq = frq / len(filtered)
            y = frq / (edges[1] - edges[0])
            new_x = (edges[:-1] + edges[1:]) / 2
            all_data.append(y)

            axes[len(relations)][col].bar(
                edges[:-1],
                y,
                width=np.diff(edges),
                # ec="k",
                align="edge")

            baseline = (-1.0 < new_x) & (new_x < -0.5)
            baseline = np.mean(y[baseline])
            response = (0 < new_x) & (new_x < 0.2)
            response = np.mean(y[response])

            axes[len(relations)][col].plot([-1.0, -0.5], [baseline, baseline], linewidth=1, color="Grey")
            axes[len(relations)][col].plot([0, 0.2], [response, response], linewidth=1, color="Blue")

            ax = axes[len(relations)][col].twinx()
            ax.axis("off")
            ax.set_ylim(0, 1)
            ax.text(x[0] / 1.1, 1 / 1.1, len(filtered),
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=16)

    all_data = np.concatenate(all_data)

    for row, rel in enumerate(relations):
        for col, cal in enumerate(call_types):
            axes[row][col].set_ylim(
                    np.min(all_data) / 1.1,
                    np.max(all_data) * 1.1
            )
    for col, cal in enumerate(call_types):
        axes[len(relations)][col].set_ylim(
                np.min(all_data) / 1.1,
                np.max(all_data) * 1.1
        )


    # axes[len(relations) - 1][-1].axis("off")
    for row, rel in enumerate(relations):
        axes[row][0].set_ylabel(rel, fontsize=20)
        axes[row][-1].bar(-0.5 + np.arange(len(baseline_matrix[row, :])), baseline_matrix[row, :], 0.3)
        axes[row][-1].bar(0 + np.arange(len(sound_matrix[row, :])), sound_matrix[row, :], 0.3)

    for col, cal in enumerate(call_types):
        axes[0][col].set_title(cal, fontsize=20)
        axes[len(relations) + 1][col].bar(-0.5 + np.arange(len(baseline_matrix[:, col])), baseline_matrix[:, col], 0.3)
        axes[len(relations) + 1][col].bar(0 + np.arange(len(sound_matrix[:, col])), sound_matrix[:, col], 0.3)

    return fig, axes


from analysis.spiketrains.plots import make_sound_raster_psth_subplot, psth_plot, raster_plot

def make_stim_rasters(
        df,
        min_time=-1.0,
        max_time=2.0,
        group_column="file",
        plot_sound_fn=None,
        key="spikes",
        figsize=(8, 6),
        height_ratios=(0.5, 1, 1),
        bin_size=0.01,
        show_details=False,
        psth_ylim=None,
        psth_color=None,
        exclude_empty_trials=True,
    ):

    '''
    relations = df["relation_short"].unique()
    call_types = df["call_type"].unique()
    categories = df.groupby([
        'relation_short',
        'call_type'
    ]).size().reset_index().rename(columns={0:'count'})
    '''
    group_keys = sorted(df[group_column].unique())

    rows, cols = get_square_dims(len(group_keys)) 
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

    group_idx = 0
    for row in range(rows):
        for col in range(cols):
            if group_idx > len(group_keys) - 1:
                break


            # display(df)
            filtered = relative_spike_times(
                df,
                (df[group_column] == group_keys[group_idx]),
                spike_time_key=key,
                exclude_empty=exclude_empty_trials,
            )

            # hack in the max time
            if group_keys[group_idx].startswith("DC"):
                duration = 5.0
            elif group_keys[group_idx].startswith("randripple"):
                duration = 4.0
            else:
                duration = 8.0

            if not len(filtered):
                group_idx += 1
                continue

            if plot_sound_fn is None:
                sub_axes = make_sound_raster_psth_subplot(
                    sound_idx=None,
                    raster_idx=1,
                    psth_idx=0,
                    outer_grid_spec=gs[row, col],
                    height_ratios=height_ratios[-2:],
                )
            else:
                sub_axes = make_sound_raster_psth_subplot(
                    sound_idx=0,
                    raster_idx=2,
                    psth_idx=1,
                    outer_grid_spec=gs[row, col],
                    height_ratios=height_ratios,
                )

            raster_ax = fig.add_subplot(sub_axes["raster_ax"])
            raster_plot(
                    filtered,
                    min_time=min_time,
                    max_time=duration,
                    color="k",
                    linewidth=.5,
                    ax=raster_ax,
                    mode="vlines",
                    fix_raster_trials=10,

            )
            raster_ax.set_yticks([])
            raster_ax.set_xlim(min_time, duration)

            psth_ax = fig.add_subplot(sub_axes["psth_ax"])
            psth_plot(
                    filtered,
                    min_time=min_time,
                    max_time=duration,
                    bins=int((duration - min_time) / bin_size),
                    color=psth_color or "k",
                    linewidth=2.0,
                    ax=psth_ax,
                    mode="gaussian",
                    conv_width=20,
            )
            psth_ax.set_xlim(min_time, duration)
            if psth_ylim is not None:
                psth_ax.set_ylim(*psth_ylim)

            if plot_sound_fn is None:
                text_ax = psth_ax
            else:
                sound_ax = fig.add_subplot(sub_axes["sound_ax"])
                plot_sound_fn(df[df[group_column] == group_keys[group_idx]], sound_ax)
                sound_ax.set_xlim(min_time, duration)
                border(sound_ax, 0, 0, 0, 1)
                sound_ax.set_yticks([])
                sound_ax.set_xticks([])

                text_ax = sound_ax
            
            text_ax.text(
                0.05,
                1.05,
                "{}".format(group_keys[group_idx]),
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=plt.gca().transAxes
            )
            psth_ax.set_xticks([])
            max_ylim = max(max_ylim, psth_ax.get_ylim()[1])
            psth_axes.append(psth_ax)
            
            if min_time < 0:
                raster_ax.vlines([0], *raster_ax.get_ylim(),
                        linestyle=":", color="red")

            border(raster_ax, 0, 0, 0, 1)
            border(psth_ax, 1, 0, 0, 0)
            group_idx += 1

    for ax in psth_axes:
        ax.set_ylim(0, max_ylim)
        ax.vlines([0], *ax.get_ylim(),
            linestyle=":", color="red")

    return fig


def generate_category_comparisons(
        events,
        spike_times,
        spike_waveforms,
    ):
    df = create_stimulus_dataframe(events)
    df = inject_spikes(df, spike_times, t_buffer=1.5)

    rates, wfs = get_unit_info(
        df,
        spike_times,
        spike_waveforms,
        t_buffer=1.0
    )

    fig, axes = make_comparison_plot(df)

    # Plot the spike shape in the last column
    last_ax = axes[-2][-1]
    last_ax = last_ax.twinx()
    last_ax.plot(np.array(wfs).T, alpha=0.2, color="Black")
    last_ax.plot(np.mean(wfs, axis=0), color="Magenta", linewidth=2)
    old_ylim = last_ax.get_ylim()

    if np.max(np.abs(np.array(old_ylim))) > 300:
        last_ax.set_ylim(-300, 200)

    new_ax = axes[-1][0]

    new_ax.scatter(
        [rate[0] for rate in rates],
        [rate[1] for rate in rates],
    )

    return fig, axes


