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

from preprocess.sound.playback_utils import _parse_julie_filename_metadata
from util import border, get_square_dims

import suss.io


def create_stimulus_dataframe(events):
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
        metadata = _parse_julie_filename_metadata(event["playback_file"])
        rows.append((
            os.path.basename(event["playback_file"]),
            "{}".format(metadata["emitter"][:4]),
            metadata["callid"],
            event["start_time"].item() + 0.5,
            event["duration"].item() - 1.0,
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



def select_with_spikes(df, selector, spike_time_key="spikes"):
    """Select subset of dataframe, dropping rows that do not fit
    the selection criteria and have no spikes
    """
    subset = df.where(selector & df[spike_time_key].apply(len) > 0).dropna(how="all")
    return subset


def relative_spike_times(df, selector, spike_time_key="spikes"):
    """Get spike times relative to stimulus onset for selected subset of data
    """
    subset = select_with_spikes(df, selector, spike_time_key=spike_time_key)
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


def _add_subplot(fig, oldnsubplots, new):
    new_idx = len(fig.axes) + 1
    return fig.add_subplot(new_idx, 1)


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
