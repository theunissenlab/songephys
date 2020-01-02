"""
Ephys Data Pipeline

Run like this:

Need to test:
    Merging functions
    Spike sorting functions
"""

from collections import defaultdict
import os
import luigi

import numpy as np
import matplotlib.pyplot as plt

from soundsig.sound import spectrogram
from soundsig.signal import highpass_filter, lowpass_filter, bandpass_filter
import suss.io
from suss.core import SpikeDataset

from format.io import load_yaml
from format.lazy import (
    LazySignal,
    LazySite,
)
from preprocess.ephys.filter import context_based_filtering
from preprocess.ephys.detect import extract_spikes
from preprocess.ephys.sort import sort_set
from preprocess.ephys.newsort import sort as new_sort, create_hierarchy
from preprocess_sound import MergeLiveCalls

from utils import session_basename

from file_setup import SiteConfig, GenerateLazyFiles
import config



class FilterSignals(luigi.Task):
    """Filter out noise from electrode signals

    1. Highpass
    2. Common average referencing
    3. Linear filtering during periods of correlated activity

    Inputs
    ------
    session_dir : str
        Path to session

    Outputs a filtered memmap, noise periods that were filtered
    with the linear filtering method, detected spike times and
    waveforms for all channels, and updates the lazy_signals lookup
    object.
    """

    session_dir = luigi.Parameter()

    def requires(self):
        return GenerateLazyFiles(self.session_dir)

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")

        filtered_path = luigi.LocalTarget(
            os.path.join(raw_dir, "filtered.memmap"))
        noise_path = luigi.LocalTarget(
            os.path.join(self.session_dir, "noise.npy"))

        return {
            "filtered_memmap": filtered_path,
            "noise_events": noise_path,
            "lazy_signals": self.input()["lazy_signals"],
        }

    def run(self):
        outputs = self.output()
        signals = np.load(self.input()["lazy_signals"].path, allow_pickle=True)[()]
        signal = signals["raw"]

        output_signal = LazySignal(
            signal.N,
            signal.M,
            outputs["filtered_memmap"].path,
            signal.sampling_rate,
        )

        _, noise_events = context_based_filtering(
            signal,
            output_signal,
            signal.sampling_rate,
            65536
        )

        signals["filtered"] = output_signal
        np.save(outputs["lazy_signals"].path, signals)
        np.save(outputs["noise_events"].path, noise_events)


class DetectSpikes(luigi.Task):
    """Extract spike events from a filtered signal

    Inputs
    ------
    session_dir : str
        Path to session

    Outputs one suss.core.SpikeDataset in a pickle file
    for each electrode
    """

    session_dir = luigi.Parameter()

    def requires(self):
        return FilterSignals(self.session_dir)

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")

        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return dict(
            (e, luigi.LocalTarget(os.path.join(
                sorting_dir,
                "spike_waveforms-e{}.pkl".format(e))))
            for e in config.ELECTRODES
        )

    def run(self):
        signals = np.load(self.input()["lazy_signals"].path, allow_pickle=True)[()]
        sampling_rate = signals["filtered"].sampling_rate
        spike_radius = int(config.SPIKE_WIDTH * sampling_rate)

        spikes = extract_spikes(
            signals["filtered"],
            # Compute threshold in 5 minute windows
            batch_size=int(sampling_rate * 60 * 5),
            threshold_std=config.SPIKE_THRESHOLD_STD,
            spike_radius=spike_radius,
            channel_ids=config.ELECTRODES
        )

        for elec in config.ELECTRODES:
            spike_dataset = SpikeDataset(
                spikes[elec]["indices"] / sampling_rate,
                spikes[elec]["waveforms"],
                sample_rate=sampling_rate,
                labels=None
            )
            suss.io.save_pickle(
                self.output()[elec].path,
                spike_dataset
            )


class MergeLazySignals(luigi.Task):
    """Merge Lazy signals from each session into a single lookup LazySite object
    """

    site_dir = luigi.Parameter()

    def requires(self):
        self.session_info = load_yaml(
            os.path.join(self.site_dir, "sessions.yaml")
        )
        for session in self.session_info["sessions"]:
            session_dir = os.path.join(
                    self.site_dir,
                    "..",
                    "..",
                    "sessions",
                    session["id"]
            )
            yield FilterSignals(session_dir=session_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "lazy.npy"))

    def run(self):
        lazy_signals_raw = []
        lazy_signals_mic = []
        lazy_signals_filtered = []
        signal_times = []
        for previous_result, session in zip(self.input(), self.session_info["sessions"]):
            session_sigs = np.load(previous_result["lazy_signals"].path, allow_pickle=True)[()]
            lazy_signals_mic.append(session_sigs["mic"])
            lazy_signals_raw.append(session_sigs["raw"])
            lazy_signals_filtered.append(session_sigs["filtered"])
            signal_times.append(session["t"])

        new_site = LazySite(
            signal_times,
            mic=lazy_signals_mic,
            raw=lazy_signals_raw,
            filtered=lazy_signals_filtered
        )

        np.save(self.output().path, new_site)


class CheckLiveCalls(luigi.Task):
    """Make a plot that shows examples of clusters of live vocalizations
    """

    site_dir = luigi.Parameter()

    def requires(self):
        return {
            "lazy_site": MergeLazySignals(self.site_dir),
            "calls": MergeLiveCalls(self.site_dir)
        }

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "live_calls.png"))

    def run(self):
        call_periods = np.load(self.input()["calls"].path, allow_pickle=True)[()]
        lazy_site = np.load(self.input()["lazy_site"].path, allow_pickle=True)[()]

        labels = sorted(list(set([period["label"] for period in call_periods])))

        idx = defaultdict(int)
        fig = plt.figure(figsize=(0.4, 1))

        for period in call_periods:
            if idx[period["label"]] > 20:
                continue
            else:
                ax = fig.add_axes([
                    idx[period["label"]],
                    period["label"],
                    1,
                    1
                ])
                t1 = period["start_time"].item()
                t2 = (period["start_time"] + period["duration"]).item()
                sig = lazy_site.time_slice(t1, t2, signal="mic")
                t_arr = sig.t
                sig = sig.data
                sig = sig - np.mean(sig, axis=0)
                sig = bandpass_filter(sig.T, lazy_site._signals["mic"][0].sampling_rate, 1000, 8000).T

                i = np.argmax(np.sum(sig, axis=0))

                t_spec, f_spec, spec, _ = spectrogram(
                        sig[:, i], 30000, 1000, 50, min_freq=250, max_freq=8000, cmplx=False)

                ax.pcolormesh(t_spec, f_spec, spec, cmap="Greys")

                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_xticklabels([])

                idx[period["label"]] += 1
        fig.savefig(self.output().path, format="png", bbox_inches='tight')


class MergeSpikeDatasets(luigi.Task):
    """Merge together multiple blocks via offsets

    The sessions and relative offsets must be defined in a
    sessions.yaml file in the site directory.

    Outputs one suss.core.SpikeDataset per electrode channel, each
    combining the data from all sessions defined in the site config file.
    """

    site_dir = luigi.Parameter()

    def requires(self):
        inputs = {
            "session_spikes": [],
            "session_config": SiteConfig(site_dir=self.site_dir)
        }
        session_config = load_yaml(os.path.join(self.site_dir, "sessions.yaml"))

        for session in session_config["sessions"]:
            session_dir = os.path.join(
                    self.site_dir,
                    "..",
                    "..",
                    "sessions",
                    session["id"]
            )
            inputs["session_spikes"].append(DetectSpikes(session_dir=session_dir))

        return inputs

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "spikes")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return dict(
            (e, luigi.LocalTarget(os.path.join(
                sorting_dir,
                "spike_waveforms-e{}.pkl".format(e))))
            for e in config.ELECTRODES
        )

    def run(self):
        inputs = self.input()
        session_config = load_yaml(inputs["session_config"].path)

        block_offsets = [session["t"] for session in session_config["sessions"]]

        for elec in config.ELECTRODES:
            spike_datasets = [
                suss.io.read_pickle(spike_result[elec].path)
                for spike_result in inputs["session_spikes"]
            ]

            # join together the spike times and waveforms
            # add the session start time to all times in each session
            spike_times = np.concatenate([
                data.times + offset
                for data, offset in zip(spike_datasets, block_offsets)
            ])
            waveforms = np.concatenate([data.waveforms for data in spike_datasets])
            new_dataset = SpikeDataset(
                spike_times,
                waveforms,
                sample_rate=spike_datasets[0].sample_rate
            )
            suss.io.save_pickle(self.output()[elec].path, new_dataset)


class SortSpikes(luigi.Task):
    """Run spike sorting code on a single channel

    This is the automated portion of the semi-automated spike sorting,
    that generates spike clusters to be manually joined in the next step.

    This is divided across channels because it takes a while and
    might be parallelizable?

    Inputs
    ------
    site_dir : str
        Path to site
    electrode : int
        Electrode id (i.e. 8-23) (not index!)

    Outputs a suss.core.ClusterDataset with clusters of clusters of spikes
    """

    site_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int,
        min_value=config.ELECTRODES[0],
        max_value=config.ELECTRODES[-1] + 1
    )

    def requires(self):
        return MergeSpikeDatasets(self.site_dir)

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "sort_results")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return luigi.LocalTarget(
            os.path.join(sorting_dir, "sorted-e{}.pkl".format(self.electrode))
        )

    def run(self):
        dataset = suss.io.read_pickle(self.input()[self.electrode].path)
        peaks = dataset.waveforms[:, dataset.waveforms.shape[1] // 2]
        times = dataset.times[peaks < 0]
        waveforms = dataset.waveforms[peaks < 0]
        t, w, l = new_sort(times, waveforms)
        result = create_hierarchy(t, w, l, 1000)
        suss.io.save_pickle(self.output().path, result)


class PrepareSorting(luigi.Task):
    """Placeholder to trigger automated spike sorting for on all channels
    """

    site_dir = luigi.Parameter()

    def requires(self):
        for elec in config.ELECTRODES:
            yield SortSpikes(site_dir=self.site_dir, electrode=elec)

    def run(self):
        pass


class ManualSorting(luigi.Task):
    site_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int,
        min_value=config.ELECTRODES[0],
        max_value=config.ELECTRODES[-1] + 1)

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "manually_curated")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return luigi.LocalTarget(os.path.join(
            sorting_dir,
            "curated-e{}.pkl".format(self.electrode)
        ))

    def requires(self):
        return SortSpikes(
            site_dir=self.site_dir,
            electrode=self.electrode
        )

    def run(self):
        # Manual sorting here
        pass


class PreprocessSite(luigi.Task):
    """Preprocess ephys data for entire site
    """

    site_dir = luigi.Parameter()

    def requires(self):
        return PrepareSorting(self.site_dir)
        pass

    def run(self):
        pass
