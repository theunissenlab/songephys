"""Ephys Data Pipeline

``luigi -m preprocess_ephys PreprocessSite --site-dir SITE_DIR``
"""

from collections import defaultdict
import os
import time

import luigi
import numpy as np
import matplotlib.pyplot as plt
from soundsig.sound import spectrogram
from soundsig.signal import bandpass_filter
from suss.core import SpikeDataset
import suss.io

from format.io import load_yaml
from format.wav import AudioFile, MultiWavReader
from preprocess.ephys.filter import common_average_filter
from preprocess.ephys.detect import extract_spikes
from preprocess.ephys.sort import sort, create_hierarchy

from file_setup import Nix2Wav, SiteConfig, SessionConfig, DefineSessions
from pipeline_utils import get_session_sites
import config


class FilterEphys(luigi.Task):
    """Apply filters to sections of
    """
    session_dir = luigi.Parameter()

    def requires(self):
        return {
            "config": SessionConfig(self.session_dir),
            "wavs": Nix2Wav(self.session_dir)
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        return luigi.LocalTarget(os.path.join(raw_dir, "ephys_filtered.wav"))

    def run(self):
        config = self.input()["config"].data()
        input_file = AudioFile(self.input()["wavs"]["ephys.wav"].path)
        output_file = AudioFile.create(
            self.output().path,
            samplerate=input_file.sampling_rate,
            channels=input_file.channels)

        try:
            common_average_filter(
                input_file,
                output_file,
                batch_size=65536,
                channel_groups=[np.arange(a, b) for a, b in config["filtering"]["ephys.wav"]],
            )
        except:
            output_file.close()
            os.remove(output_file.path)
            raise


class DetectSpikesOnElectrode(luigi.Task):
    session_dir = luigi.Parameter()
    electrode_idx = luigi.IntParameter()

    def requires(self):
        return FilterEphys(self.session_dir)

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return luigi.LocalTarget(os.path.join(sorting_dir, "spike_waveforms-e{}.pkl".format(self.electrode_idx)))

    def run(self):
        input_file = AudioFile(self.input().path)

        if self.electrode_idx >= input_file.channels:
            raise ValueError("Cannot read electrode {} on file {} with {} channels".format(self.electrode_idx, self.input().path, input_file.channels))

        spike_radius = int(config.SPIKE_WIDTH * input_file.sampling_rate)

        spikes = extract_spikes(
            input_file,
            channel=self.electrode_idx,
            # Compute threshold in 5 minute windows
            batch_size=int(input_file.sampling_rate * 60 * 5),
            threshold_std=config.SPIKE_THRESHOLD_STD,
            spike_radius=spike_radius,
        )

        spike_dataset = SpikeDataset(
            spikes["indices"] / input_file.sampling_rate,
            spikes["waveforms"],
            sample_rate=input_file.sampling_rate,
            labels=None
        )

        suss.io.save_pickle(
            self.output().path,
            spike_dataset
        )


class DetectSpikesInSession(luigi.Task):
    session_dir = luigi.Parameter()

    def requires(self):
        return FilterEphys(self.session_dir)

    def output(self):
        # A bit hacky -- to know what the output is going to be, this task
        # needs to know about how many ephys channels there are. However, this
        # method might be evaluated before the first file has even been downloaded
        # or the configuration file is even first set up!
        site_dirs, _ = get_session_sites(self.session_dir)
        if not len(site_dirs):
            return DetectSpikesOnElectrode(self.session_dir, 0).output()

        config_path = os.path.join(site_dirs[0], "site_config.yaml")
        if not os.path.exists(config_path):
            return DetectSpikesOnElectrode(self.session_dir, 0).output()

        config = load_yaml(config_path)
        if "nix2wav" not in config:
            return DetectSpikesOnElectrode(self.session_dir, 0).output()

        channels = 0
        for k, idx in config["nix2wav"]["ephys.wav"]:
            if isinstance(idx, int):
                channels += idx
            else:
                channels += idx[1] - idx[0]

        for ch in range(channels):
            yield DetectSpikesOnElectrode(self.session_dir, ch).output()

    def run(self):
        input_file = AudioFile(self.input().path)
        for channel in range(input_file.channels):
            yield DetectSpikesOnElectrode(session_dir=self.session_dir, electrode_idx=channel)


class CombineSiteSpikes(luigi.Task):
    """Merge together multiple sessions via offsets

    The sessions and relative offsets must be defined in a
    sessions.yaml file in the site directory.

    Outputs one suss.core.SpikeDataset per electrode channel, each
    combining the data from all sessions defined in the site config file.
    """

    site_dir = luigi.Parameter()

    def requires(self):
        return DefineSessions(self.site_dir)

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "spikes")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        # TODO: relying on a preset config.N_ELECTRODES is flimsy, but this method
        # can be run before any data is actually downloaded or configured yet.
        # Maybe it would be better to stuff this as a key in the site_config.yaml
        # file instead.
        return dict(
            (e, luigi.LocalTarget(os.path.join(
                sorting_dir,
                "spike_waveforms-e{}.pkl".format(e))))
            for e in range(config.N_ELECTRODES)
        )

    def run(self):
        config = self.input().data()
        session_offsets = [session["t"] for session in config["sessions"]]
        sessions_dir = os.path.join(os.path.dirname(os.path.dirname(self.site_dir)), "sessions")

        spike_results = []
        for session in config["sessions"]:
            detected = yield DetectSpikesInSession(os.path.join(sessions_dir, session["id"]))
            spike_results.append(list(detected))

        n_electrodes = len(list(spike_results[0]))

        for electrode_idx in range(n_electrodes):
            spike_datasets = [
                suss.io.read_pickle(spike_result[electrode_idx].path)
                for spike_result in spike_results
            ]

            # join together the spike times and waveforms
            # add the session start time to all times in each session
            spike_times = np.concatenate([
                data.times + offset
                for data, offset in zip(spike_datasets, session_offsets)
            ])
            waveforms = np.concatenate([data.waveforms for data in spike_datasets])
            new_dataset = SpikeDataset(
                spike_times,
                waveforms,
                sample_rate=spike_datasets[0].sample_rate
            )
            suss.io.save_pickle(self.output()[electrode_idx].path, new_dataset)


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
    electrode_idx = luigi.IntParameter()
    # add upgoing /  downgoing parameter?

    def requires(self):
        return CombineSiteSpikes(self.site_dir)

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "sort_results")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return luigi.LocalTarget(
            os.path.join(sorting_dir, "sorted-e{}.pkl".format(self.electrode_idx))
        )

    def run(self):
        dataset = suss.io.read_pickle(self.input()[self.electrode_idx].path)
        peaks = dataset.waveforms[:, dataset.waveforms.shape[1] // 2]
        times = dataset.times[peaks < 0]
        waveforms = dataset.waveforms[peaks < 0]
        t, w, l = sort(times, waveforms)
        result = create_hierarchy(t, w, l, 1000)
        suss.io.save_pickle(self.output().path, result)


class PrepareSorting(luigi.Task):
    """Placeholder to trigger automated spike sorting for on all channels
    """

    site_dir = luigi.Parameter()

    def requires(self):
        # TODO: relying on a preset config.N_ELECTRODES is flimsy, but this method
        # can be run before any data is actually downloaded or configured yet.
        # Maybe it would be better to stuff this as a key in the site_config.yaml
        # file instead.
        for elec in range(config.N_ELECTRODES):
            yield SortSpikes(site_dir=self.site_dir, electrode_idx=elec)

    def run(self):
        pass


class ManualSorting(luigi.Task):
    site_dir = luigi.Parameter()
    electrode_idx = luigi.IntParameter()

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
        return [
            DefineSessions(self.site_dir),
            PrepareSorting(self.site_dir)
        ]

    def run(self):
        pass
