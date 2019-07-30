import os
import pickle
from collections import defaultdict
from itertools import chain

import numpy as np
import luigi
import quantities as pq
from neo.core import Block

from block_io import load_yaml, load_block, save_block
from detect_sound_segments import DetectSoundSegments, DetectSoundSegmentsLazy
from segmentation.audio_segment import merge_segments
from sound_detection.detect import shave_silences, get_surrounding_silence
from spike_detection import ExtractSpikes
from filter_signals import LazyFilterElectrodeSignals
import suss.io
from suss.core import SpikeDataset
from lazy import LazySite

import config


def merge_blocks(blocks, t_offsets, merge_units=True):
    """
    Caution: modifies the timing of original blocks in-place
    to keep references to the same objects across channel index and segments
    """
    new_block = Block()
    for block, t_offset in zip(blocks, t_offsets):
        for segment in block.segments:
            for epoch in segment.epochs:
                epoch += t_offset
            for spiketrain in segment.spiketrains:
                spiketrain += t_offset
                spiketrain.t_start += t_offset
                spiketrain.t_stop += t_offset
            for analogsignal in segment.analogsignals:
                analogsignal.t_start += t_offset
            segment.annotations["times"] += t_offset

    new_block.segments = list(chain(*[block.segments for block in blocks]))

    new_block.channel_indexes = blocks[0].channel_indexes
    for i in range(len(new_block.channel_indexes)):
        new_block.channel_indexes[i].analogsignals = list(chain(*[
            block.channel_indexes[i].analogsignals
            for block in blocks
        ]))
        if not merge_units:
            new_block.channel_indexes[i].units = list(chain(*[
                block.channel_indexes[i].units
                for block in blocks
            ]))
        else:
            if len(new_block.channel_indexes[i].units) == 1:
                new_block.channel_indexes[i].units[0].spiketrains = list(
                    chain(*[
                        block.channel_indexes[i].units[0].spiketrains
                        for block in blocks
                    ])
                )

    return new_block


class MergeSessionBlocks(luigi.Task):
    """Merge together multiple blocks via offsets

    The sessions and relative offsets must be defined in a
    sessions.yaml file in the site directory.
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
            yield ExtractSpikes(session_dir=session_dir)

    def output(self):
        return luigi.LocalTarget(
                os.path.join(self.site_dir, "thresholded.pkl")
        )

    def run(self):
        block_offsets = [
            session["t"] * pq.s
            for session in self.session_info["sessions"]
        ]
        blocks = [load_block(s.path) for s in self.input()]
        new_block = merge_blocks(blocks, block_offsets)

        save_block(self.output().path, new_block)


class MergeDetectedSpikes(luigi.Task):
    """Merge together multiple blocks via offsets

    The sessions and relative offsets must be defined in a
    sessions.yaml file in the site directory.
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
            yield LazyFilterElectrodeSignals(session_dir=session_dir)

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "spikes")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return [
            luigi.LocalTarget(
                os.path.join(
                    sorting_dir,
                    "spike_waveforms-e{}.pkl".format(electrode)
                )
            ) for electrode in config.ELECTRODES
        ]

    def run(self):
        block_offsets = [
            session["t"]
            for session in self.session_info["sessions"]
        ]
        for elec, files, output in zip(config.ELECTRODES, list(zip(*self.input()))[4:], self.output()):
            spike_datasets = [suss.io.read_pickle(file.path) for file in files]
            spike_times = np.concatenate([
                data.times + offset for data, offset in zip(spike_datasets, block_offsets)
            ])
            waveforms = np.concatenate([data.waveforms for data in spike_datasets])
            new_dataset = SpikeDataset(
                spike_times,
                waveforms,
                sample_rate=spike_datasets[0].sample_rate
            )
            suss.io.save_pickle(output.path, new_dataset)


class MergeVocalPeriods(luigi.Task):
    """Merge together vocal period dicts

    The sessions and relative offsets must be defined in a
    sessions.yaml file in the site directory. This function
    takes the vocal periods (both live and playback) for each
    session and just adds the specified offset to each one.
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
            yield DetectSoundSegmentsLazy(session_dir=session_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "vocal_periods.npy"))

    def run(self):
        block_offsets = [
                session["t"] * pq.s
                for session in self.session_info["sessions"]
        ]
        vocal_period_data = [np.load(s.path)[()] for s in self.input()]

        combined_result = defaultdict(list)
        for vocal_periods, offset in zip(vocal_period_data, block_offsets):
            for key in vocal_periods:
                for period in vocal_periods[key]:
                    period["start_time"] += offset
                combined_result[key] += vocal_periods[key]
        combined_result = dict(combined_result)

        np.save(self.output().path, combined_result)


class CreateLazySite(luigi.Task):

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
            yield LazyFilterElectrodeSignals(session_dir=session_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "lazy_data.pkl"))

    def run(self):
        lazy_signals_raw = []
        lazy_signals_mic = []
        lazy_signals_filtered = []
        signal_times = []
        for filter_outputs, session in zip(self.input(), self.session_info["sessions"]):
            # FIXME: indexing like this into another task's output is fragile...
            # fix this later???
            input_file = filter_outputs[3]
            session_sigs = np.load(input_file.path)[()]
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

        with open(self.output().path, "wb") as picklefile:
            pickle.dump(new_site, picklefile)


class DetectVocalAndSilentPeriods(luigi.Task):

    site_dir = luigi.Parameter()

    def requires(self):
        return [
            CreateLazySite(self.site_dir),
            MergeVocalPeriods(self.site_dir)
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "disjoint_vocal_periods.npy"))

    def run(self):
        with open(self.input()[0].path, "rb") as picklefile:
            lazy_site = pickle.load(picklefile)

        vocal_periods = np.load(self.input()[1].path)[()]

        # this odesnt belong here but i need to do it
        vocal_periods["playback"] = [e for e in vocal_periods["playback"] if e["duration"].item() > 1.5]
        vocal_periods["live"] = [e for e in vocal_periods["live"] if e["duration"].item() > 1.5]

        all_events = np.array(merge_segments(
            vocal_periods["playback"],
            vocal_periods["live"]
        ))

        silent_periods = get_surrounding_silence(
            all_events,
            lazy_site,
            channel=0,
            min_duration=1.0,
            min_sound=0.002,
        )

        all_events = sorted(
            np.concatenate([all_events, silent_periods]),
            key=lambda e: e["start_time"]
        )

        all_events = np.array([
            shave_silences(event, lazy_site)
            for event in all_events
        ])

        np.save(self.output().path, all_events)

