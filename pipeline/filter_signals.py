from __future__ import print_function

import os
import time
import numpy as np

import luigi

import suss.io
import block_io
from detect_sound_segments import SplitUp
from rhd2nix import GenerateLazyFiles
from filtering.filters import (
        convert_to_epoch_obj,
        detect_noisy_epochs,
        filter_segment,
        filter_and_detect_spikes
)
from suss.core import SpikeDataset

from lazy import LazySignal
from utils import session_basename
import config


def estimate_time_remaining(started_at, fraction_complete):
    time_elapsed = time.time() - started_at
    if fraction_complete:
        est_total_time = time_elapsed / fraction_complete
    else:
        est_total_time = np.inf
    return time_elapsed, est_total_time - time_elapsed


class FilterElectrodeSignals(luigi.Task):

    session_dir = luigi.Parameter()

    def requires(self):
        return SplitUp(self.session_dir)

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        output_filename = "{}-filtered.pkl".format(
                session_basename(self.session_dir)
        )
        return luigi.LocalTarget(
                os.path.join(segmentation_dir, output_filename)
        )

    def run(self):
        block = block_io.load_block(self.input().path)

        def convert_mins(t_sec):
            if t_sec == np.inf:
                return "a long ass time"
            if t_sec / 60.0 > 60:
                hours = np.floor(t_sec / (60.0 * 60.0))
                minutes = (t_sec / 60.0) - (60.0 * int(hours))
                return "{:.0f}h{:.0f}m".format(hours, minutes)
            else:
                minutes = (t_sec / 60.0)
                return "{:.0f}m".format(minutes)

        start = time.time()
        for i, seg in enumerate(block.segments):
            elapsed, remaining = estimate_time_remaining(
                    start, float(i) / len(block.segments)
            )
            print(
                "Filtering Segment {}/{} in {}... "
                "Est. remaining: {}".format(
                    i,
                    len(block.segments),
                    convert_mins(elapsed),
                    convert_mins(remaining)
                )
            )
            clean_signal, predicted_signal = filter_segment(seg)

            # Identify noise epochs based on the residuals
            noise_epochs = detect_noisy_epochs(clean_signal, predicted_signal)
            # TODO: save noise epochs in a separate file
            epoch = convert_to_epoch_obj(noise_epochs)
            epoch.segment = seg
            seg.epochs.append(epoch)
            seg.create_relationship(force=True)

        block_io.save_block(self.output().path, block)


def touch(path):
    with open(path, "a"):
        os.utime(path, None)


class LazyFilterElectrodeSignals(luigi.Task):
    session_dir = luigi.Parameter()

    def requires(self):
        return GenerateLazyFiles(self.session_dir)

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        raw_dir = os.path.join(self.session_dir, "raw")

        # TODO this file will for now be untracked by luigi
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        filtering_finished_placeholder = os.path.join(
            sorting_dir,
            "filtering_finished"
        )

        return [
            luigi.LocalTarget(os.path.join(raw_dir, "filtered.memmap")),
            luigi.LocalTarget(os.path.join(self.session_dir, "noise.npy")),
            luigi.LocalTarget(filtering_finished_placeholder),
            luigi.LocalTarget(self.input().path),   # we are writing a new key in this object
        ] + [
            luigi.LocalTarget(os.path.join(sorting_dir, "spike_waveforms-e{}.pkl".format(elec)))
            for elec in config.ELECTRODES
        ]

    def run(self):
        signals = np.load(self.input().path)[()]
        signal = signals["raw"]

        output_signal = LazySignal(
            signal.N,
            signal.M,
            self.output()[0].path,
            signal.sampling_rate,
            mode="w+"
        )

        spike_ind, spike_wf, noise_events = filter_and_detect_spikes(
            signal,
            output_signal,
            signal.sampling_rate,
            65536
        )

        signals["filtered"] = output_signal
        np.save(self.output()[3].path, signals)
        touch(self.output()[2].path)
        np.save(self.output()[1].path, noise_events)

        for i, (elec, spikes, wfs) in enumerate(zip(config.ELECTRODES, spike_ind, spike_wf)):
            spike_dataset = SpikeDataset(
                spikes / signal.sampling_rate,
                wfs,
                sample_rate=signal.sampling_rate,
                labels=None
            )
            suss.io.save_pickle(self.output()[i + 4].path, spike_dataset)
