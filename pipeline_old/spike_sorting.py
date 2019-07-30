from __future__ import print_function

import os

import luigi
import neo.core
import quantities as pq
import numpy as np

import suss.io
from suss.core import SpikeDataset
from suss.gui.tags import ClusterTag
# from suss.sort import denoise, sort
from suss.sort3 import sort as sort3

from sort import sort_set as sort4

import block_io
import config
from merge_sessions import MergeSessionBlocks, MergeVocalPeriods, MergeDetectedSpikes
from segmentation.audio_segment import (
        join_continuous_segments,
        merge_segments,
)
from segmentation.audio_segment import segment as segment_block
from spikes.sorting.preprocess import collect_waveforms


def sort(dataset):
    return list(sort3(dataset))
    # for result in sort3(dataset):
    #     pass
    # return result


def correct_ch_idxs(block):
    ch_idxs = [ch_idx for ch_idx in block.channel_indexes if ch_idx.name.startswith("Amp Elec")]
    sorted_ch_idxes = sorted(ch_idxs, key=lambda c: int(c.name.split()[-1]))
    assert len(sorted_ch_idxes) == len(config.ELECTRODES)
    for ch_idx, true_elec in zip(sorted_ch_idxes, config.ELECTRODES):
        ch_idx.name = "Amp Elec {}".format(true_elec)

    return block


class CollectWaveforms(luigi.Task):

    site_dir = luigi.Parameter()

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

    def requires(self):
        return MergeSessionBlocks(self.site_dir)

    def run(self):
        block = block_io.load_block(self.input().path)
        correct_ch_idxs(block)

        for i, (elec_id, output) in enumerate(zip(config.ELECTRODES, self.output())):
            # a hack to get the electrode with index i
            print("this is its", [ch.name for ch in block.channel_indexes])
            unit = next(
                channel_index.units[0]
                for channel_index in block.channel_indexes
                if channel_index.name == "Amp Elec {}".format(elec_id)
            )

            waveforms, times = collect_waveforms(
                    unit.spiketrains,
                    force_single_peak=False
            )
            waveforms = waveforms[:, config.WF_SLICE]
            dataset = SpikeDataset(
                    times=times,
                    waveforms=waveforms,
                    sample_rate=config.ELEC_SAMPLE_RATE
            )
            suss.io.save_pickle(self.output()[i].path, dataset)


class SortSpikes(luigi.Task):

    site_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int, min_value=config.ELECTRODES[0], max_value=config.ELECTRODES[-1] + 1)

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "sort_results")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return luigi.LocalTarget(os.path.join(
            sorting_dir,
            "sorted-e{}.pkl".format(self.electrode)
        ))

    def requires(self):
        return MergeDetectedSpikes(self.site_dir)
        # return CollectWaveforms(self.site_dir)

    def run(self):
        idx = config.ELECTRODES.index(self.electrode)
        dataset = suss.io.read_pickle(self.input()[idx].path)
        # denoised = denoise(dataset.times, dataset.waveforms)
        result = [sort4(dataset)]
        # suss.io.save_pickle(os.path.join(os.path.dirname(self.output().path), "ungrouped-e{}.pkl".format(self.electrode)), result[-2])
        suss.io.save_pickle(self.output().path, result[-1])


class ManualCuration(luigi.Task):
    """Placeholder task for manual step"""

    site_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int, min_value=config.ELECTRODES[0], max_value=config.ELECTRODES[-1] + 1)

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "manually_curated")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return luigi.LocalTarget(os.path.join(
            sorting_dir,
            "curated-e{}.pkl".format(self.electrode)
        ))

    def requires(self):
        return SortSpikes(site_dir=self.site_dir, electrode=self.electrode)

    def run(self):
        # Manual sorting here
        pass


class PrepareForSorting(luigi.Task):

    site_dir = luigi.Parameter()

    def requires(self):
        for electrode in config.ELECTRODES:
            yield SortSpikes(site_dir=self.site_dir, electrode=electrode)

    def run(self):
        pass


class SortSite(luigi.Task):

    site_dir = luigi.Parameter()

    def output(self):
        # sorted.pkl
        return luigi.LocalTarget(os.path.join(self.site_dir, "sorted-resegmented.pkl"))

    def requires(self):
        yield MergeSessionBlocks(self.site_dir)
        for electrode in config.ELECTRODES:
            yield ManualCuration(site_dir=self.site_dir, electrode=electrode)
        # Save memory
        yield MergeVocalPeriods(self.site_dir)
        # End memory

    def run(self):
        # Save memory
        segments_data = np.load(self.input()[-1].path)[()]
        segment_dicts = merge_segments(
            segments_data["playback"],
            segments_data["live"]
        )
        # End Save memory

        block = block_io.load_block(self.input()[0].path)

        elec_channels = [
            channel_index
            for channel_index in block.channel_indexes
            if channel_index.name.startswith("Amp Elec")
        ]

        # remove old spiketrains from segments
        # remove old units from channel indexes
        for segment in block.segments:
            segment.spiketrains = []

        for channel_index in elec_channels:
            channel_index.units = []

        for elec_idx, (elec_id, clustered) in enumerate(zip(config.ELECTRODES, self.input()[1:-1])):
            print("Creating spiketrains for electrode {}".format(elec_id))
            sorted_data = suss.io.read_pickle(clustered.path)

            for cluster_idx, (cluster_label, cluster) in enumerate(zip(sorted_data.labels, sorted_data.nodes)):
                cluster_flat = cluster.flatten()
                tag_annotations = {}
                for tag in ClusterTag:
                    tag_annotations[tag.name] = tag in cluster.tags
                new_unit = neo.core.Unit(
                    name="e{}u{}".format(elec_id, cluster_label),
                    source=clustered.path,
                    label=cluster_label,
                    tags=tag_annotations
                )

                for segment in block.segments:
                    is_within_segment = (
                            (cluster_flat.times >= segment.t_start) &
                            (cluster_flat.times < segment.t_stop)
                    )
                    spike_train = neo.core.SpikeTrain(
                        cluster_flat.times[is_within_segment],
                        # TODO (kevin): this is where we can define the start
                        # and stop of a unit?
                        t_start=segment.t_start,
                        t_stop=segment.t_stop,
                        units=pq.s,
                        waveforms=cluster_flat.waveforms[is_within_segment]
                    )
                    new_unit.spiketrains.append(spike_train)
                    segment.spiketrains.append(spike_train)

                    spike_train.segment = segment
                    spike_train.unit = new_unit

                elec_channels[elec_idx].units.append(new_unit)
                new_unit.channel_index = elec_channels[elec_idx]

        print("Saving data to {}".format(self.output().path))

        # Save memory
        block = join_continuous_segments(block)
        new_block = segment_block(block, segment_dicts)

        block_io.save_block(self.output().path, new_block)
        # End Save memory

        # block_io.save_block(self.output().path, block)
