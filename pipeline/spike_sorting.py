from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

import numpy as np
import scipy.stats
import luigi
import matplotlib.pyplot as plt
import neo.core
import quantities as pq

from merge_sessions import MergeSessionBlocks, MergeVocalPeriods
# from spike_detection import ExtractSpikes
from preproephys import fix_chIndex, extract_spikes
from spikes.sorting.preprocess import collect_waveforms, sort_spikes_on_unit

import suss.io
from suss.core import SpikeDataset
try:
    from suss.sort import sexy_sort as sort
except:
    from suss.sort import default_sort as sort

import block_io
import config

from utils import session_basename


class CollectWaveforms(luigi.Task):

    site_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "spikes")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return [
            luigi.LocalTarget(
                os.path.join(sorting_dir, "spike_waveforms-e{}.pkl".format(electrode))
            ) for electrode in range(16)
        ]
            
    def requires(self):
        return MergeSessionBlocks(self.site_dir)

    def run(self):
        block = block_io.load_block(self.input().path)

        for i, output in enumerate(self.output()):
            # a hack to get the electrode with index i
            unit = next(
                channel_index.units[0]
                for channel_index in block.channel_indexes
                if channel_index.name == "Amp Elec {}".format(i)
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
        var_type=int, min_value=0, max_value=16)

    def output(self):
        sorting_dir = os.path.join(self.site_dir, "sort_results")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        return luigi.LocalTarget(os.path.join(
            sorting_dir,
            "sorted-e{}.pkl".format(self.electrode)
        ))
            
    def requires(self):
        return CollectWaveforms(self.site_dir)

    def run(self):
        dataset = suss.io.read_pickle(self.input()[self.electrode].path)
        result = sort(dataset.times, dataset.waveforms, dataset.sample_rate)[-1]
        suss.io.save_pickle(self.output().path, result)


class ManualCuration(luigi.Task):
    """Placeholder task for manual step
    """
    site_dir = luigi.Parameter()
    electrode = luigi.parameter.NumericalParameter(
        var_type=int, min_value=0, max_value=16)

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
        ## sorting
        pass


class SortSite(luigi.Task):

    site_dir = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "sorted.pkl"))

    def requires(self):
        yield MergeSessionBlocks(self.site_dir)
        for electrode in range(16):
            yield ManualCuration(site_dir=self.site_dir, electrode=electrode)

    def run(self):
        block = block_io.load_block(self.input()[0].path)

        elec_channels = [channel_index
                for channel_index in block.channel_indexes
                if channel_index.name.startswith("Amp Elec")]

        # remove old spiketrains from segments
        # remove old units from channel indexes
        for segment in block.segments:
            segment.spiketrains = []

        for channel_index in elec_channels:
            channel_index.units = []

        results = []
        for elec_id, clustered in enumerate(self.input()[1:]):
            sorted_data = suss.io.read_pickle(clustered.path)

            for cluster_idx, cluster in enumerate(sorted_data.nodes):
                cluster = cluster.flatten()
                new_unit = neo.core.Unit(
                    name="e{}u{}".format(elec_id, cluster_idx),
                    source=clustered.path
                )

                for segment in block.segments:
                    is_within_segment = (
                            (cluster.times >= segment.t_start) &
                            (cluster.times < segment.t_stop)
                    )
                    spike_train = neo.core.SpikeTrain(
                        cluster.times[is_within_segment],
                        # TODO (kevin): this is where we can define the start
                        # and stop of a unit?
                        t_start=segment.t_start,
                        t_stop=segment.t_stop,
                        units=pq.s,
                        waveforms=cluster.waveforms[is_within_segment]
                    )
                    new_unit.spiketrains.append(spike_train)
                    segment.spiketrains.append(spike_train)

                    spike_train.segment = segment
                    spike_train.unit = new_unit

                elec_channels[elec_id].units.append(new_unit)
                new_unit.channel_index = elec_channels[elec_id]

            ### Create noise cluster also
            flattened = sorted_data.flatten()
            noise_cluster = flattened.source.complement(flattened)

            new_unit = neo.core.Unit(
                name="e{}-noise".format(elec_id, cluster_idx),
                source=clustered.path
            )

            for segment in block.segments:
                is_within_segment = (
                        (noise_cluster.times >= segment.t_start) &
                        (noise_cluster.times < segment.t_stop)
                )
                spike_train = neo.core.SpikeTrain(
                    noise_cluster.times[is_within_segment],
                    # TODO (kevin): this is where we can define the start
                    # and stop of a unit?
                    t_start=segment.t_start,
                    t_stop=segment.t_stop,
                    units=pq.s,
                    waveforms=noise_cluster.waveforms[is_within_segment]
                )
                new_unit.spiketrains.append(spike_train)
                segment.spiketrains.append(spike_train)

                spike_train.segment = segment
                spike_train.unit = new_unit

            elec_channels[elec_id].units.append(new_unit)
            new_unit.channel_index = elec_channels[elec_id]

            ### end create noise cluster

        block_io.save_block(self.output().path, block)
