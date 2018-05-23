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

from filter_signals import FilterElectrodeSignals
from preproephys import fix_chIndex, extract_spikes, drop_raw_traces

from sklearn.manifold import TSNE
from spikes.write_neo import write_neo

from spikes.sorting.preprocess import collect_waveforms, sort_spikes_on_unit
from spikes.sorting.merging import isosplit_cluster
# from spikes_vis.runsort import create_server
from spikes.sorting.autocluster.process_block import time_slices, process_votes, relabel_across_time, gen_figures
from spikes.sorting.autocluster import sort
from spikes.sorting.metrics import cluster_snr, interspike_interval, firing_rate, consistency, isolation
import spikes.sorting.cluster_linking as clust

import webbrowser

import config

from utils import session_basename


class ExtractSpikes(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        output_file = "{}-thresholded.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(sorting_dir, output_file))

    def requires(self):
        return FilterElectrodeSignals(self.session_dir)

    def run(self):
        with open(self.input().path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

        fix_chIndex(block)
        extract_spikes(block)
        drop_raw_traces(block)
        # TODO (drop the raw spike waveform here)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(block, output_file, protocol=4)


class CollectWaveforms(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")

        return [
            luigi.LocalTarget(
                os.path.join(sorting_dir, "spike_waveforms-e{}.npy".format(electrode))
            ) for electrode in range(16)
        ]
            
    def requires(self):
        return ExtractSpikes(self.session_dir)

    def run(self):
        with open(self.input().path, "rb") as input_file:
            block = pickle.load(input_file)
        block.create_relationship(force=True, append=True, recursive=True)

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
            np.save(output.path, {"waveforms": waveforms, "times": times})


class MergeThresholdedSessions(luigi.Task):

    session_dirs = luigi.Parameter()

    def output(self):
        # Create a new directory that has all three unsorted blocks
        pass



    def requires(self):
        return [
            CollectWaveforms(session_dir=session_dir)
            for session_dir in self.session_dirs
        ]

    def run(self):
        pass
