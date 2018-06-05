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
from preproephys import fix_chIndex, extract_spikes, drop_electrode_signals

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
        drop_electrode_signals(block)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(block, output_file, protocol=4)
