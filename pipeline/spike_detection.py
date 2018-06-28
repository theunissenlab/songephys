from __future__ import print_function

import os

import luigi

import block_io
from filter_signals import FilterElectrodeSignals
from preproephys import fix_chIndex, extract_spikes, drop_electrode_signals

from utils import session_basename


class ExtractSpikes(luigi.Task):

    session_dir = luigi.Parameter()

    def output(self):
        sorting_dir = os.path.join(self.session_dir, "spikes")
        if not os.path.exists(sorting_dir):
            os.makedirs(sorting_dir)

        output_file = "{}-thresholded.pkl".format(
                session_basename(self.session_dir)
        )
        return luigi.LocalTarget(os.path.join(sorting_dir, output_file))

    def requires(self):
        return FilterElectrodeSignals(self.session_dir)

    def run(self):
        block = block_io.load_block(self.input().path)

        fix_chIndex(block)
        extract_spikes(block)
        drop_electrode_signals(block)

        block_io.save_block(self.output().path, block)
