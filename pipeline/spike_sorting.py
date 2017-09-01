import cPickle as pickle
import os

import luigi

from filter_signals import FilterElectrodeSignals
from preproephys import fix_chIndex, extract_spikes


class ExtractSpikes(luigi.Task):

    in_rhdfile = luigi.Parameter()

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return luigi.LocalTarget("{}-thresholded.pkl".format(input_file))

    def requires(self):
        return FilterElectrodeSignals(self.in_rhdfile)

    def run(self):
        with open(self.input().path, "rb") as inputfile:
            block = pickle.load(inputfile)
        block.create_relationship(force=True, append=True, recursive=True)

        fix_chIndex(block)
        extract_spikes(block)

        with open(self.output().path, "wb") as outputfile:
            pickle.dump(block, outputfile)
