import cPickle as pickle
import os

import luigi

from detect_sound_segments import ValidateSongs
from preproephys import high_low_filter


class FilterElectrodeSignals(luigi.Task):
    """huhh"""

    in_rhdfile = luigi.Parameter()

    def requires(self):
        return ValidateSongs(self.in_rhdfile)

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return luigi.LocalTarget("{}-segmented.pkl".format(input_file))

    def run(self):
        with open(self.input().path, "rb") as inputfile:
            block = pickle.load(inputfile)
        block.create_relationship(force=True, append=True, recursive=True)

        for seg in block.segments:
            high_low_filter(seg)

        with open(self.output().path, "wb") as outputfile:
            pickle.dump(block, outputfile)
