import cPickle as pickle
import os
import shutil

import luigi
from neo.io import NixIO

from rhd2nix import Rhd2Nix
from segmentation.audio_segment import segment


class DetectSoundSegments(luigi.Task):
    """Segment data based on audio where songs are detected"""

    in_rhdfile = luigi.Parameter()

    def requires(self):
        return Rhd2Nix(self.in_rhdfile)

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return luigi.LocalTarget("{}-segmented.pkl".format(input_file))

    def run(self):
        nix = NixIO(self.input().path)
        block = nix.read_block()
        nix.close()

        block = segment(block)

        with open(self.output().path, "wb") as outputfile:
           pickle.dump(block, outputfile)


class ValidateSongs(luigi.Task):
    """Placeholder task for manual validation of songs"""

    in_rhdfile = luigi.Parameter()

    def output(self):
        input_file = os.path.splitext(self.in_rhdfile)[0]
        return luigi.LocalTarget("{}-validated.pkl".format(input_file))

    def requires(self):
        return DetectSoundSegments(self.in_rhdfile)

    def run(self):
        shutil.copy(self.input().path, self.output().path)
