import cPickle as pickle
import os
import shutil

import luigi
from neo.io import NixIO

from rhd2nix import Rhd2Nix
from segmentation.audio_segment import segment

from utils import session_basename


class DetectSoundSegments(luigi.Task):
    """Segment data based on audio where songs are detected"""

    session_dir = luigi.Parameter()

    def requires(self):
        return Rhd2Nix(self.session_dir)

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        output_filename = "{}-segmented.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(segmentation_dir, output_filename))

    def run(self):
        nix = NixIO(self.input().path)
        block = nix.read_block()
        nix.close()

        block = segment(block)

        with open(self.output().path, "wb") as output_file:
           pickle.dump(block, output_file)


class ValidateSongs(luigi.Task):
    """Placeholder task for manual validation of songs"""

    session_dir = luigi.Parameter()

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        output_filename = "{}-validated.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(segmentation_dir, output_filename))

    def requires(self):
        return DetectSoundSegments(self.session_dir)

    def run(self):
        shutil.copy(self.input().path, self.output().path)
