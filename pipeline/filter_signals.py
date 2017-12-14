import cPickle as pickle
import os

import luigi

from detect_sound_segments import DetectSoundSegments
# from preproephys import high_low_filter
from filtering.filters import filter_segment

from utils import session_basename


class FilterElectrodeSignals(luigi.Task):

    session_dir = luigi.Parameter()

    def requires(self):
        return DetectSoundSegments(self.session_dir)

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        output_filename = "{}-filtered.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(segmentation_dir, output_filename))

    def run(self):
        with open(self.input().path, "rb") as inputfile:
            block = pickle.load(inputfile)
        block.create_relationship(force=True, append=True, recursive=True)

        for seg in block.segments:
            filter_segment(seg)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(block, output_file)
