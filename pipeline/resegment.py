try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import os
import time

import luigi

from detect_sound_segments import DetectSoundSegments
from spike_sorting import WriteSemiAutoSorted
from segmentation.audio_segment import segment, merge_segments, one_long_segment

from utils import session_basename


class Resegment(luigi.Task):
    """Recut segments to match previously identified
    playback and live vocalization periods
    """

    session_dir = luigi.Parameter()
    full = luigi.BoolParameter(default=True)

    def requires(self):
        return [
            DetectSoundSegments(self.session_dir),
            WriteSemiAutoSorted(self.session_dir)
        ]

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "spikes")
        output_filename = "{}-sorted-resegmented.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(segmentation_dir, output_filename))

    def run(self):
        segments_data = np.load(self.input()[0][1].path)[()]

        with open(self.input()[1].path, "rb") as inputfile:
            block = pickle.load(inputfile)
        block.create_relationship(force=True, append=True, recursive=True)

        segment_dicts = merge_segments(
            segments_data["playback"],
            segments_data["live"]
        )

        # check if block.segments continuously covers entire duration
        if self.full:
            block = one_long_segment(block)

        new_block = segment(block, segment_dicts)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(new_block, output_file, protocol=4)
