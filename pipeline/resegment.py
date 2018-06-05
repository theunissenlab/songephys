try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import os
import time

import luigi

import block_io
from detect_sound_segments import DetectSoundSegments
# from spike_sorting import WriteSemiAutoSorted
from spike_sorting import SortSite
from merge_sessions import MergeVocalPeriods
from segmentation.audio_segment import segment, merge_segments, one_long_segment
from utils import session_basename
'''

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

'''

class ResegmentSite(luigi.Task):
    """Recut segments to match previously identified
    playback and live vocalization periods
    """

    site_dir = luigi.Parameter()

    def requires(self):
        return [
            SortSite(self.site_dir),
            MergeVocalPeriods(self.site_dir)
        ]

    def output(self):
        output_filename = "sorted-resegmented.pkl"
        return luigi.LocalTarget(os.path.join(self.site_dir, output_filename))

    def run(self):
        segments_data = np.load(self.input()[1].path)[()]

        block = block_io.load_block(self.input()[0].path)

        segment_dicts = merge_segments(
            segments_data["playback"],
            segments_data["live"]
        )

        # check if block.segments continuously covers entire duration
        block = one_long_segment(block)
        new_block = segment(block, segment_dicts)

        block_io.save_block(self.output().path, new_block)
