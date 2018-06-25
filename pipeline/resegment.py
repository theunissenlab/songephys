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
from spike_sorting import SortSite
from merge_sessions import MergeVocalPeriods
from segmentation.audio_segment import segment, merge_segments, join_continuous_segments
from utils import session_basename


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
        block = join_continuous_segments(block)
        new_block = segment(block, segment_dicts)

        block_io.save_block(self.output().path, new_block)
