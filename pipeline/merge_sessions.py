import os
from collections import defaultdict
from itertools import chain

import numpy as np
import luigi
import quantities as pq
from neo.core import Block

from block_io import load_yaml, load_block, save_block
from detect_sound_segments import DetectSoundSegments
from spike_detection import ExtractSpikes


def merge_blocks(blocks, t_offsets, merge_units=True):
    """
    Caution: modifies the timing of original blocks in-place
    to keep references to the same objects across channel index and segments
    """
    new_block = Block()
    for block, t_offset in zip(blocks, t_offsets):
        for segment in block.segments:
            for epoch in segment.epochs:
                epoch += t_offset
            for spiketrain in segment.spiketrains:
                spiketrain += t_offset
                spiketrain.t_start += t_offset
                spiketrain.t_stop += t_offset
            for analogsignal in segment.analogsignals:
                analogsignal.t_start += t_offset
            segment.annotations["times"] += t_offset

    new_block.segments = list(chain(*[block.segments for block in blocks]))

    new_block.channel_indexes = blocks[0].channel_indexes
    for i in range(len(new_block.channel_indexes)):
        new_block.channel_indexes[i].analogsignals = list(chain(*[
            block.channel_indexes[i].analogsignals
            for block in blocks
        ]))
        if not merge_units:
            new_block.channel_indexes[i].units = list(chain(*[
                block.channel_indexes[i].units
                for block in blocks
            ]))
        else:
            if len(new_block.channel_indexes[i].units) == 1:
                new_block.channel_indexes[i].units[0].spiketrains = list(
                    chain(*[
                        block.channel_indexes[i].units[0].spiketrains
                        for block in blocks
                    ])
                )

    return new_block


class MergeSessionBlocks(luigi.Task):
    """Merge together multiple blocks via offsets

    The sessions and relative offsets must be defined in a
    sessions.yaml file in the site directory.
    """

    site_dir = luigi.Parameter()

    def requires(self):
        self.session_info = load_yaml(
                os.path.join(self.site_dir, "sessions.yaml")
        )
        for session in self.session_info["sessions"]:
            session_dir = os.path.join(
                    self.site_dir,
                    "..",
                    "..",
                    "sessions",
                    session["id"]
            )
            yield ExtractSpikes(session_dir=session_dir)

    def output(self):
        return luigi.LocalTarget(
                os.path.join(self.site_dir, "thresholded.pkl")
        )

    def run(self):
        block_offsets = [
            session["t"] * pq.s
            for session in self.session_info["sessions"]
        ]
        blocks = [load_block(s.path) for s in self.input()]
        new_block = merge_blocks(blocks, block_offsets)

        save_block(self.output().path, new_block)


class MergeVocalPeriods(luigi.Task):
    """Merge together vocal period dicts

    The sessions and relative offsets must be defined in a
    sessions.yaml file in the site directory. This function
    takes the vocal periods (both live and playback) for each
    session and just adds the specified offset to each one.
    """

    site_dir = luigi.Parameter()

    def requires(self):
        self.session_info = load_yaml(
                os.path.join(self.site_dir, "sessions.yaml")
        )
        for session in self.session_info["sessions"]:
            session_dir = os.path.join(
                    self.site_dir,
                    "..",
                    "..",
                    "sessions",
                    session["id"]
            )
            yield DetectSoundSegments(session_dir=session_dir)

    def output(self):
        return luigi.LocalTarget(
                os.path.join(self.site_dir, "vocal_periods.npy")
        )

    def run(self):
        block_offsets = [
                session["t"] * pq.s
                for session in self.session_info["sessions"]
        ]
        vocal_period_data = [np.load(s.path)[()] for s in self.input()]

        combined_result = defaultdict(list)
        for vocal_periods, offset in zip(vocal_period_data, block_offsets):
            for key in vocal_periods:
                for period in vocal_periods[key]:
                    period["start_time"] += offset
                combined_result[key] += vocal_periods[key]

        np.save(self.output().path, combined_result)
