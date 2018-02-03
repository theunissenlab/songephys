from __future__ import print_function

import cPickle as pickle
import datetime
import glob
import os
import shutil

import luigi
import numpy as np
import pandas as pd
from neo.io import NixIO

import config

from rhd2nix import Rhd2Nix
from setup import CombineTrialMetadata
from segmentation.audio_segment import (
    consolidate_segments,
    segment,
    generate_stim_dicts,
    generate_live_vocalization_dicts,
    generate_all_segment_dicts,
    generate_vocal_periods_dicts,
    merge_segments,
    load_stimulus_database,
    validate_digital_codes
)
from segmentation.autodetect import separate_song_sources

from utils import session_basename


class DetectSoundSegments(luigi.Task):
    """Segment data based on audio where songs are detected"""

    session_dir = luigi.Parameter()
    full = luigi.BoolParameter(default=False)

    def requires(self):
        return [Rhd2Nix(self.session_dir), CombineTrialMetadata(self.session_dir)]

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        output_filename = "{}-segmented.pkl".format(session_basename(self.session_dir))
        segment_data_file = "{}-segments.npy".format(session_basename(self.session_dir))
        return [
            luigi.LocalTarget(os.path.join(segmentation_dir, output_filename)),
            luigi.LocalTarget(os.path.join(segmentation_dir, segment_data_file))
        ]

    def run(self):
        nix = NixIO(self.input()[0].path)
        block = nix.read_block()
        nix.close()

        sound = block.segments[0].analogsignals[1]  # FIXME: the analogsignal index doesnt seem reliable
        sound_arr = sound.as_array()[:, 0]
        sampling_rate = sound.sampling_rate.item()

        # FIXME: my memory is getting super screwed by this... right now im saving myself
        live_dicts = generate_vocal_periods_dicts(sound_arr, sampling_rate)
        print("found {} live vocalizations".format(len(live_dicts)))
        live_dicts = [
            seg_dict for seg_dict in live_dicts
            if config.MIN_SEGMENT_LENGTH <= seg_dict["duration"].item() <= config.MAX_SEGMENT_LENGTH
        ]
        print("{} between {} and {}".format(len(live_dicts), config.MIN_SEGMENT_LENGTH, config.MAX_SEGMENT_LENGTH))
        del sound_arr

        playbacks_csv = pd.read_csv(self.input()[1][1].path)

        # FIXME: not sure if this will alwyas work
        if "time" in playbacks_csv:
            playbacks_csv["time"] = playbacks_csv["time"].apply(
                    lambda time_str: datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f"))

        if len(playbacks_csv) == 0:
            print("No pyoperant file found (no playbacks)")
            playback_dicts = []
            segment_dicts = live_dicts
        else:
            playbacks_db = load_stimulus_database(self.input()[1][0].path)
            digital_signals = block.segments[0].events[0]

            # the timing_offsets should be all close to zero
            # timing_offsets = validate_digital_codes(digital_signals, playbacks_csv)

            playback_dicts = generate_stim_dicts(digital_signals, playbacks_db.database)
            print("Identified {} playback periods".format(len(playback_dicts)))

            segment_dicts = merge_segments(playback_dicts, live_dicts)
            # TODO (kevin): Save segment_dicts to a file...


            print("Identified {} segments, ({} playback, {} live)".format(
                    len(segment_dicts),
                    len(playback_dicts),
                    len(segment_dicts) - len(playback_dicts)
            ))

        if self.full:
            create_segments = generate_all_segment_dicts(block, t_width=5.0)
        else:
            create_segments = consolidate_segments(segment_dicts)
        print("Creating {} segments without overlap".format(len(create_segments)))
        new_block = segment(block, create_segments)

        np.save(
            self.output()[1].path,
            {
                "live": live_dicts,
                "playback": playback_dicts,
                "consolidated": create_segments
            }
        )

        with open(self.output()[0].path, "wb") as output_file:
            pickle.dump(new_block, output_file)
