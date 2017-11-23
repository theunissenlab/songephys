import cPickle as pickle
import glob
import os
import shutil

import luigi
import pandas as pd
from neo.io import NixIO

import config

from rhd2nix import Rhd2Nix
from setup import CombineTrialMetadata
from segmentation.audio_segment import (
    segment,
    generate_stim_dicts,
    generate_live_vocalization_dicts,
    generate_vocal_periods_dicts,
    merge_segments,
    load_stimulus_database
)

from utils import session_basename


class DetectSoundSegments(luigi.Task):
    """Segment data based on audio where songs are detected"""

    session_dir = luigi.Parameter()

    def requires(self):
        return [Rhd2Nix(self.session_dir), CombineTrialMetadata(self.session_dir)]

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        output_filename = "{}-segmented.pkl".format(session_basename(self.session_dir))
        return luigi.LocalTarget(os.path.join(segmentation_dir, output_filename))

    def run(self):
        nix = NixIO(self.input()[0].path)
        block = nix.read_block()
        nix.close()

        sound = block.segments[0].analogsignals[1]  # FIXME: the analogsignal index doesnt seem reliable
        sound_arr = sound.as_array()[:, 0]
        sampling_rate = sound.sampling_rate.item()

        # TODO: will at some point want live dicts to be categorized/labeled
        # live_dicts = generate_live_vocalization_dicts(df)
        # FIXME: my memory is getting super screwed by this... im going to have to cut down on them for now
        live_dicts = generate_vocal_periods_dicts(sound_arr, sampling_rate)
        print len(live_dicts)
        live_dicts = [
            seg_dict for seg_dict in live_dicts
            if config.MIN_SEGMENT_LENGTH <= seg_dict["duration"].item() <= config.MAX_SEGMENT_LENGTH
        ]
        print len(live_dicts)
        # if len(live_dicts) > 500:
            # raise Exception("Too many vocalization periods found ({}). Filter more out or make the code better.".format(len(live_dicts)))

        print "Identified {} vocalization periods".format(len(live_dicts))
        del sound_arr

        playbacks_csv = pd.read_csv(self.input()[1][1].path)

        if len(playbacks_csv) == 0:
            print "No pyoperant file found (no playbacks)"
            segment_dicts = live_dicts
        else:
            playbacks_db = load_stimulus_database(self.input()[1][0].path)
            digital_signals = block.segments[0].events[0]

            # TODO: validate playbacks csv and digital signals here

            playback_dicts = generate_stim_dicts(digital_signals, playbacks_db.database)
            print "Identified {} playback periods".format(len(playback_dicts))

            segment_dicts = merge_segments(playback_dicts, live_dicts)
            print "Creating {} segments, ({} playback, {} live)".format(
                    len(segment_dicts),
                    len(playback_dicts),
                    len(segment_dicts) - len(playback_dicts)
            )

        new_block = segment(block, segment_dicts)

        with open(self.output().path, "wb") as output_file:
            pickle.dump(new_block, output_file)


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
