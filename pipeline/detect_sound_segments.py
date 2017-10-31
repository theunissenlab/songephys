import cPickle as pickle
import glob
import os
import shutil

import luigi
from neo.io import NixIO

import config

from rhd2nix import Rhd2Nix
from segmentation.audio_segment import (
    segment,
    generate_stim_dicts,
    generate_live_vocalization_dicts,
    generate_vocal_periods_dicts,
    merge_segments,
    save_stimulus_database
)

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

        input_directory = os.path.join(self.session_dir, "raw")
        csv_files = glob.glob(os.path.join(input_directory, "*.csv"))

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
        if len(live_dicts) > 500:
            raise Exception("Too many vocalization periods found ({}). Filter more out or make the code better.".format(len(live_dicts)))

        print "Identified {} vocalization periods".format(len(live_dicts))
        del sound_arr

        if len(csv_files) == 0:
            print "No pyoperant file found"
            # then there are no playbacks
            segment_dicts = live_dicts
        elif len(csv_files) == 1:
            # FIXME dont hardcode stim dir
            stim_dir = os.path.join(self.session_dir, "stimuli", config.STIM_DIR)

            # locate pyoperant file
            pyoperant_file = csv_files[0]

            # FIXME: stimulus database location must be an output of this task!
            sm = save_stimulus_database(pyoperant_file,
                    stim_dir,
                    "{}_db".format(session_basename(self.session_dir)))
            digital_signals = block.segments[0].events[0]

            playback_dicts = generate_stim_dicts(digital_signals, sm.database)
            print "Identified {} playback periods".format(len(playback_dicts))

            segment_dicts = merge_segments(playback_dicts, live_dicts)
            print "Creating {} segments, ({} playback, {} live)".format(
                    len(segment_dicts),
                    len(playback_dicts),
                    len(segment_dicts) - len(playback_dicts)
            )
        else:
            # to handle multiple, will need to modify generate_stim_dicts??
            raise Exception("The code is not equipped for more than one playback file yet")

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
