import cPickle as pickle
import glob
import os
import shutil

import luigi
from neo.io import NixIO

from rhd2nix import Rhd2Nix
from segmentation.audio_segment import (
    segment,
    generate_stim_dicts,
    generate_live_vocalization_dicts,
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
        assert len(csv_files) == 1

        # FIXME dont hardcode stim dir
        stim_dir = os.path.join(self.session_dir, "stimuli", "TutFamBos")

        # locate pyoperant file
        pyoperant_file = csv_files[0]

        sm = save_stimulus_database(pyoperant_file,
                stim_dir,
                "{}_db".format(session_basename(self.session_dir)))
        digital_signals = block.segments[0].events[0]

        playback_dicts = generate_stim_dicts(digital_signals, sm.database)

        # TODO: need to generate dataframe or input to live vocalization detector
        # live_dicts = generate_live_vocalization_dicts(df)
        # segment_dicts = merge_segments(playback_dicts, live_dict)
        segment_dicts = playback_dicts

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
