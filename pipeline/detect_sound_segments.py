from __future__ import print_function

import glob
import logging
import os

import luigi
import numpy as np
import pandas as pd
from neo.io import NixIO

import block_io
import config
from rhd2nix import Rhd2Nix
from segmentation.audio_segment import (
    convert_digital_to_metadata,
    segment,
    generate_stim_dicts,
    generate_all_segment_dicts,
    generate_vocal_periods_dicts,
    merge_segments,
)
from stimuli import convert_pyoperant_csvs
from utils import session_basename


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class CombineTrialMetadata(luigi.Task):
    """Combines data from pyoperant csvs and the neo block into one table

    Each pyoperant csv in raw/ has a list of stimulus presentations.
    Each time a stimulus set is run, it creates a new csv, so that
    occasionally multiple csvs are played during one session, and
    sometimes a set of stimuli can span two sessions.

    This task resolves these issues by comparing the data found in
    the pyoperant csvs to the digital pulses received in the recording
    software. It puts all the stimulus presentations in the session
    into a pandas dataframe that are in order of presentation.
    """

    session_dir = luigi.Parameter()

    def requires(self):
        return Rhd2Nix(self.session_dir)

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        if not os.path.exists(segmentation_dir):
            os.makedirs(segmentation_dir)

        combined_db = "playbacks.pkl"

        return luigi.LocalTarget(
                os.path.join(segmentation_dir, combined_db)
        )

    def run(self):
        input_directory = os.path.join(self.session_dir, "raw")
        stimuli_directory = os.path.join(self.session_dir, "stimuli")

        csv_files = glob.glob(os.path.join(input_directory, "*.csv"))

        if not len(csv_files):
            # no playback csvs -> create a dummy dataframe
            pd.DataFrame([]).to_csv(self.output().path)
            return

        # Identify and locate the stimulus directory (e.g. TutFamBos)
        # associated with each pyoperant csv
        stim_dirs = []
        possible_stim_dirs = [
            os.path.basename(f)
            for f in glob.glob(os.path.join(stimuli_directory, "*"))]

        for csv_file in csv_files:
            logger.info("Looking for stim dir for {}".format(csv_file))
            stimulus_names = pd.read_csv(csv_file)["stimulus_name"]
            if not len(stimulus_names):
                continue

            stim_dir_used = None
            for stim_dir in possible_stim_dirs:
                if any([(stim_dir in x) for x in stimulus_names]):
                    logger.info("Matched {}".format(stim_dir))
                    stim_dir_used = os.path.join(stimuli_directory, stim_dir)
                    break

            if stim_dir_used is None:
                raise Exception(
                    "Couldn't find the stim dir for {}".format(csv_file))

            stim_dirs.append(stim_dir_used)

        # determine how many playback sets there were and how many
        nix = NixIO(self.input().path)
        block = nix.read_block(lazy=True)
        events = nix.load_lazy_object(block.segments[0].events[0])
        stimulus_codes = convert_digital_to_metadata(events)

        # Combine the data from the csvs into a single pickled dataframe
        logger.info("Saving stim database {}".format(self.output().path))
        convert_pyoperant_csvs(
                stimulus_codes,
                csv_files,
                stim_dirs,
                self.output().path)


class SplitUp(luigi.Task):
    """Divide neo block up from one long segment into many short ones"""

    session_dir = luigi.Parameter()

    def requires(self):
        return Rhd2Nix(self.session_dir)

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        output_filename = "{}-segmented.pkl".format(
                session_basename(self.session_dir)
        )

        return luigi.LocalTarget(
                os.path.join(segmentation_dir, output_filename)
        )

    def run(self):
        nix = NixIO(self.input().path)
        block = nix.read_block()
        nix.close()

        create_segments = generate_all_segment_dicts(block, t_width=5.0)
        logger.info("Creating {} segments without overlap".format(
            len(create_segments))
        )
        new_block = segment(block, create_segments)
        block_io.save_block(self.output().path, new_block)


class DetectSoundSegments(luigi.Task):
    """Locate time periods containing either live or playback sounds
    
    Represents each time period by a dictionary with a start and stop
    time. Saves a list of live periods and playback periods.
    """

    session_dir = luigi.Parameter()

    def requires(self):
        return [
            Rhd2Nix(self.session_dir),
            CombineTrialMetadata(self.session_dir)
        ]

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        segment_data_file = "{}-segments.npy".format(
                session_basename(self.session_dir)
        )
        return luigi.LocalTarget(
                os.path.join(segmentation_dir, segment_data_file)
        )

    def run(self):
        nix = NixIO(self.input()[0].path)
        block = nix.read_block(lazy=True)
        # FIXME: We hardcoded 1 for the microphone analogsignal.
        # This index could conceivably change, so shouldn't be hardcoded
        sound = nix.load_lazy_object(block.segments[0].analogsignals[1])
        nix.close()

        sound_arr = sound.as_array()[:, 0]
        sampling_rate = sound.sampling_rate.item()

        ######################################################
        ### First search for periods of live vocalizations ###
        ######################################################

        live_dicts = generate_vocal_periods_dicts(
                sound_arr,
                sampling_rate
        )
        live_dicts = [
            seg_dict for seg_dict in live_dicts
            if (
                config.MIN_SEGMENT_LENGTH <=
                seg_dict["duration"].item() <=
                config.MAX_SEGMENT_LENGTH
            )
        ]

        logger.info("Found {} live vocalizations".format(len(live_dicts)))
        logger.info("{} between {} and {}".format(
            len(live_dicts),
            config.MIN_SEGMENT_LENGTH,
            config.MAX_SEGMENT_LENGTH
        ))

        #################################
        ### Identify playback periods ###
        #################################

        try:
            playbacks_db = pd.read_pickle(self.input()[1].path)
        except:
            logger.warning("No pyoperant file found (no playbacks)")
            playback_dicts = []
            segment_dicts = live_dicts
        else:
            digital_signals = block.segments[0].events[0]
            playback_dicts = generate_stim_dicts(
                    digital_signals,
                    playbacks_db
            )
            logger.info("Identified {} playback periods".format(
                len(playback_dicts))
            )
            segment_dicts = merge_segments(playback_dicts, live_dicts)

        ####################################
        ### Save identified time periods ###
        ####################################
        
        logger.info("Identified {} segments, ({} playback, {} live)".format(
                len(segment_dicts),
                len(playback_dicts),
                len(segment_dicts) - len(playback_dicts)
        ))

        np.save(
            self.output().path,
            {
                "live": live_dicts,
                "playback": playback_dicts
            }
        )
