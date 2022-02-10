import glob
import logging
import os
from collections import defaultdict

import luigi
import numpy as np
import pandas as pd
import quantities as pq
import soundfile as sf
from neo.io import NixIO

from format.io import load_yaml
from format.intan_utils import timestamp_from_intan_filename
from format.wav import AudioFile
from preprocess.sound.playback_utils import (
    convert_digital_to_metadata,
    convert_pyoperant_csvs,
    read_pyoperant_csv
)
from preprocess.sound.playback_alignment import (
    compute_realigned_trial_times,
    create_replacement_codes
)
from preprocess.sound.detect import (
    detect_call_and_syllable_dicts_from_lazy,
    generate_stim_dicts,
    playback_columns,
    autodetected_columns,
    # generate_vocal_periods_dicts_from_lazy,
)
from preprocess.sound.stimuli import merge_segments
# from preprocess_ephys import MergeLazySignals

from file_setup import Nix2Wav, Rhd2Nix, SiteConfig, SessionConfig
from pipeline_utils import SiteParameter, SessionParameter
import config


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _get_stim_dirs(csv_files, stimuli_directory):
    """Locates the specific stimulus directories found in the csv files

    Locates one stimulus directory per csv (even if there might be multiple
    stimulus directories used in the csv)

    Returns one stimulus directory per csv_file
    """
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
                # stim_dir_used = os.path.join(stimuli_directory, stim_dir)
                # We are just checking it exists the stim_dir is added in locate_stimulus_directory in playback_utils.py
                stim_dir_used = stimuli_directory
                break

        if stim_dir_used is None:
            raise Exception(
                "Couldn't find the stim dir for {}".format(csv_file))

        stim_dirs.append(stim_dir_used)

    return stim_dirs


class CreateMicrophoneAlignedCodes(luigi.Task):
    """Align stimulus playback info with microphone data

    Attempts to align the stimulus wav files to the audio picked up on the
    microphones to determine what timestamps the trials occured at.

    These timestmaps can be used to validate playback times recorded by
    our digital codes or to fill in digital codes if they are not there.

    Inputs
    ------
    session_dir : str
        Path to session. The session_dir/raw directory must contain csv
        files for all playbacks played during the session

    Outputs a numpy array of trial codes for alignment of data to trials
    """
    session_dir = SessionParameter()

    def requires(self):
        return Nix2Wav(self.session_dir)

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")

        return luigi.LocalTarget(
            os.path.join(raw_dir, "microphone_aligned_codes.npy")
        )

    def run(self):
        config_file = yield SessionConfig(self.session_dir)
        session_config = config_file.data()

        input_file = AudioFile(os.path.join(self.input()["mic.wav"].path))

        input_directory = os.path.join(self.session_dir, "raw")
        stimuli_directory = os.path.join(self.session_dir, "stimuli")
        csv_files = sorted(glob.glob(os.path.join(input_directory, "*.csv")))

        # Identify and locate the stimulus directory (e.g. TutFamBos)
        # associated with each pyoperant csv
        stim_dirs = _get_stim_dirs(csv_files, stimuli_directory)

        max_time = input_file.frames / input_file.sampling_rate

        new_codes = []
        for csv_file, stim_dir in zip(csv_files, stim_dirs):
            playback_df = read_pyoperant_csv(csv_file, stim_dir)
            session_start_timestamp = timestamp_from_intan_filename(self.session_dir)

            aligned_times = compute_realigned_trial_times(
                    playback_df,
                    session_start_timestamp,
                    input_file,
                    playback_channels=session_config.get("playback_mic_channels"),
                    window=1.0
            )

            new_codes.extend(create_replacement_codes(
                playback_df,
                aligned_times,
                input_file.sampling_rate,
                max_time
            ))

        new_codes = sorted(new_codes, key=lambda x: x[0])

        np.save(self.output().path, new_codes)


class CombinePlaybackMetadata(luigi.Task):
    """Combine data from multiple csvs and the recorded digital input signal

    Every time a playback is played, a digital code is sent and recorded in the intan
    system. A playback session created by our pyoperant code makes a csv
    with one row for each playback trial - there can be multiple playback sessions
    (i.e. multiple csvs) in one session. A second issue is that the beginning or
    end of a playback session may be cut off at the ends of the session recorded.

    This task attempts to resolve this by comparing the data from multiple csvs to the
    digital codes received in the recording software. It puts all the stimulus
    presentations in the session into a single pandas dataframe in order of presentation.

    Inputs
    ------
    session_dir : str
        Path to session. The session_dir/raw directory must contain csv files for
        all playbacks played during the session.

    Outputs a pickle file containing a pandas dataframe with the playback metadata
    for all playbacks in the session,
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            "nix_file": Rhd2Nix(self.session_dir),
            "aligned_codes": CreateMicrophoneAlignedCodes(self.session_dir)
        }

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

        csv_files = sorted(glob.glob(os.path.join(input_directory, "*.csv")))

        if not len(csv_files):
            # no playback csvs -> create a dummy dataframe
            pd.DataFrame([]).to_csv(self.output().path)
            return

        # Identify and locate the stimulus directory (e.g. TutFamBos)
        # associated with each pyoperant csv
        stim_dirs = _get_stim_dirs(csv_files, stimuli_directory)

        # determine how many playback sets there were and how many
        nix = NixIO(self.input()["nix_file"].path)
        block = nix.read_block(lazy=True)

        if not len(block.segments[0].events):
            # If we messed up the digital codes...
            stimulus_codes = np.load(self.input()["aligned_codes"].path, allow_pickle=True)[()]
        else:
            events = nix.load_lazy_object(block.segments[0].events[0])
            stimulus_codes = convert_digital_to_metadata(events)

        # Combine the data from the csvs into a single pickled dataframe
        logger.info("Saving stim database {}".format(self.output().path))
        convert_pyoperant_csvs(
                stimulus_codes,
                csv_files,
                stim_dirs,
                self.output().path)


class PlaybacksCSV(luigi.Task):
    """Locate time periods containing playback sounds

    Represents each time period by a dictionary with a start and stop
    time. Saves a list of playback periods.
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            "wav_files": Nix2Wav(self.session_dir),
            "aligned_codes": CreateMicrophoneAlignedCodes(self.session_dir),
            "playback_metadata": CombinePlaybackMetadata(self.session_dir),
            "nix_file": Rhd2Nix(self.session_dir)
        }

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        segment_data_file = "playback_segments.csv"
        return luigi.LocalTarget(
            os.path.join(segmentation_dir, segment_data_file)
        )

    def run(self):
        # Live vocalization detection is now delegated to the FindLiveCalls task
        mic = AudioFile(self.input()["wav_files"]["mic.wav"].path)

        try:
            playbacks_db = pd.read_pickle(self.input()["playback_metadata"].path)
        except:
            logger.warning("No pyoperant file found (no playbacks)")
            playback_dicts = []
        else:
            nix = NixIO(self.input()["nix_file"].path)
            block = nix.read_block(lazy=True)

            if not len(block.segments[0].events):
                stimulus_codes = np.load(self.input()["aligned_codes"].path, allow_pickle=True)[()]
                sampling_rate = mic.sampling_rate
            else:
                digital_signals = nix.load_lazy_object(block.segments[0].events[0])
                stimulus_codes = convert_digital_to_metadata(digital_signals)
                sampling_rate = digital_signals.annotations["sampling_rate"].item()

            playback_dicts = generate_stim_dicts(
                    stimulus_codes,
                    sampling_rate,
                    playbacks_db
            )
            logger.info("Identified {} playback periods".format(
                len(playback_dicts))
            )
            nix.close()

        df = pd.DataFrame(playback_dicts, columns=playback_columns)
        df.to_csv(self.output().path, index=False)


class FindLiveCalls(luigi.Task):

    session_dir = SessionParameter()

    def requires(self):
        return {
            "wav_files": Nix2Wav(self.session_dir),
            "vocal_periods": FindVocalPeriods(self.session_dir),
        }

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        segment_data_file = "autodetected_segments.csv"
        return luigi.LocalTarget(
            os.path.join(segmentation_dir, segment_data_file)
        )

    def run(self):
        mic = AudioFile(self.input()["wav_files"]["mic.wav"].path)

        playback_periods = pd.read_csv(self.input()["vocal_periods"].path)

        ######################################################
        ### First search for periods of live vocalizations ###
        ######################################################

        exclude_intervals = [
            (
                period.start_time,
                period.start_time + period.duration
            ) for period in playback_periods.itertuples()
        ]

        # This function tries to find the smallest pieces of vocalizations, e.g. syllables
        # Should modify this function if you want to group them up as songs
        seg_dicts = detect_call_and_syllable_dicts_from_lazy(
            mic, exclude_intervals, split_into_smallest_chunks=True, window=5
        )

        df = pd.DataFrame(seg_dicts, columns=autodetected_columns)
        df.to_csv(self.output().path, index=False)


class MergePlaybacksCSV(luigi.Task):
    """Merge together vocal period dicts from all sessions in a site

    The sessions and relative offsets must be defined in a
    sessions.yaml file in the site directory. This function
    takes the playback periods for each
    session and just adds the specified offset to each one.
    """

    site_dir = SiteParameter()

    def requires(self):
        config_task = SiteConfig(self.site_dir, required_keys=["sessions"])

        yield config_task

        site_config = config_task.output().data()

        for session in site_config["sessions"]:
            session_dir = os.path.join(os.path.dirname(os.path.dirname(self.site_dir)), "sessions", session["id"])
            yield PlaybacksCSV(session_dir=session_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "playback_segments.csv"))

    def run(self):
        site_config = self.input()[0].data()
        playback_csvs = self.input()[1:]
        session_offsets = [session["t"] for session in site_config["sessions"]]

        # vocal_period_data = [np.load(s.path, allow_pickle=True)[()] for s in self.input()]
        vocal_period_data = [pd.read_csv(s.path) for s in playback_csvs]

        for vocal_periods, offset in zip(vocal_period_data, session_offsets):
            vocal_periods["start_time"] += offset

        df = pd.concat(vocal_period_data, axis=0)
        df.to_csv(self.output().path, index=False)


class MergeLiveCalls(luigi.Task):
    """Merge together vocal period dicts

    The sessions and relative offsets must be defined in a
    sessions.yaml file in the site directory. This function
    takes the vocal periods (both live and playback) for each
    session and just adds the specified offset to each one.
    """

    site_dir = SiteParameter()

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
            yield FindLiveCalls(session_dir=session_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "autodetected_segments.csv"))

    def run(self):
        block_offsets = [
            session["t"]
            for session in self.session_info["sessions"]
        ]
        # vocal_period_data = [list(np.load(s.path, allow_pickle=True)[()]) for s in self.input()]
        vocal_period_data = [pd.read_csv(s.path) for s in self.input()]

        for vocal_periods, offset in zip(vocal_period_data, block_offsets):
            vocal_periods["start_time"] += offset

        df = pd.concat(vocal_period_data, axis=0)
        df.to_csv(self.output().path, index=False)
        # np.save(self.output().path, combined_result)


class CreateAutodetectedPeriods(luigi.Task):
    site_dir = SiteParameter()

    def requires(self):
        return MergeLiveCalls(self.site_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "autodetected.pkl"))

    def run(self):
        """
        Removes the dependency on pyquantities from the a.npy file and
        converts it into a pandas dataframe
        """
        periods = np.load(self.input().path, allow_pickle=True)[()]
        for period in periods:
            period["t_start"] = period["start_time"].item()
            period["t_stop"] = period["start_time"].item() + period["duration"].item()
            del period["start_time"]
            del period["duration"]

        df = pd.DataFrame(list(periods))
        df.to_pickle(self.output().path)


class CreateSoundsepSaveFile(luigi.Task):
    """Creates a save file (used for soundsep) that populates
    a soundsep project folder.

    Creates a folder for soundsep project, a soundsep config file,
    and symlinks to each sessions raw mic files

    SESSION_DIR/
        soundsep_labeling/
            soundsep.yaml
            audio/
                0_SESSION_NAME.mic.wav -> SESSION_DIR/raw/mic.wav
                ...
                        
    """
    session_dir = SessionParameter()

    def requires(self):
        return {
            "playbacks": PlaybacksCSV(self.session_dir),
            "wav_files": Nix2Wav(self.session_dir),
        }

    def output(self):
        if not os.path.exists(os.path.join(self.session_dir, "soundsep_labeling")):
            os.makedirs(os.path.join(self.session_dir, "soundsep_labeling"))
            os.makedirs(os.path.join(self.session_dir, "soundsep_labeling", "audio"))
            os.makedirs(os.path.join(self.session_dir, "soundsep_labeling", "_appdata", "save"))

        return luigi.LocalTarget(os.path.join(self.session_dir, "soundsep_labeling", "soundsep.yaml"))

    def run(self):
        config_file = yield SessionConfig(self.session_dir)
        session_config = config_file.data()
        playback_periods = pd.read_csv(self.input()["playbacks"].path)
        playback_channels = session_config.get("playback_mic_channels", 0)
        if isinstance(playback_channels, int):
            playback_channel = playback_channels
        else:
            playback_channel = playback_channels[0] if len(playback_channels) else 0

        sampling_rate = AudioFile(os.path.join(self.input()["wav_files"]["mic.wav"].path)).sampling_rate

        # Write a default config
        with open(self.output().path, "w+") as f:
            f.write("""
audio_directory: ../raw
recursive_search: false
filename_pattern: mic.wav
block_keys: null
channel_keys: null
""")

        soundsep_save_rows = []
        for i in range(len(playback_periods)):
            row = playback_periods.iloc[i]
            soundsep_save_rows.append({
                "SourceName": "Playback",
                "SourceChannel": playback_channel,
                "StartIndex": int(row["start_time"] * sampling_rate),
                "StopIndex": int((row["start_time"] + row["duration"]) * sampling_rate),
            })

        pd.DataFrame(soundsep_save_rows).to_csv(os.path.join(
            self.session_dir,
            "soundsep_labeling",
            "_appdata",
            "save",
            "segments.csv"))

        source_csv = pd.DataFrame([
            {
                "SourceName": "Playbacks",
                "SourceChannel": playback_channel,
                "SourceIndex": 0
            }
        ]).to_csv(os.path.join(
            self.session_dir,
            "soundsep_labeling",
            "_appdata",
            "sources.csv"
        ))
