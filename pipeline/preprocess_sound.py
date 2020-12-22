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
    generate_vocal_periods_dicts_from_lazy,
    get_surrounding_silence,
    shave_silences,
)
from preprocess.sound.stimuli import merge_segments
from preprocess_ephys import MergeLazySignals
from file_setup import GenerateLazyFiles, Rhd2Nix

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
                stim_dir_used = os.path.join(stimuli_directory, stim_dir)
                break

        if stim_dir_used is None:
            raise Exception(
                "Couldn't find the stim dir for {}".format(csv_file))

        stim_dirs.append(stim_dir_used)

    return stim_dirs


class ExportMicrophonesToWav(luigi.Task):
    """Write the microphone data from Nix file to a wav file
    """
    session_dir = luigi.Parameter()

    def requires(self):
        return Rhd2Nix(self.session_dir)

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        if not os.path.exists(segmentation_dir):
            os.makedirs(segmentation_dir)

        return luigi.LocalTarget(os.path.join(segmentation_dir, "mic.wav"))

    def run(self):
        nix = NixIO(self.input().path)
        block = nix.read_block(lazy=True)

        for signal in block.segments[0].analogsignals:
            if signal.annotations["neo_name"] == "Board ADC":
                mic_signal = signal

        mic_signal = nix.load_lazy_object(mic_signal)
        mic_data = np.array(mic_signal)
        mic_data_minus_mean = mic_data - np.mean(mic_data, axis=0)

        samplerate = int(mic_signal.sampling_rate.item())
        channels = mic_data_minus_mean.shape[1]

        with sf.SoundFile(self.output().path, "w", samplerate=samplerate, channels=channels) as soundfile:
            logger.info("Writing mic data to {}".format(self.output().path))
            soundfile.write(np.array(mic_data_minus_mean))


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
    session_dir = luigi.Parameter()

    def requires(self):
        return {
            "lazy_signals": GenerateLazyFiles(self.session_dir),
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")

        return luigi.LocalTarget(
            os.path.join(raw_dir, "microphone_aligned_codes.npy")
        )

    def run(self):
        input_directory = os.path.join(self.session_dir, "raw")
        stimuli_directory = os.path.join(self.session_dir, "stimuli")

        csv_files = sorted(glob.glob(os.path.join(input_directory, "*.csv")))

        # Identify and locate the stimulus directory (e.g. TutFamBos)
        # associated with each pyoperant csv
        stim_dirs = _get_stim_dirs(csv_files, stimuli_directory)

        signals = np.load(
            self.input()["lazy_signals"]["lazy_signals"].path,
            allow_pickle=True
        )[()]
        lazy_mic = signals["mic"]
        max_time = lazy_mic.N / lazy_mic.sampling_rate

        new_codes = []
        for csv_file, stim_dir in zip(csv_files, stim_dirs):
            playback_df = read_pyoperant_csv(
                csv_file,
                stim_dir
            )
            session_start_timestamp = timestamp_from_intan_filename(self.session_dir)

            aligned_times = compute_realigned_trial_times(
                playback_df,
                session_start_timestamp,
                lazy_mic
            )
            new_codes.extend(create_replacement_codes(
                playback_df,
                aligned_times,
                lazy_mic.sampling_rate,
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

    session_dir = luigi.Parameter()

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


class FindVocalPeriods(luigi.Task):
    """Locate time periods containing either live or playback sounds

    Represents each time period by a dictionary with a start and stop
    time. Saves a list of live periods and playback periods.
    """

    session_dir = luigi.Parameter()

    def requires(self):
        return {
            "aligned_codes": CreateMicrophoneAlignedCodes(self.session_dir),
            "lazy_signals": GenerateLazyFiles(self.session_dir),
            "playback_metadata": CombinePlaybackMetadata(self.session_dir),
            "nix_file": Rhd2Nix(self.session_dir)
        }

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        segment_data_file = "vocal_periods.npy"
        return luigi.LocalTarget(
            os.path.join(segmentation_dir, segment_data_file)
        )

    def run(self):
        signals = np.load(self.input()["lazy_signals"]["lazy_signals"].path, allow_pickle=True)[()]
        mic = signals["mic"]

        ######################################################
        ### First search for periods of live vocalizations ###
        ######################################################

        live_dicts = generate_vocal_periods_dicts_from_lazy(mic)
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
            playbacks_db = pd.read_pickle(self.input()["playback_metadata"].path)
        except:
            logger.warning("No pyoperant file found (no playbacks)")
            playback_dicts = []
            segment_dicts = live_dicts
        else:
            nix = NixIO(self.input()["nix_file"].path)
            block = nix.read_block(lazy=True)

            if not len(block.segments[0].events):
                stimulus_codes = np.load(self.input()["aligned_codes"].path, allow_pickle=True)[()]
                sampling_rate = mic.sampling_rate / pq.s
            else:
                digital_signals = nix.load_lazy_object(block.segments[0].events[0])
                stimulus_codes = convert_digital_to_metadata(digital_signals)
                sampling_rate = digital_signals.annotations["sampling_rate"]

            playback_dicts = generate_stim_dicts(
                    stimulus_codes,
                    sampling_rate,
                    playbacks_db
            )
            logger.info("Identified {} playback periods".format(
                len(playback_dicts))
            )
            # TODO: be more careful about how we do overlaps
            # maybe just wnat to include all overlaps, even if duplciated
            segment_dicts = merge_segments(playback_dicts, live_dicts)

            nix.close()

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


class FindLiveCalls(luigi.Task):

    session_dir = luigi.Parameter()

    def requires(self):
        return {
            "lazy_signals": GenerateLazyFiles(self.session_dir),
            "vocal_periods": FindVocalPeriods(self.session_dir),
        }

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        segment_data_file = "call_periods.npy"
        return luigi.LocalTarget(
            os.path.join(segmentation_dir, segment_data_file)
        )

    def run(self):
        signals = np.load(self.input()["lazy_signals"]["lazy_signals"].path, allow_pickle=True)[()]
        mic = signals["mic"]

        playback_periods = np.load(
            self.input()["vocal_periods"].path,
            allow_pickle=True
        )[()]["playback"]

        ######################################################
        ### First search for periods of live vocalizations ###
        ######################################################

        exclude_intervals = [
            (
                period["start_time"].item() + 0.5,
                period["start_time"].item() + period["duration"].item() - 0.5
            ) for period in playback_periods
        ]

        seg_dicts = detect_call_and_syllable_dicts_from_lazy(
            mic, exclude_intervals, window=5
        )

        np.save(
            self.output().path,
            seg_dicts
        )


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
            yield FindVocalPeriods(session_dir=session_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "vocal_periods.npy"))

    def run(self):
        block_offsets = [
                session["t"] * pq.s
                for session in self.session_info["sessions"]
        ]
        vocal_period_data = [np.load(s.path, allow_pickle=True)[()] for s in self.input()]

        combined_result = defaultdict(list)
        for vocal_periods, offset in zip(vocal_period_data, block_offsets):
            for key in vocal_periods:
                for period in vocal_periods[key]:
                    period["start_time"] += offset
                combined_result[key] += vocal_periods[key]
        combined_result = dict(combined_result)

        np.save(self.output().path, combined_result)


class MergeLiveCalls(luigi.Task):
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
            yield FindLiveCalls(session_dir=session_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "call_periods.npy"))

    def run(self):
        block_offsets = [
                session["t"] * pq.s
                for session in self.session_info["sessions"]
        ]
        vocal_period_data = [list(np.load(s.path, allow_pickle=True)[()]) for s in self.input()]

        combined_result = []
        for vocal_periods, offset in zip(vocal_period_data, block_offsets):
            for period in vocal_periods:
                period["start_time"] += offset
            combined_result = combined_result + vocal_periods

        np.save(self.output().path, combined_result)


class CreateSoundsepAutodetectedPeriods(luigi.Task):
    site_dir = luigi.Parameter()

    def requires(self):
        return MergeLiveCalls(self.site_dir)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "autodetected.pkl"))

    def run(self):
        """
        Removes the dependency on pyquantities from the call_periods.npy file and
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


class CreateLabelingSaveFile(luigi.Task):
    """Creates a save file (used for soundsep) that populates
    the live call periods (from FindLiveCalls/MergeLiveCalls) as well as storing
    the playback periods as a preset source
    """
    site_dir = luigi.Parameter()

    def requires(self):
        return {
            "live_calls": MergeLiveCalls(self.site_dir),
            "vocal_periods": MergeVocalPeriods(self.site_dir),
        }

    def output(self):
        if not os.path.exists(os.path.join(self.site_dir, "outputs")):
            os.makedirs(os.path.join(self.site_dir, "outputs"))

        return luigi.LocalTarget(os.path.join(self.site_dir, "outputs", "save.npy"))

    def run(self):
        live_calls = np.load(self.input()["live_calls"].path, allow_pickle=True)[()]
        vocal_periods = np.load(self.input()["vocal_periods"].path, allow_pickle=True)[()]

        playback_df = pd.DataFrame([
            {
                "t_start": period["start_time"].item() + 0.5,
                "t_stop": period["start_time"].item() + period["duration"].item() - 0.5
            } for period in vocal_periods["playback"]
        ])

        playback_source_dict = {
            "name": "Playbacks",
            "channel": 0,
            "hidden": False,
            "intervals": playback_df,
            "readonly": True
        }

        autodetect_table = pd.DataFrame([
            {
                "t_start": period["start_time"].item(),
                "t_stop": period["start_time"].item() + period["duration"].item()
            } for period in live_calls
        ])

        np.save(self.output().path, {
            "sources": [playback_source_dict],
            "autodetect": autodetect_table
        })


class DetectVocalAndSilentPeriods(luigi.Task):

    site_dir = luigi.Parameter()

    def requires(self):
        return {
            "lazy_site": MergeLazySignals(self.site_dir),
            "vocal_periods": MergeVocalPeriods(self.site_dir)
        }

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "vocal_periods_with_silence.npy"))

    def run(self):
        lazy_site = np.load(
            self.input()["lazy_site"].path,
            allow_pickle=True
        )[()]
        vocal_periods = np.load(
            self.input()["vocal_periods"].path,
            allow_pickle=True
        )[()]

        all_events = np.array(merge_segments(
            vocal_periods["playback"],
            vocal_periods["live"]
        ))

        silent_periods = get_surrounding_silence(
            all_events,
            lazy_site,
            channel=0,
            min_duration=1.0,
            min_sound=0.002,
        )

        all_events = sorted(
            np.concatenate([all_events, silent_periods]),
            key=lambda e: e["start_time"]
        )

        all_events = np.array([
            shave_silences(event, lazy_site)
            for event in all_events
        ])

        np.save(self.output().path, all_events)
