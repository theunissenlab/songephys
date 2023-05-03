import glob
import logging
import os
from collections import defaultdict
import datetime

import luigi
import numpy as np
import pandas as pd
import quantities as pq
import soundfile as sf
from neo.io import IntanIO
from scipy.fft import rfft

from ..utils import get_session_starttime
from ..format.io import load_yaml
from ..format.intan_utils import timestamp_from_intan_filename
from ..format.wav import AudioFile
from ..preprocess.sound.playback_utils import (
    convert_digital_to_metadata,
    convert_pyoperant_playback_csvs,
    convert_pyoperant_operant_csvs,
    read_pyoperant_playback_csv,
    create_digital_events_df,
    calculate_event_timedelta,
    fix_stimulus_codes,
    create_gaussian_window
)
from ..preprocess.sound.playback_alignment import (
    compute_realigned_trial_times,
    create_replacement_codes
)
from ..preprocess.sound.detect import (
    detect_call_and_syllable_dicts,
    generate_stim_dicts,
    generate_event_dicts_operant,
    playback_columns,
    operant_columns,
    autodetected_columns,
)
from ..preprocess.sound.stimuli import merge_segments

from .file_setup import Rhd2Wav, SiteConfig, SessionConfig, DownloadSession, GenerateSiteMemmaps
from .pipeline_utils import SiteParameter, SessionParameter


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
            print('Looking for stimulus names', stimulus_names)
            print('Not found in', stimuli_directory)
            print('Which has the following files', possible_stim_dirs)
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
        return Rhd2Wav(self.session_dir)

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
            playback_df = read_pyoperant_playback_csv(csv_file, stim_dir)
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
            "files": DownloadSession(self.session_dir),
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
        rhd_file = self.input()["files"]["rhd"].path
        rhd = IntanIO(rhd_file)
        # if this is a rhd we can check the samplerate of one of the signal channels
        # TODO maybe vet which channel we are checking. this is usually one of the 
        #   ephys channels
        sampling_rate = rhd.header['signal_channels'][0][2]
        block = rhd.read_block(lazy=True)

        if not len(block.segments[0].events):
            # If we messed up the digital codes...
            stimulus_codes = np.load(self.input()["aligned_codes"].path, allow_pickle=True)[()]
        else:
            events = block.segments[0].events[0].load()
            stimulus_codes = convert_digital_to_metadata(events,sample_rate=sampling_rate)

        # Combine the data from the csvs into a single pickled dataframe
        logger.info("Saving stim database {}".format(self.output().path))
        convert_pyoperant_playback_csvs(
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
            "wav_files": Rhd2Wav(self.session_dir),
            "aligned_codes": CreateMicrophoneAlignedCodes(self.session_dir),
            "playback_metadata": CombinePlaybackMetadata(self.session_dir)
        }

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        segment_data_file = "playback_segments.csv"
        return luigi.LocalTarget(
            os.path.join(segmentation_dir, segment_data_file)
        )

    def run(self):
        

        try:
            playbacks_db = pd.read_pickle(self.input()["playback_metadata"].path)
        except:
            logger.warning("No pyoperant file found (no playbacks)")
            playback_dicts = []
        else:
            rhd_file = self.input()["files"]["rhd"].path
            rhd = IntanIO(rhd_file)
            # if this is a rhd we can check the samplerate of one of the signal channels
            # TODO maybe vet which channel we are checking. this is usually one of the 
            #   ephys channels
            sampling_rate = rhd.header['signal_channels'][0][2]
            block = rhd.read_block(lazy = True)

            if not len(block.segments[0].events):
                # Live vocalization detection is now delegated to the FindLiveCalls task
                mic = AudioFile(self.input()["wav_files"]["mic.wav"].path)
                stimulus_codes = np.load(self.input()["aligned_codes"].path, allow_pickle=True)[()]
            else:
                # load the digital signals (lazy loaded)
                digital_signals = block.segments[0].events[0].load()
                stimulus_codes = convert_digital_to_metadata(digital_signals,sample_rate=sampling_rate)

            playback_dicts = generate_stim_dicts(
                    stimulus_codes,
                    sampling_rate,
                    playbacks_db
            )
            logger.info("Identified {} playback periods".format(
                len(playback_dicts))
            )

        df = pd.DataFrame(playback_dicts, columns=playback_columns)
        df.to_csv(self.output().path, index=False)

class CombineOperantMetadata(luigi.Task):
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
            "files": DownloadSession(self.session_dir)
        }

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        if not os.path.exists(segmentation_dir):
            os.makedirs(segmentation_dir)

        combined_db = "playbacks.pkl"

        return {
            "playbacks": luigi.LocalTarget(
                os.path.join(segmentation_dir, combined_db)
            ),
            "events": luigi.LocalTarget(
                os.path.join(segmentation_dir, "events.pkl")
            )
        }

    def run(self):
        input_directory = os.path.join(self.session_dir, "raw")
        stimuli_directory = os.path.join(self.session_dir, "stimuli")

        csv_files = sorted(glob.glob(os.path.join(input_directory, "*_trialdata.csv")))
        log_files = sorted(glob.glob(os.path.join(input_directory, "*_events.log")))

        if not len(csv_files) and not len(log_files):
            # no playback csvs -> create a dummy dataframe
            pd.DataFrame([]).to_csv(self.output()['playbacks'].path)
            pd.DataFrame([]).to_csv(self.output()['events'].path)
            return

        # Identify and locate the stimulus directory (e.g. TutFamBos)
        # associated with each pyoperant csv
        # TODO these dirs should be the same for csv and log files
        # but may want to assert thats true here.
        stim_dirs = _get_stim_dirs(csv_files, stimuli_directory)

        # determine how many playback sets there were and how many
        rhd_file = self.input()["files"]["rhd"].path
        rhd = IntanIO(rhd_file)
        # if this is a rhd we can check the samplerate of one of the signal channels
        # TODO maybe vet which channel we are checking. this is usually one of the 
        #   ephys channels
        sampling_rate = rhd.header['signal_channels'][0][2]
        block = rhd.read_block(lazy=True)

        if not len(block.segments[0].events):
            raise IOError("No Events in %s" % rhd_file)
            
        # The events channel of our rhd files is always the digital codes
        events = block.segments[0].events[0].load()
        # note in the HPIWHI5668M files, these codes are all 0.... oops
        stimulus_codes = convert_digital_to_metadata(events,sample_rate=sampling_rate)
        
        # now to process epochs
        digital_epochs = [ epoch.load() for epoch in block.segments[0].epochs ]
        # First of all, we need to calculate how far off all of our times are in the csv files from the rhd files
        # This can be done using the TTL pulse, we know is every 10s 
        # TODO the THIS DICT SHOULD BE PART OF THE CONFIG
        digital_epoch_channels = {
            "I-1" : "TTL",
            "I-2": "PeckPort",
            "I-3" : "Feeder" 
        }
        if "TTL" not in digital_epoch_channels.values():
            logger.info("TTL is not on one of the digital channels, assuming that 'I-1' is the TTL Channel" )
            digital_epoch_channels['I-1'] = "TTL"
        file_timeoffset = datetime.timedelta()

        digital_epochs_to_save = dict()

        session_start = get_session_starttime(self.session_dir)
        for dep in digital_epochs:
            # IF this is the TTL pulse we dont save it (should we? maybe to align with video etc...)
            if digital_epoch_channels.get(dep.annotations['id'],None) == "TTL":
                file_timeoffset = calculate_event_timedelta(dep.times, log_files, session_start )
            else:
            # IF it is not the TTL pulse and its present in the digital_epoch channels, then save it
                if dep.annotations['id'] in digital_epoch_channels:
                    save_name = digital_epoch_channels[dep.annotations['id']]
                    digital_epochs_to_save[save_name] = dep
                else:
                    print("Not saving %s"%dep)
        # here we will do some processing to spoof the trial data... hopefully this is not necessary in the future
        # TODO not sure where to add this file_timeoffset.... to the df or not 
        #   we may want to subtract them from the dataframes
        if len(stimulus_codes) > 0 and np.all([t[1]['trial'] == 0 for t in stimulus_codes]):
            logger.info("Fixing stim codes because they are all 0")
            stimulus_codes = fix_stimulus_codes(stim_codes=stimulus_codes, 
                                                csv_files=csv_files,stim_dirs=stim_dirs,
                                                sample_rate=sampling_rate, 
                                                session_starttime=session_start,
                                                session_offset=file_timeoffset)


        # Combine the data from the csvs into a single pickled dataframe
        logger.info("Saving stim database {}".format(self.output()["playbacks"].path))
        convert_pyoperant_operant_csvs(
                stimulus_codes,
                csv_files,
                stim_dirs,
                self.output()["playbacks"].path)

        
        logger.info("Saving event databse {}".format(self.output()["events"].path))
        create_digital_events_df(
            digital_epochs_to_save,
            self.output()["events"].path
        )

class OperantCSV(luigi.Task):
    """Locate time periods containing playback sounds and time periods of events

    Represents each time period by a dictionary with a start and stop
    time. Saves a list of playback periods.
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            #"wav_files": Rhd2Wav(self.session_dir),
            "metadata": CombineOperantMetadata(self.session_dir),
            "files": DownloadSession(self.session_dir)
        }

    def output(self):
        segmentation_dir = os.path.join(self.session_dir, "segmentation")
        segment_data_file = "playback_segments.csv"
        event_datafile = "event_segments.csv"
        return {
            "playbacks": luigi.LocalTarget( os.path.join(segmentation_dir, segment_data_file)),
            "events":  luigi.LocalTarget( os.path.join(segmentation_dir, event_datafile)),
        }

    def run(self):
        try:
            playbacks_db = pd.read_pickle(self.input()["metadata"]['playbacks'].path)
        except:
            logger.warning("No pyoperant file found (no playbacks)")
            playback_dicts = []
        else:
            rhd_file = self.input()["files"]["rhd"].path
            rhd = IntanIO(rhd_file)
            block = rhd.read_block(lazy = True)
            # if this is a rhd we can check the samplerate of one of the signal channels
            # TODO maybe vet which channel we are checking. this is usually one of the 
            #   ephys channels
            sampling_rate = rhd.header['signal_channels'][0][2]

            if not len(block.segments[0].events):
                raise IOError("No Events in %s" % rhd_file)

            # load the digital signals (lazy loaded)
            digital_signals = block.segments[0].events[0].load()
            stimulus_codes = convert_digital_to_metadata(digital_signals,sampling_rate)
            
            playback_dicts = generate_event_dicts_operant(
                    stimulus_codes,
                    sampling_rate,
                    playbacks_db
            )
            logger.info("Identified {} playback periods".format(
                len(playback_dicts))
            )

        df = pd.DataFrame(playback_dicts, columns=operant_columns)
        df.to_csv(self.output()['playbacks'].path, index=False)
        try:
            events_db = pd.read_pickle(self.input()["metadata"]['events'].path)
        except:
            logger.warning("No pyoperant file found (no events)")
            events_db = pd.DataFrame(None, columns=["start_time","duration","event_name"])
        events_db.to_csv(self.output()['events'].path, index=False)


class FindLiveCalls(luigi.Task):

    session_dir = SessionParameter()

    def requires(self):
        return {
            "wav_files": RhD2Wav(self.session_dir),
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
        seg_dicts = detect_call_and_syllable_dicts(mic)

        df = pd.DataFrame(seg_dicts, columns=autodetected_columns)
        df.to_csv(self.output().path, index=False)

class GenerateFullSpectrogram(luigi.Task):
    """Generates one full spectrogram for this site
    """

    site_dir = SiteParameter()
    # TODO things to put in settings
    half_window = 100#128 looks good but is big file
    fft_size = half_window * 2 + 1
    fft_step = 30

    def requires(self):
        return {
            "wav_files": GenerateSiteMemmaps(self.site_dir),
            "config": SiteConfig(self.site_dir, required_keys=["sessions"]),
        }
    def output(self):
        # TODO get SR through site config or something for full audio name
        return {
            "spectrogram": luigi.LocalTarget(os.path.join(self.site_dir,"audio_spectrogram_st%d_w%d.npy"%(self.fft_step,self.fft_size))),
            "full_audio" : luigi.LocalTarget(os.path.join(self.site_dir,'site_audio_30000.npy'))
        }
    
    def run(self):
        
        gaussian_window = create_gaussian_window(self.half_window, nstd=6)
        padding = np.zeros(self.half_window)

        # first load all wav files into a contiguous array
        site_config = self.input()['config'].data()
        ignore_channels = site_config['playback_mic_channels']
        session_ids = [session["id"] for session in site_config["sessions"]]
        audio_sr = None
        # pad the beginning
        audio_data = [padding]
        for session in session_ids:
            audio_file = 'sessions/%s/raw/%s_audio.wav'%(session,session)
            data, sr = sf.read(audio_file)
            # remove the channels that are just playback
            inds= np.arange(data.shape[1])
            inds = inds[np.in1d(inds,ignore_channels,invert=True)]
            if audio_sr is None:
                audio_sr = sr
            assert( audio_sr == sr )

            #average the data
            avg_mic = np.mean(data[:,inds],axis=1)
            audio_data.append(avg_mic)
        # pad the end
        audio_data.append(padding)

        full_audio = np.concatenate(audio_data)
        np.save(self.output()['full_audio'].path,full_audio)
        # cleanup some messy overhead memory
        del audio_data

        # get sliding window array view
        def apply_func(window):
            return np.abs(rfft(window*gaussian_window,self.fft_size,axis=0))
        # perform STFT on windows in the array, writing to output array
        # Produce a view of shape (steps, samples, channels)
        strides = np.lib.stride_tricks.sliding_window_view(full_audio, self.fft_size)[::self.fft_step]
        fft_output = np.apply_along_axis(apply_func,1,strides)
        #for ix, window in enumerate(strides):
        #    window_data = window * gaussian_window
        #    fft_output[ix,:] = np.abs(rfft(window_data, n=self.fft_size, axis=0))

        # saving to output
        np.save(self.output()['spectrogram'].path,fft_output)
        return

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

class MergeOperantCSV(luigi.Task):
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
            yield OperantCSV(session_dir=session_dir)

    def output(self):
        return {
            "playbacks": luigi.LocalTarget(os.path.join(self.site_dir, "playback_segments.csv")),
            "events": luigi.LocalTarget(os.path.join(self.site_dir, "event_segments.csv"))
        }

    def run(self):
        site_config = self.input()[0].data()
        playback_csvs = [input['playbacks'] for input in self.input()[1:]]
        session_offsets = [session["t"] for session in site_config["sessions"]]

        # First concatenate all playback data
        # vocal_period_data = [np.load(s.path, allow_pickle=True)[()] for s in self.input()]
        vocal_period_data = [pd.read_csv(s.path) for s in playback_csvs]

        for vocal_periods, offset in zip(vocal_period_data, session_offsets):
            vocal_periods["start_time"] += offset

        df = pd.concat(vocal_period_data, axis=0)
        df.to_csv(self.output()["playbacks"].path, index=False)

        # now concatenate all event data
        event_csvs = [input['events'] for input in self.input()[1:]]
        event_period_data = [pd.read_csv(s.path) for s in event_csvs]

        for event_periods, offset in zip(event_period_data, session_offsets):
            event_periods["start_time"] += offset
        
        df = pd.concat(event_period_data, axis=0)
        df.to_csv(self.output()['events'].path, index=False)


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


class CreateSoundsepFiles(luigi.Task):
    """Creates a save file (used for soundsep) that populates
    a soundsep project folder.

    Creates a folder for soundsep project, a soundsep config file,
    and symlinks to each sessions raw mic files

    SESSION_DIR/
        soundsep_labeling/
            soundsep.yaml
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
            f.write("""audio_directory: ../raw
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
