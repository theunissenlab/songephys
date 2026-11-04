import os
import datetime
import time
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import shutil

import numpy as np
import luigi
import soundfile
import kilosort
import wavemap
import pandas as pd
from neo.io import NixIO, IntanIO, NWBIO
from neo.rawio import IntanRawIO
from spikeinterface.extractors import read_intan
from spikeinterface.core import ConcatenateSegmentRecording, BinaryRecordingExtractor
from spikeinterface.preprocessing import highpass_filter, bandpass_filter, resample, phase_shift, common_reference
from spikeinterface.preprocessing import unsigned_to_signed

from ..format.download import downloaders
from ..format.io import load_yaml, save_yaml
from ..format.wav import get_wavfile_data_from_neo,get_raw_data_from_neo
from .targets import YamlTarget
from ..preprocess.sound.playback_utils import convert_raw_digital_to_metadata, convert_pyoperant_operant_csvs, _get_stim_dirs, convert_pyoperant_playback_csvs
from ..preprocess.sound.detect import (
    update_operant_db,
    detect_call_and_syllable_dicts,
    generate_stim_dicts,
    generate_event_dicts_operant,
    playback_columns,
    operant_columns,
    autodetected_columns,
) 
from .pipeline_utils import (
    SessionParameter,
    SiteParameter,
    associate_pyoperant_csvs_to_rhds,
    get_session_sites,
    parse_filelist,
    session_basename,
    RemoteTask
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_recording_info(recording):
    # now save info
    info = dict()
    info['num_channels'] = int(recording.get_traces(start_frame=0, end_frame=1, segment_index=0).shape[1])
    info['num_samples'] = int(recording.get_total_samples())
    info['num_segments'] = int(recording.get_num_segments())
    info['sampling_rate'] = int(recording.get_sampling_frequency())
    info['dtype'] = str(recording.get_dtype()) # Not sure this is the best way to serialize a dtype object
    return info

def get_sorted_rhd_files(rhd_files):
    """Sort the rhd files in the order they were recorded"""
    # sort the rhd files by the timestamp in the filename
    rhd_files = sorted(rhd_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return rhd_files

from pathlib import Path
from spikeinterface.extractors.neoextractors.neobaseextractor import NeoBaseRecordingExtractor
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment

# Modification of the IntanRecordingExtractor class to support fast reading of intan files
class FastIntanRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a intan board. Supports rhd and rhs format.

    Based on :py:class:`neo.rawio.IntanRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    stream_id: str, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name: str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, default: False
        Load exhaustively all annotations from neo.
    """

    mode = "file"
    NeoRawIOClass = "IntanRawIO"
    name = "intan"

    def __init__(self, file_path, stream_id=None, stream_name=None, all_annotations=False, use_names_as_ids=False, ignore_integrity_checks=True):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            ignore_integrity_checks=ignore_integrity_checks,
            **neo_kwargs,
        )

        self._kwargs.update(dict(file_path=str(Path(file_path).absolute())))

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs

from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
class RemoveJumps(BasePreprocessor):
    def __init__(self, recording, threshold):
        BasePreprocessor.__init__(self, recording)
        self._kwargs = dict(recording=recording, threshold=threshold)
        self.threshold = threshold
        for parent_segment in recording._recording_segments:
            preprocessor_segment = self._make_preprocessor_segment(parent_segment)
            self.add_recording_segment(preprocessor_segment)

    def _make_preprocessor_segment(self, parent_segment):
        return RemoveJumpsSegment(parent_segment, self.threshold)



class RemoveJumpsSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, threshold):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.threshold = threshold

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        for i in range(traces.shape[1]):
            trace = traces[:, i]
            # traces arrive already signed (unsigned_to_signed runs upstream), so
            # no 32768 offset here -- applying it twice wraps and, under numpy 2,
            # raises OverflowError since 32768 does not fit in int16.
            # astype() copies, which matters: the -= below must not touch `traces`.
            trace_int16=trace.astype(np.int16)
            trace_int16 -= np.int16(np.mean(trace_int16))
            trace_diff = np.diff(np.abs(trace_int16))
            # find all jumps up
            jumps = np.where(trace_diff > self.threshold)[0]
            jumps_down = np.where(trace_diff < -self.threshold)[0]
            if len(jumps) == 0 or len(jumps_down) == 0:
                continue

            # remove all jump ups that are after the last jump down
            jumps = jumps[jumps < jumps_down[-1]]
            if len(jumps)==0:
                continue
            # remove all jump downs that are before the first jump up
            jumps_down = jumps_down[jumps_down > jumps[0]]
            if len(jumps_down)==0:
                continue

        
            if len(jumps) > 1:
                jumps = np.concatenate(([jumps[0]], jumps[1:][np.diff(jumps) > 1]))
            if len(jumps_down) > 1:
                jumps_down = np.concatenate((jumps_down[:-1][np.diff(jumps_down) > 1], [jumps_down[-1]]))
            # if your first jump down is before your first jump up, ignore it
            if jumps_down[0] < jumps[0]:
                jumps_down = jumps_down[1:]
            # if your last jump up is after your last jump down, ignore it
            if jumps_down[-1] < jumps[-1]:
                jumps = jumps[:-1]
            # now we pair the jumps and jumps_down
            paired_jumps = []
            paired_jumps_down = []
            for jump in jumps:
                # find the closest jump_down after the jump
                # if there are no jumps_down after this jump, we ignore it
                possible_jd = jumps_down[jumps_down>jump]
                if len(possible_jd) == 0:
                    continue
                jump_down = possible_jd[0]
                if jump_down in paired_jumps_down:
                    continue
                if jump_down - jump > 30: # if its more than 1ms, we ignore it
                    #print(f"Jump too long: {jump+start_frame-buffer_start} -> {jump_down+start_frame-buffer_start}")
                    continue    
                paired_jumps.append(jump)
                paired_jumps_down.append(jump_down)

            assert(len(paired_jumps) == len(paired_jumps_down))
            for jump in zip(paired_jumps, paired_jumps_down):
                # interpolate between the two points
                trace[(jump[0]+1):(jump[1]+1)] = trace[jump[0]]/2 + trace[jump[1] + 1]/2
            traces[:, i] = trace
        return traces



class SiteConfig(luigi.Task):
    """Verify config file exists and specific keys exist

    Use as a dependency in requires() to make sure a task can only be run
    if the given configuration keys are defined

    Example Usage
    -------------
    >>> class ExampleTask(luigi.Task):
    ...     REQUIRED_KEYS = ["required_key_1", "required_key_2"]
    ...     site_dir = SiteParameter()
    ...     def requires(self):
    ...         return SiteConfig(self.site_dir, required_keys=self.REQUIRED_KEYS)
    ...     def run(self):
    ...         config = self.input().data()
    ...         assert "required_key_1" in config
    ...         assert "required_key_2" in config

    Arguments
    ---------
    site_dir : str
        Path to site
    required_keys : List[str] (default = ["download_type", "experiment_name", "filelist"])
        List of keys that must exist in the config file

    Output
    ------
    The keys in `required_keys` defined in `{site_dir}/site_config.yaml`.
    """
    site_dir = SiteParameter()
    required_keys = luigi.ListParameter(
        default=["download_type", "experiment_name", "filelist"])

    def config_path(self):
        return os.path.join(self.site_dir, "site_config.yaml")

    def output(self):
        return YamlTarget(self.config_path(), *self.required_keys)

    def run(self):
        raise Exception("""
        The keys {} are required but not present in the config file found for {}
        Config file: {}
        """.format(self.required_keys, self.site_dir, self.config_path()))


class SessionConfig(luigi.Task):
    """Verify config file exists and specific keys exist

    The configuration is located by finding a site whose config contains
    the given session as one of its sessions.

    Use as a dependency in requires() to make sure a task can only be run
    if the given configuration keys are defined

    Example Usage
    -------------
    >>> class ExampleTask(luigi.Task):
    ...     REQUIRED_KEYS = ["required_key_1", "required_key_2"]
    ...     session_dir = SessionParameter()
    ...     def requires(self):
    ...         return SessionConfig(self.session_dir, required_keys=self.REQUIRED_KEYS)
    ...     def run(self):
    ...         session_config = self.input().data()
    ...         assert "required_key_1" in session_config
    ...         assert "required_key_2" in session_config

    Arguments
    ---------
    session_dir : str
        Path to session
    required_keys : List[str] (default = ["download_type", "experiment_name", "filelist"])
        List of keys that must exist in the config file

    Output
    ------
    The keys in `required_keys` defined in `{site_dir}/site_config.yaml`, where site_dir is
    a site whose site_config.yaml contains `session_dir` in its `sessions` list.
    """
    session_dir = SessionParameter()
    required_keys = luigi.ListParameter(
        default=["download_type", "experiment_name", "filelist", "sessions"])

    def config_path(self):
        site_dirs, _, _ = get_session_sites(self.session_dir)
        if not len(site_dirs):
            raise Exception(
                "No valid site with a config file found for session {}".format(self.session_dir))
        return os.path.join(site_dirs[0], "site_config.yaml")

    def output(self):
        return YamlTarget(self.config_path(), *self.required_keys)

    def run(self):
        raise Exception("""
        The keys {} are required but not present in the config file found for {}
        Config file: {}
        """.format(self.required_keys, self.session_dir, self.config_path()))


class SignalsConfig(luigi.Task):
    """Verify config file exists with electrode layout configured properly

    Use as a dependency in requires() to make sure a task can only be run
    if the config variables already exist

    Example Usage
    -------------
    >>> class ExampleTask(luigi.Task):
    ...     site_dir = SiteParameter()
    ...     def requires(self):
    ...         return SignalsConfig(self.site_dir)
    ...     def run(self):
    ...         site_config = self.input().data()
    ...         assert "wavmap" in site_config
    ...         assert "filtering" in site_config
    ...         assert "playback_mic_channels" in site_config

    Arguments
    ---------
    site_dir : str
        Path to site

    Output
    ------
    "wavmap", "filtering", and "playback_mic_channels" must exist in the config file.
    """
    site_dir = SiteParameter()

    def output(self):
        return YamlTarget(
            os.path.join(self.site_dir, "site_config.yaml"),
            "wavmap",
            "filtering",
        )

    def run(self):
        raise Exception("""The nix2wav and filtering keys in site_config.yaml must be manually created.

        Example:

        wavmap:
            ephys.wav:
                - ["Amplifier", [0, 15]]
                - ["Amplifier", [16, 31]]
            mic.wav:
                - ["Amplifier", 15]
                - ["Amplifier", 31]
                - ["Board ADC", [0, 3]]
            motion.wav:
                - ["AUX Input", [0, 6]]
        filtering:
            ephys.wav:
                - [0, 15]
                - [15, 30]
        """)


class SiteProjectConfig(luigi.Task):
    """Verify config file exists and specific keys exist

    Use as a dependency in requires() to make sure a task can only be run
    if the given configuration keys are defined

    Example Usage
    -------------
    >>> class ExampleTask(luigi.Task):
    ...     REQUIRED_KEYS = ["required_key_1", "required_key_2"]
    ...     site_dir = SiteParameter()
    ...     def requires(self):
    ...         return ProjectConfig(self.site_dir, required_keys=self.REQUIRED_KEYS)
    ...     def run(self):
    ...         config = self.input().data()
    ...         assert "required_key_1" in config
    ...         assert "required_key_2" in config

    Arguments
    ---------
    site_dir : str
        Path to site
    required_keys : List[str] (default = ["EXPERIMENT_NAME", "DATADIR", "EPHYS"])
        List of keys that must exist in the config file

    Output
    ------
    The keys in `required_keys` defined in `{site_dir}/site_config.yaml`.
    """
    site_dir = SiteParameter()
    required_keys = luigi.ListParameter(
        default=["EXPERIMENT_NAME", "DATADIR", "EPHYS"])

    def config_path(self):
        # get project_dir from site_dir
        project_dir = self.site_dir.split("sites")[0]
        if os.path.exists(os.path.join(project_dir, "project_config.yaml")):
            return os.path.join(project_dir, "project_config.yaml")
        elif os.path.exists(os.path.join(project_dir, "remote_config.yaml")):
            remote_cfg = load_yaml(os.path.join(project_dir, "remote_config.yaml"))
            project_path = remote_cfg["REMOTE_PATH"]
            return os.path.join(project_path, "project_config.yaml")
        return os.path.join(project_dir, "project_config.yaml")

    def output(self):
        return YamlTarget(self.config_path(), *self.required_keys)

    def run(self):
        raise Exception("""
        The keys {} are required but not present in the config file found for {}
        Config file: {}
        """.format(self.required_keys, self.site_dir, self.config_path()))


class SessionProjectConfig(luigi.Task):
    """Verify config file exists and specific keys exist

    Use as a dependency in requires() to make sure a task can only be run
    if the given configuration keys are defined

    Example Usage
    -------------
    >>> class ExampleTask(luigi.Task):
    ...     REQUIRED_KEYS = ["required_key_1", "required_key_2"]
    ...     session_dir = SessionParameter()
    ...     def requires(self):
    ...         return SessionProjectConfig(self.session_dir, required_keys=self.REQUIRED_KEYS)
    ...     def run(self):
    ...         config = self.input().data()
    ...         assert "required_key_1" in config
    ...         assert "required_key_2" in config

    Arguments
    ---------
    site_dir : str
        Path to site
    required_keys : List[str] (default = ["EXPERIMENT_NAME", "DATADIR", "EPHYS"])
        List of keys that must exist in the config file

    Output
    ------
    The keys in `required_keys` defined in `{site_dir}/site_config.yaml`.
    """
    session_dir = SessionParameter()
    required_keys = luigi.ListParameter(
        default=["EXPERIMENT_NAME", "DATADIR", "EPHYS"])

    def config_path(self):
        # get project_dir from session_dir
        project_dir = self.session_dir.split("sessions")[0]
        return os.path.join(project_dir, "project_config.yaml")

    def output(self):
        return YamlTarget(self.config_path(), *self.required_keys)

    def run(self):
        raise Exception("""
        The keys {} are required but not present in the config file found for {}
        Config file: {}
        """.format(self.required_keys, self.session_dir, self.config_path()))


class DefineSessions(luigi.Task):
    """Converts a list of RHD and CSV files into a list of session dicts

    Uses timestamps of RHD filenames and timestamps within pyoperant CSV
    files to figure out which CSVs go with which RHDs

    Example
    -------
    luigi --module file_setup DefineSessions --site-dir SITE_DIR

    Output
    ------
    "sessions" key in site_config.yaml, each element is a dict with "id", "sessions", and "playbacks"
    """
    site_dir = SiteParameter()

    def requires(self):
        return { 
            "site_config": SiteConfig(self.site_dir), 
            "project_config": SiteProjectConfig(self.site_dir)
            }

    def output(self):
        return YamlTarget(os.path.join(self.site_dir, "site_config.yaml"), "sessions")

    def run(self):
        site_config = self.input()['site_config'].data()
        project_config = self.input()['project_config'].data()

        if site_config["download_type"] not in downloaders.keys():
            raise ValueError("The 'download_type' in {} must be one of {}".format(
                self.input().path, ", ".join(downloaders.keys())
            ))
        # instantiate the downloader
        downloader = downloaders[site_config['download_type']](cfg = project_config)

        session_dicts = associate_pyoperant_csvs_to_rhds(
            csv_files=[os.path.basename(f) for f in site_config["filelist"]
                       if f.endswith(".csv")],
            rhd_files=[f for f in site_config["filelist"]
                       if f.endswith(".rhd")],
            experiment_name=site_config["experiment_name"],
            downloader=downloader,
            include_events=np.any([f.endswith("events.log") for f in site_config['filelist']])
        )
        site_config["sessions"] = session_dicts

        save_yaml(self.output().path, site_config)


class DownloadSession(luigi.Task):
    """
    Download a session and pyoperant data for that session using a downloader defined in format.download

    Creates a directory under `DATA_DIR/birds/EXPERIMENT_NAME/sessions/SESSION_DIR/raw` and
    saves the rhd file, any number of pyoperant CSV files, and stimulus folders

    Example
    -------
    ``luigi --module file_setup DownloadSession --session-dir SESSION_DIR``

    Arguments
    ---------
    session_dir : str
    force : bool (default False)
        Force the task to run whether or not the files had been downloaded already

    Output
    ------
    Folders: `session_dir/raw/SESSION_NAME.rhd`, `session_dir/raw/*.csv`, and
    `session_dir/stimuli/*`
    """

    session_dir = SessionParameter()
    force = luigi.BoolParameter(default=False)

    def requires(self):
        self.session_name = session_basename(self.session_dir)
        output_directory = os.path.join(self.session_dir, "raw")
        self.site_dirs, self.pyoperants, self.events = get_session_sites(self.session_dir)

        return {
            "define_sessions": DefineSessions(self.site_dirs[0]),
            "project_config": SessionProjectConfig(self.session_dir)
        }

    def output(self):
        if self.force:
            return luigi.LocalTarget("dummy")

        self.session_name = session_basename(self.session_dir)
        output_directory = os.path.join(self.session_dir, "raw")
        self.site_dirs, self.pyoperants, self.events = get_session_sites(self.session_dir)
        print(self.pyoperants)
        return {
            "rhd": luigi.LocalTarget(os.path.join(
                output_directory, "{}.rhd".format(self.session_name)
            )),
            "pyoperants": [
                luigi.LocalTarget(os.path.join(
                    output_directory, "{}.csv".format(filename)
                )) for filename in self.pyoperants
            ],
            "events": [
                luigi.LocalTarget(os.path.join(
                    output_directory, "{}.log".format(filename)
                )) for filename in self.events
            ]
        }

    def run(self):
        """Use the yaml file to download RHDs and pyoperant csvs for this session
        """
        session_config = self.input()['define_sessions'].data()
        project_cfg = self.input()['project_config'].data()
        if not len(self.site_dirs):
            raise Exception(
                "No sites were found with site_config.yaml containing {}".format(self.session_dir))

        # instantiate the downloader with the project_config
        downloader = downloaders[session_config["download_type"]](cfg = project_cfg)

        downloader.download_session(
            session_config["experiment_name"],
            self.session_name,
            save_to=self.session_dir,
            pyoperant_files=self.pyoperants,
            event_files = self.events
        )

# class DownloadKilosortSite(luigi.Task):
#     site_dir = SiteParameter()
#     force = luigi.BoolParameter(default=False)

#     def output(self):
#         # Force this task to always run if requested
#         return luigi.LocalTarget("dummy")
    
#     def requires(self):
#         return DefineSessions(self.site_dir)
    
#     def run(self):
#         site_config = self.input().data()
#         if not self.force:
#             tasks = []
#         sessions_dir = os.path.join(self.site_dir, "sessions")

#         for session in site_config["sessions"]:
#             session_dir = os.path.join(os.path.dirname(
#                 os.path.dirname(self.site_dir)), "sessions", session["id"])
#             if not os.path.exists(session_dir):
#                 os.makedirs(session_dir)

#             if self.force:
#                 try:
#                     DownloadSession(session_dir, force=self.force).run()
#                 except Exception as e:
#                     print("Download session {} had an error {}. But going to try to download all because force=True".format(
#                         session_dir,
#                         str(e)))
#             else:
#                 tasks.append(DownloadSession(session_dir, force=self.force))
#         if not self.force:
#             yield tasks

class DownloadSite(luigi.Task):
    site_dir = SiteParameter()
    force = luigi.BoolParameter(default=False)

    def output(self):
        # Force this task to always run if requested
        return luigi.LocalTarget("dummy")

    def requires(self):
        return DefineSessions(self.site_dir)

    def run(self):
        site_config = self.input().data()
        if not self.force:
            tasks = []
    
        for session in site_config["sessions"]:
            session_dir = os.path.join(os.path.dirname(
                os.path.dirname(self.site_dir)), "sessions", session["id"])
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)

            if self.force:
                try:
                    DownloadSession(session_dir, force=self.force).run()
                except Exception as e:
                    print("Download session {} had an error {}. But going to try to download all because force=True".format(
                        session_dir,
                        str(e)))
            else:
                tasks.append(DownloadSession(session_dir, force=self.force))
        if not self.force:
            yield tasks


class GenerateSiteMemmaps(luigi.Task):
    """
    Generate numpy binary arrays of ephys data.
    
    This is used as input to kilosort
    Generates on .dat file for each session containing all channels of ephys
    """
    site_dir = SiteParameter()
    force = luigi.BoolParameter(default=False)

    def requires(self):
        return {
            "site_config": DefineSessions(self.site_dir),
            "project_config": SiteProjectConfig(self.site_dir)
        }
    
    def output(self):
        site_config = self.input()["site_config"].path
        # Force this task to always run if requested
        # TODO: Fix this by checking for files
        if self.force:
            return luigi.LocalTarget("dummy")
        return YamlTarget(site_config,'bin_arrays_generated')
    
    def run(self):
        site_config = self.input()["site_config"].data()
        sessions_dir = os.path.join(os.path.dirname(os.path.dirname(self.site_dir)), "sessions")
        all_sessions = site_config["sessions"]
        for i,session in enumerate(all_sessions):
            bin_output = yield Rhd2Binary(os.path.join(sessions_dir, session["id"]))
            all_sessions[i]['raw_bin'] = bin_output["ephys.bin"].path
        site_config['sessions'] = all_sessions
        site_config['bin_arrays_generated'] = True
        save_yaml(self.input()["site_config"].path, site_config)

class CheckSiteRHDs(luigi.Task):
    """ Check if all RHD files have continuous timestamps"""
    site_dir = SiteParameter()
    def requires(self):
        return {
            "site_config": SiteConfig(self.site_dir), 
            "project_config": SiteProjectConfig(self.site_dir)
        }
    def output(self):
        return YamlTarget(os.path.join(self.site_dir, "site_config.yaml"), "rhd_timestamps_continuous")
    def run(self):
        site_config = self.input()["site_config"].data()
        project_cfg = self.input()["project_config"].data()
        rhd_dir = project_cfg['LOCAL_RAW_FILES']['rhd_drive']
        rhd_files = [f for f in site_config['filelist'] if f.endswith('.rhd')]
        sorted_rhds = get_sorted_rhd_files(rhd_files)
        continuous_timestamps = True
        for rhd_file in sorted_rhds:
            print("Testing file: ", rhd_file)
            rhd_file = os.path.join(rhd_dir, rhd_file)
            rhd = IntanRawIO(rhd_file,ignore_integrity_checks=False)
            rhd._parse_header()
            rhd._assert_timestamp_continuity() # This will raise an error if timestamps are not continuous
            continuous_timestamps = continuous_timestamps and (rhd.discontinuous_timestamps == False)
        site_config['rhd_timestamps_continuous'] = continuous_timestamps
        save_yaml(self.output().path, site_config)

class GenerateSingleSiteLFP(luigi.Task):
    site_dir = SiteParameter()
    lfp_sample_rate = luigi.IntParameter(default=1000)
    filter_margin_ms = luigi.FloatParameter(default=1000.0)
    bin_chunksize = luigi.IntParameter(default=6000)
    def requires(self):
        return {
            "site_config": SiteConfig(self.site_dir), 
            "project_config": SiteProjectConfig(self.site_dir)
        }
    def output(self):
        return {
            'bin': luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "lfp_%s_fs%d.bin"%(os.path.basename(self.site_dir),self.lfp_sample_rate))),
            'info': luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "lfp_%s_info.json"%os.path.basename(self.site_dir)))
        }
    def run(self):
        site_config = self.input()["site_config"].data()
        project_cfg = self.input()["project_config"].data()
        rhd_dir = project_cfg['LOCAL_RAW_FILES']['rhd_drive']

        # get rhd files in order
        rhd_files = [f for f in site_config['filelist'] if f.endswith('.rhd')]
        sorted_rhds = get_sorted_rhd_files(rhd_files)

        all_intan_io = []
        # iterate through rhd files and load the digital events
        for rhd_file in sorted_rhds:
            rhd_file = os.path.join(rhd_dir, rhd_file)
            rhd = FastIntanRecordingExtractor(rhd_file, stream_name='RHD2000 amplifier channel', ignore_integrity_checks=True)
            all_intan_io.append(rhd)

        # concatenate the recording extractors in a single one
        multirecording = ConcatenateSegmentRecording(recording_list=all_intan_io)
        multirecording = unsigned_to_signed(multirecording)
        if 'bad_channels' in site_config and site_config['bad_channels'] != []:
            multirecording = multirecording.remove_channels(['A-%03d'%x for x in site_config['bad_channels']])
        # add a band pass filter
        hp_freq = 1
        lp_freq = 300
        multirecording_f = bandpass_filter(multirecording, freq_min=hp_freq, freq_max=lp_freq,ignore_low_freq_error=True, margin_ms=self.filter_margin_ms)
        # downsample to 1 kHz
        recording_lfp = resample(multirecording_f, resample_rate=self.lfp_sample_rate)

        # now save info
        info = get_recording_info(recording_lfp)

        self.output()['info'].makedirs()
        with open(self.output()['info'].path, 'w') as f:
            json.dump(info, f)

        # then save to binary if it doesnt exist
        if not self.output()['bin'].exists():
            filename, N, c, s, fs, probe_path = kilosort.io.spikeinterface_to_binary(
                recording_lfp, os.path.dirname(self.output()['bin'].path), data_name=os.path.basename(self.output()['bin'].path), dtype=np.int16,
                chunksize=self.bin_chunksize, max_workers=2  )
class GenerateSingleSiteAudioFile(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            "site_config": SiteConfig(self.site_dir), 
            "project_config": SiteProjectConfig(self.site_dir)
        }
    def output(self):
        return {
            'bin': luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "audio_%s.wav"%os.path.basename(self.site_dir))),
            'info':luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "audio_%s_info.json"%os.path.basename(self.site_dir)))
        }

    def run(self):
        site_config = self.input()["site_config"].data()
        project_cfg = self.input()["project_config"].data()
        rhd_dir = project_cfg['LOCAL_RAW_FILES']['rhd_drive']

        # get rhd files in order
        rhd_files = [f for f in site_config['filelist'] if f.endswith('.rhd')]
        sorted_rhds = get_sorted_rhd_files(rhd_files)

        all_intan_io = []
        # iterate through rhd files and load the digital events
        for rhd_file in sorted_rhds:
            print(rhd_file)
            rhd_file = os.path.join(rhd_dir, rhd_file)
            rhd = FastIntanRecordingExtractor(rhd_file, stream_name='USB board ADC input channel', ignore_integrity_checks=True)
            all_intan_io.append(rhd)

        # concatenate the recording extractors in a single one
        multirecording = ConcatenateSegmentRecording(recording_list=all_intan_io)
        multirecording = unsigned_to_signed(multirecording)

        # add a high pass filter
        multirecording_f = bandpass_filter(multirecording, freq_min=200, freq_max=8000)

        # now save info
        info = get_recording_info(multirecording_f)

        self.output()['info'].makedirs()

        # save to wav file if it doesnt exist
        if not os.path.exists(self.output()['bin'].path):
            max_workers = None
            chunksize = 240000
            print('='*40)
            print('Loading recording with SpikeInterface...')

            # Using actual data shape is less fragile than relying on .get_num_channels()
            N = multirecording_f.get_total_samples()
            print(f'number of samples: {N}')
            c = multirecording_f.get_traces(start_frame=0, end_frame=1, segment_index=0).shape[1]
            print(f'number of channels: {c}')
            s = multirecording_f.get_num_segments()
            print(f'numbef of segments: {s}')
            fs = multirecording_f.get_sampling_frequency()
            print(f'sampling rate: {fs}')
            dtype = multirecording_f.get_dtype()
            print(f'dtype: {dtype}')

            # if N is bigger than 2**32, we need to split the wav files
            file_start_ends = []
            if 2*N*c > 2**32:
                n_wav = int(np.ceil(2*N*c / 2**32))
                n_per_file = int(np.ceil(N / n_wav))
                for i in range(n_wav):
                    start_frame = i * n_per_file
                    end_frame = min((i + 1) * n_per_file, N)
                    file_start_ends.append((start_frame, end_frame))
                    if i > 0:
                        fn = f"audio_{os.path.basename(self.site_dir)}-{i}.wav"
                    else:
                        fn = f"audio_{os.path.basename(self.site_dir)}.wav"
                    wav_file = os.path.join(self.site_dir, 'raw', fn)
                    #soundfile.write(wav_file, multirecording_f.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=0), fs)

                    # Determine start/end indices for each segment
                    indices = []
                    for k in range(s):
                        n = end_frame
                        i = start_frame + k*chunksize
                        while i < n:
                            j = i + chunksize if (i + chunksize) < end_frame else end_frame
                            indices.append((i, j, k,start_frame))
                            i += chunksize

                    # Copy each chunk of data to memmory mapped binary file,
                    # use multithreading to speed it up.
                    def copy_chunk(memmap, i, j, k, sf):
                        t = multirecording_f.get_traces(start_frame=i, end_frame=j, segment_index=k)
                        memmap[i-sf:j-sf,:] = t
                        memmap.flush()
                        del(t)
                    y = wavemap(wav_file, dtype=dtype, mode='w+', shape=(end_frame-start_frame,c), sample_rate=int(multirecording_f.get_sampling_frequency()))
                    total_chunks = len(indices)
                    print('='*40)
                    print(f'Converting {total_chunks} data chunks '
                        f'with a chunksize of {chunksize} samples...')
                    with ThreadPoolExecutor(max_workers=max_workers) as exe:
                        futures = [exe.submit(copy_chunk, y, i, j, k, sf) for i, j, k, sf in indices]
                        # TODO: every some-amount-of-time, check list of futures
                        #       for completion
                        while exe._work_queue.qsize() > 0:
                            time.sleep(10)
                            print(f'{total_chunks - exe._work_queue.qsize()} of {total_chunks}'
                                ' chunks converted...')
                    print(f'Data conversion finished.')
                    print('='*40)

                    del(y)  # Close memmap after copying
            else:
                # Determine start/end indices for each segment
                indices = []
                for k in range(s):
                    n = multirecording_f.get_num_samples(segment_index=k)
                    i = 0 + k*chunksize
                    while i < n:
                        j = i + chunksize if (i + chunksize) < n else n
                        indices.append((i, j, k))
                        i += chunksize

                # Copy each chunk of data to memmory mapped binary file,
                # use multithreading to speed it up.
                def copy_chunk(memmap, i, j, k):
                    t = multirecording_f.get_traces(start_frame=i, end_frame=j, segment_index=k)
                    memmap[i:j,:] = t
                    memmap.flush()
                    del(t)
                self.output()['bin'].makedirs()
                y = wavemap(self.output()['bin'].path, dtype=dtype, mode='w+', shape=(N,c), sample_rate=int(multirecording_f.get_sampling_frequency()))
                total_chunks = len(indices)
                print('='*40)
                print(f'Converting {total_chunks} data chunks '
                    f'with a chunksize of {chunksize} samples...')
                with ThreadPoolExecutor(max_workers=max_workers) as exe:
                    futures = [exe.submit(copy_chunk, y, i, j, k) for i, j, k in indices]
                    # TODO: every some-amount-of-time, check list of futures
                    #       for completion
                    while exe._work_queue.qsize() > 0:
                        time.sleep(10)
                        print(f'{total_chunks - exe._work_queue.qsize()} of {total_chunks}'
                            ' chunks converted...')
                print(f'Data conversion finished.')
                print('='*40)

                del(y)  # Close memmap after copying
        if len(file_start_ends) > 0:
            info['file_start_ends'] = file_start_ends
        with open(self.output()['info'].path, 'w') as f:
            json.dump(info, f)



class GenerateSingleSiteOperantCSV(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            "site_config": SiteConfig(self.site_dir),
            'project_config': SiteProjectConfig(self.site_dir),
            "dig_memmap": GenerateSingleSiteDigitalMemmap(self.site_dir)
        }

    def output(self):
        return {
            "playbacks": luigi.LocalTarget(os.path.join(self.site_dir, "playback_segments_v2.csv")),
            "events": luigi.LocalTarget(os.path.join(self.site_dir, "event_segments_v2.csv"))
        }
    
    def run(self):
        site_config = self.input()["site_config"].data()
        project_cfg = self.input()["project_config"].data()

        logger.info("Loading digital data from {}".format(self.input()["dig_memmap"].path))
        #dig_trace = np.fromfile(self.input()["dig_memmap"].path, dtype='uint16')
        dig_trace = np.memmap(self.input()["dig_memmap"].path, dtype='uint16', mode='r')
        #dig_trace = multirecording.get_traces().flatten()
        fs = 30000 #TODO get this from the recording
        num_chans = int(np.log2(np.max(dig_trace)))
        traces = dict()
        # NB: np.unpackbits does this for 8 bit unsigned ints. This works for us as long as we are not using more than 8 channels
        for i in range(num_chans):
            dig_data = np.not_equal(np.bitwise_and(dig_trace, (1<<i)),0)
            # if data is mostly ones then flip it
            if dig_data.mean() > .5:
                dig_data = ~dig_data
            traces[i] = dig_data.astype(np.int8)
        
        logger.info("Converting digital data to metadata")
        # now handle the digital codes on the first channel
        stimulus_codes = convert_raw_digital_to_metadata(traces[0],fs)


        # now handle the digital events TODO put this in site_config
        event_key = dict({
            'TTL': 1,
            'PeckPort':2,
            'Feeder':3
        })

        #   Here, if there is an issue with stim codes, we can spoof them. NOT USUAL
        if len(stimulus_codes) > 0 and np.all([t[1]['trial'] == 0 for t in stimulus_codes]):
            if 'TTL' in event_key:
                log_files = [f for f in site_config["filelist"] if f.endswith("events.log")]

                # this is a hack to get the trial numbers from the digital data
                # we will assume that the first trial starts at the first TTL
                # first we will get the TTL pulses
                TTL = np.diff(traces[event_key['TTL']])
                file_timeoffset = calculate_event_timedelta(np.where(TTL==1)/fs, log_files, 0 )
                
                logger.info("Fixing stim codes because they are all 0")
                #TODO make sure this works
                stimulus_codes = fix_stimulus_codes(stim_codes=stimulus_codes, 
                                                    csv_files=csv_files,stim_dirs=stim_dirs,
                                                    sample_rate=sampling_rate, 
                                                    session_starttime=session_start,
                                                    session_offset=file_timeoffset) 
            else:
                logger.error("ERROR: No TTL channel found in digital data, and no stim codes found")

        operant_csvs = [f for f in site_config["filelist"] if f.endswith("trialdata.csv")]
        operant_csvs = [os.path.join(project_cfg['LOCAL_RAW_FILES']['pyoperant_drive'],f) for f in operant_csvs]
        stim_dirs = _get_stim_dirs(operant_csvs, project_cfg['LOCAL_RAW_FILES']['stimuli_drive'])
        # Combine the data from the csvs into a single pickled dataframe
        logger.info("Saving stim database {}".format(self.output()["playbacks"].path))
        playbacks_db = convert_pyoperant_operant_csvs(
                stimulus_codes,
                operant_csvs,
                stim_dirs,
                None)
        
        playback_dict = update_operant_db(playbacks_db,fs)
        
        df = pd.DataFrame(playback_dict, columns=operant_columns)
        df.to_csv(self.output()['playbacks'].path, index=False)

        # now handle the digital events
        event_df = dict({
            'start_time': [],
            'duration': [],
            'event_name': []
        })
        for event_type in event_key.keys():
            if event_type != 'TTL':
                event_ind = event_key[event_type]
                if event_ind < len(traces):
                    event_data = np.diff(traces[event_ind])
                    event_on_times = np.where(event_data == 1)[0]
                    event_off_times = np.where(event_data == -1)[0]
                    if len(event_on_times) == 0 or len(event_off_times) == 0:
                        logger.info("INFO: No event on or off times found for {}".format(event_type))
                        continue
                    # get rid of all off times that are before the first start time
                    if event_off_times[0] < event_on_times[0]:
                        event_off_times = event_off_times[1:]
                    # get rid of all on times that are after the last off time
                    if event_on_times[-1] > event_off_times[-1]:
                        event_on_times = event_on_times[:-1]
                    # make sure the lengths are the same
                    if len(event_on_times) != len(event_off_times):
                        logger.error("ERROR: Event on and off times are not the same length")
                        continue
                    event_df['start_time'].extend(event_on_times/fs)
                    event_df['duration'].extend((event_off_times - event_on_times)/fs)
                    event_df['event_name'].extend([event_type]*len(event_on_times))
        event_df = pd.DataFrame(event_df)
        event_df.to_csv(self.output()["events"].path, index=False)

class GenerateSingleSitePlaybackCSV(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            "site_config": SiteConfig(self.site_dir),
            'project_config': SiteProjectConfig(self.site_dir),
            "dig_memmap": GenerateSingleSiteDigitalMemmap(self.site_dir)
        }

    def output(self):
        return {
            "playbacks": luigi.LocalTarget(os.path.join(self.site_dir, "playback_segments_v2.csv")),
        }
    
    def run(self):
        site_config = self.input()["site_config"].data()
        project_cfg = self.input()["project_config"].data()

        logger.info("Loading digital data from {}".format(self.input()["dig_memmap"].path))
        #dig_trace = np.fromfile(self.input()["dig_memmap"].path, dtype='uint16')
        dig_trace = np.memmap(self.input()["dig_memmap"].path, dtype='uint16', mode='r')
        #dig_trace = multirecording.get_traces().flatten()
        fs = 30000 #TODO get this from the recording
        num_chans = int(np.log2(np.max(dig_trace)))
        traces = dict()
        # NB: np.unpackbits does this for 8 bit unsigned ints. This works for us as long as we are not using more than 8 channels
        for i in range(num_chans):
            dig_data = np.not_equal(np.bitwise_and(dig_trace, (1<<i)),0)
            # if data is mostly ones then flip it
            if dig_data.mean() > .5:
                dig_data = ~dig_data
            traces[i] = dig_data.astype(np.int8)
        
        logger.info("Converting digital data to metadata")
        # now handle the digital codes on the first channel
        stimulus_codes = convert_raw_digital_to_metadata(traces[0],fs)


        # now handle the digital events TODO put this in site_config
        event_key = dict({
            'TTL': 1,
            'PeckPort':2,
            'Feeder':3
        })

        #   Here, if there is an issue with stim codes, we can spoof them. NOT USUAL
        if len(stimulus_codes) > 0 and np.all([t[1]['trial'] == 0 for t in stimulus_codes]):
            if 'TTL' in event_key:
                log_files = [f for f in site_config["filelist"] if f.endswith("events.log")]

                # this is a hack to get the trial numbers from the digital data
                # we will assume that the first trial starts at the first TTL
                # first we will get the TTL pulses
                TTL = np.diff(traces[event_key['TTL']])
                file_timeoffset = calculate_event_timedelta(np.where(TTL==1)/fs, log_files, 0 )
                
                logger.info("Fixing stim codes because they are all 0")
                #TODO make sure this works
                stimulus_codes = fix_stimulus_codes(stim_codes=stimulus_codes, 
                                                    csv_files=csv_files,stim_dirs=stim_dirs,
                                                    sample_rate=sampling_rate, 
                                                    session_starttime=session_start,
                                                    session_offset=file_timeoffset) 
            else:
                logger.error("ERROR: No TTL channel found in digital data, and no stim codes found")

        playback_csvs = [f for f in site_config["filelist"] if f.endswith("trialdata.csv")]
        playback_csvs = [os.path.join(project_cfg['LOCAL_RAW_FILES']['pyoperant_drive'],f) for f in playback_csvs]
        stim_dirs = _get_stim_dirs(playback_csvs, project_cfg['LOCAL_RAW_FILES']['stimuli_drive'])
        # Combine the data from the csvs into a single pickled dataframe
        logger.info("Saving stim database {}".format(self.output()["playbacks"].path))
        playbacks_db = convert_pyoperant_playback_csvs(
                stimulus_codes,
                playback_csvs,
                stim_dirs,
                None)
        
        playback_dict = update_operant_db(playbacks_db,fs)
        
        df = pd.DataFrame(playback_dict, columns=operant_columns)
        df.to_csv(self.output()['playbacks'].path, index=False)

        # # now handle the digital events
        # event_df = dict({
        #     'start_time': [],
        #     'duration': [],
        #     'event_name': []
        # })
        # for event_type in event_key.keys():
        #     if event_type != 'TTL':
        #         event_ind = event_key[event_type]
        #         if event_ind < len(traces):
        #             event_data = np.diff(traces[event_ind])
        #             event_on_times = np.where(event_data == 1)[0]
        #             event_off_times = np.where(event_data == -1)[0]
        #             if len(event_on_times) == 0 or len(event_off_times) == 0:
        #                 logger.info("INFO: No event on or off times found for {}".format(event_type))
        #                 continue
        #             # get rid of all off times that are before the first start time
        #             if event_off_times[0] < event_on_times[0]:
        #                 event_off_times = event_off_times[1:]
        #             # get rid of all on times that are after the last off time
        #             if event_on_times[-1] > event_off_times[-1]:
        #                 event_on_times = event_on_times[:-1]
        #             # make sure the lengths are the same
        #             if len(event_on_times) != len(event_off_times):
        #                 logger.error("ERROR: Event on and off times are not the same length")
        #                 continue
        #             event_df['start_time'].extend(event_on_times/fs)
        #             event_df['duration'].extend((event_off_times - event_on_times)/fs)
        #             event_df['event_name'].extend([event_type]*len(event_on_times))
        # event_df = pd.DataFrame(event_df)
        # event_df.to_csv(self.output()["events"].path, index=False)

class GenerateSingleSiteGLabCSV(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            "site_config": SiteConfig(self.site_dir),
            'project_config': SiteProjectConfig(self.site_dir),
            "dig_memmap": GenerateSingleSiteDigitalMemmap(self.site_dir)
        }

    def output(self):
        return {
            "playbacks": luigi.LocalTarget(os.path.join(self.site_dir, "playback_segments_v2.csv")),
        }
    
    def run(self):
        site_config = self.input()["site_config"].data()
        project_cfg = self.input()["project_config"].data()

        logger.info("Loading digital data from {}".format(self.input()["dig_memmap"].path))
        dig_info = json.load(open(self.input()['dig_memmap']['info'].path, 'r'))
        fs = dig_info['sampling_rate']
        n_samples = dig_info['num_samples']
        n_chan = dig_info['num_channels']
        dig_dtype = dig_info['dtype']
        #dig_trace = np.fromfile(self.input()["dig_memmap"].path, dtype='uint16')
        dig_trace = np.memmap(self.input()["dig_memmap"].path, dtype=dig_dtype, shape=(n_samples, n_chan), mode='r')

        
        

class GenerateSingleSiteDigitalMemmap(luigi.Task):
    """
    Generate numpy binary arrays of digital signals.
    """
    site_dir = SiteParameter()
    def requires(self):
        return {
            "site_config": SiteConfig(self.site_dir), 
            "project_config": SiteProjectConfig(self.site_dir)
        }

    def output(self):
        return {
            'bin': luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "digital_%s.bin"%os.path.basename(self.site_dir))),
            'info':luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "digital_%s_info.json"%os.path.basename(self.site_dir)))
        }
    
    def run(self):
        site_config = self.input()["site_config"].data()
        project_cfg = self.input()["project_config"].data()
        rhd_dir = project_cfg['LOCAL_RAW_FILES']['rhd_drive']

        # get rhd files in order
        rhd_files = [f for f in site_config['filelist'] if f.endswith('.rhd')]
        sorted_rhds = get_sorted_rhd_files(rhd_files)

        all_intan_io = []
        # iterate through rhd files and load the digital events
        for rhd_file in sorted_rhds:
            rhd_file = os.path.join(rhd_dir, rhd_file)
            rhd = FastIntanRecordingExtractor(rhd_file, stream_name='USB board digital input channel', ignore_integrity_checks=True)
            all_intan_io.append(rhd)
        
        # concatenate the recording extractors in a single one
        multirecording = ConcatenateSegmentRecording(recording_list=all_intan_io)

        # now save info
        info = get_recording_info(multirecording)

        self.output()['info'].makedirs()
        with open(self.output()['info'].path, 'w') as f:
            json.dump(info, f)
        # then save to binary if the file does not exist
        if not self.output()['bin'].exists():
            filename, N, c, s, fs, probe_path = kilosort.io.spikeinterface_to_binary(
                multirecording, os.path.dirname(self.output()['bin'].path), data_name=os.path.basename(self.output()['bin'].path), dtype=np.uint16,
                chunksize=120000   )

class GenerateSingleSiteADCMemmap(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            "site_config": SiteConfig(self.site_dir), 
            "project_config": SiteProjectConfig(self.site_dir)
        }
    def output(self):
        return {
            'bin': luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "audio_%s.wav"%os.path.basename(self.site_dir))),
            'info':luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "audio_%s_info.json"%os.path.basename(self.site_dir)))
        }

    def run(self):
        site_config = self.input()["site_config"].data()
        project_cfg = self.input()["project_config"].data()
        rhd_dir = project_cfg['LOCAL_RAW_FILES']['rhd_drive']

        # get rhd files in order
        rhd_files = [f for f in site_config['filelist'] if f.endswith('.rhd')]
        sorted_rhds = get_sorted_rhd_files(rhd_files)

        all_intan_io = []
        # iterate through rhd files and load the digital events
        for rhd_file in sorted_rhds:
            print(rhd_file)
            rhd_file = os.path.join(rhd_dir, rhd_file)
            rhd = FastIntanRecordingExtractor(rhd_file, stream_name='USB board ADC input channel', ignore_integrity_checks=True)
            all_intan_io.append(rhd)

        # concatenate the recording extractors in a single one
        multirecording = ConcatenateSegmentRecording(recording_list=all_intan_io)

        # add a high pass filter
        multirecording_f = bandpass_filter(multirecording, freq_min=200, freq_max=8000)

        # now save info
        info = get_recording_info(multirecording_f)

        self.output()['info'].makedirs()
        with open(self.output()['info'].path, 'w') as f:
            json.dump(info, f)

        # save to wav file if it doesnt exist
        if not os.path.exists(self.output()['bin'].path):
            filename, N, c, s, fs, probe_path = kilosort.io.spikeinterface_to_binary(
                multirecording, os.path.dirname(self.output()['bin'].path), data_name=os.path.basename(self.output()['bin'].path), dtype=np.uint16,
                chunksize=120000   )

class GenerateSingleSiteAuxMemmap(luigi.Task):
    """
    Generate numpy binary arrays of digital signals.
    """
    site_dir = SiteParameter()
    sample_rate = luigi.IntParameter(default=7500)
    def requires(self):
        return {
            "site_config": SiteConfig(self.site_dir), 
            "project_config": SiteProjectConfig(self.site_dir)
        }

    def output(self):
        return {
            'bin': luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "aux_%s.bin"%os.path.basename(self.site_dir))),
            'info':luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "aux_%s_info.json"%os.path.basename(self.site_dir)))
        }

    
    def run(self):
        site_config = self.input()["site_config"].data()
        project_cfg = self.input()["project_config"].data()
        rhd_dir = project_cfg['LOCAL_RAW_FILES']['rhd_drive']

        # get rhd files in order
        rhd_files = [f for f in site_config['filelist'] if f.endswith('.rhd')]
        sorted_rhds = get_sorted_rhd_files(rhd_files)

        all_intan_io = []
        # iterate through rhd files and load the digital events
        for rhd_file in sorted_rhds:
            rhd_file = os.path.join(rhd_dir, rhd_file)
            rhd = FastIntanRecordingExtractor(rhd_file, stream_name='RHD2000 auxiliary input channel', ignore_integrity_checks=True)
            all_intan_io.append(rhd)
        
        # concatenate the recording extractors in a single one
        multirecording = ConcatenateSegmentRecording(recording_list=all_intan_io)
        # Auxiliary channels are unipolar ADC readings (e.g. accelerometer), not
        # bipolar neural signals, so they stay unsigned to match dtype=np.uint16 below.
        # TODO could add a filter and downsample here
        # then save to binary
        # now save info
        info = get_recording_info(multirecording)

        self.output()['info'].makedirs()
        with open(self.output()['info'].path, 'w') as f:
            json.dump(info, f)
        # then save to binary if the file does not exist
        if not self.output()['bin'].exists():
            filename, N, c, s, fs, probe_path = kilosort.io.spikeinterface_to_binary(
                multirecording, os.path.dirname(self.output()['bin'].path), data_name=os.path.basename(self.output()['bin'].path), dtype=np.uint16,
                chunksize=120000   )


class GenerateSingleSiteMemmap(luigi.Task):
    """
    Generate numpy binary arrays of ephys data.
    
    This is used as input to kilosort
    Generates on .dat file for each session containing all channels of ephys
    """
    site_dir = SiteParameter()

    def requires(self):
        return {
            "site_config": SiteConfig(self.site_dir), 
            "project_config": SiteProjectConfig(self.site_dir)
        }
    
    def output(self):
        return {
            'bin':luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "%s.bin"%os.path.basename(self.site_dir))),
            'info':luigi.LocalTarget(os.path.join(self.site_dir, 'raw', "%s_info.json"%os.path.basename(self.site_dir)))
        }

    def run(self):
        site_config = self.input()["site_config"].data()
        project_cfg = self.input()["project_config"].data()
        rhd_dir = project_cfg['LOCAL_RAW_FILES']['rhd_drive']

        # get rhd files in order
        rhd_files = [f for f in site_config['filelist'] if f.endswith('.rhd')]
        sorted_rhds = get_sorted_rhd_files(rhd_files)

        all_intan_io = []
        # iterate through rhd files and load the digital events
        for rhd_file in sorted_rhds:
            rhd_file = os.path.join(rhd_dir, rhd_file)
            rhd = FastIntanRecordingExtractor(rhd_file, stream_name='RHD2000 amplifier channel', ignore_integrity_checks=True)
            all_intan_io.append(rhd)

        # concatenate the recording extractors in a single one
        multirecording = ConcatenateSegmentRecording(recording_list=all_intan_io)
        multirecording = unsigned_to_signed(multirecording)
        # according to INTAN: 
        # Each sampling cycle is separated into 35 time slices
        # in the first 32 time slices, the data from each channel is sampled in sequence
        # for 64 channels, there are 2 32 channel groups, so 0 and 32 are sampled at the same time, etc.
        time_bins = np.linspace(0,1,36) # split the time into 35 bins
        time_shifts = dict({})
        for i in range(0,32):
            time_shifts['A-%03d'%i] = time_bins[i]
            time_shifts['A-%03d'%(i+32)] = time_bins[i]
        
        # lets try a bandpass instead of highpass
        multirecording = RemoveJumps(multirecording, threshold=6000) # about 1.1 mV if you jump up and down by that much within 1 same you are removed
        multirecording = bandpass_filter(multirecording, freq_min=300, freq_max=8000)
        if 'bad_channels' in site_config and site_config['bad_channels'] != []:
            multirecording = multirecording.remove_channels(['A-%03d'%x for x in site_config['bad_channels']])
        # inter_sample_shift = [time_shifts[x] for x in multirecording.get_channel_ids()]
        # multirecording = phase_shift(multirecording,inter_sample_shift=inter_sample_shift)
        multirecording = common_reference(recording=multirecording)#,groups=pared_groups) # could group by shank, but dont think its necessary

        # now save info
        info = get_recording_info(multirecording)

        self.output()['info'].makedirs()
        with open(self.output()['info'].path, 'w') as f:
            json.dump(info, f)
        # then save to binary if the file does not exist
        if not self.output()['bin'].exists():
            filename, N, c, s, fs, probe_path = kilosort.io.spikeinterface_to_binary(
                multirecording, os.path.dirname(self.output()['bin'].path), data_name=os.path.basename(self.output()['bin'].path), dtype=np.int16,
                chunksize=30000   )       

class RemoteRawData(RemoteTask):
    """
    This task is used to get a raw data file. it will copy the file from the remote location to the local location
    """
    def requires(self):
        return {
            "full_bin_array": GenerateSingleSiteMemmap(self.remote_site_dir)
        }
    def output(self):
        if self.is_remote and self.remote_config['SPIKE_SORTING']['COPY_RAW']:
            return {
                'bin': luigi.LocalTarget(os.path.join(self.site_dir,'raw',"%s.bin"%self.site_name)),
                'info': luigi.LocalTarget(os.path.join(self.site_dir,'raw',"%s_info.json"%self.site_name))
            }
        else:
            # we are not copying the raw data, just point to the output of GenerateSingleSiteMemmap
            return{
                'bin': self.input()['full_bin_array']['bin'],
                'info': self.input()['full_bin_array']['info']
            }

    def run(self):
        site_config = self.site_config

        out_dir = os.path.join(self.site_dir,'raw')
        os.makedirs(out_dir,exist_ok=True)
        if self.is_remote is not None and self.remote_config['SPIKE_SORTING']['COPY_RAW']:
            # copy the raw data to the local location
            for key in ['bin','info']:
                remote_path = self.input()['full_bin_array'][key].path
                local_path = self.output()[key].path
                if not os.path.exists(local_path):
                    print(f"Copying {remote_path} to {local_path}")
                    shutil.copy(remote_path,local_path)
                else:
                    print(f"File already exists: {local_path}")
        else:
            print("Not copying raw data due to remote config setting, or file already exists")



class GenerateSingleSiteFiles(luigi.WrapperTask):
    """Generate all the files for a single site"""
    site_dir = SiteParameter()

    def requires(self):
        return [
            GenerateSingleSiteMemmap(self.site_dir),
            GenerateSingleSiteDigitalMemmap(self.site_dir),
            GenerateSingleSiteAuxMemmap(self.site_dir),
            GenerateSingleSiteLFP(self.site_dir),
            GenerateSingleSiteAudioFile(self.site_dir),
            CheckSiteRHDs(self.site_dir)
            #GenerateSingleSiteOperantCSV(self.site_dir),
            #GenerateSingleSitePlaybackCSV(self.site_dir),
        ]
     

class Rhd2Nix(luigi.Task):
    """Converts an RHD file to a NIX file, and clean up the RHD file

    Inputs
    ------
    session_dir : str
        Path to session

    Outputs a nix file as raw/{session_name}.nix
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            "files": DownloadSession(self.session_dir),
            "config": SessionConfig(self.session_dir),
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        nix_filename = "{}.nix".format(session_basename(self.session_dir))

        # Temporarily move nix file from segmentation dir to raw dir
        if os.path.exists(os.path.join(self.session_dir, "segmentation", nix_filename)):
            shutil.move(os.path.join(self.session_dir, "segmentation",
                        nix_filename), os.path.join(raw_dir, nix_filename))

        return luigi.LocalTarget(os.path.join(raw_dir, nix_filename))

    def run(self):
        rhd_file = self.input()["files"]["rhd"].path
        session_config = self.input()["config"].data()

        # Delete the digital input form the rhd file if from Seewiesen.
        # These are not used for stim id but to sync with bat logger
        # TODO: this is not a great place for this in case we want to keep
        # the digital codes at some point? and seewiesen also doesnt apply to the owncloud data either...
        if (session_config["download_type"] == "seewiesen"):
            print("Seewiesen Data: digital events in rhd will be deleted")
            storeDIG = False
        else:
            storeDIG = True

        rhd = IntanIO(rhd_file)
        blk = rhd.read_block()
        blk.segments[0].annotations['signal_channels'] = [x[0]
                                                          for x in rhd.header['signal_channels']]
        blk.segments[0].annotations['event_channels'] = [x[0]
                                                         for x in rhd.header['event_channels']]
        nix = NixIO(self.output().path)
        nix.write_block(blk)
        nix.close()

        # Remove the RHD file now that we have a NIX file
        # os.remove(inputs["rhd"].path)


class Nix2Wav(luigi.Task):
    """Write analogsignals to disk as wav files (RF64 format)

    Outputs files for the microphone and electrode data arrays,
    and also writes the mapping between the channels in these files
    to their original channels into the configuration file.

    The microphone data is also normalized and zero-centered according
    to the filter groups defined in the config "filtering" key

    Inputs
    ------
    session_dir : str
        Path to session
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            "nix": Rhd2Nix(self.session_dir),
            "config": SessionConfig(self.session_dir, ["wavmap", "filtering"])
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        mic_path = os.path.join(raw_dir, "mic.wav")
        elec_path = os.path.join(raw_dir, "ephys.wav")
        motion_path = os.path.join(raw_dir, "motion.wav")

        site_dirs, _, _ = get_session_sites(self.session_dir)
        config_path = os.path.join(site_dirs[0], "site_config.yaml")

        return {
            "mic.wav": luigi.LocalTarget(mic_path),
            "ephys.wav": luigi.LocalTarget(elec_path),
            "motion.wav": luigi.LocalTarget(motion_path),
            "config_original_channels": YamlTarget(config_path, "wav_to_original_channels")
        }

    def run(self):
        outputs = self.output()
        nix_file = self.input()["nix"].path
        session_config = self.input()["config"].data()

        arrays = get_wavfile_data_from_neo(
            self.input()["nix"].path, session_config["wavmap"])

        with soundfile.SoundFile(
                self.output()["mic.wav"].path,
                "w+",
                samplerate=arrays["mic"]["sampling_rate"],
                format="RF64",
                subtype="FLOAT",
                channels=arrays["mic"]["data"].shape[1]) as f:
            print(type(arrays["mic"]["data"]))
            arrays["mic"]["data"] = np.array(arrays["mic"]["data"])
            arrays["mic"]["data"] -= np.median(arrays["mic"]["data"], axis=0)
            for group_start, group_end in session_config["filtering"]["mic.wav"]:
                arrays["mic"]["data"][:, group_start:group_end] /= np.max(
                    np.abs(arrays["mic"]["data"][:, group_start:group_end]))
            f.write(arrays["mic"]["data"])

        if not os.path.exists(self.output()["ephys.wav"].path):
            with soundfile.SoundFile(
                    self.output()["ephys.wav"].path,
                    "w+",
                    samplerate=arrays["ephys"]["sampling_rate"],
                    format="RF64",
                    subtype="FLOAT",
                    channels=arrays["ephys"]["data"].shape[1]) as f:
                f.write(arrays["ephys"]["data"])

        if "motion" in arrays and arrays["motion"]["data"] is not None:
            with soundfile.SoundFile(
                    self.output()["motion.wav"].path,
                    "w+",
                    samplerate=arrays["motion"]["sampling_rate"],
                    format="RF64",
                    subtype="FLOAT",
                    channels=arrays["motion"]["data"].shape[1]) as f:
                f.write(arrays["motion"]["data"])
        else:
            open(self.output()["motion.wav"].path, 'a').close()

        # Finally, write the channel name mappings to the configuration file
        # This is kind of an extra thing; not sure if I should put it in the
        # task outputs / set it as a new task, or what?

        # FIXME: The NixIO seems to change the order of the signals...
        # The order in the .rhd file is fine, why is it when saved to/loaded from nix
        # the order gets messed up? This can mess up any tasks that rely
        # on plain integer indexing (like filtering) so need to investigate this
        config_out = self.output()["config_original_channels"].data()
        config_out["wav_to_original_channels"] = {
            "{}.wav".format(k): v["channel_names"] for k, v in arrays.items()
        }
        save_yaml(self.output()["config_original_channels"].path, config_out)


class Rhd2Wav(luigi.Task):
    """Write analogsignals to disk as wav files (RF64 format)

    Outputs files for the microphone and electrode data arrays,
    and also writes the mapping between the channels in these files
    to their original channels into the configuration file.

    The microphone data is also normalized and zero-centered according
    to the filter groups defined in the config "filtering" key

    Inputs
    ------
    session_dir : str
        Path to session
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            "files": DownloadSession(self.session_dir),
            "config": SessionConfig(self.session_dir),
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        mic_path = os.path.join(raw_dir, "mic.wav")
        elec_path = os.path.join(raw_dir, "ephys.wav")
        motion_path = os.path.join(raw_dir, "motion.wav")

        site_dirs, _, _ = get_session_sites(self.session_dir)
        config_path = os.path.join(site_dirs[0], "site_config.yaml")

        return {
            "mic.wav": luigi.LocalTarget(mic_path),
            "ephys.wav": luigi.LocalTarget(elec_path),
            "motion.wav": luigi.LocalTarget(motion_path),
            "config_original_channels": YamlTarget(config_path, "wav_to_original_channels")
        }

    def run(self):

        rhd_file = self.input()["files"]["rhd"].path
        session_config = self.input()["config"].data()

        arrays = get_wavfile_data_from_neo(rhd_file, session_config["wavmap"])

        with soundfile.SoundFile(
                self.output()["mic.wav"].path,
                "w+",
                samplerate=arrays["mic"]["sampling_rate"],
                format="RF64",
                subtype="FLOAT",
                channels=arrays["mic"]["data"].shape[1]) as f:
            arrays["mic"]["data"] = np.array(arrays["mic"]["data"])
            arrays["mic"]["data"] -= np.median(arrays["mic"]["data"], axis=0)
            for group_start, group_end in session_config["filtering"]["mic.wav"]:
                arrays["mic"]["data"][:, group_start:group_end] /= np.max(
                    np.abs(arrays["mic"]["data"][:, group_start:group_end]))
            f.write(arrays["mic"]["data"])

        if not os.path.exists(self.output()["ephys.wav"].path):
            with soundfile.SoundFile(
                    self.output()["ephys.wav"].path,
                    "w+",
                    samplerate=arrays["ephys"]["sampling_rate"],
                    format="RF64",
                    subtype="FLOAT",
                    channels=arrays["ephys"]["data"].shape[1]) as f:
                f.write(arrays["ephys"]["data"])

        if "motion" in arrays and arrays["motion"]["data"] is not None:
            with soundfile.SoundFile(
                    self.output()["motion.wav"].path,
                    "w+",
                    samplerate=arrays["motion"]["sampling_rate"],
                    format="RF64",
                    subtype="FLOAT",
                    channels=arrays["motion"]["data"].shape[1]) as f:
                f.write(arrays["motion"]["data"])
        else:
            open(self.output()["motion.wav"].path, 'a').close()

        # Finally, write the channel name mappings to the configuration file
        # This is kind of an extra thing; not sure if I should put it in the
        # task outputs / set it as a new task, or what?

        # FIXME: The NixIO seems to change the order of the signals...
        # The order in the .rhd file is fine, why is it when saved to/loaded from nix
        # the order gets messed up? This can mess up any tasks that rely
        # on plain integer indexing (like filtering) so need to investigate this
        config_out = self.output()["config_original_channels"].data()
        config_out["wav_to_original_channels"] = {
            "{}.wav".format(k): v["channel_names"] for k, v in arrays.items()
        }
        save_yaml(self.output()["config_original_channels"].path, config_out)

class Rhd2Numpy(luigi.Task):
    """Write analogsignals to disk as numpy memmaps

    Outputs files for the microphone and electrode data arrays,
    and also writes the mapping between the channels in these files
    to their original channels into the configuration file.

    The microphone data is also normalized and zero-centered according
    to the filter groups defined in the config "filtering" key

    Inputs
    ------
    session_dir : str
        Path to session
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            "files": DownloadSession(self.session_dir),
            "config": SessionConfig(self.session_dir),
        }

    def output(self):
        sess_name = os.path.basename(self.session_dir)
        raw_dir = os.path.join(self.session_dir, "raw")
        elec_path = os.path.join(raw_dir, "%s_16bit.npy"%sess_name)

        site_dirs, _, _ = get_session_sites(self.session_dir)
        config_path = os.path.join(site_dirs[0], "site_config.yaml")

        return {
            "ephys.npy": luigi.LocalTarget(elec_path),
            "config_original_channels": YamlTarget(config_path, "dat_to_original_channels")
        }

    def run(self):

        rhd_file = self.input()["files"]["rhd"].path
        session_config = self.input()["config"].data()

        arrays = get_raw_data_from_neo(rhd_file, session_config["wavmap"])

        if not os.path.exists(self.output()["ephys.npy"].path):
            with open(self.output()["ephys.npy"].path,'wb') as f:
                np.save(f,arrays['ephys']['data'])
        

        # Finally, write the channel name mappings to the configuration file
        # This is kind of an extra thing; not sure if I should put it in the
        # task outputs / set it as a new task, or what?

        # FIXME: The NixIO seems to change the order of the signals...
        # The order in the .rhd file is fine, why is it when saved to/loaded from nix
        # the order gets messed up? This can mess up any tasks that rely
        # on plain integer indexing (like filtering) so need to investigate this
        config_out = self.output()["config_original_channels"].data()
        config_out["dat_to_original_channels"] = {
            "{}.wav".format(k): v["channel_names"] for k, v in arrays.items()
        }
        
        save_yaml(self.output()["config_original_channels"].path, config_out)

class Rhd2Binary(luigi.Task):
    """Write analogsignals to disk as binary memmaps

    Outputs files for the microphone and electrode data arrays,
    and also writes the mapping between the channels in these files
    to their original channels into the configuration file.

    The microphone data is also normalized and zero-centered according
    to the filter groups defined in the config "filtering" key

    Inputs
    ------
    session_dir : str
        Path to session
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            "files": DownloadSession(self.session_dir),
            "config": SessionConfig(self.session_dir),
        }

    def output(self):
        sess_name = os.path.basename(self.session_dir)
        raw_dir = os.path.join(self.session_dir, "raw")
        elec_path = os.path.join(raw_dir, "%s.bin"%sess_name)

        audio_path = os.path.join(raw_dir, "%s_audio.wav"%sess_name)
        
        motion_path = os.path.join(raw_dir, "%s_motion.npy"%sess_name)

        lfp_path = os.path.join(raw_dir,'%s_LFP.bin'%sess_name)
        site_dirs, _, _ = get_session_sites(self.session_dir)
        config_path = os.path.join(site_dirs[0], "site_config.yaml")

        return {
            "ephys.bin": luigi.LocalTarget(elec_path),
            "audio.wav": luigi.LocalTarget(audio_path),
            "motion.bin": luigi.LocalTarget(motion_path),
            "config_original_channels": YamlTarget(config_path, "bin_to_original_channels")
        }

    def run(self):

        rhd_file = self.input()["files"]["rhd"].path
        session_config = self.input()["config"].data()
        
        bad_chans = session_config.get('bad_channels',None)
        print("BAD CHANNS %s"%bad_chans)
        arrays = get_raw_data_from_neo(rhd_file, session_config["wavmap"],bad_chans)

        if not os.path.exists(self.output()["ephys.bin"].path):
            arrays['ephys']['data'].tofile(self.output()["ephys.bin"].path)
                
        if not os.path.exists(self.output()['audio.wav'].path):
            with soundfile.SoundFile(
                            self.output()["audio.wav"].path,
                            "w+",
                            samplerate=arrays["mic"]["sampling_rate"],
                            format="RF64",
                            subtype="FLOAT",
                            channels=arrays["mic"]["data"].shape[1]) as f:
                arrays["mic"]["data"] = np.array(arrays["mic"]["data"])
                arrays["mic"]["data"] -= np.median(arrays["mic"]["data"], axis=0)
                for group_start, group_end in session_config["filtering"]["mic.wav"]:
                    arrays["mic"]["data"][:, group_start:group_end] /= np.max(
                        np.abs(arrays["mic"]["data"][:, group_start:group_end]))
                f.write(arrays["mic"]["data"])
        
        if not os.path.exists(self.output()['motion.bin'].path):
            if "motion" in arrays and arrays["motion"]["data"] is not None:
                np.save(self.output()['motion.bin'].path,arrays['motion']['data'])
                # previously it was a wav file
                # with soundfile.SoundFile(
                #         self.output()["motion.bin"].path,
                #         "w+",
                #         samplerate=arrays["motion"]["sampling_rate"],
                #         format="RF64",
                #         subtype="FLOAT",
                #         channels=arrays["motion"]["data"].shape[1]) as f:
                #     f.write(arrays["motion"]["data"])

        # Finally, write the channel name mappings to the configuration file
        # This is kind of an extra thing; not sure if I should put it in the
        # task outputs / set it as a new task, or what?

        # FIXME: The NixIO seems to change the order of the signals...
        # The order in the .rhd file is fine, why is it when saved to/loaded from nix
        # the order gets messed up? This can mess up any tasks that rely
        # on plain integer indexing (like filtering) so need to investigate this
        config_out = self.output()["config_original_channels"].data()
        config_out["bin_to_original_channels"] = {
            "{}.wav".format(k): v["channel_names"] for k, v in arrays.items()
        }
        
        save_yaml(self.output()["config_original_channels"].path, config_out)


class GenerateLFP(luigi.Task):
    session_dir = SessionParameter()
    def requires(self):
        return Rhd2Binary(self.session_dir)
    def output(self):
        sess_name = os.path.basename(self.session_dir)
        raw_dir = os.path.join(self.session_dir, "raw")
        lfp_path = os.path.join(raw_dir,'%s_LFP.npy'%sess_name)
        return luigi.LocalTarget(lfp_path)
    def run(self):
        from scipy.signal import butter, filtfilt,decimate
        in_data = np.fromfile(self.input()['ephys.bin'].path,dtype=np.uint16)
        in_data = in_data.reshape((in_data.shape[0]//64,64))
        hp_freq = 40
        lp_freq = 300
        filt_order = 4
        FS = 30000
        Fn = FS/2.
        LFP_freq = 1000
        b,a = butter(filt_order,(hp_freq/Fn,lp_freq/Fn),btype='bandpass')
        filtered_data = decimate(filtfilt(b=b,a=a,x=in_data,axis=0),int(FS//LFP_freq),axis=0)
        np.save(self.output().path,filtered_data.astype(np.uint16))



class Rhd2Nwb(luigi.Task):
    """Converts an RHD file to a NWB file, and clean up the RHD file

    Inputs
    ------
    session_dir : str
        Path to session

    Outputs a nix file as raw/{session_name}.nix
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            "files": DownloadSession(self.session_dir),
            "config": SessionConfig(self.session_dir),
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        nwb_filename = "{}.nwb".format(session_basename(self.session_dir))

        # Temporarily move nwb file from segmentation dir to raw dir
        if os.path.exists(os.path.join(self.session_dir, "segmentation", nwb_filename)):
            import shutil
            shutil.move(os.path.join(self.session_dir, "segmentation",
                        nwb_filename), os.path.join(raw_dir, nwb_filename))

        return luigi.LocalTarget(os.path.join(raw_dir, nwb_filename))

    def run(self):
        rhd_file = self.input()["files"]["rhd"].path
        session_config = self.input()["config"].data()

        # Delete the digital input form the rhd file if from Seewiesen.
        # These are not used for stim id but to sync with bat logger
        # TODO: this is not a great place for this in case we want to keep
        # the digital codes at some point? and seewiesen also doesnt apply to the owncloud data either...
        if (session_config["download_type"] == "seewiesen"):
            print("Seewiesen Data: digital events in rhd will be deleted")
            storeDIG = False
        else:
            storeDIG = True

        rhd = IntanIO(rhd_file)
        blk = rhd.read_block()
        # the end of rhd files is the timestamp
        rhd_split = os.path.basename(rhd_file).split('.')[0].split('_')
        t_start = datetime.datetime.strptime(
            "%s%s" % (rhd_split[-2], rhd_split[-1]), '%y%m%d%H%M%S')

        nwb = NWBIO(self.output().path, mode='w')
        nwb.write_block(blk)
        nwb.close()

        # Remove the RHD file now that we have a NIX file
        # os.remove(inputs["rhd"].path)


class Nwb2Wav(luigi.Task):
    """Write analogsignals to disk as wav files (RF64 format)

    Outputs files for the microphone and electrode data arrays,
    and also writes the mapping between the channels in these files
    to their original channels into the configuration file.

    The microphone data is also normalized and zero-centered according
    to the filter groups defined in the config "filtering" key

    Inputs
    ------
    session_dir : str
        Path to session
    """

    session_dir = SessionParameter()

    def requires(self):
        return {
            "nix": Rhd2Nix(self.session_dir),  # TODO NWB
            "config": SessionConfig(self.session_dir, ["wavmap", "filtering"])
        }

    def output(self):
        raw_dir = os.path.join(self.session_dir, "raw")
        mic_path = os.path.join(raw_dir, "mic.wav")
        elec_path = os.path.join(raw_dir, "ephys.wav")
        motion_path = os.path.join(raw_dir, "motion.wav")

        site_dirs, _, _ = get_session_sites(self.session_dir)
        config_path = os.path.join(site_dirs[0], "site_config.yaml")

        return {
            "mic.wav": luigi.LocalTarget(mic_path),
            "ephys.wav": luigi.LocalTarget(elec_path),
            "motion.wav": luigi.LocalTarget(motion_path),
            "config_original_channels": YamlTarget(config_path, "wav_to_original_channels")
        }

    def run(self):
        outputs = self.output()
        nix_file = self.input()["nix"].path
        session_config = self.input()["config"].data()

        arrays = get_wavfile_data_from_nix(
            self.input()["nix"].path, session_config["wavmap"])

        with soundfile.SoundFile(
                self.output()["mic.wav"].path,
                "w+",
                samplerate=arrays["mic"]["sampling_rate"],
                format="RF64",
                subtype="FLOAT",
                channels=arrays["mic"]["data"].shape[1]) as f:
            print(type(arrays["mic"]["data"]))
            arrays["mic"]["data"] = np.array(arrays["mic"]["data"])
            arrays["mic"]["data"] -= np.median(arrays["mic"]["data"], axis=0)
            for group_start, group_end in session_config["filtering"]["mic.wav"]:
                arrays["mic"]["data"][:, group_start:group_end] /= np.max(
                    np.abs(arrays["mic"]["data"][:, group_start:group_end]))
            f.write(arrays["mic"]["data"])

        if not os.path.exists(self.output()["ephys.wav"].path):
            with soundfile.SoundFile(
                    self.output()["ephys.wav"].path,
                    "w+",
                    samplerate=arrays["ephys"]["sampling_rate"],
                    format="RF64",
                    subtype="FLOAT",
                    channels=arrays["ephys"]["data"].shape[1]) as f:
                f.write(arrays["ephys"]["data"])

        if "motion" in arrays and arrays["motion"]["data"] is not None:
            with soundfile.SoundFile(
                    self.output()["motion.wav"].path,
                    "w+",
                    samplerate=arrays["motion"]["sampling_rate"],
                    format="RF64",
                    subtype="FLOAT",
                    channels=arrays["motion"]["data"].shape[1]) as f:
                f.write(arrays["motion"]["data"])
        else:
            open(self.output()["motion.wav"].path, 'a').close()

        # Finally, write the channel name mappings to the configuration file
        # This is kind of an extra thing; not sure if I should put it in the
        # task outputs / set it as a new task, or what?

        # FIXME: The NixIO seems to change the order of the signals...
        # The order in the .rhd file is fine, why is it when saved to/loaded from nix
        # the order gets messed up? This can mess up any tasks that rely
        # on plain integer indexing (like filtering) so need to investigate this
        config_out = self.output()["config_original_channels"].data()
        config_out["wav_to_original_channels"] = {
            "{}.wav".format(k): v["channel_names"] for k, v in arrays.items()
        }
        save_yaml(self.output()["config_original_channels"].path, config_out)


class Site2Wav(luigi.Task):
    """Save all of site's raw data into wav files
    """
    site_dir = SiteParameter()

    def requires(self):
        return SiteConfig(site_dir=self.site_dir, required_keys=["wavmap", "filtering"])

    def run(self):
        site_config = self.input().data()
        for session in site_config["sessions"]:
            session_dir = os.path.join(
                os.path.dirname(os.path.dirname(self.site_dir)
                                ), "sessions", session["id"]
            )
            yield Nix2Wav(session_dir=session_dir)

