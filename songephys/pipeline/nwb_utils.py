from datetime import datetime, timedelta
from uuid import uuid4
import os
import re
import shutil
import sys
import json
import glob
from pathlib import Path

import pandas as pd
import luigi
import numpy as np
import tqdm
import soundfile as sf
from dateutil.tz import tzlocal
from scipy.stats import zscore
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor

from pynwb import NWBHDF5IO, NWBFile
from pynwb import validate
from pynwb.file import LabMetaData
from pynwb.base import TimeSeries, DynamicTable
from pynwb.epoch import TimeIntervals
from pynwb.behavior import BehavioralTimeSeries
from pynwb.ecephys import LFP, ElectricalSeries
import kilosort

#from neo.rawio.intanrawio import read_rhd_signal_groups

from ..preprocess.sound.playback_utils import convert_raw_digital_to_metadata, convert_pyoperant_operant_csvs, _get_stim_dirs, convert_pyoperant_playback_csvs

from .pipeline_utils import (
    LocalTask,
    RemoteTask,
    SiteParameter,
    parse_filelist,
    session_basename,
)
from .file_setup import (
    SiteProjectConfig,
    SiteConfig,
    GenerateSingleSiteMemmap,
    GenerateSingleSiteDigitalMemmap, 
    GenerateSingleSiteADCMemmap, 
    GenerateSingleSiteLFP, 
    GenerateSingleSiteAuxMemmap, 
    GenerateSingleSiteAudioFile,
    RemoteRawData,
    DetectStimEvents,
)
# get_digital_channel_bits and ttl_on_off moved to stim_artifact (file_setup needs
# them too and cannot import this module); re-exported here for existing callers
from ..preprocess.ephys.stim_artifact import (
    get_digital_channel_bits,
    ttl_on_off,
    load_stim_events,
    stim_nwb_rows,
)

from ..format.io import load_yaml
from .truncation import get_site_tmax
from ..format.intan_dat import recording_name

def get_rhd_start_from_fn(rhd_file):
    # start is from the filename (or the folder name for .dat recordings, FOLDER/info.rhd)
    date_string = recording_name(rhd_file).split('__')[-1]
    start_time = datetime.strptime(date_string, "%y%m%d_%H%M%S")
    return start_time

def get_pyop_start_from_fn(trialdata_csv):
    date_string = trialdata_csv.split('_')[-2]
    start_time = datetime.strptime(date_string, "%Y%m%d%H%M%S")
    return start_time
def get_pyop_start_end(trialdata_csv):
    start_time = get_pyop_start_from_fn(trialdata_csv)
    df = pd.read_csv(trialdata_csv)
    if len(df) == 0:
        return None
    end_time = datetime.fromisoformat(df['time'].iloc[-1])
    return start_time, end_time

def pyoperant_type(trialdata):
    if type(trialdata) == str:
        trialdata = pd.read_csv(trialdata)
    if 'response' in trialdata.columns:
        return 'operant'
    else:
        return 'playback'





class Experiment(object):
    def __init__(self, file):
        pass
class PyoperantExperiment(Experiment):
    # TODO could add file start and end time based on logs, also could load yaml file for config info
    def __init__(self, filename):
        self.filename = filename
        self.uid = os.path.basename(filename).split('_trialdata.csv')[0]
        self.data = pd.read_csv(filename, index_col='index')
    
    @property
    def ntrials(self):
        return len(self.data)
    
    @property
    def experiment_type(self):
        return pyoperant_type(self.data)
    
    @property
    def experiment_start_time(self):
        date_string = self.filename.split('_')[-2]
        start_time = datetime.strptime(date_string, "%Y%m%d%H%M%S")
        return start_time
    
    @property
    def experiment_end_time(self):
        # TODO get this from the eventslog, but for now we will return end of the last trial
        return self.trial_end_time(self.ntrials)
    
    def get_trial(self, trial_num):
        if trial_num > self.ntrials:
            raise ValueError("Trial number exceeds number of trials")
        return self.data[self.data.index == trial_num].iloc[0]

    def trial_start_time(self, trial_num):
        return datetime.fromisoformat(self.get_trial(trial_num).time)
    
    def trial_duration(self, trial_num):
        trial = self.get_trial(trial_num)
        if self.experiment_type == 'operant':
            if trial.response:
                rt = datetime.strptime(trial.rt, "%H:%M:%S.%f")
                rtd = timedelta(hours=rt.hour, minutes=rt.minute, seconds=rt.second, microseconds=rt.microsecond)
                return rtd
            else:
                return timedelta(seconds=trial.max_wait)
        else:
            # TODO read stimui and add this to length for playbacks
            return timedelta(seconds=4)

    def trial_end_time(self, trial_num):
        return self.trial_start_time(trial_num) + self.trial_duration(trial_num)
        
    

class NWBTarget(luigi.Target):
    def __init__(self, nwbfile, *target_names):
        self.path = nwbfile
        self.target_names = target_names
    def exists(self):
        if not os.path.exists(self.path):
            return False
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        children = [x.name for x in nwbfile.all_children()]
        io.close()
        return all([x in children for x in self.target_names])
    
class NWBAcquisitionTarget(luigi.Target):
    def __init__(self, nwbfile, *target_names):
        self.path = nwbfile
        self.target_names = target_names
    def exists(self):
        if not os.path.exists(self.path):
            return False
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        exists = all([x in nwbfile.acquisition for x in self.target_names])
        io.close()
        return exists
    
    def data(self):
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        return dict({
            x: nwbfile.acquisition[x] for x in self.target_names
        })
    
    def delete(self):
        io = NWBHDF5IO(self.path, 'r+')
        nwbfile = io.read()
        for x in self.target_names:
            try:
                nwbfile.acquisition.pop(x)
            except KeyError:
                pass
        io.write(nwbfile)
        io.close()


class NWBIntervalsTarget(luigi.Target):
    def __init__(self, nwbfile, *target_names):
        self.path = nwbfile
        self.target_names = target_names
    def exists(self):
        if not os.path.exists(self.path):
            return False
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        exists = all([x in nwbfile.intervals for x in self.target_names])
        io.close()
        return exists
    
    def data(self):
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        return dict({
            x: nwbfile.intervals[x] for x in self.target_names
        })
    
    def delete(self):
        io = NWBHDF5IO(self.path, 'r+')
        nwbfile = io.read()
        for x in self.target_names:
            try:
                nwbfile.intervals.pop(x)
            except KeyError:
                pass
        io.write(nwbfile)
        io.close()
class PyoperantStimuli(TimeSeries):
    def __init__(self, name, data, rate, bird_name,rendition_id, **kwargs):
        super().__init__(name=name, data=data, unit='wav', rate=float(rate), **kwargs)
        self.bird_name = bird_name
        self.rendition_id = rendition_id

class NWBProcessingTarget(luigi.Target):
    def __init__(self, nwbfile, processing_module, *target_names):
        self.path = nwbfile
        self.processing_module = processing_module
        self.target_names = target_names
    
    def exists(self):
        if not os.path.exists(self.path):
            return False
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        if self.processing_module not in nwbfile.processing:
            io.close()
            return False
        exists = all([x in nwbfile.processing[self.processing_module].data_interfaces for x in self.target_names])
        io.close()
        return exists
    
    def data(self):
        io = NWBHDF5IO(self.path, 'r')
        nwbfile = io.read()
        return dict({
            x: nwbfile.processing[self.processing_module].data_interfaces[x] for x in self.target_names
        })
    
    def delete(self):
        io = NWBHDF5IO(self.path, 'r+')
        nwbfile = io.read()
        for x in self.target_names:
            try:
                nwbfile.processing[self.processing_module].data_interfaces.pop(x)
            except KeyError:
                pass
        io.write(nwbfile)
        io.close()

class CreateSiteNWB(LocalTask):
    def requires(self):
        self.prerun_check()
        return None

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, '%s.nwb'%os.path.basename(self.site_dir)))
    
    def run(self):
        site_config = self.site_config
        project_config = self.project_config
        
        # get the recording start time from the rhd files
        rhd_files = [f for f in site_config['filelist'] if f.endswith('.rhd')]
        if len(rhd_files) == 0:
            raise ValueError("No rhd files found in filelist")
        start_times = [get_rhd_start_from_fn(x) for x in rhd_files]
        start_time = min(start_times)

        # first init the NWB file
        nwbfile = NWBFile(
            session_description=project_config['EXPERIMENT_NAME'],
            identifier=str(uuid4()), # random uuid
            session_start_time=start_time, # TODO: Get time from the rhd file
            experimenter=[
                "Thomas, Logan",
            ],
            lab="Theunissen Lab",
            institution="University of California, Berkeley"
        )

        # first load the probe file
        # handle the probe file
        # site_config probe <- project EPHYS PROBE (see preprocess/ephys/probes.py)
        from ..preprocess.ephys.probes import load_site_probe
        prb, _, probe_path = load_site_probe(site_config, project_config, default='H15')
        probe_name = os.path.basename(probe_path)

        # first make the device
        device = nwbfile.create_device(name=probe_name, manufacturer='Cambridge Neurotech')

        # then make the electrode group for the probe
        ml = project_config['EPHYS'].get('ML', .3)
        ap = project_config['EPHYS'].get('AP', 0)
        depth = site_config['depth_mm']
        electrode_group = nwbfile.create_electrode_group(
                name=probe_name,
                description="%s probe in %s"%(probe_name,"NCM"),#TODO site_config['brain_area']),
                device=device,
                location="NCM", # TODO put this in site_config
            )

        # remove bad channels from probe
        good_chans = [True for _ in prb['chanMap']]
        if 'bad_channels' in site_config and site_config['bad_channels'] != []:
            good_chans = [x not in site_config['bad_channels'] for x in prb['chanMap']]
            # now we have to remap the chanMap
            _, new_map = np.unique(prb['chanMap'][good_chans], return_inverse=True)
            prb['chanMap'][good_chans] = new_map

        # Add electrodes to the electrode group
        # for this experiment prb file, x is A-P, y is D-V
        # in the nwb file: +x is posterior +y is inferior +z is right
        for (cid, ap, dv) in zip(prb['chanMap'][good_chans], prb['xc'][good_chans], prb['yc'][good_chans]):
            nwbfile.add_electrode(
                    id = cid,
                    group=electrode_group,
                    location="brain area",
                    x = float(ap)/1000,
                    y = depth - float(dv)/1000,
                    z = float(ml),
                    imp = np.nan,
                    filtering = 'Intan RHD2000',
                )
            
        # save the nwb
        with NWBHDF5IO(self.output().path, 'w') as io:
            io.write(nwbfile)

        
class AddDigitalEventsNWB(LocalTask):
    audcode_channel = luigi.IntParameter(default=0)
    def requires(self):
        self.prerun_check()
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'site_digital_memmap': GenerateSingleSiteDigitalMemmap(site_dir=self.site_dir)
        }

    def output(self):
        return NWBAcquisitionTarget(self.input()['site_nwb'].path, 'ttl_events')
    
    def run(self):
        site_config = self.site_config
        project_config = self.project_config
        file_list = site_config["filelist"]

        # get the name of the digital channels
        if 'digital_channel_names' in site_config:
            digital_channel_names = site_config['digital_channel_names']
        else:
            digital_channel_names = ['AUDCODE', 'TTL', 'BUTTON', 'FEEDER', 'SEED', 'HOUSE']

        # now make the ttls table
        ttls_description = "Precise time and metadata about TTL pulses"
        channel_metadata = site_config.get('channel_metadata') or dict()
        digital_metadata = [(native, md) for native, md in channel_metadata.items() if native.startswith('DIGITAL-IN')]
        if len(digital_metadata):
            # descriptions of each digital line, from the recording's channels.csv
            ttls_description += ". Channels: " + "; ".join(
                "%s = %s%s" % (native, md.get('name') or 'unnamed',
                               " (%s)" % md['description'] if md.get('description') else "")
                for native, md in digital_metadata)
        ttls_table = DynamicTable(name="ttl_events", description=ttls_description)
        ttls_table.add_column(name="timestamp", description='time of the TTL pulse', data=np.array([],dtype=float))
        ttls_table.add_column(name="ttl_type", description='type of the TTL pulse', data=np.array([],dtype=int))
        ttls_table.add_column(name='ttl_name', description='name of the TTL pulse', data=np.array([],dtype=str))
        ttls_table.add_column(name="duration", description='duration of the TTL pulse', data=np.array([],dtype=float))  
        if self.audcode_channel >= 0:
            # if no audcode channel dont add this
            audcode_table = DynamicTable(name="audcode_events", description="Precise time and metadata about stimuli played")
            audcode_table.add_column(name="timestamp", description='time of the TTL pulse', data=np.array([],dtype=float))
            audcode_table.add_column(name="code", description='stimuli code',data=np.array([],dtype=str))
            audcode_table.add_column(name="object", description='object name', data=np.array([],dtype=str))
            audcode_table.add_column(name="action", description='action name', data=np.array([],dtype=str))
            audcode_table.add_column(name="repetition", description='repetition number', data=np.array([],dtype=int))
            audcode_table.add_column(name="trial", description='trial number',  data=np.array([],dtype=int))
            audcode_table.add_column(name="md5", description='md5 hash of the stimulus file', data=np.array([],dtype=str))
        else:
            print("Not adding audcode events table because audcode_channel is set to -1")
        
        # now load the digital memmap and fill in the ttl table and events table
        digital_memmap_info = json.load(open(self.input()['site_digital_memmap']['info'].path, 'r'))
        fs = digital_memmap_info['sampling_rate']
        num_samples = digital_memmap_info['num_samples']
        num_chans = digital_memmap_info['num_channels']

        # Build a list of (ttl_type, name, function returning the 0/1 trace) for each digital channel
        if num_chans == 1:
            # the digital memmap is the packed uint16 word (bit i = DIGITAL-IN-(i+1)), so unpack it.
            # ttl_type is the bit index, so audcode_channel refers to a bit too.
            dig_trace = np.memmap(self.input()['site_digital_memmap']['bin'].path, dtype='uint16', mode='r')
            def unpack(bit):
                return lambda: np.not_equal(np.bitwise_and(dig_trace, 1 << bit), 0).astype(np.int8)
            digital_channels = [(bit, name, unpack(bit)) for bit, name in
                                get_digital_channel_bits(site_config, digital_channel_names)]
        else:
            # one column per enabled digital channel, already de-interleaved
            # TODO maybe get dtype too
            digital_memmap = np.memmap(self.input()['site_digital_memmap']['bin'].path, dtype='uint16', mode='r', shape=(num_samples, num_chans))
            def column(i):
                return lambda: (digital_memmap[:, i] != 0).astype(np.int8)
            digital_channels = [(i, name, column(i)) for i, name in enumerate(digital_channel_names[:num_chans])]

        for ttl_type, name, get_trace in digital_channels:
            dig_data = get_trace()
            if ttl_type == self.audcode_channel:
                # 0 is auditory codes usually
                stimulus_codes = convert_raw_digital_to_metadata(dig_data,fs)
                for t, metadata in stimulus_codes:
                    audcode_table.add_row(
                        timestamp=t/fs,
                        code=metadata['code'],
                        object=metadata['object'],
                        action=metadata['action'],
                        repetition=metadata['repetition'],
                        trial=metadata['trial'],
                        md5=metadata['md5']
                    )
            else:
                # Otherwise add it to our ttls table
                rising, falling = ttl_on_off(dig_data)
                print("%s (ttl_type %d): %d pulses" % (name, ttl_type, len(rising)))
                for r, f in zip(rising, falling):
                    ttls_table.add_row(
                        timestamp=r/fs,
                        ttl_type=ttl_type,
                        ttl_name=name,
                        duration=(f-r)/fs
                    )
            del dig_data
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            # add the tables to the nwbfile
            nwbfile.add_acquisition(ttls_table)
            if self.audcode_channel >= 0:
                nwbfile.add_acquisition(audcode_table)
            nwb_io.write(nwbfile)

class AddLFPNWB(LocalTask):
    def requires(self):
        self.prerun_check()
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'lfp': GenerateSingleSiteLFP(site_dir=self.site_dir)
        }

    def output(self):
        return NWBAcquisitionTarget(self.input()['site_nwb'].path, 'LFP')
    
    def run(self):

        # load the LFP data
        lfp_info = json.load(open(self.input()['lfp']['info'].path,'r'))
        lfp_shape = (lfp_info['num_samples'], lfp_info['num_channels'])
        lfp_data = np.memmap(self.input()['lfp']['bin'].path,shape=lfp_shape, dtype=lfp_info['dtype'], mode='r')
        
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            electrode_region = nwbfile.create_electrode_table_region(
                region=list(range(lfp_shape[1])),
                description="all electrodes",
            )
            lfp_ts = ElectricalSeries(
                name='LFP',
                data=lfp_data,
                electrodes=electrode_region,
                rate=float(lfp_info['sampling_rate']),
                description='LFP data from the site'
            )
            nwbfile.add_acquisition(lfp_ts)
            nwb_io.write(nwbfile)

class AddMotionNWB(LocalTask):
    def requires(self):
        self.prerun_check()
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'motion': GenerateSingleSiteAuxMemmap(site_dir=self.site_dir)
        }

    def output(self):
        return NWBProcessingTarget(self.input()['site_nwb'].path, 'behavior', 'Motion Time Series')
    
    def run(self):

        # load the motion data
        motion_info = json.load(open(self.input()['motion']['info'].path,'r'))
        motion_shape = (motion_info['num_samples'], motion_info['num_channels'])
        motion_data = np.memmap(self.input()['motion']['bin'].path,shape=motion_shape, dtype=motion_info['dtype'], mode='r')

        # rescale to 0-1
        motion_data = motion_data / 2**16 # todo: could use np.iinfo(motion_data.dtype).max
        jerk = np.diff(motion_data,axis=0)
        normed_jerk = np.linalg.norm(jerk, axis=1)
        zs_jerk = zscore(normed_jerk, axis=0)

        motion_ts = TimeSeries(
            name="motion",
            data=zs_jerk,
            rate=float(motion_info['sampling_rate']),
            description="Motion energy calculated from the derivative of the accelerometer data",
            unit="zscore",
        )

        behavioral_time_series = BehavioralTimeSeries( # TODO maybe just add the time_series
            time_series=motion_ts,
            name="Motion Time Series",
        )
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            if 'behavior' not in nwbfile.processing:
                nwbfile.create_processing_module(
                    name="behavior", description="Processed behavioral data"
                )
            nwbfile.processing['behavior'].add(behavioral_time_series)
            nwb_io.write(nwbfile)

class AddAudioNWB(LocalTask):
    def requires(self):
        self.prerun_check()
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'audio': GenerateSingleSiteAudioFile(site_dir=self.site_dir)
        }

    def output(self):
        return NWBAcquisitionTarget(self.input()['site_nwb'].path, 'audio')
    
    def run(self):

        # load the audio data
        audio_info = json.load(open(self.input()['audio']['info'].path,'r'))
        audio_shape = (audio_info['num_samples'], audio_info['num_channels'])
        if 'file_start_ends' in audio_info:
            audio_data = np.zeros(audio_shape, dtype=audio_info['dtype'])
            for idx, [start, end] in enumerate(audio_info['file_start_ends']):
                if idx == 0:
                    fn = self.input()['audio']['bin'].path
                else:
                    fn = self.input()['audio']['bin'].path.replace('.wav','-%d.wav'%idx)
                audio_data[start:end] = np.memmap(fn,shape=(end-start, audio_info['num_channels']), dtype=audio_info['dtype'], mode='r')
        else:
            # single file
            audio_data = np.memmap(self.input()['audio']['bin'].path,shape=audio_shape, dtype=audio_info['dtype'], mode='r')
        audio_ts = TimeSeries(
            name='audio',
            data=audio_data,
            rate=float(audio_info['sampling_rate']),
            description='Audio data from the site',
            unit='mV'
        )
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            nwbfile.add_acquisition(audio_ts)
            nwb_io.write(nwbfile)



class RemoveProcessingModule(LocalTask):
    module_name = luigi.Parameter()
    def requires(self):
        self.prerun_check()
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
        }
    def output(self):
        return luigi.LocalTarget("dummy")
    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            # remove the processing module
            nwbfile.processing.pop(self.module_name)
            # export to new file
            print("Writing to file")
            export_fn = self.input()['site_nwb'].path.replace('.nwb','_temp.nwb')
            with NWBHDF5IO(export_fn, 'w') as export_io:
                export_io.export(src_io=nwb_io, nwbfile=nwbfile)
                export_io.close()
            nwb_io.close()
        os.rename(export_fn, self.input()['site_nwb'].path)

class AddPyoperantTrials(LocalTask):
    def requires(self):
        self.prerun_check()
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'digital_events': AddDigitalEventsNWB(site_dir=self.site_dir)
        }
    def output(self):
        return NWBIntervalsTarget(self.input()['site_nwb'].path, 'operant_trials', 'playback_trials', 'pyoperant_experiments')
    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            audcode_events = nwbfile.acquisition['audcode_events'].to_dataframe()
            
            # create the different timeintervals we will keep track of
            pyoperant_experiments_intervals = TimeIntervals(
                name="pyoperant_experiments",
                description="Top level intervales for operant experiments, approximate start and stop times",
            )
            pyoperant_experiments_intervals.add_column(name="experiment_type", description="Type of experiment (operant or playback)",data=np.array([],dtype=str))
            pyoperant_experiments_intervals.add_column(name="uid", description="Unique Identifier for experiment (base of filename)",data=np.array([],dtype=str))

            op_trial_intervals = TimeIntervals(
                name="operant_trials",
                description="Intervals for individual operant trials"
            )
            op_trial_intervals.add_column(name="trial_num", description="Trial number",data=np.array([],dtype=int))
            op_trial_intervals.add_column(name="experiment_uid", description="Unique Identifier for experiment (base of filename)",data=np.array([],dtype=str))
            op_trial_intervals.add_column(name="stimuli_path", description="Path of Stimuli presented",data=np.array([],dtype=str))
            op_trial_intervals.add_column(name="stimuli_name", description="Name of Stimuli presented",data=np.array([],dtype=str))
            op_trial_intervals.add_column(name="bird_name", description="Name of bird",data=np.array([],dtype=str))
            op_trial_intervals.add_column(name="stim_class", description="Name of bird",data=np.array([],dtype=str))
            op_trial_intervals.add_column(name="trial_type", description="Name of bird",data=np.array([],dtype=str))
            op_trial_intervals.add_column(name="response", description="Response made",data=np.array([],dtype=bool))
            op_trial_intervals.add_column(name="correct", description="Correct response",data=np.array([],dtype=bool))

            pb_trial_intervals = TimeIntervals(
                name="playback_trials",
                description="Intervals for individual playback trials"
            )
            pb_trial_intervals.add_column(name="trial_num", description="Trial number",data=np.array([],dtype=int))
            pb_trial_intervals.add_column(name="experiment_uid", description="Unique Identifier for experiment (base of filename)",data=np.array([],dtype=str))
            pb_trial_intervals.add_column(name="stimuli_path", description="Path of Stimuli presented",data=np.array([],dtype=str))
            pb_trial_intervals.add_column(name="stimuli_name", description="Name of Stimuli presented",data=np.array([],dtype=str))
            pb_trial_intervals.add_column(name="bird_name", description="Name of bird",data=np.array([],dtype=str))
            pb_trial_intervals.add_column(name="stim_class", description="Name of bird",data=np.array([],dtype=str))
            pb_trial_intervals.add_column(name="trial_type", description="Name of bird",data=np.array([],dtype=str))

            if len(audcode_events) == 0:
                print("No audcode events found... skipping adding experiments")
            else:
                session_start = nwbfile.session_start_time.replace(tzinfo=None)
                audcode_events['datetime'] = [session_start + timedelta(seconds=x) for x in audcode_events['timestamp']]

                # all of these TimeIntervals are relevant to all time series
                all_time_series = [c for c in nwbfile.all_children() if isinstance(c, TimeSeries)]

                # find contiguous blocks of audcode trials
                trial_blocks = np.split(audcode_events,np.where(np.diff(audcode_events.trial) != 1)[0]+1)
                if len(trial_blocks) == 0:
                    trial_block_starts = []
                    trial_block_ends = []
                else:
                    trial_block_starts = [x.datetime.iloc[0] for x in trial_blocks]
                    trial_block_ends = [x.datetime.iloc[-1] for x in trial_blocks]

                # now we need to match each of these blocks with the corresponding csv files
                # first we need to find the csv files
                site_config = self.site_config
                project_config = self.project_config
                stim_root_dir = project_config['LOCAL_RAW_FILES']['stimuli_drive']
                if 'stim_dir' not in site_config:
                    stim_dir = stim_root_dir
                else:
                    stim_dir = os.path.join(stim_root_dir, site_config['stim_dir'])
                pyoperant_csvs = [os.path.join(project_config['LOCAL_RAW_FILES']['pyoperant_drive'],f) for f in site_config['filelist'] if f.endswith('trialdata.csv')]
                pyoperant_experiments = [PyoperantExperiment(x) for x in pyoperant_csvs]

                # a truncated site (site_config tmax) has no data after tmax: drop
                # experiments/trials that start after it and clip stop times to it
                tmax = get_site_tmax(site_config)
                clip = (lambda t: min(t, tmax)) if tmax is not None else (lambda t: t)

                # now we need to match the trial blocks with the operant experiments
                for exp in pyoperant_experiments:
                    exp_start = (exp.experiment_start_time - session_start).total_seconds()
                    if tmax is not None and exp_start >= tmax:
                        print("Experiment %s starts after site tmax (%.1f s); skipped" % (exp.uid, tmax))
                        continue
                    if exp.ntrials > 0:
                        # add the top level interval
                        pyoperant_experiments_intervals.add_interval(
                            start_time=exp_start,
                            stop_time=clip((exp.experiment_end_time - session_start).total_seconds()),
                            experiment_type=exp.experiment_type,
                            uid=exp.uid,
                            timeseries=all_time_series
                        )

                        # now find the trialblock that corresponds to this experiment
                        first_trial_time = exp.trial_start_time(1)
                        last_trial_time = exp.trial_end_time(exp.ntrials)
                        # find the trial block that starts at the closest time to the first trial and ends at the closest time to the last trial
                        trial_block_ix = np.argmin([abs(x-first_trial_time) for x in trial_block_starts])
                        trial_block = trial_blocks[trial_block_ix]
                        # now lets see if all the trials are here
                        if len(trial_block) != exp.ntrials:
                            print("Experiment %s has fewer trials than expected"%exp.uid)
                            print("Expected %d, found %d"%(exp.ntrials, len(trial_block)))

                        for ix, trial in exp.data.iterrows():
                            if ix in trial_block.trial.values:
                                # TODO can assert if there is more than one trial with the same number
                                audcode_event = trial_block[trial_block.trial == ix].iloc[0]
                            else:
                                # TODO can do some funny shit with TTL Pulses here
                                print("Trial %d not found in trial block for experiment %s"%(ix, exp.uid))
                                audcode_event = None

                            start = audcode_event.timestamp if audcode_event is not None else (exp.trial_start_time(ix) - session_start).total_seconds()
                            if tmax is not None and start >= tmax:
                                continue
                            stim_path = trial.stimulus_name
                            stim_path = stim_path.replace('\\','/')
                            if 'STIM_REMAP' in self.project_config:
                                stim_and_dir = os.path.join(stim_path.split("/")[-2], stim_path.split("/")[-1])
                                if stim_and_dir in self.project_config['STIM_REMAP']:
                                    stim_and_dir_new = self.project_config['STIM_REMAP'][stim_and_dir]
                                    print("Renaming %s to %s"%(stim_path,stim_and_dir_new))
                                    stim_path = stim_path[:-len(stim_and_dir)]
                                    stim_path = stim_path + stim_and_dir_new
                            stim_name = stim_path.split('/')[-1]
                            bird_name = stim_name.split('_')[3]
                            stim_class = stim_path.split('/')[-2]

                            # add the stim to the nwb file if it is not present already
                            if stim_name not in nwbfile.stimulus:
                                bird_name = stim_name.split('_')[3]
                                rendition_id = stim_name.strip('.wav').split('_')[-1]
                                stim_files = glob.glob(os.path.join(stim_dir,'**',stim_name),recursive=True)
                                if len(stim_files) == 0:
                                    print("Stimulus %s not found"%stim_name)
                                    continue
                                # TODO: for now just take the first one
                                #       we should take the shortest one...
                                #       but maybe this isnt an issue

                                stim_file = stim_files[0]
                                # now load the wav file
                                stim_data, stim_fs = sf.read(stim_file)
                                stim_ts = PyoperantStimuli(
                                    name=stim_name,
                                    data=stim_data,
                                    rate=stim_fs,
                                    bird_name = bird_name,
                                    rendition_id=rendition_id
                                )
                                nwbfile.add_stimulus(stimulus=stim_ts)


                            if exp.experiment_type == 'operant':
                                end = clip(start + exp.trial_duration(ix).total_seconds())
                                op_trial_intervals.add_interval(
                                    start_time=start,
                                    stop_time=end,
                                    trial_num=ix,
                                    experiment_uid=exp.uid,
                                    stimuli_path=stim_path,
                                    stimuli_name=stim_name,
                                    bird_name=bird_name,
                                    stim_class=stim_class,
                                    trial_type=exp.experiment_type,
                                    response=trial.response,
                                    correct=trial.correct,
                                    timeseries=all_time_series
                                )
                            else:
                                end = clip(start + (nwbfile.stimulus[stim_name].data.shape[0] / nwbfile.stimulus[stim_name].rate))
                                pb_trial_intervals.add_interval(
                                    start_time=start,
                                    stop_time=end,
                                    trial_num=ix,
                                    experiment_uid=exp.uid,
                                    stimuli_path=stim_path,
                                    stimuli_name=stim_name,
                                    bird_name=bird_name,
                                    stim_class=stim_class,
                                    trial_type=exp.experiment_type,
                                    timeseries=all_time_series
                                )
                
            # now save the trials to the nwb file
            nwbfile.add_time_intervals(pyoperant_experiments_intervals)
            nwbfile.add_time_intervals(op_trial_intervals)
            nwbfile.add_time_intervals(pb_trial_intervals)
            nwb_io.write(nwbfile)

class AddStimuliNWB(LocalTask):
    def requires(self):
        self.prerun_check()
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'pyoperant_trials': AddPyoperantTrials(site_dir=self.site_dir)
        }
    def output(self):
        return NWBStimulusTarget(self.input()['site_nwb'].path)
    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            # get the pyoperant trials
            pyoperant_trials = nwbfile.intervals['operant_trials'].to_dataframe()
            # get the unique stimuli
            stimuli = pyoperant_trials.stimuli.unique()
            # create the stimulus table
            stimulus_table = nwbfile.create_stimulus_template(name='stimuli')
            for stim in stimuli:
                stimulus_table.add_row(
                    name=stim,
                    description="Stimulus %s"%stim
                )
            nwb_io.write(nwbfile)

class AddStimEventsNWB(LocalTask):
    """Add electrical stimulation pulses, bouts and blanked windows to the NWB

    Writes (from DetectStimEvents; empty tables if stimulation removal is off):
      intervals['stim_pulses']  one row per TTL: delivered, artefact score, blank window, bout, polarity, current
      intervals['stim_bouts']   one row per bout: pulse counts, rate, polarity, current, notes
      invalid_times             one interval per blanked window, tagged 'stim_blank'
    Polarity / current / notes come from site_config['stim_bouts'].
    The raw TTL edges are also in acquisition['ttl_events'] (AddDigitalEventsNWB).
    """
    def requires(self):
        self.prerun_check()
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'stim_events': DetectStimEvents(site_dir=self.site_dir),
        }

    def output(self):
        return NWBIntervalsTarget(self.input()['site_nwb'].path, 'stim_pulses', 'stim_bouts')

    def run(self):
        events = load_stim_events(self.input()['stim_events'].path)
        channel = str(events['config']['digital_channel'])
        pulse_rows, bout_rows, blanks = stim_nwb_rows(
            events, self.site_config.get('stim_bouts'), channel_name=channel)

        cfg = events['config']
        pulses = TimeIntervals(
            name="stim_pulses",
            description=("Electrical stimulation pulses, one per TTL edge on %s (start/stop = TTL rise/fall). "
                         "delivered: a stimulation artefact followed the TTL (score > %s); False means the TTL "
                         "fired but no current reached the tissue (e.g. stimulator off). Delivered pulses "
                         "were blanked from %s ms before to %s ms after the TTL in the spike-band data used "
                         "for sorting (LFP is not blanked). polarity/current_ua from site_config stim_bouts."
                         % (channel, cfg['delivered_threshold'], cfg['ms_before'], cfg['ms_after'])))
        for name, desc, dtype in [
                ("delivered", "stimulation artefact detected after the TTL", bool),
                ("artifact_score", "median over channels of the largest jump within 1 ms of the TTL / noise", float),
                ("artifact_peak_uv", "median over channels of the peak deviation within 1 ms of the TTL (uV)", float),
                ("blanked", "pulse blanked in the spike-band data", bool),
                ("blank_start", "start of the blanked window (s), NaN if not blanked", float),
                ("blank_stop", "end of the blanked window (s), NaN if not blanked", float),
                ("bout", "bout index (pulses separated by > %s s start a new bout)" % cfg['bout_gap_s'], int),
                ("polarity", "stimulation polarity from site_config stim_bouts ('unknown' if not given)", str),
                ("current_ua", "stimulation current in uA from site_config stim_bouts (NaN if not given)", float),
                ("ttl_channel", "digital input carrying the stim TTL", str)]:
            pulses.add_column(name=name, description=desc, data=np.array([], dtype=dtype))
        for r in pulse_rows:
            pulses.add_interval(**r)

        bouts = TimeIntervals(name="stim_bouts", description="Bouts of electrical stimulation pulses (see stim_pulses)")
        for name, desc, dtype in [
                ("bout", "bout index, matches stim_pulses.bout", int),
                ("n_pulses", "number of TTL pulses", int),
                ("n_delivered", "number of pulses with a stimulation artefact", int),
                ("rate_hz", "pulse rate (Hz)", float),
                ("polarity", "stimulation polarity from site_config stim_bouts", str),
                ("current_ua", "stimulation current in uA from site_config stim_bouts", float),
                ("notes", "notes from site_config stim_bouts", str)]:
            bouts.add_column(name=name, description=desc, data=np.array([], dtype=dtype))
        for r in bout_rows:
            bouts.add_interval(**r)

        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            nwbfile.add_time_intervals(pulses)
            nwbfile.add_time_intervals(bouts)
            # blanked windows -> invalid_times (skip ones already there from a previous run)
            inv = nwbfile.invalid_times
            use_tags = inv is None or 'tags' in inv.colnames
            existing = set()
            if inv is not None and len(inv) and use_tags:
                df = inv.to_dataframe()
                existing = {(round(s, 6), round(e, 6)) for s, e, t in zip(df.start_time, df.stop_time, df.tags)
                            if 'stim_blank' in list(t)}
            if blanks and not use_tags:
                print("invalid_times has no tags column; adding stim blank windows without the 'stim_blank' tag")
            for s, e in blanks:
                if (round(s, 6), round(e, 6)) in existing:
                    continue
                if use_tags:
                    nwbfile.add_invalid_time_interval(start_time=s, stop_time=e, tags=['stim_blank'])
                else:
                    nwbfile.add_invalid_time_interval(start_time=s, stop_time=e)
            nwb_io.write(nwbfile)
        print("Added %d stim pulses (%d delivered), %d bouts, %d blanked windows to the NWB" % (
            len(pulse_rows), sum(r['delivered'] for r in pulse_rows), len(bout_rows), len(blanks)))


class GenerateAcquisitionNWB(luigi.WrapperTask):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'digital': AddDigitalEventsNWB(site_dir=self.site_dir),
            'stim': AddStimEventsNWB(site_dir=self.site_dir),
            'lfp': AddLFPNWB(site_dir=self.site_dir),
            'motion': AddMotionNWB(site_dir=self.site_dir),
            'audio': AddAudioNWB(site_dir=self.site_dir),
            'operant': AddPyoperantTrials(site_dir=self.site_dir),
            'spectrograms': AddStimSpectrogramsNWB(site_dir=self.site_dir),
        }
    

# Remote Tasks ================================================================

class CreateRemoteSiteNWB(RemoteTask):
    suffix = luigi.Parameter(default='')
    def requires(self):
        return CreateSiteNWB(site_dir=self.remote_site_dir)

    def output(self):
        # note this wont do anything if we are not remote
        return luigi.LocalTarget(os.path.join(self.site_dir, '%s_%s.nwb'%(os.path.basename(self.site_dir), self.suffix)))

    def run(self):
        # copy the file from the remote to the local with shutil
        shutil.copy(self.input().path, self.output().path)
class ResetUnits(RemoteTask):
    def requires(self):
        return {
            'site_nwb': CreateRemoteSiteNWB(site_dir=self.site_dir),
        }
    def complete(self):
        # check if the nwbfile has a units
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r') as nwb_io:
            nwbfile = nwb_io.read()
            if nwbfile.units is None:
                nwb_io.close()
                return True
            else:
                nwb_io.close()
                return False
    def output(self):
        return luigi.LocalTarget("dummy")
    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r') as nwb_io:
            nwbfile = nwb_io.read()
            print("Deleting units")
            nwbfile.units.reset_parent()
            nwbfile.fields['units'] = None
            print("Writing to file")
            # export to new file
            export_fn = self.input()['site_nwb'].path.replace('.nwb','_temp.nwb')
            with NWBHDF5IO(export_fn, 'w') as export_io:
                export_io.export(src_io=nwb_io, nwbfile=nwbfile)
                export_io.close()
            nwb_io.close()
        os.rename(export_fn, self.input()['site_nwb'].path)


class AddKilosortUnits(RemoteTask):
    # TODO can parallelize this by running on each shank, or by unit
    kilosort_dir = luigi.Parameter(default='ks4_final')
    overwrite = luigi.BoolParameter(default=False)
    clear_old_units = luigi.BoolParameter(default=False)
    include_waveforms = luigi.BoolParameter(default=True)
    n_spikes = luigi.IntParameter(default=1000)
    def requires(self):
        requirements = {
            'site_nwb': CreateRemoteSiteNWB(site_dir=self.site_dir, suffix=self.kilosort_dir)
        }
        if self.clear_old_units:
            requirements['reset'] = ResetUnits(site_dir=self.site_dir)
        if self.include_waveforms:
            requirements['raw_bin'] = RemoteRawData(site_dir=self.site_dir)
        return requirements
    
    def output(self):
        # Force this to always run if requested
        # TODO we should only add this if Units doesnt exist
        return luigi.LocalTarget("dummy") 

    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()

            if nwbfile.units is None:
                # assert there are no units
                nwbfile.add_unit_column(name='kilosort_dir', description='The directory where the kilosort output is stored',data=np.array([],dtype=str))
                nwbfile.add_unit_column(name='kilosort_id', description='The id of the unit in the kilosort output',data=np.array([],dtype=int))
                nwbfile.add_unit_column(name='group', description='The Kilosort Group (good, mua, noise)',data=np.array([],dtype=str))
                nwbfile.add_unit_column(name='ks_x', description='The approx X position in um from Kilosort',data=np.array([],dtype=int))
                nwbfile.add_unit_column(name='ks_y', description='The approx Y position in um from Kilosort',data=np.array([],dtype=int))
            # load the kilosort data
            # first check if the director exists
            ks_output_dir =  os.path.join(self.site_dir, self.kilosort_dir)
            if not os.path.exists(ks_output_dir):
                raise ValueError("Kilosort directory not found")

            if os.path.exists(os.path.join(ks_output_dir,'shank_0')):
                shanks = glob.glob(os.path.join(ks_output_dir,'shank_*'))
                ks_output_dirs = shanks
            else:
                ks_output_dirs = [ks_output_dir]
            # load samplerate from params.py
            sys.path.append(ks_output_dirs[0])
            from params import sample_rate
            for ksod in ks_output_dirs:
                ks_dir_name = os.path.relpath(ksod, self.site_dir)
                print("Loading Kilosort data from %s"%ks_dir_name)
                ks_spike_times = np.load(os.path.join(ksod, 'spike_times.npy'))/sample_rate
                ks_spike_cluster = np.load(os.path.join(ksod, 'spike_clusters.npy'))
                ks_cluster_info = pd.read_csv(os.path.join(ksod, 'cluster_info.tsv'),delimiter='\t')
                ks_spike_pos = np.load(os.path.join(ksod, 'spike_positions.npy'))
                
                for i, cluster in ks_cluster_info.iterrows():
                    if cluster['group'] != 'noise':
                        unit_spike_pos = ks_spike_pos[ks_spike_cluster == cluster['cluster_id']]
                        avg_spike_pos = np.mean(unit_spike_pos, axis=0)
    
                        if (cluster['cluster_id'], ks_dir_name) not in zip(nwbfile.units['kilosort_id'], nwbfile.units['kilosort_dir']):
                            print("Adding unit %d from %s"%(cluster['cluster_id'], ks_dir_name))
                            nwbfile.add_unit(
                                spike_times=ks_spike_times[ks_spike_cluster == cluster['cluster_id']],
                                electrodes=[cluster.ch],
                                kilosort_id=cluster['cluster_id'],
                                group=cluster['group'],
                                kilosort_dir=ks_dir_name,
                                ks_x=avg_spike_pos[0],
                                ks_y=avg_spike_pos[1]
                            )
                        else:
                            print("Unit %d from %s already exists"%(cluster['cluster_id'], ks_dir_name))
                            if self.overwrite:
                                print("Overwriting")
                                ix = (nwbfile.units['kilosort_id'] == cluster['cluster_id']) & (nwbfile.units['kilosort_dir'] == ks_dir_name)
                                nwbfile.units['spike_times'][ix] = ks_spike_times[ks_spike_cluster == cluster['cluster_id']]
                                nwbfile.units['group'][ix] = cluster['group']
                                nwbfile.units['ks_x'][ix] = avg_spike_pos[0]
                                nwbfile.units['ks_y'][ix] = avg_spike_pos[1]

            nwb_io.write(nwbfile)

class AddExperiments(RemoteTask):
    experiment_fn = luigi.Parameter(default='experiments.csv')
    def requires(self):
        return {
            'site_nwb': CreateRemoteSiteNWB(site_dir=self.site_dir),
            'audio': AddAudioNWB(site_dir=self.site_dir),
        }
    def output(self):
        return NWBIntervalsTarget(self.input()['site_nwb'].path, 'experiments')
    def run(self):
        # We will read the experiments.csv file and add it to the nwb file as a TimeIntervals table
        # The rows are: Start Time, End Time, Name, Notes
        # Note that Start Time and End Time are Times, so we need to convert them to seconds since the start of the session
        experiments_csv = os.path.join(self.site_dir, self.experiment_fn)
        if not os.path.exists(experiments_csv):
            raise ValueError("Experiments file not found: %s"%experiments_csv)
        experiments_df = pd.read_csv(experiments_csv)
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            experiments_table = TimeIntervals(
                name="experiments",
                description="Experiments conducted on this site"
            )
            experiments_table.add_column(name='experiment_name', description='Name of the experiment',data=np.array([],dtype=str))
            experiments_table.add_column(name='notes', description='Notes about the experiment',data=np.array([],dtype=str))
            session_start_time = nwbfile.session_start_time.replace(tzinfo=None)
            session_end_time = session_start_time + timedelta(seconds=nwbfile.acquisition['audio'].data.shape[0] / nwbfile.acquisition['audio'].rate)
            for i, row in experiments_df.iterrows():
                start_time_txt = row['Start Time']
                if start_time_txt == "START":
                    row['Start Time'] = session_start_time
                else:
                    row['Start Time'] = datetime.combine(session_start_time.date(), datetime.strptime(row['Start Time'], "%H:%M").time())   
                end_time_txt = row['End Time']
                if end_time_txt == "END":
                    row['End Time'] = session_end_time
                else:
                    row['End Time'] = datetime.combine(session_start_time.date(), datetime.strptime(row['End Time'], "%H:%M").time())

                experiments_table.add_interval(
                    start_time=row['Start Time'],
                    stop_time=row['End Time'],
                    experiment_name=row['Name'],
                    notes=row['Notes'],
                    timeseries=[c for c in nwbfile.all_children() if isinstance(c, TimeSeries)]
                )
            nwbfile.add_time_intervals(experiments_table)
            nwb_io.write(nwbfile)

class AddUnitWaveforms(RemoteTask):
    kilosort_dir = luigi.Parameter(default='ks4_final')
    ms_spike = luigi.IntParameter(default=2)
    n_spikes = luigi.IntParameter(default=1000)

    def requires(self):
        return {
            'site_nwb': CreateRemoteSiteNWB(site_dir=self.site_dir, suffix=self.kilosort_dir),
            'raw_bin': RemoteRawData(site_dir=self.site_dir)
        }
    def output(self):
        return NWBAcquisitionTarget(self.input()['site_nwb'].path, 'unit_waveforms')

    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwb = nwb_io.read()
            all_units = nwb.units.to_dataframe()
            
            wf_table = DynamicTable(name="unit_waveforms", description="Precise time and metadata about TTL pulses")
            wf_table.add_column(name="unit_id", description='time of the TTL pulse', data=np.array([],dtype=int))
            wf_table.add_column(name="electrodes", description='electrodes of units')
            wf_table.add_column(name="waveform_mean", description='waveform of units')
            wf_table.add_column(name="waveform_sd", description='std of waveform of units')

            # load the raw data
            bin_file = Path(self.input()['raw_bin']['bin'].path)
            bin_info = json.load(open(self.input()['raw_bin']['info'].path,'r'))
            bin_memmap = np.memmap(bin_file, dtype=bin_info['dtype'], mode='r', shape=(bin_info['num_samples'], bin_info['num_channels']))
            bin_fs = bin_info['sampling_rate']
            # ms_spike before and after each spike
            samp_spike = int(self.ms_spike * bin_fs / 1000)
            memmap_sliding_window_view = np.lib.stride_tricks.sliding_window_view(bin_memmap, samp_spike * 2 + 1, axis=0)

            for uid, unit in all_units.iterrows():
                # ignore spikes that are too close to the beginning or end of the recording
                # for generating waveform averages
                spike_times_samples = unit['spike_times']*bin_fs - samp_spike
                spike_times_samples = spike_times_samples.astype(int)

                # sample a few spikes (n_spikes) to get the average waveform
                spike_times_samples = spike_times_samples[np.searchsorted(spike_times_samples,0):np.searchsorted(spike_times_samples,memmap_sliding_window_view.shape[0])]
                spike_times_samples = np.random.choice(spike_times_samples, min(len(spike_times_samples),self.n_spikes), replace=False)

                waveforms = memmap_sliding_window_view[spike_times_samples,:]
                avg_spike = np.mean(waveforms, axis=0)
                std_spike = np.std(waveforms, axis=0)

                # lets take the top 4 channels to save
                channel_weights = np.sum(np.abs(avg_spike), axis=1)
                top_channels = np.argsort(channel_weights)[::-1][:4]
                avg_spike = avg_spike[top_channels,:]
                std_spike = std_spike[top_channels,:]
                electrodes = top_channels

                wf_table.add_row(
                    unit_id=uid,
                    electrodes=electrodes,
                    waveform_mean=avg_spike,
                    waveform_sd=std_spike
                )
            nwb.add_acquisition(wf_table)
            nwb_io.write(nwb)


    
class AddUnitValidIntervals(RemoteTask):
    kilosort_dir = luigi.Parameter(default='ks4_final')
    def requires(self):
        return {
            'site_nwb': CreateRemoteSiteNWB(site_dir=self.site_dir, suffix=self.kilosort_dir),
        }
    def output(self):
        return NWBIntervalsTarget(self.input()['site_nwb'].path, 'unit_intervals')
    def run(self):
        # load the nwb file
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            # go through the units
            nwb = nwb_io.read()
            all_units = nwb.units.to_dataframe()

            # create the intervals table
            unit_intervals = TimeIntervals(
                name="unit_intervals",
                description="Valid intervals for each unit"
            )
            unit_intervals.add_column(name='unit_id', description='The id of the unit in the nwb file that this interval corresponds to',data=np.array([],dtype=int))


            for uid, unit in all_units.iterrows():
                # lets make a psth for this unit
                spike_times = nwb.units['spike_times'][uid]

                # Get total recording time for this nwb file rounded to the nearest second
                tot_time = int(nwb.acquisition['LFP'].data.shape[0] / nwb.acquisition['LFP'].rate)

                # bin the spikes by 5
                t_bin = 3
                bins = np.arange(0, tot_time, t_bin)
                spike_counts, _ = np.histogram(spike_times, bins=bins)
                # smoothed = ndimage.gaussian_filter1d(spike_counts, 2)

                # take a sliding window of 30 seconds and determine if the rate is significantly nonzero using a t test
                window = 30 // t_bin
                step = 15 // t_bin
                active = []
                pvals = []
                active_rates = []
                active_bins = [0]
                for i in range(0, len(spike_counts) - window, step):
                    t, p = stats.ttest_1samp(spike_counts[i:i+window], 0)
                    active_rates.append(np.mean(spike_counts[i:i+window]))
                    active_bins.append(i*t_bin)
                    active.append(p < 0.2)
                    pvals.append(p)
                # plot the active periods
                active = np.array(active, dtype=int)
                active_rates = np.log10(np.array(active_rates))
                active_mean = np.mean(active_rates[active == 1])
                active_std = np.std(active_rates[active == 1])
                # reject active bins that are less than 1 std 
                active = active * (active_rates > active_mean - 2.5*active_std)

                # calculate contiguous active periods
                if np.all(active == 1):
                    active_periods = [(active_bins[0], active_bins[len(active)])]
                else:
                    active_periods = []
                    start = 0
                    for i in range(1, len(active)):
                        if active[i] == 1 and active[i-1] == 0:
                            start = i
                        if active[i] == 0 and active[i-1] == 1:
                            active_periods.append((active_bins[start], active_bins[i+1]))
                    if active[-1] == 1:
                        active_periods.append((active_bins[start], active_bins[-1]))
                if len(active_periods) == 0:
                    print("Unit %d has no active periods"%unit.kilosort_id)
                    continue
                # go through the active periods, and if the inactive period between two active periods is less than 5minutes, then merge them
                agglomerated_active_periods = [active_periods[0]]
                for i in range(1, len(active_periods)):
                    # check if the inactive period is less than 3 minutes
                    if active_periods[i][0] - agglomerated_active_periods[-1][1] < 180:
                        agglomerated_active_periods[-1] = (agglomerated_active_periods[-1][0], active_periods[i][1])
                    else:
                        agglomerated_active_periods.append(active_periods[i])
                # now remove any active periods that are less than 5 minutes
                agglomerated_active_periods = [x for x in agglomerated_active_periods if x[1] - x[0] > 300]
                for start, stop in agglomerated_active_periods:
                    unit_intervals.add_interval(
                        start_time=float(start),
                        stop_time=float(stop),
                        unit_id=uid
                    )
            # add the table to the nwbfile
            nwb.add_time_intervals(unit_intervals)
            nwb_io.write(nwb)
