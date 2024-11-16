from datetime import datetime, timedelta
from uuid import uuid4
import os
import sys
import json
import glob

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
from pynwb.base import TimeSeries
from pynwb.epoch import TimeIntervals
from pynwb.behavior import BehavioralTimeSeries
from pynwb.ecephys import LFP, ElectricalSeries
import kilosort

#from neo.rawio.intanrawio import read_rhd_signal_groups
from ndx_events import TtlsTable, TtlTypesTable, EventsTable, EventTypesTable, Task

from ..preprocess.sound.playback_utils import convert_raw_digital_to_metadata, convert_pyoperant_operant_csvs, _get_stim_dirs, convert_pyoperant_playback_csvs

from .pipeline_utils import (
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
    GenerateSingleSiteAudioFile
)

def get_rhd_start_from_fn(rhd_file):
    # start is from the filename
    date_string = rhd_file.split('__')[-1].split('.rhd')[0]
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

class CreateSiteNWB(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_config': SiteConfig(site_dir=self.site_dir),
            'project_config': SiteProjectConfig(site_dir=self.site_dir)
        }

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, '%s.nwb'%os.path.basename(self.site_dir)))
    
    def run(self):
        site_config = self.input()['site_config'].data()
        project_config = self.input()['project_config'].data()
        
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
        probes = dict({
            "P1": 'cambridge_neurotech_assy_236_P1.prb', # 4 shank
            "H10": 'cambridge_H10.prb', # 2 shank
            "M16": 'microprobes_16ch.prb' # microprobes with 16 wires
        })
        probe_name = probes[project_config["EPHYS"].get('PROBETYPE','H10')]
        prb = kilosort.io.load_probe(os.path.join(kilosort.PROBE_DIR,probe_name))

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
        
class AddDigitalEventsNWB(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_config': SiteConfig(site_dir=self.site_dir),
            'project_config': SiteProjectConfig(site_dir=self.site_dir),
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'site_digital_memmap': GenerateSingleSiteDigitalMemmap(site_dir=self.site_dir)
        }

    def output(self):
        return NWBAcquisitionTarget(self.input()['site_nwb'].path, 'audcode_events', 'ttl_events')
    
    def run(self):
        site_config = self.input()['site_config'].data()
        project_config = self.input()['project_config'].data()
        file_list = site_config["filelist"]

        # get the name of the digital channels
        if 'digital_channel_names' in site_config:
            digital_channel_names = site_config['digital_channel_names']
        else:
            digital_channel_names = ['AUDCODE', 'TTL', 'BUTTON', 'FEEDER', 'SEED', 'HOUSE']

        # make the ttl types table
        ttl_types_table = EventTypesTable(name="event_types", description="Metadata about TTL types")
        ttl_types_table.add_row(
            event_name="AUDCODE",
            event_type_description="Auditory stimulus code"
        )
        ttl_types_table.add_row(
            event_name="TTL",
            event_type_description="Continuous TTL signal from behavioral arduino"
        )
        ttl_types_table.add_row(
            event_name="BUTTON",
            event_type_description="Peckport beam break"
        )
        ttl_types_table.add_row(
            event_name="FEEDER",
            event_type_description="Feeder activation motor activation / LED on signal"
        )
        ttl_types_table.add_row(
            event_name="SEED",
            event_type_description="Seed dispenser beam break"
        )
        ttl_types_table.add_row(
            event_name="HOUSE",
            event_type_description="House light on signal"
        )
        # now make the ttls table
        ttls_table = EventsTable(name='ttl_events',description="Metadata about TTL events", target_tables={"event_type": ttl_types_table})
        
        audcode_table = EventsTable(name="audcode_events", description="Precise time and metadata about stimuli played", target_tables={"event_type": ttl_types_table})
        audcode_table.add_column(name="code", description='stimuli code',data=np.array([],dtype=str))
        audcode_table.add_column(name="object", description='object name', data=np.array([],dtype=str))
        audcode_table.add_column(name="action", description='action name', data=np.array([],dtype=str))
        audcode_table.add_column(name="repetition", description='repetition number', data=np.array([],dtype=int))
        audcode_table.add_column(name="trial", description='trial number',  data=np.array([],dtype=int))
        audcode_table.add_column(name="md5", description='md5 hash of the stimulus file', data=np.array([],dtype=str))
        
        # now load the digital memmap and fill in the ttl table and events table
        digital_memmap_info = json.load(open(self.input()['site_digital_memmap']['info'].path, 'r'))
        fs = digital_memmap_info['sampling_rate']

        if digital_memmap_info['num_channels'] == 1:
            # if digital memmap is only one channel then we need to deinterleave them, if not then it is done for us
            digital_memmap = np.memmap(self.input()['site_digital_memmap']['bin'].path, dtype='uint16', mode='r')
            num_chans = int(np.log2(np.max(digital_memmap)))

            dig_trace = np.memmap(self.input()["dig_memmap"].path, dtype='uint16', mode='r')
            #dig_trace = multirecording.get_traces().flatten()
            # NB: np.unpackbits does this for 8 bit unsigned ints. This works for us as long as we are not using more than 8 channels
            for i in range(num_chans):
                dig_data = np.not_equal(np.bitwise_and(dig_trace, (1<<i)),0).astype(np.int8)
                if i == 0:
                    # 0 is auditory codes usually
                    stimulus_codes = convert_raw_digital_to_metadata(dig_data,fs)
                    for stim in stimulus_codes:
                        t, metadata = stim
                        audcode_table.add_row(
                            timestamp=t/fs,
                            event_type=0,
                            code=metadata['code'],
                            object=metadata['object'],
                            action=metadata['action'],
                            repetition=metadata['repetition'],
                            trial=metadata['trial'],
                            md5=metadata['md5']
                        )
                else:
                    # Otherwise add it to our ttls table
                    # if data is mostly ones then flip it
                    if dig_data.mean() > .5:
                        dig_data = ~dig_data
                    # now find the rising and falling edges
                    rising = np.where(np.diff(dig_data) == 1)[0]
                    falling = np.where(np.diff(dig_data) == -1)[0]
                    if len(rising) != len(falling) and len(rising) > 0:
                        # it has to be off by one. we can figure out how to fix it with a few tests
                        # if the first rising is after the first falling then we need to remove the first falling
                        if len(falling) == 0:
                            assert(len(rising) == 1)
                            falling = [len(dig_data)]
                            break
                        if rising[0] > falling[0]:
                            falling = falling[1:]
                            assert(len(rising) == len(falling))
                        # if the last falling is before the last rising then we need to add a falling at the end that is the total duration
                        if rising[-1] < falling[-1]:
                            falling = np.append(falling, len(dig_data))
                            assert(len(rising) == len(falling))              
                    for r, f in zip(rising, falling):
                        ttls_table.add_row(
                            timestamp=r/fs,
                            event_type=i,
                            duration=(f-r)/fs
                        )
        else:
            num_samples = digital_memmap_info['num_samples']
            num_chans = digital_memmap_info['num_channels']
            # TODO maybe get dtype too
            digital_memmap = np.memmap(self.input()['site_digital_memmap']['bin'].path, dtype='uint16', mode='r', shape=(num_samples, num_chans))
            # now we can just iterate over the channels
            for i, chan in enumerate(digital_channel_names):
                if i == 0: # audcode chan
                    stimulus_codes = convert_raw_digital_to_metadata(digital_memmap[:,i].astype(np.int8),fs)
                    for stim in stimulus_codes:
                        t, metadata = stim
                        audcode_table.add_row(
                            timestamp=t/fs,
                            event_type=0,
                            code=metadata['code'],
                            object=metadata['object'],
                            action=metadata['action'],
                            repetition=metadata['repetition'],
                            trial=metadata['trial'],
                            md5=metadata['md5']
                        )
                else:
                    # Otherwise add it to our ttls table
                    # if data is mostly ones then flip it
                    dig_data = digital_memmap[:,i].astype(np.int8)
                    if digital_memmap[:,i].mean() > .5:
                        dig_data = ~dig_data
                    # now find the rising and falling edges
                    rising = np.where(np.diff(dig_data) == 1)[0]
                    falling = np.where(np.diff(dig_data) == -1)[0]
                    if len(rising) != len(falling) and len(rising) > 0:
                        # it has to be off by one. we can figure out how to fix it with a few tests
                        # if the first rising is after the first falling then we need to remove the first falling
                        if len(falling) == 0:
                            assert(len(rising) == 1)
                            falling = [len(dig_data)]
                        else:
                            if rising[0] > falling[0]:
                                falling = falling[1:]
                                assert(len(rising) == len(falling))
                            # if the last falling is before the last rising then we need to add a falling at the end that is the total duration
                            if rising[-1] < falling[-1]:
                                falling = np.append(falling, len(dig_data))
                                assert(len(rising) == len(falling))              
                    for r, f in zip(rising, falling):
                        ttls_table.add_row(
                            timestamp=r/fs,
                            event_type=i-1,
                            duration=(f-r)/fs
                        )
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            # add the tables to the nwbfile
            task = Task()
            task.event_types = ttl_types_table
            nwbfile.add_lab_meta_data(task)
            nwbfile.add_acquisition(ttls_table)
            nwbfile.add_acquisition(audcode_table)
            nwb_io.write(nwbfile)

class AddLFPNWB(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
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

class AddMotionNWB(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
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

class AddAudioNWB(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
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

#TODO move this to a utils file
def detect_ripples_on_channel(ch, lfp_data, fs, ripple_threshold):
    print("Detecting ripples on channel %d"%ch)
    # hamming filter from 100 to 200 hz\\\n",
    taps = signal.firwin(100, [100/(fs/2), 200/(fs/2)],pass_zero=False)
    conv_result = signal.convolve(lfp_data, taps, mode='same',method='fft')
    # do hilber transform on result
    hilb = signal.hilbert(conv_result)
    filtered = ndimage.gaussian_filter1d(np.abs(hilb), 4)
    zscored = zscore(filtered)

    # now we need to find the peaks
    above_thresh = np.where(zscored > ripple_threshold)[0]
    # how many continuous events are there
    diff = np.diff(above_thresh, axis=0)
    events_boundaries = np.concatenate([[0],np.where(diff > 1)[0]+1, [above_thresh.shape[0]]])
    event_durations = np.diff(events_boundaries)
    events = events_boundaries[:-1]

    # take only ones that last for more than 10 samples above threshold
    events = events[event_durations > 10]
    event_durations = event_durations[event_durations > 10]

    # now expand each event to the previous 0 crossing and the next 0 crossing
    # TODO we can increase 0 to .5 or 1 if these events are too big
    zero_crossings = np.where(np.diff(np.sign(zscored)))[0]

    # now we find the index of the zero crossing that is closest to the start and end of the event
    event_starts = above_thresh[events]
    event_ends = above_thresh[events] + event_durations
    zc_start_inds = np.searchsorted(zero_crossings, event_starts) - 1
    event_starts[zc_start_inds >= 0] = zero_crossings[zc_start_inds[zc_start_inds >= 0]]
    zc_end_inds = np.searchsorted(zero_crossings, event_ends)
    event_ends[zc_end_inds < zero_crossings.shape[0]] = zero_crossings[zc_end_inds[zc_end_inds < zero_crossings.shape[0]]]

    events = event_starts
    event_durations = event_ends - event_starts

    # remove duplicates
    events, inds = np.unique(events, return_index=True)
    event_durations = event_durations[inds]
    event_output = []
    for event, duration in zip(events, event_durations):
        event_output.append(dict({
            'event': event,
            'duration': duration,
            'zscored': zscored[event:event+duration]
        }))
    print("Found %d events on ch %d"%(len(event_output), ch))
    return event_output


class RemoveProcessingModule(luigi.Task):
    site_dir = SiteParameter()
    module_name = luigi.Parameter()
    def requires(self):
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

class RemoveRippleModule(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
        }
    def output(self):
        return luigi.LocalTarget("dummy")
    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            # remove the processing module
            nwbfile.processing.pop('ripples')
            # reset parent

            print("Writing to file")
            # export to new file
            export_fn = self.input()['site_nwb'].path.replace('.nwb','_temp.nwb')
            with NWBHDF5IO(export_fn, 'w') as export_io:
                export_io.export(src_io=nwb_io, nwbfile=nwbfile)
                export_io.close()
            nwb_io.close()
        os.rename(export_fn, self.input()['site_nwb'].path)

class AddRippleEventsNWB(luigi.Task):
    site_dir = SiteParameter()
    ripple_threshold = luigi.FloatParameter(default=2.5)
    overwrite = luigi.BoolParameter(default=False)
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'lfp': AddLFPNWB(site_dir=self.site_dir),
            'motion': AddMotionNWB(site_dir=self.site_dir),
            'site_config': SiteConfig(site_dir=self.site_dir)
        }

    def output(self):
        return NWBProcessingTarget(self.input()['site_nwb'].path, 'ripples', 'putative_ripples')
    
    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwb = nwb_io.read()
            
            site_config = self.input()['site_config'].data()
            # lets take our lfp and filter it
            lfp = nwb.acquisition['LFP']
            fs = lfp.rate
            motion = nwb.processing['behavior']['Motion Time Series'].time_series['motion']

            if 'bad_channels' not in site_config:
                # first detect bad channels
                # take a significantly sized chunk of the data, maybe 10 seconds?
                chunk_size = int(10*fs)
                chunk_start = np.random.randint(0,lfp.data.shape[0]-chunk_size)
                random_chunk = lfp.data[chunk_start:chunk_start+chunk_size,:]
                # get the average of all the channels
                channel_means = np.mean(random_chunk, axis=1)
                # get the correlation of each channel with the average
                corrs = np.zeros(lfp.data.shape[1])
                for i in range(lfp.data.shape[1]):
                    corrs[i] = np.corrcoef(channel_means, random_chunk[:,i])[0,1]        
                zscored_corrs = zscore(corrs)
                bad_channels = np.where(zscored_corrs < -2)[0]
            else:
                # bad channels should be already removed from the lfp
                bad_channels = []

            # # now add the detected ripples to the nwb file
            # putative_ripples = TimeIntervals(
            #     name="putative_ripples",
            #     description="Putative ripple events detected from LFP data"
            # )
            # putative_ripples.add_column(name="duration", description="Duration of the ripple event")
            # putative_ripples.add_column(name="peak_time", description="Time of the peak of the ripple event")
            # putative_ripples.add_column(name="peak_value", description="Value of the peak of the ripple event")
            # putative_ripples.add_column(name="channel", description="Channel of the ripple event")
            # putative_ripples.add_column(name="AOC", description="Area over the curve of the ripple event")
            # putative_ripples.add_column(name="motion_peak", description="Peak of the motion energy during the ripple event")
            # putative_ripples.add_column(name="motion_peak_time", description="Area over the curve of the motion energy during the ripple event")
            # putative_ripples.add_column(name="motion_avg", description="Average motion energy during the ripple event")
            

            # lets do this on each channel individually TODO could do this for all at the same time. but gets annoying with bad channels
            
            # in parallel detect ripples on each channel
            results = []
            with ProcessPoolExecutor() as executor:
                future_results = []
                for ch in range(lfp.data.shape[1]):
                    future_results.append(executor.submit(detect_ripples_on_channel,ch, lfp.data[:,ch], lfp.rate, self.ripple_threshold))
                results = [res.result() for res in future_results]

            # now add the results to the putative ripples
            putative_ripples_list = []
            for ch, result in enumerate(results):
                for e in result:
                    conv_factor = motion.rate / fs
                    motion_ts = motion.data[int(e['event'] * conv_factor):int((e['event']+e['duration'])*conv_factor)]
                    putative_ripples_list.append({
                        "start_time":e['event']/fs,
                        "stop_time":(e['event']+e['duration'])/fs,
                        "duration":e['duration']/fs,
                        "peak_time":e['event']/fs + np.argmax(e['zscored'])/fs,
                        "peak_value":np.max(e['zscored']),
                        "channel":ch,
                        "AOC":np.sum(e['zscored']),
                        "motion_peak":np.max(motion_ts),
                        "motion_peak_time":np.argmax(motion_ts)/motion.rate,
                        "motion_avg":np.mean(motion_ts),
                        # "timeseries":[lfp, motion]
                    }
                    )
            putative_ripples_df = pd.DataFrame(putative_ripples_list)
            putative_ripples = TimeIntervals.from_dataframe(putative_ripples_df, name="putative_ripples")
            if 'ripples' not in nwb.processing:
                nwb.create_processing_module(
                    name="ripples", description="Processed ripple events"
                )
            nwb.processing['ripples'].add(putative_ripples)
            nwb_io.write(nwb)

class AgglomerateRippleEventsNWB(luigi.Task):
    site_dir = SiteParameter()
    ripple_threshold = luigi.FloatParameter(default=2.5)
    force = luigi.BoolParameter(default=False)
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'putative_ripples': AddRippleEventsNWB(site_dir=self.site_dir)
        }

    def output(self):
        return NWBProcessingTarget(self.input()['site_nwb'].path,'ripples', 'ripple_events')
    
    def run(self):
        if self.force and self.output().exists():
            self.output().delete()
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwb = nwb_io.read()
            
            # get operant and playback events
            operant_events = nwb.intervals['operant_trials'].to_dataframe()
            playback_events = nwb.intervals['playback_trials'].to_dataframe()
            # get the putative ripples
            rip_df = nwb.processing['ripples']['putative_ripples'].to_dataframe()
            #nwb_io.close()

            # lets mark all ripples that occur during playback or operant events
            # NB: This marks ripples that bridge between trials as in trial
            all_trials = np.concatenate([operant_events[['start_time', 'stop_time']].values, playback_events[['start_time', 'stop_time']].values])
            def check_if_ripple_in_trial(st, end, all_trials):
                return any([any([(st > trial[0]) & (st < trial[1]) for trial in all_trials]) | any([(end > trial[0]) & (end < trial[1]) for trial in all_trials])])
            rip_df['in_trial'] = rip_df.apply(lambda x: check_if_ripple_in_trial(x['start_time'], x['stop_time'], all_trials), axis=1)

            # agglomerate ripples into events
            # now to group ripples into events, we can filter out ripples with motion peaks greater than 1.5 stdev
            rip_df['event'] = -1
            event = 0
            # lets sort by peak time
            ripples_to_check = rip_df[rip_df['motion_peak'] < 10]
            ripples_to_check = ripples_to_check.sort_values('peak_time')

            for ix, ripple in ripples_to_check.iterrows():
                if rip_df.loc[ix, 'event'] > -1:
                    continue
                #rip_df.loc[ix, 'event'] = event
                # lets take all ripples that have a peak time within this ripple
                rip_start = ripple['start_time']
                rip_stop = ripple['stop_time']
                n_rips = 1
                while True: # keep going until we have all ripples
                    lhs = np.searchsorted(ripples_to_check['peak_time'].values, rip_start)
                    rhs = np.searchsorted(ripples_to_check['peak_time'].values, rip_stop, side='right')
                    ripples = ripples_to_check.iloc[lhs:rhs]
                    if len(ripples) == n_rips:
                        break
                    n_rips = len(ripples)
                    rip_start = ripples['start_time'].min()
                    rip_stop = ripples['stop_time'].max()
                # assign all ripples from any event present above -1 to the new event
                # get all rip_df events that have events that are shared with ripples
                rip_df.loc[ripples.index, 'event'] = event
                event += 1
            
            # now add the ripple events to a new table
            ripple_events_list = []
            for event, event_df in rip_df.groupby('event'):
                if event == -1:
                    continue
                ripple_events_list.append({
                    'start_time':event_df.start_time.min(),
                    'stop_time':event_df.stop_time.max(),
                    'duration':event_df.stop_time.max() - event_df.start_time.min(),
                    'n_ripples':event_df.shape[0],
                    'in_trial':event_df.in_trial.max(),
                    'AOC':event_df.AOC.max(),
                    'peak_value':event_df.peak_value.max(),
                    'peak_channel':event_df.channel[event_df.peak_value.idxmax()],
                    'peak_time': event_df.peak_time[event_df.peak_value.idxmax()],
                    'event':event#,
                })
            ripple_events = TimeIntervals.from_dataframe(pd.DataFrame(ripple_events_list), 'ripple_events')

            # add the tables to the nwbfile
            nwb.processing['ripples'].add(ripple_events)

            # # add the event column to the putative ripples
            if 'event' not in nwb.processing['ripples']['putative_ripples']:
                nwb.processing['ripples']['putative_ripples'].add_column(name="event", description="Event number of the ripple in the ripple_events table", data = rip_df.event.values)
            else:
                nwb.processing['ripples']['putative_ripples']['event'].data[:] = rip_df.event.values
            
            nwb_io.write(nwb)
            nwb_io.close()






class ResetUnits(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
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


class AddKilosortUnits(luigi.Task):
    site_dir = SiteParameter()
    kilosort_dir = luigi.Parameter(default='ks4_final')
    overwrite = luigi.BoolParameter(default=False)
    clear_old_units = luigi.BoolParameter(default=False)
    def requires(self):
        if self.clear_old_units:
            return {
                'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
                'reset': ResetUnits(site_dir=self.site_dir),
            }
        else:
            return {
                'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            }
    def output(self):
        # Force this to always run if requested
        return luigi.LocalTarget("dummy") 

    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()

            if nwbfile.units is None:
                # assert there are no units
                nwbfile.add_unit_column(name='kilosort_dir', description='The directory where the kilosort output is stored')
                nwbfile.add_unit_column(name='kilosort_id', description='The id of the unit in the kilosort output')
                nwbfile.add_unit_column(name='group', description='The Kilosort Group (good, mua, noise)')
                nwbfile.add_unit_column(name='contampct', description='The Contamination Percentage from Kilosort')

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
                
                for i, cluster in ks_cluster_info.iterrows():
                    if cluster['group'] != 'noise':
                        if (cluster['cluster_id'], ks_dir_name) not in zip(nwbfile.units['kilosort_id'], nwbfile.units['kilosort_dir']):
                            print("Adding unit %d from %s"%(cluster['cluster_id'], ks_dir_name))
                            nwbfile.add_unit(
                                spike_times=ks_spike_times[ks_spike_cluster == cluster['cluster_id']],
                                electrodes=[cluster.ch], # for now lets just put the main electrode
                                kilosort_id=cluster['cluster_id'],
                                group=cluster['group'],
                                contampct=cluster['ContamPct'],
                                kilosort_dir=ks_dir_name
                            )
                        else:
                            print("Unit %d from %s already exists"%(cluster['cluster_id'], ks_dir_name))
                            if self.overwrite:
                                print("Overwriting")
                                ix = (nwbfile.units['kilosort_id'] == cluster['cluster_id']) & (nwbfile.units['kilosort_dir'] == ks_dir_name)
                                nwbfile.units['spike_times'][ix] = ks_spike_times[ks_spike_cluster == cluster['cluster_id']]
                                nwbfile.units['contampct'][ix] = cluster['ContamPct']
                                nwbfile.units['group'][ix] = cluster['group']
            nwb_io.write(nwbfile)

class AddUnitValidIntervals(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
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
                active = active * (active_rates > active_mean - active_std)

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
                    return []
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
class AddPyoperantTrials(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_config': SiteConfig(site_dir=self.site_dir),
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'project_config': SiteProjectConfig(site_dir=self.site_dir),
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
                site_config = self.input()['site_config'].data()
                project_config = self.input()['project_config'].data()
                stim_root_dir = project_config['LOCAL_RAW_FILES']['stimuli_drive']
                pyoperant_csvs = [os.path.join(project_config['LOCAL_RAW_FILES']['pyoperant_drive'],f) for f in site_config['filelist'] if f.endswith('trialdata.csv')]
                pyoperant_experiments = [PyoperantExperiment(x) for x in pyoperant_csvs]

                

                # now we need to match the trial blocks with the operant experiments
                for exp in pyoperant_experiments:
                    if exp.ntrials > 0:
                        # add the top level interval
                        pyoperant_experiments_intervals.add_interval(
                            start_time=(exp.experiment_start_time - session_start).total_seconds(),
                            stop_time=(exp.experiment_end_time - session_start).total_seconds(),
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
                            stim_path = trial.stimulus_name
                            stim_name = trial.stimulus_name.split('\\')[-1]
                            

                            # add the stim to the nwb file if it is not present already
                            if stim_name not in nwbfile.stimulus:
                                bird_name = stim_name.split('_')[3]
                                rendition_id = stim_name.strip('.wav').split('_')[-1]
                                stim_files = glob.glob(os.path.join(stim_root_dir,'**',stim_name),recursive=True)
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
                                end = (start + exp.trial_duration(ix).total_seconds())
                                op_trial_intervals.add_interval(
                                    start_time=start,
                                    stop_time=end,
                                    trial_num=ix,
                                    experiment_uid=exp.uid,
                                    stimuli_path=trial.stimulus_name,
                                    stimuli_name=trial.stimulus_name.split('\\')[-1],
                                    response=trial.response,
                                    correct=trial.correct,
                                    timeseries=all_time_series
                                )
                            else:
                                end = (start + (nwbfile.stimulus[stim_name].data.shape[0] / nwbfile.stimulus[stim_name].rate))
                                pb_trial_intervals.add_interval(
                                    start_time=start,
                                    stop_time=end,
                                    trial_num=ix,
                                    experiment_uid=exp.uid,
                                    stimuli_path=trial.stimulus_name,
                                    stimuli_name=trial.stimulus_name.split('\\')[-1],
                                    timeseries=all_time_series
                                )
                
            # now save the trials to the nwb file
            nwbfile.add_time_intervals(pyoperant_experiments_intervals)
            nwbfile.add_time_intervals(op_trial_intervals)
            nwbfile.add_time_intervals(pb_trial_intervals)
            nwb_io.write(nwbfile)

class AddStimuliNWB(luigi.Task):
    def requires(self):
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

class GenerateAcquisitionNWB(luigi.WrapperTask):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'digital': AddDigitalEventsNWB(site_dir=self.site_dir),
            'lfp': AddLFPNWB(site_dir=self.site_dir),
            'motion': AddMotionNWB(site_dir=self.site_dir),
            'audio': AddAudioNWB(site_dir=self.site_dir),
            'ripples': AgglomerateRippleEventsNWB(site_dir=self.site_dir),
            'operant': AddPyoperantTrials(site_dir=self.site_dir)
        }
    