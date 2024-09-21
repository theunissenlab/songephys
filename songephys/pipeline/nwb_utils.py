from datetime import datetime, timedelta
from uuid import uuid4
import os
import sys
import json

import pandas as pd
import luigi
import numpy as np
import tqdm
from dateutil.tz import tzlocal
from scipy.stats import zscore
import scipy.signal as signal
import scipy.ndimage as ndimage

from pynwb import NWBHDF5IO, NWBFile
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
            return timedelta(seconds=2)

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
        ml = site_config.get('ml_mm',.3) # mm, same for all probes TODO: Make this part of site_config optionally
        depth = site_config['depth_mm']
        electrode_group = nwbfile.create_electrode_group(
                name=probe_name,
                description="%s probe in %s"%(probe_name,"NCM"),#TODO site_config['brain_area']),
                device=device,
                location="NCM", # TODO put this in site_config
            )

        # Add electrodes to the electrode group
        # for this experiment prb file, x is A-P, y is D-V
        # in the nwb file: +x is posterior +y is inferior +z is right
        for (cid, ap, dv) in zip(prb['chanMap'], prb['xc'], prb['yc']):
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
        audcode_table.add_column(name="code", description='stimuli code')
        audcode_table.add_column(name="object", description='object name')
        audcode_table.add_column(name="action", description='action name')
        audcode_table.add_column(name="repetition", description='repetition number')
        audcode_table.add_column(name="trial", description='trial number')
        audcode_table.add_column(name="md5", description='md5 hash of the stimulus file')
        
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

class AddRippleEventsNWB(luigi.Task):
    site_dir = SiteParameter()
    ripple_threshold = luigi.FloatParameter(default=2.5)
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'lfp': AddLFPNWB(site_dir=self.site_dir),
            'motion': AddMotionNWB(site_dir=self.site_dir)
        }

    def output(self):
        return NWBProcessingTarget(self.input()['site_nwb'].path, 'ripples', 'putative_ripples')
    
    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwb = nwb_io.read()

            # lets take our lfp and filter it
            lfp = nwb.acquisition['LFP']
            fs = lfp.rate
            motion = nwb.processing['behavior']['Motion Time Series'].time_series['motion']

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

            # now add the detected ripples to the nwb file
            putative_ripples = TimeIntervals(
                name="putative_ripples",
                description="Putative ripple events detected from LFP data"
            )
            putative_ripples.add_column(name="duration", description="Duration of the ripple event")
            putative_ripples.add_column(name="peak_time", description="Time of the peak of the ripple event")
            putative_ripples.add_column(name="peak_value", description="Value of the peak of the ripple event")
            putative_ripples.add_column(name="channel", description="Channel of the ripple event")
            putative_ripples.add_column(name="AOC", description="Area over the curve of the ripple event")
            putative_ripples.add_column(name="motion_peak", description="Peak of the motion energy during the ripple event")
            putative_ripples.add_column(name="motion_peak_time", description="Area over the curve of the motion energy during the ripple event")
            putative_ripples.add_column(name="motion_avg", description="Average motion energy during the ripple event")
            

            # lets do this on each channel individually TODO could do this for all at the same time. but gets annoying with bad channels
            for ch in range(lfp.data.shape[1]):
                if ch in bad_channels:
                    print("Skipping BAD channel %d"%ch)
                    continue
                print("Detecting ripples on channel %d"%ch)
                lfp_data = lfp.data[:,ch]
                # hamming filter from 100 to 200 hz\\\n",
                taps = signal.firwin(100, [100/(fs/2), 200/(fs/2)],pass_zero=False)
                conv_result = signal.convolve(lfp_data, taps, mode='same',method='fft')
                # do hilber transform on result
                hilb = signal.hilbert(conv_result)
                filtered = ndimage.gaussian_filter1d(np.abs(hilb), 4)
                zscored = zscore(filtered)

                # now we need to find the peaks
                above_thresh = np.where(zscored > self.ripple_threshold)[0]
                # how many continuous events are there
                diff = np.diff(above_thresh, axis=0)
                events_boundaries = np.concatenate([[0],np.where(diff > 1)[0]+1, [above_thresh.shape[0]]])
                event_durations = np.diff(events_boundaries)
                events = events_boundaries[:-1]

                # take only ones that last for more than 10 ms above 3 std
                events = events[event_durations > 30]
                event_durations = event_durations[event_durations > 20]

                # now expand each event to the previous 0 crossing and the next 0 crossing
                # TODO we can increase 0 to .5 or 1 if these events are too big
                zero_crossings = np.where(np.diff(np.sign(zscored)))[0]
                for i in range(events.shape[0]):
                    start = above_thresh[events[i]]
                    end = above_thresh[events[i]] + event_durations[i]
                    start = zero_crossings[zero_crossings <= start][-1]
                    end = zero_crossings[zero_crossings >= end][0]
                    events[i] = start
                    event_durations[i] = end-start

                # remove duplicates
                events, inds = np.unique(events, return_index=True)
                event_durations = event_durations[inds]

                # now we need to calculate the area over the curve for each event
                for i, e in enumerate(events):
                    conv_factor = motion.rate / fs
                    motion_ts = motion.data[int(e * conv_factor):int((e+event_durations[i])*conv_factor)]
                    putative_ripples.add_interval(
                                start_time=e/fs,
                                stop_time=(e+event_durations[i])/fs,
                                duration=event_durations[i]/fs,
                                peak_time=e/fs + np.argmax(zscored[e:e+event_durations[i]])/fs,
                                peak_value=np.max(zscored[e:e+event_durations[i]]),
                                channel=ch,
                                AOC=np.sum(zscored[e:e+event_durations[i]]),
                                motion_peak=np.max(motion_ts),
                                motion_peak_time=np.argmax(motion_ts)/motion.rate,
                                motion_avg=np.mean(motion_ts),
                                timeseries=[lfp, motion]
                            )
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
        return NWBProcessingTarget(self.input()['site_nwb'].path, 'ripples', 'ripple_events')
    
    def run(self):
        if self.force and self.output().exists():
            self.output().delete()
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            # get operant and playback events
            operant_events = nwbfile.intervals['operant_trials'].to_dataframe()
            playback_events = nwbfile.intervals['playback_trials'].to_dataframe()
            # get the putative ripples
            rip_df = nwbfile.processing['ripples']['putative_ripples'].to_dataframe()
            # lets mark all ripples that occur during playback or operant events
            all_trials = np.concatenate([operant_events[['start_time', 'stop_time']].values, playback_events[['start_time', 'stop_time']].values])
            def check_if_ripple_in_trial(st, end, all_trials):
                return any([any([(st > trial[0]) & (st < trial[1]) for trial in all_trials]) | any([(end > trial[0]) & (end < trial[1]) for trial in all_trials])])
            rip_df['in_trial'] = rip_df.apply(lambda x: check_if_ripple_in_trial(x['start_time'], x['stop_time'], all_trials), axis=1)

            # agglomerate ripples into events
            # now to group ripples into events, we can filter out ripples with motion peaks greater than 1.5 stdev
            rip_df['event'] = -1
            for ix, ripple in rip_df[rip_df['motion_peak'] < 10].iterrows():
                event = rip_df.event.max() + 1
                rip_df.loc[ix, 'event'] = event
                ## get all ripples that start before this one ends and end after this one starts
                #ripples = rip_df[(rip_df.start_time < ripple.stop_time) & (rip_df.stop_time > ripple.start_time)]
                # more conservative: Get all ripples where the peak is within this ripple
                ripples = rip_df[(rip_df.peak_time > ripple.start_time) & (rip_df.peak_time < ripple.stop_time)]
                # assert that this ripple is in ripples
                assert ix in ripples.index
                # if all events are the same, continue
                if ripples.event.nunique() == 1:
                    # A single ripple event
                    continue
                event = min(ripples[ripples.event>-1].event.min(), event)
                # assign all ripples from any event present above -1 to the new event
                # get all rip_df events that have events that are shared with ripples
                shared_events = rip_df[rip_df.event.isin(ripples[ripples.event>-1].event.unique())]
                # assign all of these the same event number
                rip_df.loc[ripples.index, 'event'] = event
                rip_df.loc[shared_events.index, 'event'] = event
            
            # now make a ripple_events table for the agglomerated events
            ripple_events = TimeIntervals(
                name="ripple_events",
                description="Agglomerated ripple events"
            )
            ripple_events.add_column(name="duration", description="Duration of the ripple event")
            ripple_events.add_column(name="n_ripples", description="Number of ripples in the event")
            ripple_events.add_column(name="in_trial", description="Whether the event occurs during a trial")
            ripple_events.add_column(name="channels", description="Channels that the ripples occur on")
            ripple_events.add_column(name="AOC", description="Area over the curve of the ripple event")
            ripple_events.add_column(name="peak_value", description="Value of the peak of the ripple event")
            ripple_events.add_column(name='peak_time', description='Time of the peak of the ripple event')
            ripple_events.add_column(name="event", description="Event number")
            for event, event_df in rip_df.groupby('event'):
                if event == -1:
                    continue
                ripple_events.add_interval(
                    start_time=event_df.start_time.min(),
                    stop_time=event_df.stop_time.max(),
                    duration=event_df.stop_time.max() - event_df.start_time.min(),
                    n_ripples=event_df.shape[0],
                    in_trial=event_df.in_trial.max(),
                    channels=event_df.channel.unique(),
                    AOC=event_df.AOC.max(),
                    peak_value=event_df.peak_value.max(),
                    peak_time=event_df.peak_time.mean(),#TODO could do this too event_df.peak_time[event_df.peak_value.idxmax()],
                    event=event
                )
            nwbfile.processing['ripples'].add(ripple_events)

            # add the event column to the putative ripples
            if 'event' not in nwbfile.processing['ripples']['putative_ripples']:
                nwbfile.processing['ripples']['putative_ripples'].add_column(name="event", description="Event number of the ripple in the ripple_events table", data = rip_df.event.values)
            else:
                nwbfile.processing['ripples']['putative_ripples']['event'] = rip_df.event.values
            
            nwb_io.write(nwbfile)








class AddKilosortUnits(luigi.Task):
    site_dir = SiteParameter()
    kilosort_dir = luigi.Parameter(default='ks4_final')
    overwrite = luigi.BoolParameter(default=False)
    def requires(self):
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
            # load samplerate from params.py
            sys.path.append(ks_output_dir)
            from params import sample_rate
            ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))/sample_rate
            ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
            ks_cluster_info = pd.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')
            
            for i, cluster in ks_cluster_info.iterrows():
                if cluster['group'] != 'noise':
                    if (cluster['cluster_id'], ks_output_dir) not in zip(nwbfile.units['kilosort_id'], nwbfile.units['kilosort_dir']):
                        print("Adding unit %d from %s"%(cluster['cluster_id'], ks_output_dir))
                        nwbfile.add_unit(
                            spike_times=ks_spike_times[ks_spike_cluster == cluster['cluster_id']],
                            electrodes=[cluster.ch], # for now lets just put the main electrode
                            kilosort_id=cluster['cluster_id'],
                            group=cluster['group'],
                            contampct=cluster['ContamPct'],
                            kilosort_dir=ks_output_dir
                        )
                    else:
                        print("Unit %d from %s already exists"%(cluster['cluster_id'], ks_output_dir))
                        if self.overwrite:
                            print("Overwriting")
                            ix = (nwbfile.units['kilosort_id'] == cluster['cluster_id']) & (nwbfile.units['kilosort_dir'] == ks_output_dir)
                            nwbfile.units['spike_times'][ix] = ks_spike_times[ks_spike_cluster == cluster['cluster_id']]
                            nwbfile.units['contampct'][ix] = cluster['ContamPct']
                            nwbfile.units['group'][ix] = cluster['group']
            nwb_io.write(nwbfile)

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
        return NWBTarget(self.input()['site_nwb'].path, 'operant_trials')
    def run(self):
        with NWBHDF5IO(self.input()['site_nwb'].path, 'r+') as nwb_io:
            nwbfile = nwb_io.read()
            audcode_events = nwbfile.acquisition['audcode_events'].to_dataframe()
            session_start = nwbfile.session_start_time.replace(tzinfo=None)
            audcode_events['datetime'] = [session_start + timedelta(seconds=x) for x in audcode_events['timestamp']]

            # all of these TimeIntervals are relevant to all time series
            all_time_series = [c for c in nwbfile.all_children() if isinstance(c, TimeSeries)]

            # find contiguous blocks of audcode trials
            trial_blocks = np.split(audcode_events,np.where(np.diff(audcode_events.trial) != 1)[0]+1)
            trial_block_starts = [x.datetime.iloc[0] for x in trial_blocks]
            trial_block_ends = [x.datetime.iloc[-1] for x in trial_blocks]

            # now we need to match each of these blocks with the corresponding csv files
            # first we need to find the csv files
            site_config = self.input()['site_config'].data()
            project_config = self.input()['project_config'].data()
            pyoperant_csvs = [os.path.join(project_config['LOCAL_RAW_FILES']['pyoperant_drive'],f) for f in site_config['filelist'] if f.endswith('trialdata.csv')]
            pyoperant_experiments = [PyoperantExperiment(x) for x in pyoperant_csvs]

            # create the different timeintervales we will keep track of
            pyoperant_experiments_intervals = TimeIntervals(
                name="pyoperant_experiments",
                description="Top level intervales for operant experiments, approximate start and stop times",
            )
            pyoperant_experiments_intervals.add_column(name="experiment_type", description="Type of experiment (operant or playback)")
            pyoperant_experiments_intervals.add_column(name="uid", description="Unique Identifier for experiment (base of filename)")

            op_trial_intervals = TimeIntervals(
                name="operant_trials",
                description="Intervals for individual operant trials"
            )
            op_trial_intervals.add_column(name="trial_num", description="Trial number")
            op_trial_intervals.add_column(name="experiment_uid", description="Unique Identifier for experiment (base of filename)")
            op_trial_intervals.add_column(name="stimuli", description="Stimuli presented")
            op_trial_intervals.add_column(name="response", description="Response made")
            op_trial_intervals.add_column(name="correct", description="Correct response")

            pb_trial_intervals = TimeIntervals(
                name="playback_trials",
                description="Intervals for individual playback trials"
            )
            pb_trial_intervals.add_column(name="trial_num", description="Trial number")
            pb_trial_intervals.add_column(name="experiment_uid", description="Unique Identifier for experiment (base of filename)")
            pb_trial_intervals.add_column(name="stimuli", description="Stimuli presented")

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
                        end = (start + exp.trial_duration(ix).total_seconds())
                        if exp.experiment_type == 'operant':
                            op_trial_intervals.add_interval(
                                start_time=start,
                                stop_time=end,
                                trial_num=ix,
                                experiment_uid=exp.uid,
                                stimuli=trial.stimulus_name,# TODO Make this truncated or something
                                response=trial.response,
                                correct=trial.correct,
                                timeseries=all_time_series
                            )
                        else:
                            pb_trial_intervals.add_interval(
                                start_time=start,
                                stop_time=end,
                                trial_num=ix,
                                experiment_uid=exp.uid,
                                stimuli=trial.stimulus_name,
                                timeseries=all_time_series
                            )
            # now save them to the nwb file
            nwbfile.add_time_intervals(pyoperant_experiments_intervals)
            nwbfile.add_time_intervals(op_trial_intervals)
            nwbfile.add_time_intervals(pb_trial_intervals)
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
            'ripples': AddRippleEventsNWB(site_dir=self.site_dir),
            'operant': AddPyoperantTrials(site_dir=self.site_dir)
        }
    