from datetime import datetime
from uuid import uuid4
import os
import json

import luigi
import numpy as np
import tqdm
from dateutil.tz import tzlocal

from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import LabMetaData
from pynwb.base import TimeSeries
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

        # first init the NWB file
        nwbfile = NWBFile(
            session_description=project_config['EXPERIMENT_NAME'],
            identifier=str(uuid4()), # random uuid
            session_start_time=datetime.now(tzlocal()), # TODO: Get time from the rhd file
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
        
class NWBAddDigitalEvents(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_config': SiteConfig(site_dir=self.site_dir),
            'project_config': SiteProjectConfig(site_dir=self.site_dir),
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'site_digital_memmap': GenerateSingleSiteDigitalMemmap(site_dir=self.site_dir)
        }

    def output(self):
        return NWBTarget(self.input()['site_nwb'].path, 'audcode_events', 'ttl_events', 'event_types')
    
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
        return NWBTarget(self.input()['site_nwb'].path, 'LFP')
    
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
    sample_rate = luigi.IntParameter(default=7500)
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'motion': GenerateSingleSiteAuxMemmap(site_dir=self.site_dir)
        }

    def output(self):
        return NWBTarget(self.input()['site_nwb'].path, 'Motion Time Series')
    
    def run(self):

        # load the motion data
        motion_info = json.load(open(self.input()['motion']['info'].path,'r'))
        motion_shape = (motion_info['num_samples'], motion_info['num_channels'])
        motion_data = np.memmap(self.input()['motion']['bin'].path,shape=motion_shape, dtype=motion_info['dtype'], mode='r')

        # take the motion data, and calculate a downsampled proxy for total motion
        # this is the sqrt of the sum of the squares of the motion data
        if self.sample_rate != motion_info['sampling_rate']:
            # TODO: I dont think this needs to be in chunks
            ratio = motion_info['sampling_rate']/self.sample_rate
            rectified_motion = np.zeros(int(motion_data.shape[0]//ratio))
            chunk_size = 20000
            tqdm.tqdm.write("Calculating motion energy")
            for i in tqdm.tqdm(range(0, rectified_motion.shape[0], chunk_size)):
                if i+chunk_size > rectified_motion.shape[0]:
                    chunk_size = rectified_motion.shape[0] - i
                if i*ratio + chunk_size*ratio > motion_data.shape[0]:
                    chunk_size = int((motion_data.shape[0] - i*ratio)//ratio)
                chunk = np.sqrt(np.sum(motion_data[int(i*ratio):int((i+chunk_size)*ratio)]**2, axis=1))

                # NB: This is not correct, but its less computationaly expensive than the correct way
                # interpolation, and I dont care abou tthe sub-ms accuracy of this signal. PLease dont do this for spiking data
                inds = (np.round(np.arange(i,i+chunk_size)*ratio) - np.round(i*ratio)).astype(int) 
                rectified_motion[i:i+chunk_size] = chunk[inds]
        else:
            rectified_motion = np.sqrt(np.sum(motion_data.astype(np.float32)**2, axis=1))


        motion_ts = TimeSeries(
            name="motion",
            data=rectified_motion,
            rate=float(self.sample_rate),
            description="Motion energy calculated from the motion data",
            unit="v?",
        )

        behavioral_time_series = BehavioralTimeSeries(
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
        return NWBTarget(self.input()['site_nwb'].path, 'audio')
    
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

class GenerateAcquisitionNWB(luigi.WrapperTask):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir),
            'digital': NWBAddDigitalEvents(site_dir=self.site_dir),
            'lfp': AddLFPNWB(site_dir=self.site_dir),
            'motion': AddMotionNWB(site_dir=self.site_dir),
            'audio': AddAudioNWB(site_dir=self.site_dir)
        }