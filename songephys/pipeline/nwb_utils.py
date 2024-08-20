from datetime import datetime
from uuid import uuid4
import os
import json

import luigi
import numpy as np
from dateutil.tz import tzlocal

from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import LFP, ElectricalSeries
from pykilosort.io.probes import cambridge_neurotech_assy_236_h10, cambridge_neurotech_assy_236_P1, microprobes_16ch
#from neo.rawio.intanrawio import read_rhd_signal_groups
from ndx_events import TtlsTable, TtlTypesTable, EventsTable, EventTypesTable, Task

from ..preprocess.sound.playback_utils import convert_raw_digital_to_metadata, convert_pyoperant_operant_csvs, _get_stim_dirs, convert_pyoperant_playback_csvs

from .pipeline_utils import (
    SiteParameter,
    parse_filelist,
    session_basename,
)

from .file_setup import GenerateSingleSiteMemmap, GenerateSingleSiteDigitalMemmap

def get_probe_file(probe_name):
    probes = dict({
            "P1": cambridge_neurotech_assy_236_P1, # 4 shank
            "H10": cambridge_neurotech_assy_236_h10, # 2 shank 
            "M16": microprobes_16ch # microprobes with 16 wires
    })
    return probes[probe_name]



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
        file_list = site_config["filelist"]

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
        probe_name = project_config["EPHYS"].get('PROBETYPE','H10')
        prb = kilosort.io.load_probe(get_probe_file(probe_name))

        # first make the device
        device = nwbfile.create_device(name=probe_name, manufacturer='Cambridge Neurotech')

        # then make the electrode group for the probe
        ml = .3 # mm, same for all probes TODO: Make this part of site_config optionally
        depth = site_config['depth_mm']
        electrode_group = nwbfile.create_electrode_group(
                name=probe_name,
                description="%s probe in %s"%(probe_name,"NCM"),#TODO site_config['brain_area']),
                device=device,
                location="Right hemisphere, cerebellum"
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
        return luigi.LocalTarget(os.path.join(self.site_dir, '%s.nwb'%os.path.basename(self.site_dir)))
    
    def run(self):
        site_config = self.input()['site_config'].data()
        project_config = self.input()['project_config'].data()
        file_list = site_config["filelist"]

        nwbfile = NWBHDF5IO(self.input()['site_nwb'].path, 'r').read()

        # get the name of the digital channels
        if ['digital_channel_names'] in site_config:
            digital_channel_names = site_config['digital_channel_names']
        else:
            digital_channel_names = ['AUDCODE', 'TTL', 'BUTTON', 'FEEDER', 'SEED', 'HOUSE']

        # make the ttl types table
        ttl_types_table = EventTypesTable(name="ttl_types", description="Metadata about TTL types")
        ttl_types_table.add_row(
            event_name="TTL",
            event_type_description="Continuous TTL signal from behavioral arduino"
        )
        ttl_types.add_row(
            event_name="BUTTON",
            event_type_description="Peckport beam break"
        )
        ttl_types.add_row(
            event_name="FEEDER",
            event_type_description="Feeder activation motor activation / LED on signal"
        )
        ttl_types.add_row(
            event_name="SEED",
            event_type_description="Seed dispenser beam break"
        )
        ttl_types.add_row(
            event_name="HOUSE",
            event_type_description="House light on signal"
        )
        # now make the ttls table
        ttls_table = EventsTable(description="Metadata about TTL events", target_tables={"ttl_type": ttl_types_table})

        # Make the events types table
        events_types_table = EventTypesTable(name="audcode_event_types", description="Precise time and metadata about stimuli played")
        events_types_table.add_row(
            event_name="AUDCODE",
            event_type_description="Auditory stimulus code"
        )
        events_table = EventsTable(name="audcode_events", description="Precise time and metadata about stimuli played", target_tables={"event_type": events_types_table})
        events_table.add_column(name="code", description='stimuli code')
        events_table.add_column(name="object", description='object name')
        events_table.add_column(name="action", description='action name')
        events_table.add_column(name="repetition", description='repetition number')
        events_table.add_column(name="trial", description='trial number')
        events_table.add_column(name="md5", description='md5 hash of the stimulus file')
        
        # now load the digital memmap and fill in the ttl table and events table
        digital_memmap_info = json.load(open(self.input()['site_digital_memmap']['info'],'r'))
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
                        t, metadat = stim
                        events_table.add_row(
                            time=t/fs,
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
                    if len(rising) != len(falling):
                        # it has to be off by one. we can figure out how to fix it with a few tests
                        # if the first rising is after the first falling then we need to remove the first falling
                        if rising[0] > falling[0]:
                            falling = falling[1:]
                            assert(len(rising) == len(falling))
                        # if the last falling is before the last rising then we need to add a falling at the end that is the total duration
                        if rising[-1] < falling[-1]:
                            falling = np.append(falling, len(dig_data))
                            assert(len(rising) == len(falling))              
                    for r, f in zip(rising, falling):
                        ttls_table.add_row(
                            time=r/fs,
                            event_type=i-1,
                            duration=(f-r)/fs
                        )
        else:
            num_samples = digital_memmap_info['num_samples']
            num_chans = digital_memmap_info['num_channels']
            # TODO maybe get dtype too
            digital_memmap = np.memmap(self.input()['site_digital_memmap']['bin'].path, dtype='uint16', mode='r', shape=(num_samples, num_chans))
            # now we can just iterate over the channels
            for i, chan in enumerate(digital_channel_names):
                if chan == 'AUDCODE':
                    stimulus_codes = convert_raw_digital_to_metadata(digital_memmap[:,i],fs)
                    for stim in stimulus_codes:
                        t, metadat = stim
                        events_table.add_row(
                            time=t/fs,
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
                    if digital_memmap[:,i].mean() > .5:
                        dig_data = ~digital_memmap[:,i]
                    # now find the rising and falling edges
                    rising = np.where(np.diff(dig_data) == 1)[0]
                    falling = np.where(np.diff(dig_data) == -1)[0]
                    if len(rising) != len(falling):
                        # it has to be off by one. we can figure out how to fix it with a few tests
                        # if the first rising is after the first falling then we need to remove the first falling
                        if rising[0] > falling[0]:
                            falling = falling[1:]
                            assert(len(rising) == len(falling))
                        # if the last falling is before the last rising then we need to add a falling at the end that is the total duration
                        if rising[-1] < falling[-1]:
                            falling = np.append(falling, len(dig_data))
                            assert(len(rising) == len(falling))              
                    for r, f in zip(rising, falling):
                        ttls_table.add_row(
                            time=r/fs,
                            event_type=i-1,
                            duration=(f-r)/fs
                        )
        # add the tables to the nwbfile
        nwbfile.add_acquisition(ttls_table)
        nwbfile.add_acquisition(events_table)
        with NWBHDF5IO(self.output().path, 'w') as io:
            io.write(nwbfile)



        
        #for ix, chan in enumerate(digital_channel_names):
        
        # add the digital events
        nwbfile.add_acquisition(
            TimeSeries(
                name='DigitalEvents',
                data=digital_memmap,
                unit='bitfield',
                rate=30000.0,
                timestamps=np.arange(len(digital_memmap))/30000.0,
                description='Digital events from the site'
            )
        )
        with NWBHDF5IO(self.output().path, 'w') as io:
            io.write(nwbfile)

class AddLFPNWB(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return {
            'site_config': SiteConfig(site_dir=self.site_dir),
            'project_config': SiteProjectConfig(site_dir=self.site_dir),
            'site_nwb': CreateSiteNWB(site_dir=self.site_dir)
        }

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, '%s.nwb'%os.path.basename(self.site_dir)))
    
    def run(self):
        site_config = self.input()['site_config'].data()
        project_config = self.input()['project_config'].data()
        file_list = site_config["filelist"]
        nwbfile = NWBHDF5IO(self.input()['site_nwb'].path, 'r').read()

        # load the LFP data
        lfp_data = np.random.rand(1000, 16) # 1000 samples, 16 channels
        lfp_ts = ElectricalSeries(
            name='LFP',
            data=lfp_data,
            electrodes=nwbfile.electrodes,
            rate=1000.0,
            description='LFP data from the site'
        )
        nwbfile.add_acquisition(lfp_ts)
        with NWBHDF5IO(self.output().path, 'w') as io:
            io.write(nwbfile)
