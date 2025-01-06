import os
import numpy as np
import pandas as pd
from scipy import signal, ndimage
from scipy.stats import zscore
from concurrent.futures import ProcessPoolExecutor
import luigi
from pynwb import NWBHDF5IO
from pynwb.epoch import TimeIntervals

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
from .nwb_utils import (
    CreateSiteNWB,
    AddLFPNWB,
    AddMotionNWB,
    NWBProcessingTarget
)

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
            all_trials = all_trials[all_trials[:,0].argsort()]
            rip_df['in_trial'] = np.searchsorted(all_trials[:,0], rip_df['stop_time'].values) - np.searchsorted(all_trials[:,1], rip_df['start_time'])  == 1


            # def check_if_ripple_in_trial(st, end, all_trials):

            #     return any([any([(st > trial[0]) & (st < trial[1]) for trial in all_trials]) | any([(end > trial[0]) & (end < trial[1]) for trial in all_trials])])
            # rip_df['in_trial'] = rip_df.apply(lambda x: check_if_ripple_in_trial(x['start_time'], x['stop_time'], all_trials), axis=1)

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



