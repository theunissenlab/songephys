import os

import luigi
import pandas
import numpy as np
import math
import pickle
from scipy.io import wavfile
from scipy import signal,stats
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
from sklearn.decomposition import PCA
import seaborn
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from scipy.stats import zscore

from .file_setup import SiteProjectConfig, SiteConfig, GenerateSingleSiteOperantCSV, GenerateSingleSitePlaybackCSV
from .pipeline_utils import SiteParameter, BaseTask
from .preprocess_sound import MergeOperantCSV, MergePlaybacksCSV
from .preprocess_ephys import KiloSort4Sorting,GenerateSiteLFP, KiloSortSorting
from ..plot_utils import plot_pecking_test_data, color_by_reward

"""
This file Will have all jobs for post (afer spike-sorting) analysis
for the operant ephys sites
This includes:
- GenerateFigs : WrapperTask that runs all jobs to generate figures for a given site
    - PlotAuditorySpikerate: Plots 
    - PlotPCA: 
"""

# region Plotting utils. 
# TODO these can be moved to a separate file eventually
def _weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def _get_spectrogram(arr: np.ndarray, fs: int, db: int=120, fc: int=12500) -> [np.ndarray, np.ndarray, np.ndarray]:
    # create the spectrogram for a 1d signal at a given sampling rate
    # arr: 1d signal
    # fs: sampling frequency
    # db: db present in the output spectrogram (default 120)
    # fc: frequency cutoff (defualt 12500)

    f, t, Sxx = signal.spectrogram(arr, fs, nperseg = 1000)
    # ignore frequencies about 12.5 kHZ
    f_inds = f<fc
    f = f[f_inds]
    Sxx = 20 * np.log10(Sxx)
    minDB = Sxx.max() - db
    Sxx[Sxx < minDB] = minDB
    return (f, t, Sxx[f_inds,:])


def _plot_trials(spikes: np.ndarray,peck_times: np.ndarray,title: str,spect: np.ndarray,t_start: float,t_end: float,mode: str='rasters',colors=None) -> None:
    # spikes is a list of lists of spike_times to plot
    # colors is a list of classes for spikes
    # peck_times is time into the trial a peck occurs
    # title is plot title
    # spect is the necessary spectrogram plot
    # mode is either heatmap or rasters
    if colors is None:
        colors = np.ones(len(spikes))
    
    # Set up 4 subplots as axis objects using GridSpec:
    gs = gridspec.GridSpec(3, 2, width_ratios=[1,6], height_ratios=[1,3,1])
    # Add space between scatter plot and KDE plots to accommodate axis labels:
    gs.update(hspace=0.3, wspace=0.3)

    # Set background canvas colour to White instead of grey default
    fig = plt.figure(title)
    fig.patch.set_facecolor('white')
    
    # SET UP AXIS -----------
    ax = plt.subplot(gs[1,1]) # Instantiate raster_plot area
    ax.set_xlim(t_start,t_end)#-.25, all_spikes.max()+.25)
    ax.set_ylim(-.5,len(spikes)-.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Trial')
    if spect is None:
        ax.set_title(title)
    
    axl = plt.subplot(gs[1,0], sharey=ax) # Instantiate left plot area
    axl.spines["right"].set_visible(False)
    axl.spines["top"].set_visible(False)
    axl.spines["bottom"].set_visible(False)

    if spect is not None:
        axu = plt.subplot(gs[0,1],sharex=ax)
        f,t,Sxx = spect
        axu.pcolormesh(t, f, Sxx, shading='gouraud')
        axu.set_ylabel('Frequency [Hz]')
        axu.set_title(title)
    
    axb = plt.subplot(gs[2,1], sharex=ax) # Instantiate bottom KDE plot area
    axb.get_xaxis().set_visible(False) # Hide tick marks and spines
    axb.get_yaxis().set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axb.spines["left"].set_visible(False)
    axb.set_xlabel('Time [sec]')

    axc = plt.subplot(gs[2,0]) # Instantiate legend plot area
    axc.axis('off') # Hide tick marks and spines
    #-------------------------
    if mode == 'heatmap':
        # Bin spikes for heatmap   
        time_bin_size = .15 #100 ms
        t_bins = np.arange(t_start,t_end,time_bin_size)
        heatmap = np.zeros((len(spikes),len(t_bins)-1))
        for i in range(len(spikes)):
            hist = np.histogram(spikes[i],t_bins)[0]
            peck_time = peck_times[i]
            if peck_time > 0:
                bin = np.where(t_bins > peck_time)[0][0]
                hist[bin:] = 0
                #ax.vlines([peck_time],i-.5,i+.5,colors='red')
            heatmap[i,:] = hist
        ax.imshow(heatmap, interpolation='nearest',  aspect='auto',extent = (t_start,t_end,-.5,len(spikes)-.5),origin='lower')
    elif mode == 'rasters':
        ax.eventplot(spikes)
        for i in range(len(peck_times)):
            peck_time = peck_times.iloc[i]
            if peck_time > 0:
                ax.vlines([peck_time],i-.5,i+.5,colors='red')
        early_rate = []
        for i in range(len(spikes)):
            # get firing rate in first 500 ms
            early_rate.append(np.sum(np.logical_and(spikes.iloc[i]>0,spikes.iloc[i]<.5)))
        axl.plot(early_rate, np.arange(len(early_rate)))
        
    # Plot data for each categorical variable as scatter and marginal KDE plots:
    for c in np.unique(colors):
        c_spikes = spikes[colors==c]
        
        concatenated_spike_times = np.concatenate(np.asarray(c_spikes))
        peck_times = np.asarray(peck_times)
        peck_times[peck_times == 0.0] = np.nan
        weights = np.array([1/(len(peck_times) - sum(peck_times < x)) for x in concatenated_spike_times])

        x_d = np.linspace(0,6,301)
        #kde = KernelDensity(bandwidth=.1).fit(c_x,sample_weight=weights)
        #kdens = kde.score_samples(x_d.reshape(-1, 1))
        kdens = sum(norm(xi,scale=.05).pdf(x_d)*w for xi,w in zip(concatenated_spike_times,weights))
        axb.plot(x_d,kdens)


    # Copy legend object from scatter plot to lower left subplot and display:
    # NB 'scatterpoints = 1' customises legend box to show only 1 handle (icon) per label 
    #handles, labels = ax.get_legend_handles_labels()
    lines, labels = axb.get_legend_handles_labels()
    axc.legend(lines, labels, loc = 'center', fontsize = 12)

def _plot_motion_trials(motion: np.ndarray, peck_times: np.ndarray, title: str, spect: np.ndarray, t_start:float, t_end: float) -> None:
    # spikes is a list of lists of spike_times to plot
    # colors is a list of classes for spikes
    # peck_times is time into the trial a peck occurs
    # title is plot title
    # spect is the necessary spectrogram plot
    # mode is either heatmap or rasters
    
    # Set up 4 subplots as axis objects using GridSpec:
    gs = gridspec.GridSpec(3, 1, height_ratios=[1,3,1])
    # Add space between scatter plot and KDE plots to accommodate axis labels:
    gs.update(hspace=0.3, wspace=0.3)

    # Set background canvas colour to White instead of grey default
    fig = plt.figure(title)
    fig.patch.set_facecolor('white')
    
    # SET UP AXIS -----------
    ax = plt.subplot(gs[1,0]) # Instantiate raster_plot area
    ax.set_xlim(t_start,t_end)#-.25, all_spikes.max()+.25)
    ax.set_ylim(-.5,len(motion)-.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Trial')
    if spect is None:
        ax.set_title(title)

    if spect is not None:
        axu = plt.subplot(gs[0,0],sharex=ax)
        f,t,Sxx = spect
        axu.pcolormesh(t, f, Sxx, shading='gouraud')
        axu.set_ylabel('Frequency [Hz]')
        axu.set_title(title)
    
    axb = plt.subplot(gs[2,0], sharex=ax) # Instantiate bottom KDE plot area
    axb.get_xaxis().set_visible(False) # Hide tick marks and spines
    axb.get_yaxis().set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axb.spines["left"].set_visible(False)
    axb.set_xlabel('Time [sec]')

    #-------------------------
    # Bin spikes for heatmap   
    # time_bin_size = .15 #100 ms
    # t_bins = np.arange(t_start,t_end,time_bin_size)
    # heatmap = np.zeros((len(spikes),len(t_bins)-1))
    # for i in range(len(spikes)):
    #     hist = np.histogram(spikes[i],t_bins)[0]
    #     peck_time = peck_times[i]
    #     if peck_time > 0:
    #         bin = np.where(t_bins > peck_time)[0][0]
    #         hist[bin:] = 0
    #         #ax.vlines([peck_time],i-.5,i+.5,colors='red')
    #     heatmap[i,:] = hist
    ax.imshow(motion, interpolation='nearest',  aspect='auto',extent = (t_start,t_end,-.5,len(motion)-.5),origin='lower')
    
    # for i in range(len(peck_times)):
    #     peck_time = peck_times.iloc[i]
    #     if peck_time > 0:
    #         ax.vlines([peck_time],i-.5,i+.5,colors='red')

# endregion
# region Prep Jobs

def is_operant(site_dir: str) -> bool:
    # TODO: Could do this smarter or in a different way
    return 'operant' in os.path.basename(os.path.normpath(site_dir))

def is_playback(site_dir: str) -> bool:
    return 'playback' in os.path.basename(os.path.normpath(site_dir))

def is_communication(site_dir: str) -> bool:
    return 'communication' in os.path.basename(os.path.normpath(site_dir))


class CreateSoundsepSegments(luigi.Task):
    """Creates a save file (used for soundsep) that populates
    a soundsep project folder.

    Creates a folder for soundsep project, a soundsep config file,
    and symlinks to each sessions raw mic files

    SESSION_DIR/
        soundsep_labeling/
            soundsep.yaml
    """
    site_dir = SiteParameter()

    def requires(self):
        # if operant use merge operant
        # if playback use merge playback
        if is_operant(self.site_dir):
            events =  GenerateSingleSiteOperantCSV(site_dir = self.site_dir)
        elif is_playback(self.site_dir):
            events =  GenerateSingleSitePlaybackCSV(site_dir = self.site_dir)
        return {
            "events": events,
            "site_config": SiteConfig(self.site_dir, required_keys=["sessions"]),
        }

    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir, "soundsep_labeling", "soundsep.yaml"))

    def run(self):
        site_config = self.input()["site_config"].data()
        playback_periods = pd.read_csv(self.input()["playbacks"].path)
        event_periods = pd.read_csv(self.input()["events"].path)

        playback_channels = site_config.get("playback_mic_channels", 0)
        if isinstance(playback_channels, int):
            playback_channel = playback_channels
        else:
            playback_channel = playback_channels[0] if len(playback_channels) else 0

        audio_fn = self.input()["wav_files"]["audio.wav"].path
        sampling_rate = AudioFile(os.path.join(audio_fn)).sampling_rate

        # Write a default config
        with open(self.output().path, "w+") as f:
            f.write("""audio_directory: ../raw""")
class AnalysisTask(BaseTask):
    site_dir = SiteParameter()
    analysis_dir = luigi.Parameter(default='analysis_ks4')
    ksversion = luigi.Parameter(default='ks4')

    def get_analysis_dir(self):
        return os.path.join(self.site_dir,self.analysis_dir)

    def get_ks_task(self):
        if self.ksversion == 'ks4':
            return KiloSort4Sorting(site_dir = self.site_dir)
        elif self.ksversion == 'ks2.5':
            return KiloSortSorting(site_dir = self.site_dir)
        else:
            raise ValueError("Kilosort version not recognized")
        
# class AnalysisConfig(AnalysisTask):
#     def output(self):
#         return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"config.pkl"))
#     def run(self):
#         # generate a default config
#         config = dict()
#         config['analysis_dir'] = 
#         with open(self.output().path,'wb') as f:
#             pickle.dump(self.param_kwargs,f)

class GenStimInfoDB(AnalysisTask):
    """ Generates database of playbacks.csv with more info for each stim"""
    def requires(self):
        # TODO this is janky. get the config first and set a variable there for if its playback or operant
        if is_playback(self.site_dir):
            merge_task = GenerateSingleSitePlaybackCSV(site_dir = self.site_dir)
        elif is_operant(self.site_dir):
            merge_task = GenerateSingleSiteOperantCSV(site_dir = self.site_dir)
        else:
            print("NOT SURE WHAT %s is"%self.site_dir)

        return {
            "merge_csv": merge_task,
            "site_config": SiteConfig(site_dir = self.site_dir),
            "project_config": SiteProjectConfig(site_dir = self.site_dir)
        }
    
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"raw","dfPlaybackInfo.pkl"))
    
    def run(self):
        # load the project config to get stim_drive location
        project_config = self.input()['project_config'].data()

        # read the locations of the renditions within each stimuli
        timings_df = pandas.read_csv(os.path.join(project_config['LOCAL_RAW_FILES']['stimuli_drive'],"StimInfo.csv"))

        # load playback info from the MergeOperantCSV task
        playback_fn = self.input()['merge_csv']['playbacks'].path
        playbacks = pandas.read_csv(playback_fn)

        # clean the playbacksdb to have the same stim names as the timings_df
        #playbacks['name'] = [pb.strip('playback_') for pb in playbacks.name]
        playbacks['name'] = [sn.split('/')[-1] for sn in playbacks.playback_file]

        # join the playbacks and timings into one dataframe
        playbacks_and_timings = playbacks.join(timings_df.set_index('Filename'),on='name')

        if is_playback(self.site_dir):
            conditions = pandas.read_csv(os.path.join(project_config['LOCAL_RAW_FILES']['stimuli_drive'],"StimCategories.csv"))
            playbacks_and_timings = playbacks_and_timings.join(conditions.set_index('StimName'),on='bird_name')
            playbacks_and_timings = playbacks_and_timings.rename(columns={"Category":"condition_name"})
            playbacks_and_timings['trial'] = playbacks_and_timings.index
            playbacks_and_timings['response'] = False
        # save the databse to the output path
        self.output().makedirs()
        playbacks_and_timings.to_pickle(self.output().path)

class GenBehaviorDB(AnalysisTask):
    def requires(self):
        return GenStimInfoDB(site_dir = self.site_dir)
    def output(self):
        # TODO I could put a list here of jobs ran
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","dfBehavior.pkl"))
    def run(self):
        playback_and_timings = pandas.read_pickle(self.input().path)
        playback_and_timings['novel'] = "Learned"
        if is_operant(self.site_dir):
            project_dir = self.site_dir.split("sites")[0]
            novel_stim_db = pandas.read_csv(os.path.join(project_dir,"novel_stims.csv"))
            site = self.site_dir.split('sites/')[-1]
            novel_stim_db = novel_stim_db[novel_stim_db.site == site]
            # go through and inject informative trials
            informative_dfs = []
            for bird, birddf in playback_and_timings.groupby('bird_name'):
                didnt_respond_previously = (~birddf.response).shift()
                didnt_respond_previously.iloc[0] = 0
                informative_trials = didnt_respond_previously.cumsum() # the first interrupted should be 0
                informative_trials.name = 'informative_trials'
                informative_dfs.append(informative_trials)
                is_novel = novel_stim_db[(novel_stim_db.site == site)&(novel_stim_db.bird==bird)].novel.iloc[0]
                playback_and_timings.loc[playback_and_timings.bird_name == bird,'novel'] = "Novel" if is_novel else "Learned"
            playback_and_timings = playback_and_timings.join(pandas.concat(informative_dfs))
        elif is_playback(self.site_dir):
            # go through and inject informative trials
            informative_dfs = []
            for bird, birddf in playback_and_timings.groupby('bird_name'):
                didnt_respond_previously = (~birddf.response).shift()
                didnt_respond_previously.iloc[0] = 0
                informative_trials = didnt_respond_previously.cumsum() # the first interrupted should be 0
                informative_trials.name = 'informative_trials'
                informative_dfs.append(informative_trials)
                site = self.site_dir.split('sites/')[-1]
            playback_and_timings = playback_and_timings.join(pandas.concat(informative_dfs))
        self.output().makedirs()
        playback_and_timings.to_pickle(self.output().path)

class GenSpectrogramDB(AnalysisTask):
    """ Generates database of spectrograms for each stim"""
    def requires(self):
        return {
            "project_config": SiteProjectConfig(site_dir = self.site_dir),
            "playback_info": GenStimInfoDB(site_dir = self.site_dir)
        }
    
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"raw","dfSongSpect.pkl"))
    
    def run(self):
        # load the project config to get stim_drive location
        project_config = self.input()['project_config'].data()
        stim_dir = os.path.join(project_config['LOCAL_RAW_FILES']['stimuli_drive'],"Operant")

        playbacks_and_timings = pandas.read_pickle(self.input()['playback_info'].path)

        df_song_spect = {"Bird":[],"Name":[],"Spectrogram":[]}
        # Go through all emitters
        for bird in np.unique(playbacks_and_timings.bird_name):
            bird_df = playbacks_and_timings[(playbacks_and_timings.bird_name == bird)]
            # all unique stim files for each emitter
            for song in np.unique(bird_df.name):
                # get all playbacks of this stim file
                song_df = bird_df[bird_df.name == song]
                row = song_df.iloc[0]
                song_rend = int(song.split('_')[2])

                # read the wav file to get the spectrogram
                fs,arr = wavfile.read(row.playback_file)#os.path.join(stim_dir,row.playback_file.split('stimuli/Operant/')[-1]))

                spectrogram = _get_spectrogram(arr,fs)
                df_song_spect['Bird'].append(bird)
                df_song_spect['Name'].append(song)
                df_song_spect['Spectrogram'].append(spectrogram)
            
        df_song_spect = pandas.DataFrame(df_song_spect)

        self.output().makedirs()
        df_song_spect.to_pickle(self.output().path)

# endregion Prep Jobs
# region Descriptive Stats
class GenClusterAuditorySpikeRatesDB(AnalysisTask):
    """ OUTPUTS: 
            data/<CLUSTER>/dfAuditorySpikes.pkl
    """
    cluster = luigi.DictParameter()

    def requires(self):
        return {
            "project_config": SiteProjectConfig(site_dir = self.site_dir),
            "playback_info": GenBehaviorDB(site_dir = self.site_dir),
            "kilosort": self.get_ks_task()
        }
    
    def output(self):
        # TODO could break this into playbacks and silence dfs, and do a bunch of other stuff with the playbacks
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","Cluster_%d"%self.cluster['cluster_id'],"dfAuditorySpikes.pkl"))
    
    def run(self):
        ks_output_dir =  os.path.dirname(self.input()['kilosort'].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        # TODO get samplerate from project config?
        cluster_spike_times = ks_spike_times[ks_spike_cluster==self.cluster['cluster_id']] / 30000 # in seconds

        playbacks_and_timings = pandas.read_pickle(self.input()['playback_info'].path)
        # first calculate the silent (outside of stims) rate of firing for this unit
        sound_times = []
        for i,trial_row in playbacks_and_timings.iterrows():
            # first add rend 1
            row_end = trial_row.start_time + trial_row.duration
            start_time = trial_row.start_time + trial_row.rend1_st # padded start time
            end_time = start_time + trial_row.rend1_dur # unpadded end time
            # check if this trial got past first rend
            if row_end < end_time:
                end_time = row_end
            sound_times.append([start_time,end_time,trial_row.condition_name])

            start_time = trial_row.start_time + trial_row.rend2_st # padded start time
            if row_end > start_time:
                end_time = start_time + trial_row.rend2_dur # unpadded end time
                # check if this trial got past first rend
                if row_end < end_time:
                    end_time = row_end
                sound_times.append([start_time,end_time,trial_row.condition_name])

            start_time = trial_row.start_time + trial_row.rend3_st # padded start time
            if row_end > start_time:
                end_time = start_time + trial_row.rend3_dur # unpadded end time
                # check if this trial got past first rend
                if row_end < end_time:
                    end_time = row_end
                sound_times.append([start_time,end_time,trial_row.condition_name])
            if is_operant(self.site_dir):
                if not trial_row.response and trial_row.reward:
                    sound_end = end_time+1.5
                    if i < len(playbacks_and_timings)-1:
                        next_row = playbacks_and_timings.iloc[i+1]
                        if next_row.start_time < sound_end:
                            sound_end = next_row.start_time
                    sound_times.append([end_time,sound_end,'RewardSound'])
        # now make opposite, silent_times
        prev_stop = 0
        silent_times = []
        for start,stop,_ in sound_times:
            if prev_stop < start-.1: #minimum .1s time
                silent_times.append((prev_stop,start))
                prev_stop = stop
            # TODO think about this, I am skipping the last part of the recording
        
        # now sound_times is a list of tuples of start and stop of playback sounds. I can check for all spikes how many spikes are in sound times and silent times
        spike_dict = dict({
            "NSpikes": [],
            "Duration": [],
            "Label": [],
            "Time": []
        })
        for start, stop in silent_times:
            spike_dict['NSpikes'].append(np.logical_and(cluster_spike_times > start,cluster_spike_times < stop).sum())
            spike_dict['Duration'].append(stop - start)
            spike_dict['Label'].append('Silent')
            spike_dict['Time'].append(start)

        for start, stop,sound_label in sound_times:
            spike_dict['NSpikes'].append(np.logical_and(cluster_spike_times > start,cluster_spike_times < stop).sum())
            spike_dict['Duration'].append(stop - start)
            spike_dict['Label'].append(sound_label)
            spike_dict['Time'].append(start)

        spike_df = pandas.DataFrame(spike_dict)
        spike_df['SpikeRate'] = spike_df['NSpikes'] / spike_df['Duration']
        
        self.output().makedirs()
        spike_df.to_pickle(self.output().path)

class PlotClusterAuditorySpikeRates(AnalysisTask):
    """ OUTPUTS: 
            AuditorySpikerates.png
    """
    cluster = luigi.DictParameter()

    def requires(self):
        return GenClusterAuditorySpikeRatesDB(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"AuditorySpikerates.png"))
    
    def run(self):
        dfSpikeRate = pandas.read_pickle(self.input().path)
        # filter out periods less than .15 second
        dfSpikeRate = dfSpikeRate[dfSpikeRate.Duration > .15]

        plt.figure()
        #seaborn.ecdfplot(x=dfSpikeRate.NSpikes/dfSpikeRate.Duration,hue=dfSpikeRate.Label)

        seaborn.displot(x=dfSpikeRate.NSpikes/dfSpikeRate.Duration,hue=dfSpikeRate.Label,kind='kde',common_norm=False)
        #seaborn.violinplot(x=dfSpikeRate.Label,y=dfSpikeRate.NSpikes/dfSpikeRate.Duration,hue=dfSpikeRate.Label)
        #plt.bar([1,3,4,5],[ _get_spikerate(dfSpikeRate.Label == 'Silent'), _get_spikerate(dfSpikeRate.Label != 'Silent'), _get_spikerate(dfSpikeRate.Label == 'Rewarded'), _get_spikerate(dfSpikeRate.Label == 'Unrewarded')],
        #            label=['No Stim','All Sound','Rewarded','Unrewarded'], color=['black','blue','green','red'])
        #plt.legend()

        self.output().makedirs()
        plt.savefig(self.output().path)
        plt.close()

class PlotClusterRelativeRates(AnalysisTask):
    """ OUTPUTS:
            RelativeAuditorySpikerates.png
    """
    cluster = luigi.DictParameter()

    def requires(self):
        return GenClusterAuditorySpikeRatesDB(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"RelativeSpikerates.png"))
    def run(self):
        dfSpikeRate = pandas.read_pickle(self.input().path)
class PlotClusterSummary(AnalysisTask):
    """
    Plotting task for Cluster_Summary.png

    Plots the templates in the cluster, the isi, and the spikerate
    TODO can put some useful stats on here too
    """
    cluster = luigi.DictParameter()

    def requires(self):
        return {
            'kilosort': self.get_ks_task(),
            'config': SiteConfig(site_dir = self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),'figures','clusters_%s'%self.cluster['group'],'Cluster_%d'%self.cluster['cluster_id'],'Cluster_Summary.png'))
    def run(self):
        ks_output_dir =  os.path.dirname(self.input()['kilosort'].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')
        ks_spike_templates = np.load(os.path.join(ks_output_dir,'spike_templates.npy'))
        chan_map = np.load(os.path.join(ks_output_dir,'channel_map.npy'))
        ks_templates = np.load(os.path.join(ks_output_dir,'templates.npy'))
        ks_chan_map = np.load(os.path.join(ks_output_dir,'channel_positions.npy'))
        site_config = self.input()['config'].data()

        gs = gridspec.GridSpec(2,2)
        #gs.update(hspace=.3,wspace=.3)
        title = "Cluster %s"%self.cluster['cluster_id']
        fig = plt.figure(title)
        fig.patch.set_facecolor('white')
        
        # ISI Plot
        ax = plt.subplot(gs[0,1])
        cluster_spike_times = ks_spike_times[ks_spike_cluster==self.cluster['cluster_id']] / 30000 # in seconds
        vec_isi = cluster_spike_times[1:] - cluster_spike_times[:-1]
        ax.hist(vec_isi,bins = np.arange(0,.15,.002))

        # Spikerate over time plot
        ax_sr = plt.subplot(gs[1,1])
        ax_sr.hist(cluster_spike_times,bins=np.arange(0,max(cluster_spike_times)+1))
        
        # spike shape plot
        # first get all templates that contrib to this cluster
        templates,counts = np.unique(ks_spike_templates[ks_spike_cluster == self.cluster['cluster_id']],return_counts=True)
        cluster_info = cluster_info = ks_cluster_info[ks_cluster_info.cluster_id == self.cluster['cluster_id']]
        chan = cluster_info.ch.iloc[0]
        templates_list = []
        for template in templates:            
            actual_template = ks_templates[template,:,np.where(chan_map == chan)]
            templates_list.append(actual_template.flatten())
        template_avg = np.average(templates_list,axis=0,weights=counts/np.sum(counts))
        ax_spike = plt.subplot(gs[0,0])
        ax_spike.plot(template_avg)
        #plt.plot(template_avg)

        sizes = np.ones(len(ks_chan_map))
        sizes[np.where(chan_map == chan)] = 20
        ax_chan = plt.subplot(gs[1,0])
        depths = site_config.get('depth_mm',0) - ks_chan_map[:,1]/1000.
        ax_chan.scatter(ks_chan_map[:,0]/1000.,depths,s=sizes)
        ax_chan.invert_yaxis()


        self.output().makedirs()
        plt.savefig(self.output().path)
        plt.close()


        
class GenSiteSummary(AnalysisTask):
    """
    Goes through all clusters and generates plots summarizing all clusters
    """
    def requires(self):
        return {
            "kilosort": self.get_ks_task(),
            'config': SiteConfig(site_dir = self.site_dir)
        }
    
    def output(self):
        # TODO could break this into playbacks and silence dfs, and do a bunch of other stuff with the playbacks
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","dfClusterSummaries.pkl"))
    
    def run(self):
         # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input()['kilosort'].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')
        ks_spike_templates = np.load(os.path.join(ks_output_dir,'spike_templates.npy'))
        chan_map = np.load(os.path.join(ks_output_dir,'channel_map.npy'))
        ks_templates = np.load(os.path.join(ks_output_dir,'templates.npy'))
        site_config = self.input()['config'].data()
        new_infos = []
        # first go and yield all cluster_aud_spikerates TODO This can be done in requires with some code-fu
        for ix,cluster in ks_cluster_info.iterrows():
            if cluster['group'] != 'noise':
                yield GenClusterAuditorySpikeRatesDB(site_dir = self.site_dir,cluster=dict(cluster))
    
        for ix,cluster in ks_cluster_info.iterrows():
            if cluster['group'] != 'noise':#  and cluster.fr > 5:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
                templates,counts = np.unique(ks_spike_templates[ks_spike_cluster == cluster['cluster_id']],return_counts=True)
                cluster_info = cluster.copy()
                chan = cluster_info.ch
                templates_list = []
                for template in templates:            
                    actual_template = ks_templates[template,:,np.where(chan_map == chan)]# FOR KS2.5 do 20:
                    templates_list.append(actual_template.flatten())
                template_avg = np.average(templates_list,axis=0,weights=counts/np.sum(counts))
                #plt.plot(np.arange(len(template_avg))/30000, template_avg)

                
                sorted_inds = np.argsort(template_avg)
                min_ind = sorted_inds[0]
                max_ind = sorted_inds[sorted_inds > min_ind][-1]

                # take min and max
                min_v = template_avg[min_ind]
                max_v = template_avg[max_ind]
                
                cluster_info['PTP_ratio'] = max_v / np.abs(min_v)
                cluster_info['PTP_dur_ms'] =  (max_ind - min_ind) / 30 # 30 kHz
                cluster_info['site'] = self.site_dir
                cluster_info['template'] = [template_avg]
                cluster_info['depth'] = site_config.get('depth_mm',0) - cluster_info['depth']/1000


                #TODO do this better
                cluster_auditory_spikerates = yield GenClusterAuditorySpikeRatesDB(site_dir = self.site_dir,cluster=dict(cluster))
                cluster_rate = pandas.read_pickle(cluster_auditory_spikerates.path)
                for lbl in np.unique(cluster_rate.Label):
                    sub_df = cluster_rate[cluster_rate.Label == lbl]
                    sub_df 
                    avg_rate = np.average(sub_df.SpikeRate,weights=sub_df.Duration)
                    std_rate = np.sqrt(np.average((sub_df.SpikeRate - avg_rate)**2, weights=sub_df.Duration))
                    cluster_info["%s_avg"%lbl] = avg_rate
                    cluster_info["%s_std"%lbl] = std_rate

                new_infos.append(cluster_info)


        #cluster_info_df = pandas.concat(new_infos,ignore_index=True)
        cluster_info_df = pandas.DataFrame(new_infos)
        self.output().makedirs()
        cluster_info_df.to_pickle(self.output().path)

class PlotSiteSummary(AnalysisTask):
    def requires(self):
        return GenSiteSummary(site_dir = self.site_dir)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","AllClusters.png"))

    def run(self):
        cluster_info_df = pandas.read_pickle(self.input().path)
        plt.figure()
        fig, ax = plt.subplots(1, 2)
        ax_scatter = ax[0]
        seaborn.scatterplot(cluster_info_df,x='PTP_dur_ms',y='PTP_ratio',hue='group',ax=ax_scatter)
        ax_spike = ax[1]
        for _,x in cluster_info_df.iterrows():
            if x.group != 'noise':
                ax_spike.plot(x.template[0],c='b' if x.group=='good' else 'g')
        self.output().makedirs()
        plt.savefig(self.output().path)
        plt.close()
# class GenProjectSummary(AnalysisTask):
#     project_dir = luigi.
#     def requires(self):

#     def output(self):

#     def run(self):

# endregion 
# region Trial Raster Plots
class GenPlaybackSpikesDB(AnalysisTask):
    cluster = luigi.DictParameter()
    pad_s = luigi.FloatParameter(default=1.0)
    def requires(self):
        return {
            "project_config": SiteProjectConfig(site_dir = self.site_dir),
            "playback_info": GenBehaviorDB(site_dir = self.site_dir),
            "kilosort": self.get_ks_task(),
            "spect": GenSpectrogramDB(site_dir = self.site_dir)
        }
    
    def output(self):
        # TODO could break this into playbacks and silence dfs, and do a bunch of other stuff with the playbacks
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","Cluster_%d"%self.cluster['cluster_id'],"dfPlaybackSpikes.pkl"))
    
    def run(self):
        ks_output_dir =  os.path.dirname(self.input()['kilosort'].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        # TODO get samplerate from project config?
        cluster_spike_times = ks_spike_times[ks_spike_cluster==self.cluster['cluster_id']] / 30000 # in seconds

        playbacks_and_timings = pandas.read_pickle(self.input()['playback_info'].path)

        out_df = {"Rendition_Index":[], "Rendition_Name":[],"Trial_Number":[],
                  "Spike_Times":[],"Start":[],"End":[],"Peck":[], "Rate":[], 
                  "Bird":[], "Category":[], "Novel": [], "Informative_Trials":[],
                  "PlaybackFile":[]}
        # Go through all stims, grouped by bird
        for bird in np.unique(playbacks_and_timings.bird_name):
            # all playbacks for this vocalizer
            bird_df = playbacks_and_timings[(playbacks_and_timings.bird_name == bird)]

            # all unique stim files
            for song in np.unique(bird_df.name):
                # get all playbacks of this stim file
                song_df = bird_df[bird_df.name == song]
                row = song_df.iloc[0]
                song_rend = int(song.split('_')[2])

                if len(song_df):
                    # create a dataframe of spikes and info about each trial
                    for _,trial_row in song_df.iterrows():
                        trial_num = trial_row.trial
                        start_time = trial_row.start_time
                        end_time = start_time + trial_row.duration# 6.0 # seconds TODO get this from stim file maybe
                        if trial_row.response:
                            peck = trial_row.duration
                        else:
                            peck = 0
                            # only pad if the bird didnt peck
                            end_time += self.pad_s
                        # get spikes times for padded duration
                        spike_times = cluster_spike_times[np.logical_and(cluster_spike_times > start_time - self.pad_s,
                                                                            cluster_spike_times < end_time )]

                        # calculate spike rate from spikes that are just within the stim bounds
                        # TODO can just take spikes that happen within the renditions too
                        real_spikes = cluster_spike_times[np.logical_and(cluster_spike_times > start_time,
                                                                            cluster_spike_times < start_time + trial_row.duration)]
                        rate = len(real_spikes) / (trial_row.duration)
                        out_df['Rendition_Index'].append(song_rend)
                        out_df['Rendition_Name'].append(song)
                        out_df['Bird'].append(bird)
                        out_df['Trial_Number'].append(trial_num)
                        out_df['Start'].append(start_time)
                        out_df['End'].append(end_time)
                        out_df['Peck'].append(peck)
                        out_df['Spike_Times'].append(spike_times)
                        out_df['Rate'].append(rate)
                        out_df['PlaybackFile'].append(trial_row.playback_file)
                        out_df['Category'].append(trial_row.condition_name)
                        out_df['Novel'].append(trial_row.novel)
                        out_df['Informative_Trials'].append(trial_row.informative_trials)
        spike_df = pandas.DataFrame(out_df)
        spike_df = spike_df.sort_values(by='Start')
        self.output().makedirs()
        spike_df.to_pickle(self.output().path)

class PlotTrialSpikeRasters(AnalysisTask):
    cluster = luigi.DictParameter()
    pad_s = luigi.FloatParameter(default=1.0)
    extension = luigi.Parameter(default='png')
    def requires(self):
        return {
            'spikes': GenPlaybackSpikesDB(site_dir=self.site_dir,pad_s=self.pad_s, cluster=self.cluster),
            'spectrograms': GenSpectrogramDB(site_dir = self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"spike_rasters_%.2f_%s"%(self.pad_s,self.extension),'trial_sorted'))
    def run(self):
        # Load the spike dataframe from GenPlaybackSpikeDB
        spike_df = pandas.read_pickle(self.input()['spikes'].path)
        #spike_df.sor
        df_song_spect = pandas.read_pickle(self.input()['spectrograms'].path)
        
        # One plot for each bird, rend pair
        for bird in np.unique(spike_df['Bird']):
            bird_df = spike_df[spike_df.Bird == bird]
            if not np.all(bird_df.Spike_Times.apply(len) == 0):
                for song in np.unique(bird_df.Rendition_Name):
                    spectrogram = df_song_spect.Spectrogram[df_song_spect.Name == song].iloc[0]
                    rend_df = bird_df[bird_df.Rendition_Name == song]
                    if not np.all(rend_df.Spike_Times.apply(len) == 0):
                        _plot_trials(rend_df.Spike_Times-rend_df.Start,rend_df.Peck,song,spectrogram,-1,7,mode='rasters')
                        
                        out_fn = "Rasterplot_%s.%s"%(song,self.extension)
                        out_root = self.output().path
                        out_dir = os.path.join(out_root,"%s_%s_%s"%(rend_df.Category.iloc[0],bird,rend_df.Novel.iloc[0]))
                        os.makedirs(out_dir,exist_ok=True)
                        plt.savefig(os.path.join(out_dir,out_fn))
                        plt.close()

                # now save by bird
                _plot_trials(bird_df.Spike_Times - bird_df.Start, bird_df.Peck, bird,None,-1,7,mode='rasters')
                out_fn = "Raster_%s.%s"%(bird,self.extension)
                out_root = self.output().path
                out_dir = os.path.join(out_root,"%s_%s_%s"%(rend_df.Category.iloc[0],bird,rend_df.Novel.iloc[0]))
                os.makedirs(out_dir,exist_ok=True)
                plt.savefig(os.path.join(out_dir,out_fn))
                plt.close()
class PlotRenditionSpikeRasters(AnalysisTask):
    cluster = luigi.DictParameter()
    pad_s = luigi.FloatParameter(default=1.0)
    extension = luigi.Parameter(default='png')
    def requires(self):
        return {
            'spikes': GenPlaybackSpikesDB(site_dir=self.site_dir,pad_s=self.pad_s, cluster=self.cluster),
            'spectrograms': GenSpectrogramDB(site_dir = self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"spike_rasters_%.2f_%s"%(self.pad_s,self.extension),'by_rendition'))
    def run(self):
        # Load the spike dataframe from GenPlaybackSpikeDB
        spike_df = pandas.read_pickle(self.input()['spikes'].path)
        #spike_df.sor
        df_song_spect = pandas.read_pickle(self.input()['spectrograms'].path)
        
        # One plot for each bird, rend pair
        for bird in np.unique(spike_df['Bird']):
            ## TODO: GET RENDITIONS OF THE SAME ONE
            bird_df = spike_df[spike_df.Bird == bird]
            for song in np.unique(bird_df.Rendition_Name):
                spectrogram = df_song_spect.Spectrogram[df_song_spect.Name == song].iloc[0]
                rend_df = bird_df[bird_df.Rendition_Name == song]
                _plot_trials(rend_df.Spike_Times-rend_df.Start,rend_df.Peck,song,spectrogram,-1,7,mode='rasters')
                
                out_fn = "Rasterplot_%s.%s"%(song,self.extension)
                out_root = self.output().path
                out_dir = os.path.join(out_root,"%s_%s_%s"%(rend_df.Category.iloc[0],bird,rend_df.Novel.iloc[0]))
                os.makedirs(out_dir,exist_ok=True)
                plt.savefig(os.path.join(out_dir,out_fn))
                plt.close()

            # now save by bird
            _plot_trials(bird_df.Spike_Times - bird_df.Start, bird_df.Peck, bird,None,-1,7,mode='rasters')
            out_fn = "Raster_%s.%s"%(bird,self.extension)
            out_root = self.output().path
            out_dir = os.path.join(out_root,"%s_%s_%s"%(rend_df.Category.iloc[0],bird,rend_df.Novel.iloc[0]))
            os.makedirs(out_dir,exist_ok=True)
            plt.savefig(os.path.join(out_dir,out_fn))
            plt.close()
class PlotMotionTrials(AnalysisTask):
    pad_s = luigi.FloatParameter(default=1.0)
    def requires(self):
        return {
            #'motion': GenMotionDB(site_dir = self.site_dir), #TODO
            'playback_info': GenBehaviorDB(site_dir = self.site_dir),
            'spectrograms': GenSpectrogramDB(site_dir = self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures",'motion_plots_pad%.2f'%self.pad_s))
    def run(self):
        # first load the playbacks
        playback_df = pandas.read_pickle(self.input()['playback_info'].path)
        # now load the motion
        motion_arr = np.load(os.path.join(self.site_dir,'full_motion.npy'))
        motion_arr -= motion_arr.min()
        motion_arr /= motion_arr.max()
        # apply pca to get first PC of motion
        #pca = PCA(n_components=1)
        #TODO fix this
        #motion_arr = pca.fit_transform(motion_arr)

        # load spectrograms for plotting
        df_song_spect = pandas.read_pickle(self.input()['spectrograms'].path)
        # now make an array of the motion during each playback
        # One plot for each bird, rend pair
        for bird in np.unique(playback_df['bird_name']):
            bird_df = playback_df[playback_df.bird_name == bird]
            for song in np.unique(bird_df.name):
                spectrogram = df_song_spect.Spectrogram[df_song_spect.Name == song].iloc[0]
                song_df = bird_df[bird_df.name == song]
                motion_sampling_rate = 7500
                motion_playback = np.zeros((len(song_df),int((6+self.pad_s)*motion_sampling_rate),3))
                i=0
                for _,row in song_df.iterrows():
                    start = int((row.start_time - self.pad_s )* motion_sampling_rate)
                    end = int(start + (row.duration + self.pad_s)* motion_sampling_rate)
                    motion_playback[i,0:(end-start)] = motion_arr[start:end]
                    i+=1
                pecking = [x[1].duration if x[1].response else np.nan for x in song_df.iterrows()]
                _plot_motion_trials(motion_playback,pecking,song,spectrogram,-1,6)

                out_fn = "Motionplot_%s.png"%song
                out_root = self.output().path
                out_dir = os.path.join(out_root,"%s_%s"%(song_df.condition_name.iloc[0],bird))
                os.makedirs(out_dir,exist_ok=True)
                plt.savefig(os.path.join(out_dir,out_fn))
                plt.close()


class PlotInterruptionSpikeRasters(AnalysisTask):
    cluster = luigi.DictParameter()
    pad_s = luigi.FloatParameter(default=1.0)
    extension = luigi.Parameter(default='png')
    def requires(self):
        return {
            'spikes': GenPlaybackSpikesDB(site_dir = self.site_dir,pad_s=self.pad_s, cluster=self.cluster),
            'spectrograms': GenSpectrogramDB(site_dir = self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"spike_rasters_%.2f_%s"%(self.pad_s,self.extension),'int_sorted'))
    def run(self):
        # Load the spike dataframe from GenPlaybackSpikeDB
        spike_df = pandas.read_pickle(self.input()['spikes'].path)
        #spike_df.sor
        df_song_spect = pandas.read_pickle(self.input()['spectrograms'].path)
        
        # One plot for each bird, rend pair
        for bird in np.unique(spike_df['Bird']):
            bird_df = spike_df[spike_df.Bird == bird]
            bird_df.loc[bird_df.Peck==0,'Peck'] = np.nan
            bird_df = bird_df.sort_values('Peck')
            for song in np.unique(bird_df.Rendition_Name):
                spectrogram = df_song_spect.Spectrogram[df_song_spect.Name == song].iloc[0]
                rend_df = bird_df[bird_df.Rendition_Name == song]
                _plot_trials(rend_df.Spike_Times-rend_df.Start,rend_df.Peck,song,spectrogram,-1,7,mode='rasters')
                
                out_fn = "Rasterplot_%s.%s"%(song,self.extension)
                out_root = self.output().path
                out_dir = os.path.join(out_root,"%s_%s_%s"%(rend_df.Category.iloc[0],bird,rend_df.Novel.iloc[0]))
                os.makedirs(out_dir,exist_ok=True)
                plt.savefig(os.path.join(out_dir,out_fn))
                plt.close()

            # now save by bird
            _plot_trials(bird_df.Spike_Times - bird_df.Start, bird_df.Peck, bird,None,-1,7,mode='rasters')
            out_fn = "Raster_%s.%s"%(bird,self.extension)
            out_root = self.output().path
            out_dir = os.path.join(out_root,"%s_%s_%s"%(rend_df.Category.iloc[0],bird,rend_df.Novel.iloc[0]))
            os.makedirs(out_dir,exist_ok=True)
            plt.savefig(os.path.join(out_dir,out_fn))
            plt.close()
class PlotSpikeRateByTrial(AnalysisTask):
    cluster = luigi.DictParameter()
    pad_s = luigi.FloatParameter(default=1.0)
    extension = luigi.Parameter(default='png')
    def requires(self):
        return GenPlaybackSpikesDB(site_dir = self.site_dir,pad_s=self.pad_s, cluster=self.cluster)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"spike_rasters_%.2f_%s"%(self.pad_s, self.extension),'SpikeRate_ByBird.%s'%self.extension))
    def run(self):
        # now save avg spikerate by trial
        spike_df = pandas.read_pickle(self.input().path)
        rates = []
        for ix, row in spike_df.iterrows():
            spike_times = row.Spike_Times-row.Start
            start = 0
            end = .5
            rate = np.logical_and(spike_times > start, spike_times < end).sum() / (end-start)
            rates.append(rate)
        spike_df['rate_onset'] = rates
        
        plt.figure()
        seaborn.lineplot(spike_df,x='Start',y='rate_onset',hue='Bird')

        self.output().makedirs()
        plt.savefig(self.output().path)
        plt.close()
        

# endregion
# region PCA Population stuff
class GenPopulationPCA(AnalysisTask):
    # TODO put these in site config or something
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return self.get_ks_task()
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","PCATimeseries_bin%.2f_step%.2f.npy"%(self.bin_size_s,self.bin_step_s)))
    def run(self):
        # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input().path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')


        cluster_ids_good = []
        for ix,cluster in ks_cluster_info.iterrows():
            if cluster['group'] != 'noise':# and cluster.fr > 2.0:#  and cluster.fr > 5:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
                cluster_ids_good.append(cluster.cluster_id)
        cluster_ids_good = np.asarray(cluster_ids_good)

        # TODO could get this from project or site config or something
        t_start = np.floor(ks_spike_times.min()/30000)
        t_stop = np.ceil(ks_spike_times.max()/30000)
        tot_time = t_start - t_stop

        # T x Cluster
        spike_arr_30k = np.zeros((int(t_stop*30000),len(cluster_ids_good)),dtype=np.uint8)
        for ix,p in enumerate(cluster_ids_good):
            spike_arr_30k[:,ix][ks_spike_times[ks_spike_cluster==p]] = 1
        
        # now we window it and sum for each cluster 
        # TODO get samplerate from some config
        pop_timeseries = np.lib.stride_tricks.sliding_window_view(spike_arr_30k, int(np.round(self.bin_size_s*30000)), axis=0)[::int(np.round(self.bin_step_s*30000))].sum(axis=2)
        # Now z score the pop vector by cluster
        pop_timeseries = stats.zscore(pop_timeseries)
        #time_bins = np.arange(0,len(pop_timeseries))*bin_step_s
        # do pca on the 
        print("Running PCA on the zscored bins")
        pca = PCA(n_components=min(5,len(cluster_ids_good))) 
        # pca_timeseries is <NPC>x<NTime>
        pca_timeseries = pca.fit_transform(pop_timeseries).T

        self.output().makedirs()
        np.save(self.output().path,pca_timeseries)

class GenPlaybackPCADB(AnalysisTask):
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return {
            'pca': GenPopulationPCA(site_dir = self.site_dir, bin_size_s =self.bin_size_s, bin_step_s=self.bin_step_s),
            'playback_info': GenStimInfoDB(site_dir = self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","dfPlaybackPCA_bin%.2f_step%.2f.pkl"%(self.bin_size_s,self.bin_step_s)))
    def run(self):

        pca_timeseries = np.load(self.input()['pca'].path)
        time_bins = np.arange(0,pca_timeseries.shape[1])*self.bin_step_s
        playbacks_and_timings = pandas.read_pickle(self.input()['playback_info'].path)

        # Now label each bin as rewarded or unrewarded
        #bin_labels = np.zeros(len(time_bins),dtype=np.uint8)
        
        paths = dict({
            "Stim" : [],
            "pc_path": [],
            "Category": [],
            "Bird": [],
            "Trial":[],
            "Interrupted":[],
            }
        )
        for ix,pb_row in playbacks_and_timings.iterrows():
            # TODO first pass just doing duration plus 1s for reward 
            if is_operant(site_dir = self.site_dir):
                if pb_row.condition_name == 'Rewarded' and pb_row.response == False:
                    post_pb_inds = np.logical_and(time_bins > pb_row.start_time + pb_row.duration,time_bins < pb_row.start_time + pb_row.duration + 1.0)
                    #bin_labels[post_pb_inds] = 3
            if ix+1 < len(playbacks_and_timings):
                next_pb = playbacks_and_timings.iloc[ix+1]
                pb_end_time = min(pb_row.start_time + pb_row.duration + 2, next_pb.start_time-.05)
            else:
                pb_end_time = pb_row.start_time + pb_row.duration + 2
            pb_inds = np.logical_and( (time_bins+self.bin_size_s) > pb_row.start_time,time_bins < pb_end_time)
            #bin_labels[pb_inds] = 1 if pb_row.condition_name == 'Rewarded' else 2
            paths['Stim'].append(pb_row['name'])
            paths['Category'].append(pb_row.condition_name)
            paths['Bird'].append(pb_row.bird_name)
            paths['Trial'].append(pb_row.trial)
            paths["Interrupted"].append(pb_row.response)
            paths['pc_path'].append(pca_timeseries[:,pb_inds])


        # Make a dataframe of the pc paths
        df_pc_paths = pandas.DataFrame(paths)
        self.output().makedirs()
        df_pc_paths.to_pickle(self.output().path)

class PlotStimPopulationPCA(AnalysisTask):
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return {
            'pca': GenPlaybackPCADB(site_dir = self.site_dir,bin_size_s=self.bin_size_s,bin_step_s=self.bin_step_s),
            'spect': GenSpectrogramDB(site_dir = self.site_dir)
        }
    def output(self):
        # This is a directory that will contain all the PC Plots
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","pcplots_%.2f_step%.2f"%(self.bin_size_s,self.bin_step_s),"per_stim"))
    def run(self):
        labels = dict({
            "Unrewarded": "#a40e4c",
            "Rewarded": "#64f58d"
        })
        df_pc_paths = pandas.read_pickle(self.input()['pca'].path)
        df_song_spect = pandas.read_pickle(self.input()['spect'].path)

        for bird in np.unique(df_pc_paths.Bird):
            bird_df = df_pc_paths[df_pc_paths.Bird == bird]
            for stim in np.unique(bird_df.Stim):
                rend = stim.split('_')[2]
                stim_df = bird_df[bird_df.Stim == stim]
                # Plot pcs in a heatmap type layout
                # Second plot PC components stacked on a time axis
                out_fn = "%s_%s_%s.png"%(stim_df.Category.iloc[0],bird,rend)
                
                # Plot Spectrogram, with all PCs below it
                gs = gridspec.GridSpec(4,2,width_ratios=[10,1])
                gs.update(hspace=.3,wspace=.3)
                title = "PC Heatmap"
                fig = plt.figure(title)
                fig.patch.set_facecolor('white')
                maxlen = max([x.shape[1] for x in stim_df.pc_path])
                
                
                # PC0 Axis
                ax = plt.subplot(gs[1,0])
                ax.set_xlim(0,maxlen * self.bin_step_s)

                ax1 = plt.subplot(gs[2,0],sharex=ax)
                ax2 = plt.subplot(gs[3,0],sharex=ax)

                bx = plt.subplot(gs[1,1])
                bx1 = plt.subplot(gs[2,1])
                bx2 = plt.subplot(gs[3,1])

                # Spectrogram Axis
                spec_ax = plt.subplot(gs[0,0],sharex=ax)
                f,t,Sxx = df_song_spect[df_song_spect.Name == stim].Spectrogram.iloc[0]
                spec_ax.pcolormesh(t, f, Sxx, shading='gouraud')
                spec_ax.set_ylabel('Frequency [Hz]')
                #axu.set_xlabel('Time [sec]')
                spec_ax.set_title(title)

                cmap = cm.get_cmap('cool')
                #npcs = stim_df.pc_path.iloc[0].shape[0]
                # Only looking at top 3 pcs
                n_pcs = 3
                pc_im = np.zeros((n_pcs,len(stim_df),maxlen))
                pc_im.fill(np.nan)
                t_val = np.arange(maxlen) * self.bin_step_s
                i=0
                for ix,row in stim_df.iterrows():
                    for pc in range(n_pcs):
                        n_tval = len(row.pc_path[pc,:])
                        pc_im[pc,i,:n_tval] = row.pc_path[pc,:]
                    i+=1
                ax.imshow(pc_im[0,:,:],extent=(0,maxlen * self.bin_step_s,len(stim_df),0),aspect='auto')
                ax1.imshow(pc_im[1,:,:],extent=(0,maxlen * self.bin_step_s,len(stim_df),0),aspect='auto')
                ax2.imshow(pc_im[2,:,:],extent=(0,maxlen * self.bin_step_s,len(stim_df),0),aspect='auto')

                out_root = self.output().path
                out_dir = os.path.join(out_root,"%s_%s"%(stim_df.Category.iloc[0],bird))
                os.makedirs(out_dir,exist_ok=True)
                plt.savefig(os.path.join(out_dir,out_fn))
                lt.close()

class PlotBirdPopulationPCA(AnalysisTask):
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return {
            'pca': GenPlaybackPCADB(site_dir = self.site_dir,bin_size_s=self.bin_size_s,bin_step_s=self.bin_step_s),
            'spect': GenSpectrogramDB(site_dir = self.site_dir)
        }
    def output(self):
        # This is a directory that will contain all the PC Plots
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","pcplots_%.2f_step%.2f"%(self.bin_size_s,self.bin_step_s),"per_bird"))
    def run(self):
        labels = dict({
            "Unrewarded": "#a40e4c",
            "Rewarded": "#64f58d"
        })
        df_pc_paths = pandas.read_pickle(self.input()['pca'].path)
        df_song_spect = pandas.read_pickle(self.input()['spect'].path)

        for bird in np.unique(df_pc_paths.Bird):
            bird_df = df_pc_paths[df_pc_paths.Bird == bird]
            # Plot pcs in a heatmap type layout
            # Second plot PC components stacked on a time axis
            out_fn = "%s_%s.png"%(bird_df.Category.iloc[0],bird)
            
            # Plot Spectrogram, with all PCs below it
            gs = gridspec.GridSpec(3,2,width_ratios=[10,1])
            gs.update(hspace=.3,wspace=.3)
            title = "PC Heatmap"
            fig = plt.figure(title,figsize=(5,12))
            fig.patch.set_facecolor('white')
            maxlen = max([x.shape[1] for x in bird_df.pc_path])
            
            
            # PC0 Axis
            ax = plt.subplot(gs[0,0])
            ax.set_xlim(0,maxlen * self.bin_step_s)

            ax1 = plt.subplot(gs[1,0],sharex=ax)
            ax2 = plt.subplot(gs[2,0],sharex=ax)

            n_pcs = 3
            pc_im = np.zeros((n_pcs,len(bird_df),maxlen))
            pc_im.fill(np.nan)
            t_val = np.arange(maxlen) * self.bin_step_s
            i=0
            for ix,row in bird_df.iterrows():
                for pc in range(n_pcs):
                    n_tval = len(row.pc_path[pc,:])
                    pc_im[pc,i,:n_tval] = row.pc_path[pc,:]
                i+=1
            ax.imshow(pc_im[0,:,:],extent=(0,maxlen * self.bin_step_s,len(bird_df),0),aspect='auto')
            #ax.invert_yaxis()
            ax1.imshow(pc_im[1,:,:],extent=(0,maxlen * self.bin_step_s,len(bird_df),0),aspect='auto')
            #ax1.invert_yaxis()
            ax2.imshow(pc_im[2,:,:],extent=(0,maxlen * self.bin_step_s,len(bird_df),0),aspect='auto')
            #ax2.invert_yaxis()

            out_dir = self.output().path
            os.makedirs(out_dir,exist_ok=True)
            plt.savefig(os.path.join(out_dir,out_fn))
            plt.close()


class Plot2dPCScatter(AnalysisTask):
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return {
            'pca': GenPlaybackPCADB(site_dir = self.site_dir,bin_size_s=self.bin_size_s,bin_step_s=self.bin_step_s)
        }
    def output(self):
        # This is a directory that will contain all the PC Plots
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","pcplots_%.2f_step%.2f"%(self.bin_size_s,self.bin_step_s),"scatter"))
    def run(self):
        labels = dict({
            "Unrewarded": "#a40e4c",
            "Rewarded": "#64f58d"
        })
        df_pc_paths = pandas.read_pickle(self.input()['pca'].path)

        for bird in np.unique(df_pc_paths.Bird):
            bird_df = df_pc_paths[df_pc_paths.Bird == bird]
            # Plot pcs in a heatmap type layout
            # Second plot PC components stacked on a time axis
            out_fn = "%s_%s.png"%(bird_df.Category.iloc[0],bird)
            
            # Plot Spectrogram, with all PCs below it
            gs = gridspec.GridSpec(3,2,width_ratios=[10,1])
            gs.update(hspace=.3,wspace=.3)
            title = "PC Heatmap"
            fig = plt.figure(title,figsize=(5,12))
            fig.patch.set_facecolor('white')
            maxlen = max([x.shape[1] for x in bird_df.pc_path])
            
            
            # PC0 Axis
            ax = plt.subplot(gs[0,0])
            ax.set_xlim(0,maxlen * self.bin_step_s)

            ax1 = plt.subplot(gs[1,0],sharex=ax)
            ax2 = plt.subplot(gs[2,0],sharex=ax)

            n_pcs = 3
            pc_im = np.zeros((n_pcs,len(bird_df),maxlen))
            pc_im.fill(np.nan)
            t_val = np.arange(maxlen) * self.bin_step_s
            i=0
            for ix,row in bird_df.iterrows():
                for pc in range(n_pcs):
                    n_tval = len(row.pc_path[pc,:])
                    pc_im[pc,i,:n_tval] = row.pc_path[pc,:]
                i+=1
            ax.imshow(pc_im[0,:,:],extent=(0,maxlen * self.bin_step_s,len(bird_df),0),aspect='auto')
            #ax.invert_yaxis()
            ax1.imshow(pc_im[1,:,:],extent=(0,maxlen * self.bin_step_s,len(bird_df),0),aspect='auto')
            #ax1.invert_yaxis()
            ax2.imshow(pc_im[2,:,:],extent=(0,maxlen * self.bin_step_s,len(bird_df),0),aspect='auto')
            #ax2.invert_yaxis()

            out_dir = self.output().path
            os.makedirs(out_dir,exist_ok=True)
            plt.savefig(os.path.join(out_dir,out_fn))
            plt.close()
# endregion

# region PostStim Plots
class GenPostStimDB(AnalysisTask):
    
    cluster = luigi.DictParameter()
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the stimuli, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=.5, description="Size, in seconds, of the post stim window. Default is .5s")
    def requires(self):
        return {
            'spikes': self.get_ks_task(),
            'playback_info': GenStimInfoDB(site_dir = self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","Cluster_%d"%self.cluster['cluster_id'],"dfPostStim_start%.2f_size%.2f.pkl"%(self.window_start,self.window_size)))
    def run(self):
         # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input()['spikes'].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')

        playbacks_and_timings = pandas.read_pickle(self.input()['playback_info'].path)
        
        # TODO get samplerate from project config?
        cluster_spike_times = ks_spike_times[ks_spike_cluster==self.cluster['cluster_id']] / 30000 # in seconds

         # find all spikes from -.25 s to .25 s after a stimuli
        window_times = []

        no_response_df = playbacks_and_timings[playbacks_and_timings.response == False]
        for _,trial_row in no_response_df.iterrows():
            window_st = trial_row.start_time + trial_row.duration + self.window_start
            window_end = window_st + self.window_size
            window_times.append([window_st,window_end,trial_row.condition_name,trial_row.bird_name])
        
        # now sound_times is a list of tuples of start and stop of playback sounds. I can check for all spikes how many spikes ar ein sound times and silent times
        spike_dict = dict({
            "Trials":[],
            "NSpikes":[],
            "Duration":[],
            "LogRatio":[],
            "Category":[],
            "Bird":[]
        })
        for start, stop, trial_label, bird in window_times:
            spike_inds = np.logical_and(cluster_spike_times > start,cluster_spike_times < stop)
            spike_dict['Trials'].append(cluster_spike_times[spike_inds] - start + self.window_start)
            spike_dict['LogRatio'].append(np.log10((spike_dict['Trials'][-1] < 0).sum() / (spike_dict['Trials'][-1] > 0).sum() ))
            spike_dict['NSpikes'].append( spike_inds.sum())
            spike_dict['Duration'].append( stop - start)
            spike_dict['Category'].append(trial_label)
            spike_dict['Bird'].append(bird)
        #for trial_label in ["Rewarded", "Unrewarded"]:
        #    spike_dict[trial_label]['SpikeRate'] = spike_dict[trial_label]['NSpikes'] / spike_dict[trial_label]['Duration']
        spike_df = pandas.DataFrame(spike_dict)    
        self.output().makedirs()
        spike_df.to_pickle(self.output().path)

class PlotPostStimResponses(AnalysisTask):
    
    cluster = luigi.DictParameter()
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the stimuli, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=.5, description="Size, in seconds, of the post stim window. Default is .5s")

    def requires(self):
        return GenPostStimDB(site_dir = self.site_dir, cluster=self.cluster)
    def output(self):
        # TODO I could put a list here of jobs ran
        return {
            "rasters": luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"PostStimRasters_start%.2f_size%.2f.png"%(self.window_start,self.window_size))),
            "ratio": luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"PostStimRatio_start%.2f_size%.2f.png"%(self.window_start,self.window_size))),
            "kde": luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"PostStimKDE_start%.2f_size%.2f.png"%(self.window_start,self.window_size)))
        }
    def run(self):
        spike_df = pandas.read_pickle(self.input().path)    
    
        gs = gridspec.GridSpec(1, 2)
        ax = plt.subplot(gs[0,1])
        _,inv_cat = np.unique(spike_df.Bird, return_inverse=True)
        inv_cat = inv_cat.astype(int)
        cmap = cm.get_cmap('cool')
        ax.eventplot(spike_df[spike_df.Category =='Unrewarded']['Trials'],colors=[cmap(x/4.) for x in inv_cat[spike_df.Category =='Unrewarded']])
        ax.set_title("Unrewarded")

        ax = plt.subplot(gs[0,0])
        ax.eventplot(spike_df[spike_df.Category =='Rewarded']['Trials'],colors=[cmap(x/4.) for x in inv_cat[spike_df.Category =='Rewarded']])
        ax.set_title("Rewarded")
        self.output()['rasters'].makedirs()
        plt.savefig(self.output()['rasters'].path)
        plt.close()

        # plot the KDE for unrewarded and rewarded stims
        plt.figure()
        for trial_label in ['Unrewarded','Rewarded']:
            cat_df = spike_df[spike_df.Category == trial_label]
            for bn in np.unique(cat_df.Bird):
                c_spike_times = np.concatenate(np.asarray(cat_df[cat_df.Bird == bn]['Trials']))
                seaborn.kdeplot(c_spike_times,bw_adjust=.5,label="%s %s"%(trial_label,bn))
        plt.xlim([self.window_start,self.window_start + self.window_size])
        plt.legend()
        self.output()['kde'].makedirs()
        plt.savefig(self.output()['kde'].path)
        plt.close()

        # take spike rate pre and spike rate post and plot a distribution of that quotient
        plt.figure()
        #spikes_ratio = [[(x<0).sum()/(x>0).sum() for x in spike_dict[trial_label]['Trials']] for trial_label in ['Unrewarded','Rewarded']]
        seaborn.histplot(spike_df,x='LogRatio',hue='Category',stat='density')
        #plt.legend()
        self.output()['ratio'].makedirs()
        plt.savefig(self.output()['ratio'].path)
        plt.close()


# endregion

# region PostEvent Tasks


class GenEventSpikeDB(AnalysisTask):
    
    cluster = luigi.DictParameter()
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the feeder onset, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=1.0, description="Size, in seconds, of the post stim window. Default is .5s")

    def requires(self):
        if is_playback(self.site_dir):
            merge_task = GenerateSingleSitePlaybackCSV(site_dir = self.site_dir)
        elif is_operant(self.site_dir):
            merge_task = GenerateSingleSiteOperantCSV(site_dir = self.site_dir)
        else:
            print("dont know what to do with %s"%self.site_dir)

        return {
            "spikes": self.get_ks_task(),
            "mergecsv": merge_task
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","Cluster_%d"%self.cluster['cluster_id'],"dfEventRaster_start%.2f_size%.2f.pkl"%(self.window_start,self.window_size)))
    
    def run(self):
        # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input()['spikes'].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))

        events = pandas.read_csv(self.input()['mergecsv']['events'].path)
        
        # TODO get samplerate from project config?
        cluster_spike_times = ks_spike_times[ks_spike_cluster==self.cluster['cluster_id']] / 30000 # in seconds

        event_dict = dict({
            "Event": [],
            "Spikes": [],
            "EventTime": []
        })
        for _, event_row in events.iterrows():
            window_st = event_row.start_time + self.window_start
            window_end = window_st + self.window_size
            spike_inds = np.logical_and(cluster_spike_times > window_st, cluster_spike_times < window_end)
            event_dict['Event'].append(event_row.event_name)
            event_dict['Spikes'].append(cluster_spike_times[spike_inds] - event_row.start_time)
            event_dict['EventTime'].append(event_row.start_time)

        event_df = pandas.DataFrame(event_dict)    
        self.output().makedirs()
        event_df.to_pickle(self.output().path)

class PlotEventAlignedRaster(AnalysisTask):
    
    cluster = luigi.DictParameter()
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the feeder onset, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=1.0, description="Size, in seconds, of the post stim window. Default is .5s")
    def requires(self):
        return GenEventSpikeDB(site_dir = self.site_dir,cluster=self.cluster,window_start=self.window_start,window_size=self.window_size)
    def output(self):
        return {
            "rasters": luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"EventRasters_start%.2f_size%.2f.png"%(self.window_start,self.window_size))),
        }
    def run(self):
        plt.figure()
        event_df = pandas.read_pickle(self.input().path)    
        event_names = np.unique(event_df.Event)

        gs = gridspec.GridSpec(1, len(event_names))
        for i, event_name in enumerate(event_names):
            ax = plt.subplot(gs[0,i])
            ax.eventplot(event_df.Spikes[event_df.Event == event_name])
            ax.set_title(event_name)
        plt.savefig(self.output()['rasters'].path)
        plt.close()

class GenEventPCADB(AnalysisTask):
    
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the feeder onset, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=1.0, description="Size, in seconds, of the post stim window. Default is .5s")
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)

    def requires(self):
        if is_playback(self.site_dir):
            merge_task = GenerateSingleSitePlaybackCSV(site_dir = self.site_dir)
        elif is_operant(self.site_dir):
            merge_task = GenerateSingleSiteOperantCSV(site_dir = self.site_dir)
        else:
            print("dont know what to do with %s"%self.site_dir)

        return {
            'pca': GenPopulationPCA(site_dir = self.site_dir, bin_size_s =self.bin_size_s, bin_step_s=self.bin_step_s),
            'mergecsv': merge_task
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","dfEventPCA_start%.2f_size%.2f_bsize%.2f_bstep%.2f.pkl"%(self.window_start,self.window_size,self.bin_size_s,self.bin_step_s)))
    
    def run(self):
        events = pandas.read_csv(self.input()['mergecsv']['events'].path)
        pca_timeseries = np.load(self.input()['pca'].path)
        time_bins = np.arange(0,pca_timeseries.shape[1])*self.bin_step_s

        event_dict = dict({
            "Event": [],
            "PCA": [],
            "EventTime": []
        })
        for _, event_row in events.iterrows():
            window_st = event_row.start_time + self.window_start
            window_end = window_st + self.window_size
            spike_inds = np.logical_and(time_bins > window_st, time_bins < window_end)
            event_dict['Event'].append(event_row.event_name)
            event_dict['PCA'].append(pca_timeseries[:,spike_inds])
            event_dict['EventTime'].append(event_row.start_time)

        event_df = pandas.DataFrame(event_dict)    
        self.output().makedirs()
        event_df.to_pickle(self.output().path)
class PlotEventAlignedPCA(AnalysisTask):
    
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the feeder onset, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=1.0, description="Size, in seconds, of the post stim window. Default is .5s")
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return GenEventPCADB(site_dir = self.site_dir,window_start=self.window_start,window_size=self.window_size,
                             bin_size_s=self.bin_size_s,bin_step_s=self.bin_step_s)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","EventPCA_start%.2f_size%.2f_bsize%.2f_bstep%.2f.png"%(self.window_start,self.window_size, self.bin_size_s, self.bin_step_s)))
    
    def run(self):
        event_df = pandas.read_pickle(self.input().path)    
        event_names = np.unique(event_df.Event)

        gs = gridspec.GridSpec(3, len(event_names))
        title = "PC Heatmap"
        fig = plt.figure(title,figsize=(4,8))
        fig.patch.set_facecolor('white')

        for i, event_name in enumerate(event_names):
            pc_paths = event_df.PCA[event_df.Event == event_name]
            ax = plt.subplot(gs[0,i])
            ax.set_title(event_name)
        
            maxlen = max([x.shape[1] for x in pc_paths])
            
            ax.set_xlim(0,maxlen * self.bin_step_s)

            ax1 = plt.subplot(gs[1,i],sharex=ax)
            ax2 = plt.subplot(gs[2,i],sharex=ax)

            n_pcs = 3
            pc_im = np.zeros((n_pcs,len(pc_paths),maxlen))
            pc_im.fill(np.nan)
            t_val = np.arange(maxlen) * self.bin_step_s
            i=0
            for pc_path in pc_paths:
                for pc in range(n_pcs):
                    n_tval = len(pc_path[pc,:])
                    pc_im[pc,i,:n_tval] = pc_path[pc,:]
                i+=1
            ax.imshow(pc_im[0,:,:],extent=(0,maxlen * self.bin_step_s,0,len(pc_paths)),aspect='auto')
            ax1.imshow(pc_im[1,:,:],extent=(0,maxlen * self.bin_step_s,0,len(pc_paths)),aspect='auto')
            ax2.imshow(pc_im[2,:,:],extent=(0,maxlen * self.bin_step_s,0,len(pc_paths)),aspect='auto')
        plt.savefig(self.output().path)
        plt.close()
# endregion

# region Decoding

class GenClusterTimePCDB(AnalysisTask):
    
    def requires(self):
        return self.get_ks_task()
    def output(self):
        pass
    def run(self):
        # TODO COpy from here https://github.com/theunissenlab/jupyterNotebook/blob/master/notebooks/fet/chronic/GenerateDataBase.ipynb
        # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input().path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')


        cluster_ids_good = []
        for ix,cluster in ks_cluster_info.iterrows():
            print(cluster)
            if cluster['group'] != 'noise':#== 'good' and cluster.fr > 1.0:#  and cluster.fr > 5:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
                cluster_ids_good.append(cluster.cluster_id)
        cluster_ids_good = np.asarray(cluster_ids_good)
        # take all responses to all stimuli and generate PCs
        # TODO could get this from project or site config or something
        t_start = np.floor(ks_spike_times.min()/30000)
        t_stop = np.ceil(ks_spike_times.max()/30000)
        tot_time = t_start - t_stop

        # T x Cluster
        spike_arr_30k = np.zeros((int(t_stop*30000),len(cluster_ids_good)),dtype=np.uint8)
        for ix,p in enumerate(cluster_ids_good):
            spike_arr_30k[:,ix][ks_spike_times[ks_spike_cluster==p]] = 1
        
        # now we window it and sum for each cluster 
        # TODO get samplerate from some config
        pop_timeseries = np.lib.stride_tricks.sliding_window_view(spike_arr_30k, int(np.round(self.bin_size_s*30000)), axis=0)[::int(np.round(self.bin_step_s*30000))].sum(axis=2)
        # Now z score the pop vector by cluster
        pop_timeseries = stats.zscore(pop_timeseries)
        #time_bins = np.arange(0,len(pop_timeseries))*bin_step_s
        # do pca on the 
        print("Running PCA on the zscored bins")
        pca = PCA(n_components=min(5,len(cluster_ids_good))) 
        # pca_timeseries is <NPC>x<NTime>
        pca_timeseries = pca.fit_transform(pop_timeseries).T

        self.output().makedirs()
        np.save(self.output().path,pca_timeseries)
        pass

def _decode_labels(x,y,test_size,return_prediction=False):
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=test_size)
    # # remove rare inputs
    # X_train_inds = np.where(X_train.sum(axis=0)>0)[0]
    # X_train = X_train[:,X_train_inds]
    # X_test = X_test[:,X_train_inds]
    # zscore inputs 
    #X_mean=np.nanmean(X_train,axis=0)
    #X_std=np.nanstd(X_train,axis=0)
    #X_train =(X_train-X_mean)/X_std
    #X_test=(X_test-X_mean)/X_std

    gnb = GaussianNB()
    Y_pred = gnb.fit(X_train,Y_train).predict(X_test)
    perf = (Y_test == Y_pred).sum() / X_test.shape[0]
    return perf

class DecodeLabelsNB(AnalysisTask):
    
    bin_size_s = luigi.FloatParameter(default=.01)
    test_size = luigi.FloatParameter(default=.25)
    def requires(self):
        return {
            "spikes": self.get_ks_task(),
            "behavior": GenBehaviorDB(site_dir = self.site_dir)
        }
    def output(self):
        return {
            "ensemble":luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","dfPerf_zscore_%.3f_%.2f.pkl"%(self.bin_size_s,self.test_size))),
            "clusters":luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","dfClusterPerf_zscore_%.3f_%.2f.pkl"%(self.bin_size_s,self.test_size))),
        }
    
    def run(self):

        # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input()["spikes"].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')

        cluster_ids_good = []
        for ix,cluster in ks_cluster_info.iterrows():
            # TODO centralize this
            if cluster['group'] != 'noise':# and cluster.fr > 1.0:#  and cluster.fr > 5:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
                cluster_ids_good.append(cluster.cluster_id)
        cluster_ids_good = np.asarray(cluster_ids_good)

        # T x Cluster zscored firing rate
        # TODO Sample rate
        fs = 30000
        t_start = 0#np.floor(ks_spike_times.min()/fs)
        t_stop = np.ceil(ks_spike_times.max()/fs)
        edges = np.arange(t_start,t_stop*fs,int(self.bin_size_s*fs))
        num_bins = edges.shape[0]-1
        num_neurons = len(cluster_ids_good)
        neural_data=np.empty([num_bins,num_neurons])
        for ix,p in enumerate(cluster_ids_good):
            neural_data[:,ix] = zscore(np.histogram(ks_spike_times[ks_spike_cluster==p],edges)[0])


        # now we have binned neural data at bin_size_s bins
        # lets make an array of onset responses
        playback_and_timings = pandas.read_pickle(self.input()["behavior"].path)
        # half sec of the start of each stimuli
        bins_after_onset = int(.5 / self.bin_size_s)
        onset_times = (playback_and_timings.start_time/self.bin_size_s).astype(np.int)
        X_windows = np.lib.stride_tricks.sliding_window_view(neural_data, bins_after_onset, axis=0)[onset_times]
        # reshape it into a flat format
        X = X_windows.reshape(X_windows.shape[0],X_windows.shape[1]*X_windows.shape[2])

        cluster_df = dict({
            "Performance": [],
            "Random_mean": [],
            "Random_Std" :[],
            "Label": [],
            "Cluster": []
        })
        ensemble_df =dict({
            "Performance": [],
            "Random_mean": [],
            "Random_Std" :[],
            "Label": []
        })

        # Now test prediction performance for Category, Stim_name, Bird_Name, Novel
        for label in ['condition_name','bird_name','name','novel','response']:
            Y = np.asarray(playback_and_timings[label])
            # first evaluate the performance of each cluster
            for ix,clust_id in enumerate(cluster_ids_good):
                clust_perf = _decode_labels(X_windows[:,ix,:],Y,test_size=self.test_size)
                cluster_df['Performance'].append(clust_perf)
                cluster_df['Label'].append(label)
                cluster_df['Cluster'].append(clust_id)

            # then evaluate the performance of all good clusters together
            perf = _decode_labels(X,Y,test_size=self.test_size)
            ensemble_df["Performance"].append(perf)


            # bootstrap a conf interval
            rand_perf = []
            clust_rand_perfs = dict((cid,[])for cid in cluster_ids_good)
            for i in range(500):
                # shuffle the labels
                np.random.shuffle(Y)

                # evaluate ensemble performance
                rand_perf.append(_decode_labels(X,Y,test_size=self.test_size))

                # now eval for each cluster
                for ix,clust_id in enumerate(cluster_ids_good):
                    clust_rand_perfs[clust_id].append(_decode_labels(X_windows[:,ix,:],Y,test_size=self.test_size))
                
            ensemble_df["Random_mean"].append(np.mean(rand_perf))
            ensemble_df["Random_Std"].append(np.std(rand_perf))
            ensemble_df['Label'].append(label)

            for ix,clust_id in enumerate(cluster_ids_good):
                cluster_df["Random_mean"].append(np.mean(clust_rand_perfs[clust_id]))
                cluster_df["Random_Std"].append(np.std(clust_rand_perfs[clust_id]))

        perf_df = pandas.DataFrame(ensemble_df)
        self.output()['ensemble'].makedirs()
        perf_df.to_pickle(self.output()['ensemble'].path)

        # now save cluster df
        cluster_perf_df = pandas.DataFrame(cluster_df)
        self.output()['clusters'].makedirs()
        cluster_perf_df.to_pickle(self.output()['clusters'].path)

class GenerateClusterDecoderResults(AnalysisTask):
    
    cluster = luigi.DictParameter()
    bin_size_s = luigi.FloatParameter(default=.01)

    def requires(self):
        return {
            "spikes": self.get_ks_task(),
            "behavior": GenBehaviorDB(site_dir = self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","Cluster_%d"%self.cluster['cluster_id'],"dfBehaviorWithPrediction_bs%.2f.pkl"%self.bin_size_s))
    def run(self):
        # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input()["spikes"].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))

        # T x Cluster
        # TODO Firing rate
        fs = 30000
        t_start = 0#np.floor(ks_spike_times.min()/fs)
        t_stop = np.ceil(ks_spike_times.max()/fs)
        edges = np.arange(t_start,t_stop*fs,int(self.bin_size_s*fs))
        neural_data = zscore(np.histogram(ks_spike_times[ks_spike_cluster==self.cluster['cluster_id']],edges)[0])


        # now we have binned neural data at bin_size_s bins
        # lets make an array of onset responses
        playback_and_timings = pandas.read_pickle(self.input()["behavior"].path)
        # half sec of the start of each stimuli
        bins_after_onset = int(.5 / self.bin_size_s)
        onset_times = (playback_and_timings.start_time/self.bin_size_s).astype(np.int)
        X = np.lib.stride_tricks.sliding_window_view(neural_data, bins_after_onset, axis=0)[onset_times]

        # Now test prediction performance for Category, Stim_name, Bird_Name, Novel
        for label in ['condition_name','bird_name','name','novel','response']:
            Y = np.asarray(playback_and_timings[label])
            
            gnb = GaussianNB()
            Y_pred = gnb.fit(X,Y).predict(X)
            playback_and_timings["%s_pred"%label] = Y_pred
        playback_and_timings.to_pickle(self.output().path)

class PlotDecoderResults(AnalysisTask):
    
    bin_size_s = luigi.FloatParameter(default=.01)

    def requires(self):
        return DecodeLabelsNB(site_dir=self.site_dir, bin_size_s=self.bin_size_s)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","DecoderResults_bs%.3f.png"%self.bin_size_s))
    def run(self):
        dfDecoder= pandas.read_pickle(self.input()["ensemble"].path)
        dfClusterDecoder = pandas.read_pickle(self.input()["clusters"].path)

        # make a box plot comparing each predicition to the random distribution
        plt.figure()
        plt.title("Decoder Performance")
        plt.scatter(x=dfClusterDecoder['Label'], y=dfClusterDecoder['Performance'],c=dfClusterDecoder['Cluster'],s=25)
        plt.scatter(x=dfDecoder['Label'], y=dfDecoder['Performance'],c='r',s=50)
        plt.errorbar(x=dfDecoder['Label'], y=dfDecoder['Random_mean'],yerr=dfDecoder['Random_Std'],c='green',fmt='none')
        self.output().makedirs()
        plt.savefig(self.output().path)
        plt.close()

class PlotClusterDecoderResults(AnalysisTask):
    
    cluster = luigi.DictParameter()
    bin_size_s = luigi.FloatParameter(default=.01)
    def requires(self):
        return GenerateClusterDecoderResults(site_dir=self.site_dir,cluster=self.cluster,bin_size_s=self.bin_size_s)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"PredictionsvsBehavior_bs%.2f"%self.bin_size_s))
    def run(self):
        window = 5
        dfBehaviorWithPrediction = pandas.read_pickle(self.input().path)
        for stim in np.unique(dfBehaviorWithPrediction['bird_name'].values):
            plt.figure()

            stim_df = dfBehaviorWithPrediction[dfBehaviorWithPrediction['bird_name']==stim]
            plt.title("%s %s %s"%(stim,stim_df['novel'].iloc[0],stim_df['condition_name'].iloc[0]))
            correct = np.convolve(stim_df['correct'], np.ones((window,))/window, mode='valid')
            #ax[0].plot(range(len(correct)),correct)

            arr_im = np.zeros((5,len(correct)))
            arr_im[0,:] = correct

            # now look at how many the decoder got right
            labels = ['novel','bird_name','condition_name','response']
            for ix, label in enumerate(labels):
                decoder_correct = stim_df[label] == stim_df['%s_pred'%label]
                decoder_correct = np.convolve(decoder_correct, np.ones((window,))/window, mode='valid')
                arr_im[ix+1,:] = decoder_correct
            plt.imshow(arr_im,aspect='auto',interpolation='none')
            plt.yticks(range(len(labels)+1),['correct'] + labels)
            plt.colorbar()
            os.makedirs(self.output().path,exist_ok=True)
            plt.savefig(os.path.join(self.output().path,"%s_%s_%s.png"%(stim,stim_df['novel'].iloc[0],stim_df['condition_name'].iloc[0])))
            plt.close()
# endregion

class GenerateBinnedStimResponse(AnalysisTask):
    
    bin_size_s = luigi.FloatParameter(default=.05)
    def requires(self):
        return {
            "spikes": self.get_ks_task(),
            "behavior": GenBehaviorDB(site_dir = self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","binned_spikes_%.2f.npy"%self.bin_size_s))
    def run(self):
        pass


# region Behavior Plotting


class PlotBehaviorTrials(AnalysisTask):
    
    def requires(self):
        return GenBehaviorDB(site_dir = self.site_dir)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","Behavior_Trials.png"))
    def run(self):
        playback_and_timings = pandas.read_pickle(self.input().path)
        #playback_and_timings = playback_and_timings[playback_and_timings.novel == 'Novel']
        plot_pecking_test_data(playback_and_timings,['bird_name','condition_name','novel'])
        self.output().makedirs()
        plt.savefig(self.output().path)
        plt.close()
        



#endregion

# region Coherence
from soundsig.coherence import multitapered_coherence


class GenPlaybackOperantCoherence(AnalysisTask):
    """Generate a coherence plot for each cluster"""
    
    cluster = luigi.DictParameter()
    operant_site = SiteParameter()
    operant_cluster = luigi.DictParameter()
    window_size = luigi.IntParameter(default=1024)
    min_trials = luigi.IntParameter(default=4)
    bSmooth = luigi.BoolParameter(default=False)
    def requires(self):
        return {
            "playback": GenPlaybackSpikesDB(site_dir=self.site_dir,cluster=self.cluster),
            "operant": GenPlaybackSpikesDB(site_dir=self.operant_site,cluster=self.operant_cluster)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"coherence_pbvsop_%d_%s.pkl"%(self.window_size,"smooth" if self.bSmooth else "hist")))
    def run(self):
        # load the playback spikes
        pb_spike_df = pandas.read_pickle(self.input()['playback'].path)
        op_spike_df = pandas.read_pickle(self.input()["operant"].path)
        
        # filter pb to just be the stims that operant also had
        pb_spike_df = pb_spike_df[pb_spike_df.Rendition_Name.isin(op_spike_df.Rendition_Name)]
        # go through each stim
        # TODO: can group here by rewarded and not rewarded etc... for now lets just do all stims
        # sub_df = pb_spike_df[pb_spike_df.Rewarded == 1] or something
        output_dict = dict()
        for condition, condition_df in [('Playback',pb_spike_df),('Operant',op_spike_df)]:
            categories = {
                "Rewarded": (condition_df["Category"] == "Rewarded"),
                "Unrewarded": (condition_df["Category"] == "Unrewarded"),
                "All": np.ones(len(condition_df)).astype(np.bool)
            }
            output_dict[condition] = dict()
            for category_name, category_selector in categories.items():
                category_df = condition_df[category_selector]
                X = []
                for stim_name,stim_df in category_df.groupby("Rendition_Name"):
                    nTrials = len(stim_df)
                    if nTrials < self.min_trials:
                        continue

                    # Get actual response using a weighted sum of gaussians
                    spike_times = stim_df.Spike_Times - stim_df.Start
                    peck_times = stim_df.Peck.values.astype(float)
                    peck_times[peck_times == 0.0] = np.nan

                    #split them into two groups, even trials and odd trials
                    even_trials = np.arange(0,nTrials,2)
                    odd_trials = np.arange(1,nTrials,2)

                    concatenated_spike_times_even = np.concatenate(np.asarray(spike_times.iloc[even_trials]))
                    peck_times_even = peck_times[even_trials]
                    concatenated_spike_times_odd = np.concatenate(np.asarray(spike_times.iloc[odd_trials]))
                    peck_times_odd = peck_times[odd_trials]
                    weights_even = np.array([1/(len(peck_times_even) - sum(peck_times_even < x)) for x in concatenated_spike_times_even])
                    weights_odd = np.array([1/(len(peck_times_odd) - sum(peck_times_odd < x)) for x in concatenated_spike_times_odd])
                    x_d = np.linspace(0,6,6001)#1ms bins
                    if self.bSmooth:
                        resp_even = sum(norm(xi,scale=.008).pdf(x_d)*w for xi,w in zip(concatenated_spike_times_even,weights_even))
                        resp_odd = sum(norm(xi,scale=.008).pdf(x_d)*w for xi,w in zip(concatenated_spike_times_odd,weights_odd))
                    else:
                        resp_even = np.histogram(concatenated_spike_times_even,bins=x_d,weights=weights_even)[0]
                        resp_odd = np.histogram(concatenated_spike_times_odd,bins=x_d,weights=weights_odd)[0]
                    # TODO truncate by max peck_times
                    if np.all(resp_even == 0) or np.all(resp_odd == 0):
                        print("Stim: %s bad for %s"%(stim_name,category_name))
                    X.append(np.array([resp_even,resp_odd]))
                coherence_struct = multitapered_coherence(X,sampling_rate=1000,chunk_size=self.window_size)
                output_dict[condition][category_name] = coherence_struct
        self.output().makedirs()
        pickle.dump(output_dict,open(self.output().path,'wb'))

class PlotPlaybackOperantCoherence(AnalysisTask):
    """Generate a coherence plot for each cluster"""
    # site_dir
    cluster = luigi.DictParameter()
    operant_site = SiteParameter()
    operant_cluster = luigi.DictParameter()
    window_size = luigi.IntParameter(default=1024)
    min_trials = luigi.IntParameter(default=4)
    bSmooth = luigi.BoolParameter(default=False)  
    extension = luigi.Parameter(default="png")
    def requires(self):
        return GenPlaybackOperantCoherence(site_dir=self.site_dir,cluster=self.cluster,operant_site=self.operant_site,operant_cluster=self.operant_cluster,window_size=self.window_size,min_trials=self.min_trials,bSmooth=self.bSmooth)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","Cluster_%d"%self.cluster['cluster_id'],"coherence_pbvsop_%d_%s_All.%s"%(self.window_size,"smooth" if self.bSmooth else "hist",self.extension)))
    def run(self):
        coherence_struct = pickle.load(open(self.input().path,'rb'))

        # first plot all
        for condition in ['Playback','Operant']:
            bi = _plot_coherence(coherence_struct[condition]['All'],self.extension)
            print("All: %s %f"%(condition,bi))
        plt.title('Coherence')
        self.output().makedirs()
        plt.savefig(self.output().path)
        plt.close()

class GenOperantCoherence(AnalysisTask):
    """Generate a coherence plot for each cluster"""
    
    cluster = luigi.DictParameter()
    window_size = luigi.IntParameter(default=1024)
    min_trials = luigi.IntParameter(default=4)
    bSmooth = luigi.BoolParameter(default=False)
    def requires(self):
        return GenPlaybackSpikesDB(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","Cluster_%d"%self.cluster['cluster_id'],"coherence_%d_%s.pkl"%(self.window_size,"smooth" if self.bSmooth else "hist")))
    def run(self):
        # load the playback spikes
        pb_spike_df = pandas.read_pickle(self.input().path)

        
        # go through each stim
        # TODO: can group here by rewarded and not rewarded etc... for now lets just do all stims
        # sub_df = pb_spike_df[pb_spike_df.Rewarded == 1] or something
        categories = {
            "Rewarded": (pb_spike_df["Category"] == "Rewarded"),
            "Unrewarded": (pb_spike_df["Category"] == "Unrewarded"),
            "Novel": (pb_spike_df["Novel"] == "Novel") & (pb_spike_df["Informative_Trials"] <= 10),
            "Learned": (pb_spike_df["Novel"] == "Learned") & (pb_spike_df["Informative_Trials"] <= 10),
            "All": np.ones(len(pb_spike_df)).astype(np.bool)
        }
        output_dict = dict()
        for category_name, category_selector in categories.items():
            category_df = pb_spike_df[category_selector]
            X = []
            N = []
            for stim_name,stim_df in category_df.groupby("Rendition_Name"):
                nTrials = len(stim_df)
                if nTrials < self.min_trials:
                    continue

                # Get actual response using a weighted sum of gaussians
                spike_times = stim_df.Spike_Times - stim_df.Start
                peck_times = stim_df.Peck.values.astype(float)
                peck_times[peck_times == 0.0] = np.nan

                #split them into two groups, even trials and odd trials
                even_trials = np.arange(0,nTrials,2)
                odd_trials = np.arange(1,nTrials,2)

                concatenated_spike_times_even = np.concatenate(np.asarray(spike_times.iloc[even_trials]))
                peck_times_even = peck_times[even_trials]
                concatenated_spike_times_odd = np.concatenate(np.asarray(spike_times.iloc[odd_trials]))
                peck_times_odd = peck_times[odd_trials]
                weights_even = np.array([1/(len(peck_times_even) - sum(peck_times_even < x)) for x in concatenated_spike_times_even])
                weights_odd = np.array([1/(len(peck_times_odd) - sum(peck_times_odd < x)) for x in concatenated_spike_times_odd])
                x_d = np.linspace(0,6,6001)#1ms bins
                if self.bSmooth:
                    resp_even = sum(norm(xi,scale=.008).pdf(x_d)*w for xi,w in zip(concatenated_spike_times_even,weights_even))
                    resp_odd = sum(norm(xi,scale=.008).pdf(x_d)*w for xi,w in zip(concatenated_spike_times_odd,weights_odd))
                else:
                    resp_even = np.histogram(concatenated_spike_times_even,bins=x_d,weights=weights_even)[0]
                    resp_odd = np.histogram(concatenated_spike_times_odd,bins=x_d,weights=weights_odd)[0]
                # TODO truncate by max peck_times
                if np.all(resp_even == 0) or np.all(resp_odd == 0):
                    print("Stim: %s bad for %s"%(stim_name,category_name))
                X.append(np.array([resp_even,resp_odd]))

                # now save noise to measure noise distribution
                N.append(resp_even - resp_odd)

            coherence_struct = multitapered_coherence(X,sampling_rate=1000,chunk_size=self.window_size)
            coherence_struct['noise'] = N
            output_dict[category_name] = coherence_struct
        self.output().makedirs()
        pickle.dump(output_dict,open(self.output().path,'wb'))

class GenOperantLFPCoherence(AnalysisTask):
    """Generate a coherence plot for each cluster"""
    
    window_size = luigi.IntParameter(default=1024)
    min_trials = luigi.IntParameter(default=4)
    highpass_freq = luigi.FloatParameter(default=10.0)
    def requires(self):
        return {
            "playbacks": GenBehaviorDB(site_dir=self.site_dir),
            "lfp": GenerateSiteLFP(site_dir=self.site_dir,highpass_freq=self.highpass_freq)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"data","coherence_LFP_%d.pkl"%(self.window_size)))
    def run(self):
        # load the playback spikes
        pb_spike_df = pandas.read_pickle(self.input()["playbacks"].path)
        lfp = np.fromfile(self.input()['lfp'].path, dtype=np.float32).reshape(-1,64)
        
        # go through each stim
        # TODO: can group here by rewarded and not rewarded etc... for now lets just do all stims
        # sub_df = pb_spike_df[pb_spike_df.Rewarded == 1] or something
        categories = {
            "Rewarded": (pb_spike_df["condition_name"] == "Rewarded"),
            "Unrewarded": (pb_spike_df["condition_name"] == "Unrewarded"),
            "All": np.ones(len(pb_spike_df)).astype(np.bool)
        }
        output_dict = dict()
        for category_name, category_selector in categories.items():
            category_df = pb_spike_df[category_selector]
            X = []
            N = []
            for stim_name,trial_df in category_df.groupby("name"):
                nTrials = len(trial_df)
                if nTrials < self.min_trials:
                    continue
                
                t_st =  0
                t_end = 6.0
                time_bins = np.arange(t_st, t_end, .001)
                indiv_lfp = []

                lfp_sum_e = np.zeros((len(time_bins),64))
                lfp_count_e = np.zeros(len(time_bins))
                lfp_sum_o = np.zeros((len(time_bins),64))
                lfp_count_o = np.zeros(len(time_bins))

                # Get actual response using a weighted sum of gaussians
                i = 0
                for ix, row in trial_df.iterrows():
                    st = row.start_time + t_st
                    end = row.start_time + row.duration
                    seg_lfp = lfp[int(st*1000):int(end*1000),:]
                    if i % 2 == 0:
                        lfp_sum_e[:seg_lfp.shape[0],:] += seg_lfp
                        lfp_count_e[:seg_lfp.shape[0]] += 1
                    else:
                        lfp_sum_o[:seg_lfp.shape[0],:] += seg_lfp
                        lfp_count_o[:seg_lfp.shape[0]] += 1
                    i += 1
                zinds = np.logical_or(lfp_count_e == 0, lfp_count_o == 0)
                lfp_sum_e = lfp_sum_e[~zinds,:]
                lfp_count_e = lfp_count_e[~zinds]
                lfp_sum_o = lfp_sum_o[~zinds,:]
                lfp_count_o = lfp_count_o[~zinds]
                resp_even = lfp_sum_e / lfp_count_e[:,None]
                resp_odd = lfp_sum_o / lfp_count_o[:,None]
                X.append(np.array([resp_even,resp_odd]))

                # now save noise to measure noise distribution
                N.append(resp_even - resp_odd)
            coherence_struct = dict()
            for c in np.arange(64):
                X_c = [np.array([x[0,:,c], x[1,:,c]]) for x in X]
                coherence_struct[c] = multitapered_coherence(X_c,sampling_rate=1000,chunk_size=self.window_size)
            coherence_struct['noise'] = N
            output_dict[category_name] = coherence_struct
        self.output().makedirs()
        pickle.dump(output_dict,open(self.output().path,'wb'))

def compute_coherence_mean(x):
    '''
    x: 3dim: 
    '''
    dimx, dimy, dimz = x.shape
    n = dimx*(dimy-1.0)
    return ((np.sum(np.sum(x, 0),0) - np.sum(np.diagonal(x),1))/n)

def compute_bits_info(x, delta):
    '''
    x: positive coherence
    '''
    # using a bound to zero out coherence under 0.05 CI
    x[np.argwhere(x<np.percentile(x, 95))]=0
    return np.sum(-np.log2(1-x)) * delta

def viz(freqs, co_mean, bit, plot_error=False, error=False, output=False, save=False, format='eps'):
    x = np.fft.fftshift(freqs)
    y = np.fft.fftshift(co_mean)
    plt.plot(x, y, "-k")
    xaxis = np.max(freqs)
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    
    if plot_error:
        plt.fill_between(x, np.fft.fftshift(error[0]), np.fft.fftshift(error[1]), alpha=0.5)
    if save:
        plt.savefig(output, format=format)

def viz_wrapper(freqs,
               co_mean,
               bit,
               coherence_bounds,
               output:str=None,
               save:bool=False,
               format:str='eps'):
    error_prep = coherence_bounds
    error_all = np.vstack([compute_coherence_mean(error_prep[i]) for i in range(len(error_prep))])

    viz(freqs, co_mean, bit, plot_error=True, error=error_all, output=output, save=save, format=format)
def _plot_coherence(cstruct,extension):
    coherence_mean = compute_coherence_mean(cstruct["coherence"])
    delta = np.fft.fftshift(cstruct["freqs"])[1]-np.fft.fftshift(cstruct["freqs"])[0]
    bits_info = compute_bits_info(coherence_mean.copy(), delta)
    viz_wrapper(cstruct['freqs'], coherence_mean, bits_info,
                cstruct['coherence_bounds'],format=extension)
    return bits_info
    
class PlotOperantCoherence(AnalysisTask):
    
    cluster = luigi.DictParameter()
    window_size = luigi.IntParameter(default=1024)
    bSmooth = luigi.BoolParameter(default=False)
    extension = luigi.Parameter(default="png")
    def requires(self):
        return GenOperantCoherence(site_dir=self.site_dir,cluster=self.cluster,window_size=self.window_size,bSmooth=self.bSmooth)
    def output(self):
        return dict({
            "All": luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"Coherence","coherence_%d_%s.%s"%(self.window_size,"smooth" if self.bSmooth else "hist",self.extension))),
            "Reward": luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"Coherence","coherence_Reward_%d_%s.%s"%(self.window_size,"smooth" if self.bSmooth else "hist",self.extension))),
            "Novel": luigi.LocalTarget(os.path.join(self.get_analysis_dir(),"figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"Coherence","coherence_Novel_%d_%s.%s"%(self.window_size,"smooth" if self.bSmooth else "hist",self.extension)))
        })
    def run(self):
        coherence_struct = pickle.load(open(self.input().path,'rb'))

        # first plot all
        plt.figure()
        bi = _plot_coherence(coherence_struct['All'],self.extension)
        plt.title('Coherence')
        print("-------")
        print("All: Cluster %d: %f bits"%(self.cluster['cluster_id'],bi))
        self.output()['All'].makedirs()
        plt.savefig(self.output()['All'].path)
        plt.close()

        # now rewarded vs unrewarded
        plt.figure()
        for category_name in ['Rewarded', 'Unrewarded']:
            bi = _plot_coherence(coherence_struct[category_name],self.extension)
            print("%s: Cluster %d: %f bits"%(category_name,self.cluster['cluster_id'],bi))
        plt.title('Rewarded vs Unrewarded')
        self.output()["Reward"].makedirs()
        plt.savefig(self.output()["Reward"].path)
        plt.close()

        # now Novel vs Learned
        plt.figure()
        for category_name in ['Novel', 'Learned']:
            bi = _plot_coherence(coherence_struct[category_name],self.extension)
            print("%s: Cluster %d: %f bits"%(category_name,self.cluster['cluster_id'],bi))
        plt.title('Novel vs Learned')
        self.output()["Novel"].makedirs()
        plt.savefig(self.output()["Novel"].path)
        plt.close()


# endregion

# region Wrapper Tasks 
class SiteFigureGeneration(luigi.WrapperTask):
    site_dir = SiteParameter()
    force = luigi.BoolParameter(default=False)
    def requires(self):
        #yield KiloSort4Sorting(site_dir = self.site_dir)
        ks_output_dir = os.path.dirname(self.get_ks_task().output().path)
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')

        tasks = []
        for ix,cluster in ks_cluster_info.iterrows():
            if cluster['group'] != 'noise':#  and cluster.fr > 1.0:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
                print(cluster)
                tasks.append(PlotClusterAuditorySpikeRates(site_dir=self.site_dir,cluster=dict(cluster)))
                tasks.append(PlotTrialSpikeRasters(site_dir=self.site_dir,cluster=dict(cluster)))
                tasks.append(PlotInterruptionSpikeRasters(site_dir=self.site_dir,cluster=dict(cluster)))
                tasks.append(PlotPostStimResponses(site_dir=self.site_dir,cluster=dict(cluster)))
                if is_operant(self.site_dir):
                    tasks.append(PlotEventAlignedRaster(site_dir =self.site_dir, cluster=dict(cluster)))
                    tasks.append(PlotOperantCoherence(site_dir =self.site_dir, cluster=dict(cluster)))
                tasks.append(PlotClusterSummary(site_dir =self.site_dir, cluster=dict(cluster)))
                tasks.append(PlotSpikeRateByTrial(site_dir =self.site_dir, cluster=dict(cluster)))
                #tasks.append(PlotClusterDecoderResults(site_dir =self.site_dir, cluster=dict(cluster),bin_size_s=.005))
        tasks.append(PlotSiteSummary(site_dir = self.site_dir))
        return tasks
    
    
# endregion