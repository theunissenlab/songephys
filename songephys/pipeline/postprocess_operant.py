import os

import luigi
import pandas
import numpy as np
from scipy.io import wavfile
from scipy import signal,stats
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
from sklearn.decomposition import PCA
import seaborn

from .file_setup import SiteProjectConfig, SiteConfig
from .pipeline_utils import SiteParameter
from .preprocess_sound import MergeOperantCSV
from .preprocess_ephys import KiloSortSorting
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
def _get_spectrogram(arr,fs, db=120, fc=12500):
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


def _plot_trials(spikes,peck_times,title,spect,t_start,t_end,mode='rasters',colors=None):
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
        
        c_spike_times = np.concatenate(np.asarray(c_spikes))
        if len(c_spike_times)>1:
            seaborn.kdeplot(c_spike_times,ax=axb,bw_adjust=.2)


    # Copy legend object from scatter plot to lower left subplot and display:
    # NB 'scatterpoints = 1' customises legend box to show only 1 handle (icon) per label 
    #handles, labels = ax.get_legend_handles_labels()
    lines, labels = axb.get_legend_handles_labels()
    axc.legend(lines, labels, loc = 'center', fontsize = 12)

# endregion
# region Prep Jobs
class GenStimInfoDB(luigi.Task):
    """ Generates database of playbacks.csv with more info for each stim"""
    site_dir = SiteParameter()
    def requires(self):
        return {
            "merge_csv": MergeOperantCSV(self.site_dir),
            "site_config": SiteConfig(self.site_dir),
            "project_config": SiteProjectConfig(self.site_dir)
        }
    
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","raw","dfPlaybackInfo.pkl"))
    
    def run(self):
        # load the project config to get stim_drive location
        project_config = self.input()['project_config'].data()

        # read the locations of the renditions within each stimuli
        timings_df = pandas.read_csv(os.path.join(project_config['LOCAL_RAW_FILES']['stimuli_drive'],"StimInfo.csv"))

        # load playback info from the MergeOperantCSV task
        playback_fn = self.input()['merge_csv']['playbacks'].path
        playbacks = pandas.read_csv(playback_fn)

        # clean the playbacksdb to have the same stim names as the timings_df
        playbacks['name'] = [pb.strip('playback_') for pb in playbacks.name]

        # join the playbacks and timings into one dataframe
        playbacks_and_timings = playbacks.join(timings_df.set_index('Filename'),on='name')

        # save the databse to the output path
        self.output().makedirs()
        playbacks_and_timings.to_pickle(self.output().path)

class GenBehaviorDB(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return GenStimInfoDB(self.site_dir)
    def output(self):
        # TODO I could put a list here of jobs ran
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","dfBehavior.pkl"))
    def run(self):
        playback_and_timings = pandas.read_pickle(self.input().path)

        project_dir = self.site_dir.split("sites")[0]
        novel_stim_db = pandas.read_csv(os.path.join(project_dir,"novel_stims.csv"))

        # go through and inject informative trials
        informative_dfs = []
        playback_and_timings['novel'] = "Learned"
        for bird, birddf in playback_and_timings.groupby('bird_name'):
            didnt_respond_previously = (~birddf.response).shift()
            didnt_respond_previously.iloc[0] = 0
            informative_trials = didnt_respond_previously.cumsum() # the first interrupted should be 0
            informative_trials.name = 'informative_trials'
            informative_dfs.append(informative_trials)
            site = self.site_dir.split('sites/')[-1]
            is_novel = novel_stim_db[(novel_stim_db.site == site)&(novel_stim_db.bird==bird)].novel.iloc[0]
            playback_and_timings.loc[playback_and_timings.bird_name == bird,'novel'] = "Novel" if is_novel else "Learned"
        playback_and_timings = playback_and_timings.join(pandas.concat(informative_dfs))
        
        self.output().makedirs()
        playback_and_timings.to_pickle(self.output().path)

class GenSpectrogramDB(luigi.Task):
    """ Generates database of spectrograms for each stim"""
    site_dir = SiteParameter()
    def requires(self):
        return {
            "project_config": SiteProjectConfig(self.site_dir),
            "playback_info": GenStimInfoDB(self.site_dir)
        }
    
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","raw","dfSongSpect.pkl"))
    
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
class GenClusterAuditorySpikeRatesDB(luigi.Task):
    """ OUTPUTS: 
            data/<CLUSTER>/dfAuditorySpikes.pkl
    """
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()

    def requires(self):
        return {
            "project_config": SiteProjectConfig(self.site_dir),
            "playback_info": GenBehaviorDB(self.site_dir),
            "kilosort": KiloSortSorting(self.site_dir)
        }
    
    def output(self):
        # TODO could break this into playbacks and silence dfs, and do a bunch of other stuff with the playbacks
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster_id,"dfAuditorySpikes.pkl"))
    
    def run(self):
        ks_output_dir =  os.path.dirname(self.input()['kilosort'].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        # TODO get samplerate from project config?
        cluster_spike_times = ks_spike_times[ks_spike_cluster==self.cluster_id] / 30000 # in seconds

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
            "Label": []
        })
        for start, stop in silent_times:
            spike_dict['NSpikes'].append(np.logical_and(cluster_spike_times > start,cluster_spike_times < stop).sum())
            spike_dict['Duration'].append(stop - start)
            spike_dict['Label'].append('Silent')

        for start, stop,sound_label in sound_times:
            spike_dict['NSpikes'].append(np.logical_and(cluster_spike_times > start,cluster_spike_times < stop).sum())
            spike_dict['Duration'].append(stop - start)
            spike_dict['Label'].append(sound_label)

        spike_df = pandas.DataFrame(spike_dict)
        spike_df['SpikeRate'] = spike_df['NSpikes'] / spike_df['Duration']
        
        self.output().makedirs()
        spike_df.to_pickle(self.output().path)

class PlotClusterAuditorySpikeRates(luigi.Task):
    """ OUTPUTS: 
            AuditorySpikerates.png
    """
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()

    def requires(self):
        return GenClusterAuditorySpikeRatesDB(site_dir=self.site_dir,cluster_id=self.cluster_id)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","Cluster_%d"%self.cluster_id,"AuditorySpikerates.png"))
    
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


class PlotClusterSummary(luigi.Task):
    """
    Plotting task for Cluster_Summary.png

    Plots the templates in the cluster, the isi, and the spikerate
    TODO can put some useful stats on here too
    """
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()

    def requires(self):
        return {
            'kilosort': KiloSortSorting(self.site_dir),
            'config': SiteConfig(self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,'analysis','figures','Cluster_%d'%self.cluster_id,'Cluster_Summary.png'))
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
        title = "Cluster %s"%self.cluster_id
        fig = plt.figure(title)
        fig.patch.set_facecolor('white')
        
        # ISI Plot
        ax = plt.subplot(gs[0,1])
        cluster_spike_times = ks_spike_times[ks_spike_cluster==self.cluster_id] / 30000 # in seconds
        vec_isi = cluster_spike_times[1:] - cluster_spike_times[:-1]
        ax.hist(vec_isi,bins = np.arange(0,.15,.002))

        # Spikerate over time plot
        ax_sr = plt.subplot(gs[1,1])
        ax_sr.hist(cluster_spike_times,bins=np.arange(0,max(cluster_spike_times)+1))
        
        # spike shape plot
        # first get all templates that contrib to this cluster
        templates,counts = np.unique(ks_spike_templates[ks_spike_cluster == self.cluster_id],return_counts=True)
        cluster_info = cluster_info = ks_cluster_info[ks_cluster_info.cluster_id == self.cluster_id]
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


        
class GenSiteSummary(luigi.Task):
    """
    Goes through all clusters and generates plots summarizing all clusters
    """
    site_dir = SiteParameter()
    def requires(self):
        return {
            "kilosort": KiloSortSorting(self.site_dir),
            'config': SiteConfig(self.site_dir)
        }
    
    def output(self):
        # TODO could break this into playbacks and silence dfs, and do a bunch of other stuff with the playbacks
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","dfClusterSummaries.pkl"))
    
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
        cluster_ids_good = []
        for ix,cluster in ks_cluster_info.iterrows():
            if cluster['group'] != 'noise' and cluster.fr > 2.0:#  and cluster.fr > 5:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
                cluster_ids_good.append(cluster.cluster_id)
        cluster_ids_good = np.asarray(cluster_ids_good)

        new_infos = []
        for cid in cluster_ids_good:
            templates,counts = np.unique(ks_spike_templates[ks_spike_cluster == cid],return_counts=True)
            cluster_info = ks_cluster_info[ks_cluster_info.cluster_id == cid].copy()
            chan = cluster_info.ch.iloc[0]
            templates_list = []
            for template in templates:            
                actual_template = ks_templates[template,20:,np.where(chan_map == chan)]
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
            cluster_rate = pandas.read_pickle(GenClusterAuditorySpikeRatesDB(self.site_dir,cluster_id = cid).output().path)
            for lbl in np.unique(cluster_rate.Label):
                sub_df = cluster_rate[cluster_rate.Label == lbl]
                sub_df 
                avg_rate = np.average(sub_df.SpikeRate,weights=sub_df.Duration)
                std_rate = np.sqrt(np.average((sub_df.SpikeRate - avg_rate)**2, weights=sub_df.Duration))
                cluster_info["%s_avg"%lbl] = avg_rate
                cluster_info["%s_std"%lbl] = std_rate

            new_infos.append(cluster_info)
            #plt.plot(template_avg)
        #plt.figure()

        cluster_info_df = pandas.concat(new_infos,ignore_index=True)
        #seaborn.scatterplot(cluster_info_df,'PTP_ratio','PTP_dur_ms')
        self.output().makedirs()
        cluster_info_df.to_pickle(self.output().path)

class PlotSiteSummary(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return GenSiteSummary(self.site_dir)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","AllClusters.png"))

    def run(self):
        cluster_info_df = pandas.read_pickle(self.input().path)
        fig, ax = plt.subplots(1, 2)
        ax_scatter = ax[0]
        seaborn.scatterplot(cluster_info_df,x='PTP_dur_ms',y='PTP_ratio',ax=ax_scatter)
        ax_spike = ax[1]
        for _,x in cluster_info_df.iterrows():
            ax_spike.plot(x.template)
        self.output().makedirs()
        plt.savefig(self.output().path)
# class GenProjectSummary(luigi.Task):
#     project_dir = luigi.
#     def requires(self):

#     def output(self):

#     def run(self):

# endregion 
# region Trial Raster Plots
class GenPlaybackSpikesDB(luigi.Task):
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()
    pad_s = luigi.FloatParameter(default=1.0)
    def requires(self):
        return {
            "project_config": SiteProjectConfig(self.site_dir),
            "playback_info": GenBehaviorDB(self.site_dir),
            "kilosort": KiloSortSorting(self.site_dir),
            "spect": GenSpectrogramDB(self.site_dir)
        }
    
    def output(self):
        # TODO could break this into playbacks and silence dfs, and do a bunch of other stuff with the playbacks
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster_id,"dfPlaybackSpikes.pkl"))
    
    def run(self):
        ks_output_dir =  os.path.dirname(self.input()['kilosort'].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        # TODO get samplerate from project config?
        cluster_spike_times = ks_spike_times[ks_spike_cluster==self.cluster_id] / 30000 # in seconds

        playbacks_and_timings = pandas.read_pickle(self.input()['playback_info'].path)

        out_df = {"Rendition_Index":[], "Rendition_Name":[],"Trial_Number":[],
                  "Spike_Times":[],"Start":[],"End":[],"Peck":[], "Rate":[], 
                  "Bird":[], "Category":[], "Novel": [], "Informative_Trials":[]}
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
                        out_df['Category'].append(trial_row.condition_name)
                        out_df['Novel'].append(trial_row.novel)
                        out_df['Informative_Trials'].append(trial_row.informative_trials)
        spike_df = pandas.DataFrame(out_df)
        spike_df = spike_df.sort_values(by='Start')
        self.output().makedirs()
        spike_df.to_pickle(self.output().path)

class PlotTrialSpikeRasters(luigi.Task):
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()
    pad_s = luigi.FloatParameter(default=1.0)
    def requires(self):
        return {
            'spikes': GenPlaybackSpikesDB(self.site_dir,pad_s=self.pad_s, cluster_id=self.cluster_id),
            'spectrograms': GenSpectrogramDB(self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","Cluster_%d"%self.cluster_id,"spike_rasters_%.2f"%(self.pad_s)))
    def run(self):
        # Load the spike dataframe from GenPlaybackSpikeDB
        spike_df = pandas.read_pickle(self.input()['spikes'].path)
        #spike_df.sor
        df_song_spect = pandas.read_pickle(self.input()['spectrograms'].path)
        
        # One plot for each bird, rend pair
        for bird in np.unique(spike_df['Bird']):
            bird_df = spike_df[spike_df.Bird == bird]
            for song in np.unique(bird_df.Rendition_Name):
                spectrogram = df_song_spect.Spectrogram[df_song_spect.Name == song].iloc[0]
                rend_df = bird_df[bird_df.Rendition_Name == song]
                _plot_trials(rend_df.Spike_Times-rend_df.Start,rend_df.Peck,song,spectrogram,-1,7,mode='rasters')
                
                out_fn = "Rasterplot_%s.png"%song
                out_root = self.output().path
                out_dir = os.path.join(out_root,"%s_%s_%s"%(rend_df.Category.iloc[0],bird,rend_df.Novel.iloc[0]))
                os.makedirs(out_dir,exist_ok=True)
                plt.savefig(os.path.join(out_dir,out_fn))
                plt.close()

            # now save by bird
            _plot_trials(bird_df.Spike_Times - bird_df.Start, bird_df.Peck, bird,None,-1,7,mode='rasters')
            out_fn = "Raster_%s.png"%bird
            out_root = self.output().path
            out_dir = os.path.join(out_root,"%s_%s_%s"%(rend_df.Category.iloc[0],bird,rend_df.Novel.iloc[0]))
            os.makedirs(out_dir,exist_ok=True)
            plt.savefig(os.path.join(out_dir,out_fn))
            plt.close()

class PlotSpikeRateByTrial(luigi.Task):
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()
    pad_s = luigi.FloatParameter(default=1.0)
        
    def requires(self):
        return GenPlaybackSpikesDB(self.site_dir,pad_s=self.pad_s, cluster_id=self.cluster_id)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","Cluster_%d"%self.cluster_id,"spike_rasters_%.2f"%(self.pad_s),'SpikeRate_ByBird.png'))
    def run(self):
        # now save avg spikerate by trial
        spike_df = pandas.read_pickle(self.input().path)
        rates = []
        for ix, row in spike_df.iterrows():
            spike_times = row.Spike_Times-row.Start
            start = -.05
            end = .55
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
class GenPopulationPCA(luigi.Task):
    site_dir = SiteParameter()
    # TODO put these in site config or something
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return KiloSortSorting(self.site_dir)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","PCATimeseries_bin%.2f_step%.2f.npy"%(self.bin_size_s,self.bin_step_s)))
    def run(self):
        # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input().path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')


        cluster_ids_good = []
        for ix,cluster in ks_cluster_info.iterrows():
            if cluster['group'] != 'noise' and cluster.fr > 2.0:#  and cluster.fr > 5:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
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

class GenPlaybackPCADB(luigi.Task):
    site_dir = SiteParameter()
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return {
            'pca': GenPopulationPCA(self.site_dir, bin_size_s =self.bin_size_s, bin_step_s=self.bin_step_s),
            'playback_info': GenStimInfoDB(self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","dfPlaybackPCA_bin%.2f_step%.2f.pkl"%(self.bin_size_s,self.bin_step_s)))
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

class PlotStimPopulationPCA(luigi.Task):
    site_dir = SiteParameter()
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return {
            'pca': GenPlaybackPCADB(self.site_dir,bin_size_s=self.bin_size_s,bin_step_s=self.bin_step_s),
            'spect': GenSpectrogramDB(self.site_dir)
        }
    def output(self):
        # This is a directory that will contain all the PC Plots
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","pcplots_%.2f_step%.2f"%(self.bin_size_s,self.bin_step_s),"per_stim"))
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

class PlotBirdPopulationPCA(luigi.Task):
    site_dir = SiteParameter()
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return {
            'pca': GenPlaybackPCADB(self.site_dir,bin_size_s=self.bin_size_s,bin_step_s=self.bin_step_s),
            'spect': GenSpectrogramDB(self.site_dir)
        }
    def output(self):
        # This is a directory that will contain all the PC Plots
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","pcplots_%.2f_step%.2f"%(self.bin_size_s,self.bin_step_s),"per_bird"))
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


class Plot2dPCScatter(luigi.Task):
    site_dir = SiteParameter()
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return {
            'pca': GenPlaybackPCADB(self.site_dir,bin_size_s=self.bin_size_s,bin_step_s=self.bin_step_s)
        }
    def output(self):
        # This is a directory that will contain all the PC Plots
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","pcplots_%.2f_step%.2f"%(self.bin_size_s,self.bin_step_s),"scatter"))
    def run(self):
        labels = dict({
            "Unrewarded": "#a40e4c",
            "Rewarded": "#64f58d"
        })
        df_pc_paths = pandas.read_pickle(self.input()['pca'].path)

        plt.figure()
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
# endregion

# region PostStim Plots
class GenPostStimDB(luigi.Task):
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the stimuli, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=.5, description="Size, in seconds, of the post stim window. Default is .5s")
    def requires(self):
        return {
            'spikes': KiloSortSorting(self.site_dir),
            'playback_info': GenStimInfoDB(self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster_id,"dfPostStim_start%.2f_size%.2f.pkl"%(self.window_start,self.window_size)))
    def run(self):
         # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input()['spikes'].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')

        playbacks_and_timings = pandas.read_pickle(self.input()['playback_info'].path)
        
        # TODO get samplerate from project config?
        cluster_spike_times = ks_spike_times[ks_spike_cluster==self.cluster_id] / 30000 # in seconds

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

class PlotPostStimResponses(luigi.Task):
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the stimuli, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=.5, description="Size, in seconds, of the post stim window. Default is .5s")

    def requires(self):
        return GenPostStimDB(self.site_dir, cluster_id=self.cluster_id)
    def output(self):
        # TODO I could put a list here of jobs ran
        return {
            "rasters": luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","Cluster_%d"%self.cluster_id,"PostStimRasters_start%.2f_size%.2f.png"%(self.window_start,self.window_size))),
            "ratio": luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","Cluster_%d"%self.cluster_id,"PostStimRatio_start%.2f_size%.2f.png"%(self.window_start,self.window_size))),
            "kde": luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","Cluster_%d"%self.cluster_id,"PostStimKDE_start%.2f_size%.2f.png"%(self.window_start,self.window_size)))
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


class GenEventSpikeDB(luigi.Task):
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the feeder onset, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=1.0, description="Size, in seconds, of the post stim window. Default is .5s")

    def requires(self):
        return {
            "spikes": KiloSortSorting(self.site_dir),
            "mergecsv": MergeOperantCSV(self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster_id,"dfEventRaster_start%.2f_size%.2f.pkl"%(self.window_start,self.window_size)))
    
    def run(self):
        # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input()['spikes'].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))

        events = pandas.read_csv(self.input()['mergecsv']['events'].path)
        
        # TODO get samplerate from project config?
        cluster_spike_times = ks_spike_times[ks_spike_cluster==self.cluster_id] / 30000 # in seconds

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

class PlotEventAlignedRaster(luigi.Task):
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the feeder onset, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=1.0, description="Size, in seconds, of the post stim window. Default is .5s")
    def requires(self):
        return GenEventSpikeDB(self.site_dir,cluster_id=self.cluster_id,window_start=self.window_start,window_size=self.window_size)
    def output(self):
        return {
            "rasters": luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","Cluster_%d"%self.cluster_id,"EventRasters_start%.2f_size%.2f.png"%(self.window_start,self.window_size))),
        }
    def run(self):
        event_df = pandas.read_pickle(self.input().path)    
        event_names = np.unique(event_df.Event)

        gs = gridspec.GridSpec(1, len(event_names))
        for i, event_name in enumerate(event_names):
            ax = plt.subplot(gs[0,i])
            ax.eventplot(event_df.Spikes[event_df.Event == event_name])
            ax.set_title(event_name)
        plt.savefig(self.output()['rasters'].path)
        plt.close()

class GenEventPCADB(luigi.Task):
    site_dir = SiteParameter()
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the feeder onset, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=1.0, description="Size, in seconds, of the post stim window. Default is .5s")
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)

    def requires(self):
        return {
            'pca': GenPopulationPCA(self.site_dir, bin_size_s =self.bin_size_s, bin_step_s=self.bin_step_s),
            'mergecsv': MergeOperantCSV(self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","dfEventPCA_start%.2f_size%.2f_bsize%.2f_bstep%.2f.pkl"%(self.window_start,self.window_size,self.bin_size_s,self.bin_step_s)))
    
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
class PlotEventAlignedPCA(luigi.Task):
    site_dir = SiteParameter()
    window_start = luigi.FloatParameter(default=-.25,description="Start of the post stim window relative to the end of the feeder onset, in seconds. Default is -.25s")
    window_size = luigi.FloatParameter(default=1.0, description="Size, in seconds, of the post stim window. Default is .5s")
    bin_size_s= luigi.FloatParameter(default = .15)
    bin_step_s= luigi.FloatParameter(default =.050)
    def requires(self):
        return GenEventPCADB(self.site_dir,window_start=self.window_start,window_size=self.window_size,
                             bin_size_s=self.bin_size_s,bin_step_s=self.bin_step_s)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","EventPCA_start%.2f_size%.2f_bsize%.2f_bstep%.2f.png"%(self.window_start,self.window_size, self.bin_size_s, self.bin_step_s)))
    
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

class GenClusterTimePCDB(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return KiloSortSorting(self.site_dir)
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
            if cluster['group'] != 'noise' and cluster.fr > 2.0:#  and cluster.fr > 5:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
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


class DecodeLabelsNB(luigi.Task):
    site_dir = SiteParameter()
    bin_size_s = luigi.FloatParameter(default=.01)
    test_size = luigi.FloatParameter(default=.2)
    def requires(self):
        return {
            "spikes": KiloSortSorting(self.site_dir),
            "behavior": GenBehaviorDB(self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","dfPerf_zscore_%.2f_%.2f.pkl"%(self.bin_size_s,self.test_size)))
    def run(self):
        # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input()["spikes"].path)
        ks_spike_times = np.load(os.path.join(ks_output_dir, 'spike_times.npy'))
        ks_spike_cluster = np.load(os.path.join(ks_output_dir, 'spike_clusters.npy'))
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')

        cluster_ids_good = []
        for ix,cluster in ks_cluster_info.iterrows():
            if cluster['group'] != 'noise' and cluster.fr > 2.0:#  and cluster.fr > 5:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
                cluster_ids_good.append(cluster.cluster_id)
        cluster_ids_good = np.asarray(cluster_ids_good)

        # T x Cluster
        t_start = 0#np.floor(ks_spike_times.min()/30000)
        t_stop = np.ceil(ks_spike_times.max()/30000)
        edges = np.arange(t_start,t_stop*30000,int(self.bin_size_s*3000))
        num_bins = edges.shape[0]-1
        num_neurons = len(cluster_ids_good)
        neural_data=np.empty([num_bins,num_neurons])
        for ix,p in enumerate(cluster_ids_good):
            neural_data[:,ix] = np.histogram(ks_spike_times[ks_spike_cluster==p],edges)[0]


        # now we have binned neural data at bin_size_s bins
        # lets make an array of onset responses
        playback_and_timings = pandas.read_pickle(self.input()["behavior"].path)
        # half sec of the start of each stimuli
        bins_after_onset = int(.5 / self.bin_size_s)
        onset_times = (playback_and_timings.start_time/self.bin_size_s).astype(np.int)
        X = np.lib.stride_tricks.sliding_window_view(neural_data, bins_after_onset, axis=0)[onset_times]
        # reshape it into a flat format
        X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB

        out_df =dict({
            "Performance": [],
            "Random_mean": [],
            "Random_Std" :[],
            "Label": []
        })

        # Now test prediction performance for Category, Stim_name, Bird_Name, Novel
        for label in ['condition_name','bird_name','name','novel']:
            Y = np.asarray(playback_and_timings[label])
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=self.test_size)
            # remove rare inputs
            X_train_inds = np.where(X_train.sum(axis=0)>0)[0]
            X_train = X_train[:,X_train_inds]
            X_test = X_test[:,X_train_inds]
            # zscore inputs 
            X_mean=np.nanmean(X_train,axis=0)
            X_std=np.nanstd(X_train,axis=0)
            X_train =(X_train-X_mean)/X_std
            X_test=(X_test-X_mean)/X_std

            gnb = GaussianNB()
            Y_pred = gnb.fit(X_train,Y_train).predict(X_test)
            perf = (Y_test == Y_pred).sum() / X_test.shape[0]
            out_df["Performance"].append(perf)
            rand_perf = []
            for i in range(50):
                np.random.shuffle(Y)
                X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=self.test_size)
                # remove rare inputs
                X_train_inds = np.where(X_train.sum(axis=0)>0)[0]
                X_train = X_train[:,X_train_inds]
                X_test = X_test[:,X_train_inds]
                # zscore inputs 
                X_mean=np.nanmean(X_train,axis=0)
                X_std=np.nanstd(X_train,axis=0)
                X_train =(X_train-X_mean)/X_std
                X_test=(X_test-X_mean)/X_std
                gnb = GaussianNB()
                Y_pred = gnb.fit(X_train,Y_train).predict(X_test)
                rand_perf.append((Y_test == Y_pred).sum() / X_test.shape[0])
            out_df["Random_mean"].append(np.mean(rand_perf))
            out_df["Random_Std"].append(np.std(rand_perf))
            out_df['Label'].append(label)
        perf_df = pandas.DataFrame(out_df)
        self.output().makedirs()
        perf_df.to_pickle(self.output().path)



class GenerateBinnedStimResponse(luigi.Task):
    site_dir = SiteParameter()
    bin_size_s = luigi.FloatParameter(default=.05)
    def requires(self):
        return {
            "spikes": KiloSortSorting(self.site_dir),
            "behavior": GenBehaviorDB(self.site_dir)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","binned_spikes_%.2f.npy"%self.bin_size_s))
    def run(self):
        pass
# endregion

# region Behavior Plotting


class PlotBehaviorTrials(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return GenBehaviorDB(self.site_dir)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","Behavior_Trials.png"))
    def run(self):
        playback_and_timings = pandas.read_pickle(self.input().path)
        plot_pecking_test_data(playback_and_timings,['bird_name','condition_name',"novel"])
        self.output().makedirs()
        plt.savefig(self.output().path)
        



#endregion

# region Wrapper Tasks 
class SiteFigureGeneration(luigi.Task):
    site_dir = SiteParameter()
    def requires(self):
        return KiloSortSorting(self.site_dir)
    def output(self):
        # TODO I could put a list here of jobs ran
        return luigi.LocalTarget("dummy")
    def run(self):
        # Load necessary kilosort output files
        ks_output_dir =  os.path.dirname(self.input().path)
        ks_cluster_info = pandas.read_csv(os.path.join(ks_output_dir, 'cluster_info.tsv'),delimiter='\t')

        # First run the summary stats
        # TODO

        # then thee PCA stuff
        yield PlotStimPopulationPCA(site_dir=self.site_dir)
        yield PlotBirdPopulationPCA(site_dir=self.site_dir)
        yield PlotEventAlignedPCA(site_dir =self.site_dir)
        yield PlotBehaviorTrials(site_dir=self.site_dir)
        yield DecodeLabelsNB(site_dir = self.site_dir)
        cluster_ids_good = []
        for ix,cluster in ks_cluster_info.iterrows():
            if cluster['group'] != 'noise'  and cluster.fr > 2.0:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
                cluster_ids_good.append(cluster.cluster_id)
                yield PlotClusterAuditorySpikeRates(site_dir=self.site_dir,cluster_id=cluster.cluster_id)
                yield PlotTrialSpikeRasters(site_dir=self.site_dir,cluster_id=cluster.cluster_id)
                yield PlotPostStimResponses(site_dir=self.site_dir,cluster_id=cluster.cluster_id)
                yield PlotEventAlignedRaster(site_dir =self.site_dir, cluster_id=cluster.cluster_id)
                yield PlotClusterSummary(site_dir =self.site_dir, cluster_id=cluster.cluster_id)
                yield PlotSpikeRateByTrial(site_dir =self.site_dir, cluster_id=cluster.cluster_id)
        cluster_ids_good = np.asarray(cluster_ids_good)
        yield PlotSiteSummary(site_dir = self.site_dir)
    
# endregion