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


"""
This file Will have all jobs for post (afer spike-sorting) analysis
for the operant ephys sites
This includes:
- GenerateFigs : WrapperTask that runs all jobs to generate figures for a given site
    - PlotAuditorySpikerate: Plots 
    - PlotPCA: 
"""

#------ Plotting utils. TODO these can be moved to a separate file eventually ------
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
    
    axl = plt.subplot(gs[1,0], sharey=ax) # Instantiate left KDE plot area
    axl.spines["right"].set_visible(False)
    axl.spines["top"].set_visible(False)
    axl.spines["bottom"].set_visible(False)

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
        #if colors is not None:
        #    ax.eventplot(spikes,colors=[cat_color_lookup[cat] for cat in colors])
        #    # TODO PLot peck times
        #else:
        ax.eventplot(spikes)
        for i in range(len(peck_times)):
            peck_time = peck_times[i]
            if peck_time > 0:
                ax.vlines([peck_time],i-.5,i+.5,colors='red')
        
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

#------------------------------------------------------------------------------------
#------------------------------- Prep Jobs ------------------------------------------
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
                fs,arr = wavfile.read(os.path.join(stim_dir,row.playback_file.split('stimuli/Operant/')[-1]))

                spectrogram = _get_spectrogram(arr,fs)
                df_song_spect['Bird'].append(bird)
                df_song_spect['Name'].append(song)
                df_song_spect['Spectrogram'].append(spectrogram)
            
        df_song_spect = pandas.DataFrame(df_song_spect)

        self.output().makedirs()
        df_song_spect.to_pickle(self.output().path)

#------------------------------------------------------------------------------------
#---------------------------Descriptive Stats----------------------------------------
class GenClusterAuditorySpikeRatesDB(luigi.Task):
    """ OUTPUTS: 
            data/<CLUSTER>/dfPlaybackSpikes.pkl
    """
    site_dir = SiteParameter()
    cluster_id = luigi.NumericalParameter(var_type=int,min_value=0,max_value=4000)

    def requires(self):
        return {
            "project_config": SiteProjectConfig(self.site_dir),
            "playback_info": GenStimInfoDB(self.site_dir),
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
        # now make opposite, silent_times
        prev_stop = 0
        silent_times = []
        for start,stop,_ in sound_times:
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
        def _get_spikerate(inds):
            return dfSpikeRate[inds].NSpikes.sum() / dfSpikeRate[inds].Duration.sum()
        plt.figure()
        plt.bar([1,3,4,5],[ _get_spikerate(dfSpikeRate.Label == 'Silent'), _get_spikerate(dfSpikeRate.Label != 'Silent'), _get_spikerate(dfSpikeRate.Label == 'Rewarded'), _get_spikerate(dfSpikeRate.Label == 'Unrewarded')],
                    label=['No Stim','All Sound','Rewarded','Unrewarded'], color=['black','blue','green','red'])
        plt.legend()

        self.output().makedirs()
        plt.savefig(self.output().path)

#-------------------------- Trial Raster Plots ------------------------------------
class GenPlaybackSpikesDB(luigi.Task):
    site_dir = SiteParameter()
    cluster_id = luigi.IntParameter()
    pad_s = luigi.FloatParameter(default=1.0)
    def requires(self):
        return {
            "project_config": SiteProjectConfig(self.site_dir),
            "playback_info": GenStimInfoDB(self.site_dir),
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

        out_df = {"Rendition_Index":[], "Rendition_Name":[],"Trial_Number":[],"Spike_Times":[],"Start":[],"End":[],"Peck":[], "Rate":[], "Bird":[]}
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
                        end_time = start_time + 6.0 # seconds TODO get this from stim file maybe
                        if trial_row.response:
                            peck = trial_row.duration
                        else:
                            peck = 0
                        # get spikes times for padded duration
                        spike_times = cluster_spike_times[np.logical_and(cluster_spike_times > start_time - self.pad_s,
                                                                            cluster_spike_times < end_time + self.pad_s)]

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
        spike_df = pandas.DataFrame(out_df)
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
        df_song_spect = pandas.read_pickle(self.input()['spectrograms'].path)
        
        # One plot for each bird, rend pair
        for bird in np.unique(spike_df['Bird']):
            bird_df = spike_df[spike_df.Bird == bird]
            for song in np.unique(bird_df.Rendition_Name):
                spectrogram = df_song_spect.Spectrogram[df_song_spect.Name == song].iloc[0]
                rend_df = bird_df[bird_df.Rendition_Name == song]
                _plot_trials(rend_df.Spike_Times-rend_df.Start,spike_df.Peck,song,spectrogram,-1,7,mode='rasters')
                
                out_fn = "Rasterplot_%s.png"%song
                out_root = self.output().path
                out_dir = os.path.join(out_root,bird)
                os.makedirs(out_dir,exist_ok=True)
                plt.savefig(os.path.join(out_dir,out_fn))
                plt.close()



#-------------------------- PCA Population stuff -----------------------------------
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
            if cluster['group'] != 'noise':#  and cluster.fr > 5:#cluster.n_spikes > 8000:# or cluster['group'] == 'mua':
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
        pca = PCA(n_components=5) 
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
            # if ix+1 < len(playbacks_and_timings):
            #     next_pb = playbacks_and_timings.iloc[ix+1]
            #     pb_end_time = min(pb_row.start_time + pb_row.duration + 2, next_pb.start_time-.5)
            # else:
            #     pb_end_time = pb_row.start_time + pb_row.duration + 2
            pb_inds = np.logical_and(time_bins > pb_row.start_time,time_bins < pb_row.start_time + pb_row.duration)
            #bin_labels[pb_inds] = 1 if pb_row.condition_name == 'Rewarded' else 2
            paths['Stim'].append(pb_row['name'])
            paths['Category'].append(pb_row.condition_name)
            paths['Bird'].append(pb_row.bird_name)
            paths['Trial'].append(pb_row.trial)
            paths["Interrupted"].append(pb_row.response)
            paths['pc_path'].append(pca_timeseries[:,pb_inds])

        # pca_timeseries = pca_timeseries[:,bin_labels!=0]
        # sns.jointplot(x=pca_timeseries[0,:],y=pca_timeseries[1,:],hue=bin_labels[bin_labels!=0],kind='kde')
        # #plt.scatter(pca_timeseries[0,:],pca_timeseries[1,:],c=bin_labels,alpha=.2)
        # #plt.colorbar()
        # plt.savefig("Figs/StimPlots/%s/PCA.png"%(site))

        # Make a dataframe of the pc paths
        df_pc_paths = pandas.DataFrame(paths)
        self.output().makedirs()
        df_pc_paths.to_pickle(self.output().path)

class PlotPopulationPCA(luigi.Task):
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
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","pcplots_%.2f_step%.2f"%(self.bin_size_s,self.bin_step_s)))
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
                out_fn = "PCHeatmap_%s_%s_%s.png"%(bird,rend,stim_df.Category.iloc[0])
                
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
                ax.imshow(pc_im[0,:,:],extent=(0,maxlen * self.bin_step_s,0,len(stim_df)),aspect='auto')
                ax1.imshow(pc_im[1,:,:],extent=(0,maxlen * self.bin_step_s,0,len(stim_df)),aspect='auto')
                ax2.imshow(pc_im[2,:,:],extent=(0,maxlen * self.bin_step_s,0,len(stim_df)),aspect='auto')

                out_root = self.output().path
                out_dir = os.path.join(out_root,bird)
                os.makedirs(out_dir,exist_ok=True)
                plt.savefig(os.path.join(out_dir,out_fn))
#------------------------------------------------------------------------------------

# ------------------------------- Wrapper Tasks -------------------------------------
class SiteFigureGeneration(luigi.Task):
    def requires(self):
        pass
    def output(self):
        pass
    def run(self):
        pass
#------------------------------------------------------------------------------------
