
from scipy.signal import windows, butter, filtfilt,find_peaks
from soundsig.sound import BioSound
from soundsig.sound import spec_colormap
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

import numpy as np
import luigi
import matplotlib.pyplot as plt
import os
import pandas
import pickle
from scipy.io import wavfile

from strfpy import preprocSound

from .postprocess_strf import GenerateSTRFRequirements
from .postprocess_operant import SiteProjectConfig,GenStimInfoDB, GenPlaybackSpikesDB
from .pipeline_utils import SiteParameter
def _get_mrf_data_folder(site_dir,cluster):
    return os.path.join(site_dir,"analysis","data","Cluster_%d"%cluster['cluster_id'],"MRF")

def _get_mrf_figure_folder(site_dir,cluster):
    return os.path.join(site_dir,"analysis","figures","clusters_%s"%cluster['group'],"Cluster_%d"%cluster['cluster_id'],"MRF")
    

class DataFrameColumn(luigi.Target): # TODO but not before lab meeting...
    def __init__(self, df_fn, column):
        self.df = df_fn
        self.column = column
    def exists(self):
        return self.column in self.df.columns
    def remove(self):
        self.df.drop(columns=[self.column], inplace=True)
    def data(self):
        return self.df[self.column]

class BaseTask(luigi.Task):
    force = luigi.BoolParameter()
    def complete(self):
        outputs = luigi.task.flatten(self.output())
        for output in outputs:
            if self.force and output.exists():
                output.remove()
        return all(map(lambda output: output.exists(), outputs))
def _old_get_segments(data,sr,pad=50,amp_sr=1000,lp_hz=50, min_seg_size_ms=25):
    data = np.pad(data, (pad,0), 'constant', constant_values=(0, 0))
    # calculate ampenv with biosound
    myBioSound = BioSound(soundWave=data, fs=sr, emitter='STRF', calltype = 'song')
    myBioSound.ampenv(cutoff_freq = 20, amp_sample_rate = amp_sr)
    pass

def _get_segments(data,sr,pad=50,amp_sr=1000,lp_hz=50, min_seg_size_ms=25,force_next_up=False):
    """ Returns the segments of a wav file"""
    data = np.pad(data, (pad,0), 'constant', constant_values=(0, 0))
    # calculate ampenv with biosound
    myBioSound = BioSound(soundWave=data, fs=sr, emitter='STRF', calltype = 'song')
    myBioSound.ampenv(cutoff_freq = 20, amp_sample_rate = amp_sr)
    # lowpass filter the ampenv
    a,b = butter(6, lp_hz/(amp_sr/2), btype='low')
    filtered = filtfilt(a,b, myBioSound.amp[1:])
    # find peaks in the derivative
    diff = np.diff(filtered)
    peaks, _ = find_peaks(diff, height=np.std(diff)*.5, distance=5)
    amppad = int(pad * amp_sr / sr)
    if peaks[0] > amppad + 50:
        peaks = np.concatenate([[amppad+1],peaks])
    # adjust peaks to be the first negative before the peak
    for ix,peak in enumerate(peaks): # TODO this may be a bit too much
        zeros_below_peak = np.where(diff<0)[0][np.where(diff<0)[0] < peak]
        if len(zeros_below_peak) > 0:
            peaks[ix] = zeros_below_peak[-1]
        else:
            peaks[ix] = amppad +1

    down_peaks, _ = find_peaks(-diff, height=np.std(diff)*.5)
    # add the end to the downpeaks
    down_peaks = np.concatenate([down_peaks,[len(diff)-1-amppad]])
    # now pair the peaks
    segments = []
    for ix, peak in enumerate(peaks):
        if ix < len(peaks):
            if ix < len(peaks)-2:
                next_peak = peaks[ix+1]
            else:
                next_peak = len(diff)-1-amppad
            possible_down_peaks = down_peaks[(down_peaks > peak) & (down_peaks < next_peak)]
            if len(possible_down_peaks) > 0 and (not force_next_up or next_peak - peak > .5*amp_sr):
                # take the last possible down peak
                down_peak = possible_down_peaks[-1]
            else:
                # no down peak found, take the next  up peak
                down_peak = next_peak#-10
            tadjust = pad/sr # shift back to undo padding from earlier
            segment = [max(0,peak/amp_sr-tadjust), down_peak/amp_sr-tadjust]
            if segment[1] - segment[0] > min_seg_size_ms/1000:
                segments.append(segment)

    # TODO Join segments that are neighbors and short
    return segments

def _generate_MPS(data,sr,mps_sr=1000):
    # Create BioSound Object and calculate basic sound features
    myBioSound = BioSound(soundWave=data, fs=sr, emitter='STRF', calltype = 'song')
    myBioSound.spectroCalc(spec_sample_rate=mps_sr, freq_spacing = 50, min_freq=0, max_freq=10000)
    myBioSound.rms = myBioSound.sound.std() 
    myBioSound.ampenv(cutoff_freq = 20, amp_sample_rate =mps_sr)
    myBioSound.spectrum(f_high=10000)
    myBioSound.mpsCalc(window=0.05, Norm = True)  # Norm equals true normalizes the spectrogram but not the MPS

    # only save center of mps
    indwt = np.argwhere((myBioSound.wt > -100) & (myBioSound.wt < 100))
    indwf = np.argwhere( (myBioSound.wf >= 0) & (myBioSound.wf <6e-3))

    mps = myBioSound.mps[int(indwf[0]):int(indwf[-1]+1),int(indwt[0]):int(indwt[-1]+1)].reshape(len(indwt)*len(indwf))
    wt = myBioSound.wt[indwt]
    wf = myBioSound.wf[indwf]
    return mps, wt, wf


class GenSegmentMPSDB(BaseTask):
    """ Generates database of segments for each stim"""
    site_dir = SiteParameter()
    padding_ms = luigi.IntParameter(default=25)

    def requires(self):
        return {
            "project_config": SiteProjectConfig(self.site_dir),
            "playback_info": GenStimInfoDB(self.site_dir)
        }
    
    def output(self):# SCHEMA TAG
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","raw","dfSongMPSSegments_sc-b_pad%d.pkl"%(self.padding_ms)))
    
    def run(self):
        # load the project config to get stim_drive location
        project_config = self.input()['project_config'].data()
        stim_dir = os.path.join(project_config['LOCAL_RAW_FILES']['stimuli_drive'],"Operant")

        playbacks_and_timings = pandas.read_pickle(self.input()['playback_info'].path)

        df_song_segments = {"Bird":[],"Name":[],"tStart":[], "tEnd":[], "MPS":[], "padding_ms":[],'wt':[],'wf':[]}
        # Go through all emitters
        for bird in np.unique(playbacks_and_timings.bird_name):
            bird_df = playbacks_and_timings[(playbacks_and_timings.bird_name == bird)]
            # all unique stim files for each emitter
            for song in np.unique(bird_df.name):
                # get all playbacks of this stim file
                song_df = bird_df[bird_df.name == song]
                row = song_df.iloc[0]

                # read the wav data
                sr, data = wavfile.read(row.playback_file)

                # get the segments
                segments = _get_segments(data,sr,force_next_up=True)
                for segment in segments:
                    # Generate MRF for this segment
                    samp_start = max(0, int(segment[0]*sr) - int(self.padding_ms/1000*sr))
                    samp_end = min(int(segment[1]*sr + self.padding_ms/1000*sr), len(data))
                    data_segment = data[samp_start:samp_end]
                    mps, wt, wf = _generate_MPS(data_segment,sr)

                    df_song_segments['Bird'].append(bird)
                    df_song_segments['Name'].append(song)
                    df_song_segments['tStart'].append(segment[0])
                    df_song_segments['tEnd'].append(segment[1])
                    df_song_segments['MPS'].append(mps)
                    df_song_segments['wt'].append(wt)
                    df_song_segments['wf'].append(wf)
                    df_song_segments['padding_ms'] = self.padding_ms

            
        df_song_segments = pandas.DataFrame(df_song_segments)

        self.output().makedirs()
        df_song_segments.to_pickle(self.output().path)

class PlotSegments(BaseTask):
    """ Generates database of segments for each stim"""
    site_dir = SiteParameter()
    padding_ms = luigi.IntParameter(default=25)
    extension = luigi.Parameter(default="png")
    def requires(self):
        return {
            "segments": GenSegmentMPSDB(site_dir=self.site_dir,padding_ms=self.padding_ms),
            "project_config": SiteProjectConfig(self.site_dir),
            "playback_info": GenStimInfoDB(self.site_dir)
        }
    def output(self):
        if not self.input()["playback_info"].path:
            raise Exception("No pb_df found, Run GenStimInfoDB")
        pb_df = pandas.read_pickle(self.input()["playback_info"].path)
        num_stims = len(pb_df.name.unique())
        outputs = []
        for i in range(num_stims//10):
            outputs.append(luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","Segments","Segments_%d.%s"%(i,self.extension))))
        return outputs
    def run(self):
        # load the segments database
        df_song_segments = pandas.read_pickle(self.input()['segments'].path)
        # load the project config to get stim_drive location
        project_config = self.input()['project_config'].data()
        stim_dir = os.path.join(project_config['LOCAL_RAW_FILES']['stimuli_drive'],"Operant")
        # load playback db
        playbacks_and_timings = pandas.read_pickle(self.input()['playback_info'].path)
        
        DBNOISE = 50  # Color Threshold for spectrogram

        # group dataframe by the song name
        spectrograms = []
        segments = []
        names = []
        nstim = 0
        for name, group in df_song_segments.groupby('Name'):
            # load the playback file to get the spectrogram 
            playback_file = playbacks_and_timings[playbacks_and_timings.name == name].iloc[0].playback_file
            sr, data = wavfile.read(playback_file)
            # now use BioSound to make the spectrogram
            myBioSound = BioSound(soundWave=data, fs=sr, emitter='STRF', calltype = 'song')
            myBioSound.spectroCalc(spec_sample_rate=1000, freq_spacing = 50, min_freq=0, max_freq=10000)

            tfrep = myBioSound.spectro
            t = myBioSound.to
            f = myBioSound.fo

            spectrograms.append((tfrep,t,f))
            this_stim_segs = list(zip(group.tStart,group.tEnd))
            segments.append(this_stim_segs)
            names.append(name)
            # Color map for spectrograms
            spec_colormap()   # defined in 
            if nstim % 10 == 10 - 1 or nstim == nstim-1:
                f, arr_ax = plt.subplots(len(spectrograms), 1, sharex=True, dpi=200, figsize = (8,8))
                for i in range((nstim % 10) +1):
                    # if this is a multiple of 10 then plot the first 10 stims
                    # Plot the spectrogram
                    # The spectrogram
                    tfrep,t,f = spectrograms[i]
                    soundSpect = np.copy(tfrep)
                    maxB = soundSpect.max()
                    minB = maxB-DBNOISE
                    soundSpect[soundSpect < minB] = minB
                    minSpect = soundSpect.min()
                    cmap = plt.get_cmap('SpectroColorMap')

                    arr_ax[i].imshow(tfrep, extent=[t[0], t[-1], f[0]*1e-3, f[-1]*1e-3],
                                aspect='auto', interpolation='nearest', origin='lower', cmap=cmap, vmin=minSpect, vmax=maxB)
                    for st,end in segments[i]:
                        arr_ax[i].axvspan(st,end,alpha=.3,color=np.random.rand(3,))
                    arr_ax[i].text(np.mean(arr_ax[i].get_xlim()),arr_ax[i].get_ylim()[1]*.5,names[i],ha='center',va='top')
                plt.xlabel('Temporal Frequency (Hz)')
                plt.ylabel('Spectral Frequency (Cycles/KHz)')
                self.output()[nstim//10].makedirs()
                plt.savefig(self.output()[nstim//10].path)
                spectrograms = []
                names = []
                segments = []
            nstim += 1

class GenSegmentSpikeDB(BaseTask):
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    padding_ms = luigi.IntParameter(default=25)
    kWidth_ms = luigi.IntParameter(default=3)
    def requires(self):
        return {
            "trials": GenPlaybackSpikesDB(site_dir=self.site_dir,cluster=self.cluster),
            "segments": GenSegmentMPSDB(site_dir=self.site_dir,padding_ms=self.padding_ms)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(_get_mrf_data_folder(self.site_dir,self.cluster),"dfPreprocSegmentResponsePairs.pkl"))
    def run(self):
        playbacks_df = pandas.read_pickle(self.input()['trials'].path)
        segments_df = pandas.read_pickle(self.input()['segments'].path)

        col_segment_spikes = []
        col_ntrials = []
        col_kde = []
        col_trate = []
        col_avg_rate = []
        # Go through each segment and get all spikes for that segment
        for ix, segment in segments_df.iterrows():
            # get segment duration
            seg_dur_ms = (segment.tEnd - segment.tStart)*1000

            # pad a bit to get a good kde estimate
            padding = (self.padding_ms + self.kWidth_ms) / 1000.
            padded_start = segment.tStart - padding
            padded_end = segment.tEnd + padding

            # find all playbacks of this song
            song_df = playbacks_df[playbacks_df.Rendition_Name == segment.Name]

            # find all playbacks that overlap with this segment
            segment_playbacks = song_df[(song_df.End-song_df.Start) > padded_end]

            # get all spikes for this segment
            segment_spikes = []
            for jx, playback in segment_playbacks.iterrows():
                spikes = playback.Spike_Times - playback.Start
                segment_spikes.append(spikes[(spikes > padded_start) & (spikes < padded_end)] - segment.tStart)
            
            # Generate a time axis at wich to estimate the average rate
            #TODO parameterize srate
            tRate = np.arange(start=-self.padding_ms, stop=np.fix(seg_dur_ms+self.padding_ms), step=1)#.reshape(-1,1)
            sampRate = 1000.0/(tRate[1]-tRate[0]) # should be 1000.
            if len(segment_spikes) == 0: #TODO fix
                col_kde.append(np.zeros_like(tRate))
                col_segment_spikes.append(segment_spikes)
                col_ntrials.append(0)
                col_trate.append(tRate)
                col_avg_rate.append(0)
            else:
                ntrials = len(segment_spikes)
                concatenated_spike_times = np.concatenate(segment_spikes).reshape(-1,1)
                if len(concatenated_spike_times) == 0:
                    # Consider ignoring these
                    meanSpikeDens = np.zeros_like(tRate)
                else:
                    # Fit the density
                    kdeMean = KernelDensity(kernel='gaussian', bandwidth=self.kWidth_ms).fit(concatenated_spike_times*1000)

                    # Generate the kde rate.
                    meanSpikeDens = np.exp(kdeMean.score_samples(tRate.reshape(-1,1)))# / ntrials
                    meanSpikeDens = meanSpikeDens*len(concatenated_spike_times)*sampRate/(sum(meanSpikeDens)*ntrials)

                col_kde.append(meanSpikeDens)
                col_segment_spikes.append(segment_spikes)
                col_ntrials.append(ntrials)
                col_trate.append(tRate)
                col_avg_rate.append(np.sum(concatenated_spike_times)/(seg_dur_ms/1000.0*ntrials))

        segments_df['Spikes'] = col_segment_spikes
        segments_df['Trials'] = col_ntrials
        segments_df['KDE'] = col_kde
        segments_df['KDE_rate'] = 1000
        segments_df['KDE_t'] = col_trate
        segments_df['KDE_width'] = self.kWidth_ms
        segments_df['AVG_rate'] = col_avg_rate

        self.output().makedirs()
        print("Saving to %s"%self.output().path)
        segments_df.to_pickle(self.output().path)

class GenerateMRF(BaseTask): # In progress
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    def requires(self):
        return GenSegmentSpikeDB(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        return {
            "response": luigi.LocalTarget(os.path.join(_get_mrf_data_folder(self.site_dir,self.cluster),'respMPSPC.pkl')),
            "pred_response": luigi.LocalTarget(os.path.join(_get_mrf_data_folder(self.site_dir,self.cluster),'respMPSPCpred.npy')),
            "mrf_struct": luigi.LocalTarget(os.path.join(_get_mrf_data_folder(self.site_dir,self.cluster),'stimMRF.pkl'))
        }
    def run(self):
        # Mean Response
        seg_spike_df = pandas.read_pickle(self.input().path)
        # filter out segments with no trials
        seg_spike_df = seg_spike_df[seg_spike_df.Trials > 0]
        
        # make all responses the same size 
        kde_len = 500
        resp_kdes = np.zeros((len(seg_spike_df),kde_len))
        t_start = seg_spike_df.KDE_t.iloc[0][0]
        t_kde = np.arange(t_start,t_start+kde_len)
        for ix, row in seg_spike_df.iterrows():
            if row.KDE.shape[0] > kde_len:
                resp_kdes[ix,:] = row.KDE[:kde_len]
            else:
                resp_kdes[ix,:row.KDE.shape[0]] = row.KDE

        # Calculate the PCs
        pcaResp = PCA(n_components=10)
        respPC = pcaResp.fit_transform(resp_kdes)
        out_resp_dict = dict({
            "pcaResp": pcaResp,
            "responsePCs": respPC,
            "t_resp": t_kde,
            "avg_resp": resp_kdes.mean(axis=0)
        })
        pickle.dump(out_resp_dict,open(self.output()["response"].path,'wb'))
        
        # Perform cross-validated ridge regression for all PCs
        Alphas = (1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13)   # Ridge hyperparameter coeffiecients(1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 
        nPC = respPC.shape[1]


        stim_mps_feat = np.vstack(seg_spike_df.MPS)
        stim_mps_feat -= stim_mps_feat.mean(axis=0)
        wt = seg_spike_df.wt.iloc[0]
        wf = seg_spike_df.wf.iloc[0]

        # Allocate space for predictions
        respPCPred = np.zeros(respPC.shape)
        MRF = np.zeros((nPC, stim_mps_feat.shape[1]))

        clf = RidgeCV(alphas=Alphas)
        clf.fit(stim_mps_feat, respPC)
        respPCPred = clf.predict(stim_mps_feat)
        MRF = clf.coef_
        print('Best alpha = %g R2 = %.3f' % (clf.alpha_, clf.score(stim_mps_feat, respPC)))

        dictOutputMRF = dict({
            "MRF": MRF,
            "wt": wt,
            "wf": wf
        })
        pickle.dump(dictOutputMRF,open(self.output()["mrf_struct"].path,'wb'))
        np.save(self.output()["pred_response"].path, respPCPred) # TODO output this as a column of the seg df
        


class PlotPCMRFs(BaseTask): # good
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    extension= luigi.Parameter(default="png")
    nPCs = luigi.IntParameter(default=10)
    def requires(self):
        return GenerateMRF(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        for i in range(self.nPCs):
            yield luigi.LocalTarget(os.path.join(_get_mrf_figure_folder(self.site_dir,self.cluster),"PC_MRFs","PC%d_MRF.%s"%(i,self.extension)))
    def run(self):
        MRFStruct = pickle.load(open(self.input()["mrf_struct"].path,'rb'))
        MRF = MRFStruct["MRF"]
        wt = MRFStruct["wt"]
        wf = MRFStruct["wf"]

        respPCPred = np.load(self.input()["pred_response"].path)
        respPC = pickle.load(open(self.input()["response"].path,'rb'))["responsePCs"]
        file_gen = self.output()
        # Plot predictions and MRF for first PC
        for i in range(self.nPCs):
            plt.figure(figsize=(6,2), dpi = 100)

            plt.subplot(1,2,1)
            plt.plot(respPC[:,i], respPCPred[:,i], 'k+')
            plt.xlabel('Actual Value')
            plt.ylabel('Predicted Value')
            plt.title('PC%d predictions'%i)

            plt.subplot(1,2,2)
            MRFPlot = MRF[i,:].reshape(len(wf), len(wt))
            maxabs = np.max((MRFPlot.max(), -MRFPlot.min()))
            cmap = plt.get_cmap('jet')
            ex = (wt.min(), wt.max(), wf.min()*1e3, wf.max()*1e3)
            plt.imshow(MRFPlot,  aspect='auto', origin='lower', cmap=cmap, extent=ex) # interpolation='spline16',
            plt.ylabel('Spectral Frequency (Cycles/KHz)')
            plt.xlabel('Temporal Frequency (Hz)')
            plt.title('PC%d MRF'%i)


            plt.ylim((0,4))
            plt.xlim((-50, 50))
            plt.clim((-maxabs*0.8, maxabs*0.8))
            out_target = next(file_gen)
            out_target.makedirs()
            print(out_target.path)
            plt.savefig(out_target.path)

class PlotMRFPCs(BaseTask): # good
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    nPCs = luigi.IntParameter(default=5)
    extension = luigi.Parameter(default="png")
    def requires(self):
        return GenerateMRF(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        return luigi.LocalTarget(os.path.join(_get_mrf_figure_folder(self.site_dir,self.cluster),"RespPCs.%s"%self.extension))
    def run(self):
        dictResp = pickle.load(open(self.input()["response"].path,'rb'))
        pcResp = dictResp["pcaResp"]
        avg_resp = dictResp["avg_resp"]
        t_resp = dictResp["t_resp"]
        # Plot the PCs
        plt.figure(figsize = (3,2), dpi=100)
        plt.subplot(2,1,1)
        plt.plot(t_resp,avg_resp)
        plt.axvline(0, color = 'k')
        plt.subplot(2,1,2)

        for iPC in range(self.nPCs):
            plt.plot(t_resp,pcResp.components_[iPC,:], label='PC%d'%(iPC))
        plt.axvline(0, color = 'k')
        plt.xlabel('Time (ms)')
        #plt.legend()

        # Make figure directory
        self.output().makedirs()
        print("Saving figure to %s"%self.output().path)
        plt.savefig(self.output().path)

class PlotAverageMPS(BaseTask):  #BUG: not working
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    extension = luigi.Parameter(default="png")
    def requires(self):
        return GenerateMRFSegments(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"MRF","AverageMPS.%s"%self.extension))
    def run(self):
        stimDict = pickle.load(open(self.input()["stimuli"].path,'rb'))
        stimMPS = stimDict["stimMPS"]
        wt = stimDict["wt"]
        wf = stimDict["wf"]
        segmentBuffer = stimDict["segmentBuffer"]
        # Generating MPS PLots
        # Plot the average MPS.
        mpsAVG = stimMPS.mean(axis=0)
        mpsAVG = mpsAVG.reshape((len(wf), len(wt)))

        DBNOISE = 50
        plt.figure(dpi=300, figsize=(2,2))
        cmap = plt.get_cmap('jet')
        ex = (wt.min(), wt.max(), wf.min()*1e3, wf.max()*1e3)
        logMPS = 10.0*np.log10(mpsAVG)
        maxMPS = logMPS.max()
        minMPS = maxMPS-DBNOISE
        logMPS[logMPS < minMPS] = minMPS
        plt.imshow(logMPS, interpolation='spline16', aspect='auto', origin='lower', cmap=cmap, extent=ex)
        plt.ylabel('Spectral Frequency (Cycles/KHz)')
        plt.xlabel('Temporal Frequency (Hz)')

        plt.ylim((0,6))
        plt.xlim((-100, 100))
        plt.title('Average MPS')
        self.output().makedirs()
        plt.savefig(self.output().path)

class PlotMRFPredictedResponses(BaseTask):
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    stim_number = luigi.IntParameter(default=-1) # -1 will do all stims
    nPC = luigi.IntParameter(default=5)
    extension = luigi.Parameter(default="png")
    def requires(self):
        return {
            "mrf": GenerateMRF(site_dir=self.site_dir,cluster=self.cluster),
            "segments": GenSegmentSpikeDB(site_dir=self.site_dir,cluster=self.cluster),
            "trials": GenPlaybackSpikesDB(site_dir=self.site_dir,cluster=self.cluster)
        }
    def output(self):
        if not self.input()["trials"].path:
            raise Exception("No pb_df found, Run GenerateSTRFRequirements")
        pb_df = pandas.read_pickle(self.input()["trials"].path)
        num_stims = len(pb_df.Rendition_Name.unique())
        outputs = []
        if self.stim_number < 0:
            for i in range(num_stims//10):
                outputs.append(luigi.LocalTarget(os.path.join(_get_mrf_figure_folder(self.site_dir,self.cluster),"PredictedResponses","Preview_%d.%s"%(i,self.extension))))
            return outputs
        else:
            return luigi.LocalTarget(os.path.join(_get_mrf_figure_folder(self.site_dir,self.cluster),"PredictedResponses","PredResp_%d.%s"%(self.stim_number,self.extension)))
    def run(self):
        pb_df = pandas.read_pickle(self.input()["trials"].path)

        respPCPred = np.load(self.input()["mrf"]["pred_response"].path)
        pcResp = pickle.load(open(self.input()["mrf"]["response"].path,'rb'))["pcaResp"]

        seg_df = pickle.load(open(self.input()["segments"].path,'rb'))

        # plot predicted responses
        # Generate, Store and Plot the Predictions by rejoining the predictions for each segment.
        # Note that we are only using the positive R2

        # We are calculating these for a coherence estimation but this is not happening yet.
        concatPsthHalf1 = np.array([])
        concatPsthHalf2 = np.array([])
        pairCountTrials = np.zeros(len(pb_df))

        nChunk = len(seg_df)
        DBNOISE = 50  # Color Threshold for spectrogram

        wHann = windows.hann(21, sym=True)   # The 21 ms (number of points) hanning window used to smooth the PSTH

        # Color map for spectrograms
        spec_colormap()   # defined in sound.py
        if self.stim_number < 0:
            # If stim num is -1 we will plot all pred resp in batches of 10, with no stimuli
            pred_resps = []
            actual_resps = []
            sampleLens = []
            stim_names = []
            output_targets = self.output()
            stim_ix = 0
            nstims = len(pb_df.Rendition_Name.unique())
            for stim_name,stim_df in pb_df.groupby("Rendition_Name"):
                stim_names.append(os.path.basename(stim_name))
                # Get actual response using a weighted sum of gaussians
                spike_times = stim_df.Spike_Times - stim_df.Start
                peck_times = stim_df.Peck.values.astype(float)
                concatenated_spike_times = np.concatenate(np.asarray(spike_times))
                peck_times[peck_times == 0.0] = np.nan
                weights = np.array([1/(len(peck_times) - sum(peck_times < x)) for x in concatenated_spike_times])
                x_d = np.linspace(0,6,6001)#1ms bins
                bin_size = x_d[1]-x_d[0]
                resp = sum(norm(xi,scale=.008).pdf(x_d)*w for xi,w in zip(concatenated_spike_times,weights))

                # Get the spectrogram of the stimuli
                # first load the stim file
                wavFile = stim_df.PlaybackFile.iloc[0]
                fs , soundStim = wavfile.read(wavFile)
                # now use BioSound to make the spectrogram
                myBioSound = BioSound(soundWave=soundStim, fs=fs, emitter='STRF', calltype = 'song')
                myBioSound.spectroCalc(spec_sample_rate=1000, freq_spacing = 50, min_freq=0, max_freq=10000)

                tfrep = myBioSound.spectro
                t = myBioSound.to
                f = myBioSound.fo
                
                # Generate the prediction
                respPrediction = np.zeros_like(resp)

                stim_seg_df = seg_df[seg_df.Name == stim_name]
                for iseg,seg_row in stim_seg_df.iterrows():
                    # Start the prediction with the mean PC
                    samp_padding = int(seg_row.padding_ms)
                    samp_start = int(seg_row.tStart*1000)
                    samp_end = min(int(seg_row.tEnd*1000),samp_start+500-samp_padding)# TODO make this param

                    respIndStart = samp_padding
                    respIndEnd = respIndStart + (samp_end-samp_start)
                    respPrediction[samp_start:samp_end] += pcResp.mean_[respIndStart:respIndEnd]
                    # respPrediction[chunkID['indZero'][iChunk]:chunkID['indEnd'][iChunk]] += pcResp.mean_[segmentBuffer:chunkID['respIndEnd'][iChunk]]
                    # respPrediction[chunkID['indStart'][iChunk]:chunkID['indZero'][iChunk]] += pcResp.mean_[chunkID['respIndStart'][iChunk]:segmentBuffer]
                    
                    # Now add contribution of each PC fit if R2 above threshold
                    for iPC in range(self.nPC):
                        respPrediction[samp_start:samp_end] += respPCPred[iseg, iPC]*pcResp.components_[iPC,respIndStart:respIndEnd]
                        # respPrediction[chunkID['indZero'][iChunk]:chunkID['indEnd'][iChunk]] += respPCPred[iChunk, iPC]*pcResp.components_[iPC,segmentBuffer:chunkID['respIndEnd'][iChunk]]
                        # respPrediction[chunkID['indStart'][iChunk]:chunkID['indZero'][iChunk]] += respPCPred[iChunk, iPC]*pcResp.components_[iPC,chunkID['respIndStart'][iChunk]:segmentBuffer]

                # Get raw spike data divided into two for coherency calculations
                # if len(row.rawSpikeIndicies) > 1:
                #     psthdata = preprocSound.split_psth(row.rawSpikeIndicies, resp.shape[0])
                #     concatPsthHalf1 = np.concatenate((concatPsthHalf1, psthdata['psth_half1']))
                #     concatPsthHalf2 = np.concatenate((concatPsthHalf2, psthdata['psth_half2']))
                # numTrials = len(row.rawSpikeIndicies)  # One needs to check that all numTrials are equal
                # pairCountTrials[ix] = numTrials

                pred_resps.append(respPrediction)
                actual_resps.append(resp)
                if stim_ix % 10 == 10 - 1 or stim_ix == nstims-1:
                    # Plot the spectrogram and the prediction
                    f, arr_ax = plt.subplots(len(pred_resps), 1, sharex=True, dpi=100, figsize = (5,8))
                    for i in range(len(pred_resps)):
                        # The prediction
                        ax = arr_ax[i]
                        ix_this = stim_ix-10+i+1
                        #fac = row.sampleRate/(pairCountTrials[ix_this])   # The response is in sample rate (usually ms) and summed over trials. Here we correct to get spikes/s
                        ax.plot(x_d, actual_resps[i], 'k', label='Resp')
                        ax.plot(x_d, pred_resps[i], 'r', label='Pred')
                        ax.set_ylabel('%d' %ix_this)
                        ax.yaxis.set_ticklabels([])
                        # # put text showing how many trials
                        ax.text(np.mean(ax.get_xlim()),ax.get_ylim()[1]*.9,stim_names[i],ha='center',va='top')
                    arr_ax[-1].set_xlabel('Time (s)')
                    plt.legend()
                    file_target = output_targets[stim_ix//10]
                    file_target.makedirs()
                    plt.savefig(file_target.path)
                    pred_resps = []
                    actual_resps = []
                    stim_names = []
                    plt.close()
                stim_ix += 1
        else:
            # Get the stimulus (spectrogram) and the response
            #ds = srData['datasets'][iSet]
            stim_name = np.sort(pb_df.Rendition_Name.unique())[self.stim_number]
            stim_df = pb_df[pb_df.Rendition_Name == stim_name]
            # If stim num is -1 we will plot all pred resp in batches of 10, with no stimuli
        
            # Get actual response using a weighted sum of gaussians
            spike_times = stim_df.Spike_Times - stim_df.Start
            peck_times = stim_df.Peck.values.astype(float)
            concatenated_spike_times = np.concatenate(np.asarray(spike_times))
            peck_times[peck_times == 0.0] = np.nan
            weights = np.array([1/(len(peck_times) - sum(peck_times < x)) for x in concatenated_spike_times])
            x_d = np.linspace(0,6,6001)#1ms bins
            bin_size = x_d[1]-x_d[0]
            resp = sum(norm(xi,scale=.008).pdf(x_d)*w for xi,w in zip(concatenated_spike_times,weights))

            # Get the spectrogram of the stimuli
            # first load the stim file
            wavFile = stim_df.PlaybackFile.iloc[0]
            fs , soundStim = wavfile.read(wavFile)
            # now use BioSound to make the spectrogram
            myBioSound = BioSound(soundWave=soundStim, fs=fs, emitter='STRF', calltype = 'song')
            myBioSound.spectroCalc(spec_sample_rate=1000, freq_spacing = 50, min_freq=0, max_freq=10000)

            tfrep = myBioSound.spectro
            t = myBioSound.to
            f = myBioSound.fo
            
            # Generate the prediction
            respPrediction = np.zeros_like(resp)

            stim_seg_df = seg_df[seg_df.Name == stim_name]
            samp_start_stop = []
            for iseg,seg_row in stim_seg_df.iterrows():
                # Start the prediction with the mean PC
                samp_padding = int(seg_row.padding_ms)
                samp_start = int(seg_row.tStart*1000)
                samp_end = min(int(seg_row.tEnd*1000),samp_start+500-samp_padding)# TODO make this param
                samp_start_stop.append((seg_row.tStart,seg_row.tEnd))
                respIndStart = samp_padding
                respIndEnd = respIndStart + (samp_end-samp_start)
                respPrediction[samp_start:samp_end] += pcResp.mean_[respIndStart:respIndEnd]
                # respPrediction[chunkID['indZero'][iChunk]:chunkID['indEnd'][iChunk]] += pcResp.mean_[segmentBuffer:chunkID['respIndEnd'][iChunk]]
                # respPrediction[chunkID['indStart'][iChunk]:chunkID['indZero'][iChunk]] += pcResp.mean_[chunkID['respIndStart'][iChunk]:segmentBuffer]
                
                # Now add contribution of each PC fit if R2 above threshold
                for iPC in range(self.nPC):
                    respPrediction[samp_start:samp_end] += respPCPred[iseg, iPC]*pcResp.components_[iPC,respIndStart:respIndEnd]
                    # respPrediction[chunkID['indZero'][iChunk]:chunkID['indEnd'][iChunk]] += respPCPred[iChunk, iPC]*pcResp.components_[iPC,segmentBuffer:chunkID['respIndEnd'][iChunk]]
                    # respPrediction[chunkID['indStart'][iChunk]:chunkID['indZero'][iChunk]] += respPCPred[iChunk, iPC]*pcResp.components_[iPC,chunkID['respIndStart'][iChunk]:segmentBuffer]
        
            # Plot the spectrogram and the prediction
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=100, figsize = (8,4))

            # The spectrogram
            soundSpect = np.copy(tfrep)
            maxB = soundSpect.max()
            minB = maxB-DBNOISE
            soundSpect[soundSpect < minB] = minB
            minSpect = soundSpect.min()
            cmap = plt.get_cmap('SpectroColorMap')
            plt.title(os.path.basename(stim_name))
            ax1.imshow(tfrep, extent=[t[0], t[-1], f[0]*1e-3, f[-1]*1e-3],
                        aspect='auto', interpolation='nearest', origin='lower', cmap=cmap, vmin=minSpect, vmax=maxB)
            ax1.set_ylim(0, 8)
            for start,stop in samp_start_stop:
                ax1.axvline(start, 0,.5,c='r')
                ax1.axvline(stop, 0,.5,c='b')
            ax1.set_ylabel('Frequency (kHz)')
            # The prediction

            #fac = row.sampleRate/(pairCountTrials[ix_this])   # The response is in sample rate (usually ms) and summed over trials. Here we correct to get spikes/s
            ax2.plot(x_d, resp, 'k', label='Resp')
            ax2.plot(x_d, respPrediction, 'r', label='Pred')
            ax2.yaxis.set_ticklabels([])

            ax2.set_xlabel('Time (s)')

            rsq = np.corrcoef(resp, respPrediction)[0,1]**2
            ax2.text(np.mean(ax2.get_xlim()),ax2.get_ylim()[1]*.9,"R2 = %.3f"%rsq,ha='center',va='top')

            #plt.legend()
            self.output().makedirs()    
            plt.savefig(self.output().path)
                    


class RunMRFTasks(luigi.WrapperTask):
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    def output(self):
        # TODO I could put a list here of jobs ran
        return luigi.LocalTarget("dummy")
    def run(self):
        yield PlotPCMRFs(site_dir=self.site_dir,cluster=self.cluster)
        yield PlotMRFPCs(site_dir=self.site_dir,cluster=self.cluster)
        yield PlotMRFPredictedResponses(site_dir=self.site_dir,cluster=self.cluster,stim_number=1)
        yield PlotMRFPredictedResponses(site_dir=self.site_dir,cluster=self.cluster)
