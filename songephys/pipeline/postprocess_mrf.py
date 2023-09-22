
# region MRF
from scipy.signal import windows
from soundsig.sound import BioSound
from soundsig.sound import spec_colormap
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

import numpy as np
import luigi
import matplotlib.pyplot as plt
import os
import pandas
import pickle
from scipy.io import wavfile

from strfpy import preprocSound

from .postprocess_strf import GenerateSTRFRequirements
from .postprocess_operant import SiteProjectConfig,GenStimInfoDB
from .pipeline_utils import SiteParameter

class BaseTask(luigi.Task):
    force = luigi.BoolParameter()
    def complete(self):
        outputs = luigi.task.flatten(self.output())
        for output in outputs:
            if self.force and output.exists():
                output.remove()
        return all(map(lambda output: output.exists(), outputs))

def _get_segments(stim):
    """ Returns the segments of a wav file"""
    samplerate, data = wavfile.read(stim)
    data = np.pad(data, (50,0), 'constant', constant_values=(0, 0))
    # calculate ampenv with biosound
    myBioSound = BioSound(soundWave=data, fs=samplerate, emitter='STRF', calltype = 'song')
    myBioSound.ampenv(cutoff_freq = 20, amp_sample_rate = 1000)
    # lowpass filter the ampenv
    a,b = butter(6, 50/500, btype='low')
    filtered = filtfilt(a,b, myBioSound.amp[1:])
    # find peaks in the derivative
    diff = np.diff(filtered)
    peaks, _ = find_peaks(diff, height=np.std(diff)*.5, distance=5)
    down_peaks, _ = find_peaks(-diff, height=np.std(diff)*.5)
    # now pair the peaks
    segments = []
    for ix, peak in enumerate(peaks):
        if ix < len(peaks)-1:
            next_peak = peaks[ix+1]
            possible_down_peaks = down_peaks[(down_peaks > peak) & (down_peaks < next_peak)]
            if len(possible_down_peaks) > 0:
                # take the last possible down peak
                down_peak = possible_down_peaks[-1]
            else:
                # no down peak found, take the next peak
                down_peak = next_peak-10
            segment = [max(0,int(peak*samplerate/1000)-50), int(down_peak*fs/1000)-50]
            segments.append(segment)
    return segments



class GenSegmentsDB(luigi.Task):
    """ Generates database of segments for each stim"""
    site_dir = SiteParameter()
    def requires(self):
        return {
            "project_config": SiteProjectConfig(self.site_dir),
            "playback_info": GenStimInfoDB(self.site_dir)
        }
    
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","raw","dfSongSegments.pkl"))
    
    def run(self):
        # load the project config to get stim_drive location
        project_config = self.input()['project_config'].data()
        stim_dir = os.path.join(project_config['LOCAL_RAW_FILES']['stimuli_drive'],"Operant")

        playbacks_and_timings = pandas.read_pickle(self.input()['playback_info'].path)

        df_song_segments = {"Bird":[],"Name":[],"Segments":[]}
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

                segments = _get_segments(arr,fs)
                df_song_segments['Bird'].append(bird)
                df_song_segments['Name'].append(song)
                df_song_segments['Segments'].append(spectrogram)
            
        df_song_segments = pandas.DataFrame(df_song_segments)

        self.output().makedirs()
        df_song_segments.to_pickle(self.output().path)

class GenerateMRFSegments(BaseTask): # checked
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    def requires(self):
        return GenerateSTRFRequirements(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        return {
            "chunk_id": luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster['cluster_id'],"dfMRFChunkID.pkl")),
            "response": luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster['cluster_id'],'respMPS.npy')),
            "stimuli": luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster['cluster_id'],'stimMPSFeat.pkl'))
        }
    def run(self):
        pb_df = pandas.read_pickle(self.input().path)
        ampThresh = 24.0  # Threshold in dB where 50 is max
        minSilence = 10  # Minimum length of silence in points - time units given by stim sample rate
        minSound = 10  # Minimum length of sound in points
        segmentBuffer = 25 # Number of points on each side of segment for response and MPS - time units given by stim sample rate
        respChunkLen = 500 # Response length in number of points 

        wHann = windows.hann(21, sym=True)   # The 21 ms (number of points) hanning window used to smooth the PSTH

        stimMPS = []
        respMPS = []
        chunkID = {'set': [], 'indZero': [], 'indStart' : [], 'indEnd' : [], 'respIndStart' : [], 'respIndEnd' : []}

        for ix,row in pb_df.iterrows():
            wavFile = row.stim_rawfile

            fs , soundStim = wavfile.read(wavFile)
            soundLen = soundStim.size
            ampenv = np.mean(row.stim, axis=0)
            ampfs = row.sampleRate



            # plt.axhline(ampThresh, color = 'r')

            # Make a threshold
            soundOn = np.ones(ampenv.shape)
            soundOn[ampenv < ampThresh] = 0

            # Find edges and remove silences below 10 ms
            edges = soundOn[1:]-soundOn[0:-1]
            silBeg = np.argwhere(edges == -1)+1
            silEnd = np.argwhere(edges == 1)

            for silStart in silBeg:
                silFinish = np.argwhere(silEnd >= silStart)
                if silFinish.shape[0]:
                    dt = silEnd[silFinish[0][0]]-silStart
                    if dt < minSilence:
                        soundOn[int(silStart):int(silEnd[silFinish[0][0]]+1)] = 1.0

            # Remove sounds shorter than 10 ms
            edges = soundOn[1:]-soundOn[0:-1]
            soundEnd = np.argwhere(edges == -1)
            soundBeg = np.argwhere(edges == 1)+1

            for soundStart in soundBeg:
                soundFinish = np.argwhere(soundEnd >= soundStart)
                if soundFinish.shape[0]:
                    dt = soundEnd[soundFinish[0][0]]-soundStart
                    # print (soundStart, soundEnd[soundFinish[0][0]], dt)   
                    if dt < minSound:
                        # print(int(soundStart), int(soundEnd[soundFinish[0][0]]+1))
                        soundOn[int(soundStart):int(soundEnd[soundFinish[0][0]]+1)] = 0

            
            # Recalculate Edges
            edges = soundOn[1:]-soundOn[0:-1]
            soundEnd = np.argwhere(edges == -1)
            soundBeg = np.argwhere(edges == 1)+1

            for soundStart in soundBeg:
                soundFinish = np.argwhere(soundEnd >= soundStart)
                if soundFinish.shape[0]:
                    dt = soundEnd[soundFinish[0][0]]-soundStart
                    # print (soundStart, soundEnd[soundFinish[0][0]], dt)   
                    if dt < minSound:
                        print('WARNING - not expecting short sound here', int(soundStart), int(soundEnd[soundFinish[0][0]]+1))

                # Plot response
            resp = row.psth
            respSmoothed = np.convolve(resp, wHann, mode='same')
            respDataLen = respSmoothed.shape[0]

            # Extract the MPS for each edge use 25 ms on each side as buffer
            for soundStart in soundBeg:
                soundFinish = np.argwhere(soundEnd >= soundStart)
                if soundFinish.shape[0]:

                    # First estimate MPS for chunk
                    indStart = int((soundStart-segmentBuffer)*fs/ampfs)
                    if indStart < 0 :
                        indStart = 0
                    indEnd = int((soundEnd[soundFinish[0][0]]+segmentBuffer)*fs/ampfs)
                    if indEnd >= soundLen:
                        indEnd = soundLen - 1

                    wavSound = soundStim[indStart:indEnd+1]

                    # Create BioSound Object and calculate basic sound features
                    myBioSound = BioSound(soundWave=wavSound, fs=fs, emitter='STRF', calltype = 'song')
                    myBioSound.spectroCalc(spec_sample_rate=1000, freq_spacing = 50, min_freq=0, max_freq=10000)
                    myBioSound.rms = myBioSound.sound.std() 
                    myBioSound.ampenv(cutoff_freq = 20, amp_sample_rate = 1000)
                    myBioSound.spectrum(f_high=10000)
                    myBioSound.mpsCalc(window=0.05, Norm = True)  # Norm equals true normalizes the spectrogram but not the MPS
                
                    # Store in list
                    stimMPS.append(myBioSound.mps)

                    # Now extract response for the same chunk.
                    respChunk = np.zeros(respChunkLen)
                    indZero = int(soundStart)
        
                    indEnd = int((soundEnd[soundFinish[0][0]]+segmentBuffer))
                    if indEnd > respDataLen:
                        indEnd = respDataLen
                    respIndEnd = segmentBuffer + (indEnd-indZero)
                    if respIndEnd > respChunkLen:
                        respIndEnd = respChunkLen
                        indEnd = indZero + (respIndEnd-segmentBuffer)

                    respChunk[segmentBuffer:respIndEnd] = respSmoothed[indZero:indEnd]

                    respIndStart = 0
                    indStart = indZero - segmentBuffer
                    if indStart < 0:
                        indStart = 0
                        respIndStart = segmentBuffer - indZero

                    respChunk[respIndStart:segmentBuffer] = respSmoothed[indStart:indZero]

                    respMPS.append(respChunk)
                    chunkID['set'].append(ix)
                    chunkID['indStart'].append(indStart)
                    chunkID['indZero'].append(indZero)
                    chunkID['indEnd'].append(indEnd)
                    chunkID['respIndStart'].append(respIndStart)
                    chunkID['respIndEnd'].append(respIndEnd)

        # Save the chunkID dictionary
        self.output()["chunk_id"].makedirs()
        chunkID = pandas.DataFrame(chunkID)
        chunkID.to_pickle(self.output()["chunk_id"].path)

        # save response
        respMPS = np.array(respMPS)
        np.save(self.output()["response"].path, respMPS)
        
        # save stims
        stimMPS = np.array(stimMPS)
        # Use center of MPS for feature

        # Find the indices of the center
        wt = myBioSound.wt
        wf = myBioSound.wf
        indwt = np.argwhere((wt > -100) & (wt < 100))
        indwf = np.argwhere( (wf >= 0) & (wf <6e-3))

        stimMPSFeat = stimMPS[:,int(indwf[0]):int(indwf[-1]+1),int(indwt[0]):int(indwt[-1]+1)].reshape(stimMPS.shape[0], len(indwt)*len(indwf))
        stimMPSFeat = stimMPSFeat - stimMPSFeat.mean(axis=0)
        stimOutputStruct = dict({
            "stimMPS": stimMPSFeat,
            "wt": wt[indwt],
            "wf": wf[indwf],
            "segmentBuffer": segmentBuffer
        })
        pickle.dump(stimOutputStruct,open(self.output()["stimuli"].path,'wb'))

class GenerateMRF(BaseTask): # checked
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    def requires(self):
        return GenerateMRFSegments(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        return {
            "response": luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster['cluster_id'],'respMPSPC.pkl')),
            "pred_response": luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster['cluster_id'],'respMPSPCpred.npy')),
            "mrf_struct": luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster['cluster_id'],'stimMRF.pkl'))
        }
    def run(self):
        # Mean Response
        respMPS = np.load(self.input()["response"].path)

        # Calculate the PCs
        pcResp = PCA(n_components=10)
        respPC = pcResp.fit_transform(respMPS)
        out_resp_dict = dict({
            "pcaResp": pcResp,
            "responsePCs": respPC
        })
        pickle.dump(out_resp_dict,open(self.output()["response"].path,'wb'))

        # Perform cross-validated ridge regression for all PCs
        Alphas = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10)   # Ridge hyperparameter coeffiecients
        nPC = respPC.shape[1]

        stimDict = pickle.load(open(self.input()["stimuli"].path,'rb'))
        stimMPSFeat = stimDict["stimMPS"]
        wt = stimDict["wt"]
        wf = stimDict["wf"]

        # Allocate space for predictions
        respPCPred = np.zeros(respPC.shape)
        MRF = np.zeros((nPC, stimMPSFeat.shape[1]))


        for iPC in range(nPC):
            clf = RidgeCV(alphas=Alphas)
            clf.fit(stimMPSFeat, respPC[:,iPC])
            respPCPred[:,iPC] = clf.predict(stimMPSFeat)
            MRF[iPC,:] = clf.coef_
            print('PC %d: Best alpha = %g R2 = %.3f' % (iPC, clf.alpha_, clf.score(stimMPSFeat, respPC[:,iPC])))
        # save the calculated MRFs for each PC
        dictOutputMRF = dict({
            "MRF": MRF,
            "wt": wt,
            "wf": wf
        })
        pickle.dump(dictOutputMRF,open(self.output()["mrf_struct"].path,'wb'))
        np.save(self.output()["pred_response"].path, respPCPred)
        


class PlotPCMRFs(BaseTask): # checked
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    extension= luigi.Parameter(default="png")
    nPCs = luigi.IntParameter(default=5)
    def requires(self):
        return GenerateMRF(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        for i in range(self.nPCs):
            yield luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"MRF","PC_MRFs","PC%d_MRF.%s"%(i,self.extension)))
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
            plt.imshow(MRFPlot, interpolation='spline16', aspect='auto', origin='lower', cmap=cmap, extent=ex)
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

class PlotMRFPCs(BaseTask): # checked
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    nPCs = luigi.IntParameter(default=5)
    extension = luigi.Parameter(default="png")
    def requires(self):
        return {
            "segments": GenerateMRFSegments(site_dir=self.site_dir,cluster=self.cluster),
            "mrf": GenerateMRF(site_dir=self.site_dir,cluster=self.cluster)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"MRF","RespPCs.%s"%self.extension))
    def run(self):
        pcResp = pickle.load(open(self.input()["mrf"]["response"].path,'rb'))["pcaResp"]
        respMPS = np.load(self.input()["segments"]["response"].path)
        segmentBuffer = pickle.load(open(self.input()["segments"]["stimuli"].path,'rb'))["segmentBuffer"]
        # Plot the PCs
        plt.figure(figsize = (3,2), dpi=100)
        plt.subplot(2,1,1)
        plt.plot(respMPS.mean(axis=0))
        plt.axvline(segmentBuffer, color = 'k')
        plt.subplot(2,1,2)

        for iPC in range(self.nPCs):
            plt.plot(pcResp.components_[iPC,:], label='PC%d'%(iPC))
        plt.axvline(segmentBuffer, color = 'k')
        plt.xlabel('Time (ms)')
        plt.legend()

        # Make figure directory
        self.output().makedirs()
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
            "segments": GenerateMRFSegments(site_dir=self.site_dir,cluster=self.cluster),
            "pb_df": GenerateSTRFRequirements(site_dir=self.site_dir,cluster=self.cluster)
        }
    def output(self):
        if not self.input()["pb_df"].path:
            raise Exception("No pb_df found, Run GenerateSTRFRequirements")
        pb_df = pandas.read_pickle(self.input()["pb_df"].path)
        num_stims = len(pb_df)
        outputs = []
        if self.stim_number < 0:
            for i in range(num_stims//10):
                outputs.append(luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"MRF","PredictedResponses","Preview_%d.%s"%(i,self.extension))))
            return outputs
        else:
            return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"MRF","PredictedResponses","PredResp_%d.%s"%(self.stim_number,self.extension)))
    def run(self):
        pb_df = pandas.read_pickle(self.input()["pb_df"].path)

        MRFStruct = pickle.load(open(self.input()["mrf"]["mrf_struct"].path,'rb'))
        respPCPred = np.load(self.input()["mrf"]["pred_response"].path)
        pcResp = pickle.load(open(self.input()["mrf"]["response"].path,'rb'))["pcaResp"]

        stimDict = pickle.load(open(self.input()["segments"]["stimuli"].path,'rb'))
        stimMPSFeat = stimDict["stimMPS"]
        wt = stimDict["wt"]
        wf = stimDict["wf"]
        segmentBuffer = stimDict["segmentBuffer"]

        chunkID = pandas.read_pickle(self.input()["segments"]["chunk_id"].path)
        respMPS = np.load(self.input()["segments"]["response"].path)
        
        stimMPSFeat = stimDict["stimMPS"]
        wt = stimDict["wt"]
        # plot predicted responses
        # Generate, Store and Plot the Predictions by rejoining the predictions for each segment.
        # Note that we are only using the positive R2

        # We are calculating these for a coherence estimation but this is not happening yet.
        concatPsthHalf1 = np.array([])
        concatPsthHalf2 = np.array([])
        pairCountTrials = np.zeros(len(pb_df))

        nChunk = len(respMPS)
        DBNOISE = 50  # Color Threshold for spectrogram

        wHann = windows.hann(21, sym=True)   # The 21 ms (number of points) hanning window used to smooth the PSTH

        # Color map for spectrograms
        spec_colormap()   # defined in sound.py
        if self.stim_number < 0:
            # If stim num is -1 we will plot all pred resp in batches of 10, with no stimuli
            pred_resps = []
            actual_resps = []
            sampleLens = []
            output_targets = self.output()
            for ix,row in pb_df.iterrows():

                # Get the stimulus (spectrogram) and the response
                #ds = srData['datasets'][iSet]
                resp =row.psth
                tfrep = row.stim
                respSmoothed = np.convolve(resp, wHann, mode='same')
                
                # Generate the prediction
                respPrediction = np.zeros(respSmoothed.shape)
                for iChunk in range(nChunk):
                    if (chunkID['set'][iChunk] == ix):
                        
                        # Start the prediction with the mean PC
                        respPrediction[chunkID['indZero'][iChunk]:chunkID['indEnd'][iChunk]] += pcResp.mean_[segmentBuffer:chunkID['respIndEnd'][iChunk]]
                        respPrediction[chunkID['indStart'][iChunk]:chunkID['indZero'][iChunk]] += pcResp.mean_[chunkID['respIndStart'][iChunk]:segmentBuffer]
                        
                        # Now add contribution of each PC fit if R2 above threshold
                        for iPC in range(self.nPC):
                            #if (r2PC[iPC] > 0.01) :
                            respPrediction[chunkID['indZero'][iChunk]:chunkID['indEnd'][iChunk]] += respPCPred[iChunk, iPC]*pcResp.components_[iPC,segmentBuffer:chunkID['respIndEnd'][iChunk]]
                            respPrediction[chunkID['indStart'][iChunk]:chunkID['indZero'][iChunk]] += respPCPred[iChunk, iPC]*pcResp.components_[iPC,chunkID['respIndStart'][iChunk]:segmentBuffer]

                # Get raw spike data divided into two for coherency calculations
                if len(row.rawSpikeIndicies) > 1:
                    psthdata = preprocSound.split_psth(row.rawSpikeIndicies, resp.shape[0])
                    concatPsthHalf1 = np.concatenate((concatPsthHalf1, psthdata['psth_half1']))
                    concatPsthHalf2 = np.concatenate((concatPsthHalf2, psthdata['psth_half2']))
                numTrials = len(row.rawSpikeIndicies)  # One needs to check that all numTrials are equal
                pairCountTrials[ix] = numTrials
                
                sampleLens.append(row.sample_length)
                pred_resps.append(respPrediction)
                actual_resps.append(respSmoothed)
                if ix % 10 == 10 - 1 or ix == len(pb_df)-1:
                    # Plot the spectrogram and the prediction
                    f, arr_ax = plt.subplots(len(pred_resps), 1, sharex=True, dpi=100, figsize = (5,8))
                    for i in range(len(pred_resps)):
                        # The prediction
                        ax = arr_ax[i]
                        ix_this = ix-10+i+1
                        t = np.arange(0,sampleLens[i])
                        fac = row.sampleRate/(pairCountTrials[ix_this])   # The response is in sample rate (usually ms) and summed over trials. Here we correct to get spikes/s
                        ax.plot(t, actual_resps[i]*fac, 'k', label='Resp')
                        ax.plot(t, pred_resps[i]*fac*2, 'r', label='Pred')
                        ax.set_ylabel('%d' %ix_this)
                        ax.yaxis.set_ticklabels([])
                        ax.text(np.mean(ax.get_xlim()),ax.get_ylim()[1]*.9,"%d Trials"%pairCountTrials[ix_this],ha='center',va='top')
                    arr_ax[-1].set_xlabel('Time (s)')
                    plt.legend()
                    file_target = output_targets[ix//10]
                    file_target.makedirs()
                    plt.savefig(file_target.path)
                    pred_resps = []
                    actual_resps = []
                    sampleLens = []
                    plt.close()
        else:
            # Get the stimulus (spectrogram) and the response
            #ds = srData['datasets'][iSet]
            row = pb_df.iloc[self.stim_number]
            ix = self.stim_number
            resp =  row.psth
            tfrep = row.stim
            respSmoothed = np.convolve(resp, wHann, mode='same')
            
            # Generate the prediction
            respPrediction = np.zeros(respSmoothed.shape)
            for iChunk in range(nChunk):
                if (chunkID['set'][iChunk] == ix):
                    
                    # Start the prediction with the mean PC
                    respPrediction[chunkID['indZero'][iChunk]:chunkID['indEnd'][iChunk]] += pcResp.mean_[segmentBuffer:chunkID['respIndEnd'][iChunk]]
                    respPrediction[chunkID['indStart'][iChunk]:chunkID['indZero'][iChunk]] += pcResp.mean_[chunkID['respIndStart'][iChunk]:segmentBuffer]
                    
                    # Now add contribution of each PC fit if R2 above threshold
                    for iPC in range(nPC):
                        #if (r2PC[iPC] > 0.01) :
                        respPrediction[chunkID['indZero'][iChunk]:chunkID['indEnd'][iChunk]] += respPCPred[iChunk, iPC]*pcResp.components_[iPC,segmentBuffer:chunkID['respIndEnd'][iChunk]]
                        respPrediction[chunkID['indStart'][iChunk]:chunkID['indZero'][iChunk]] += respPCPred[iChunk, iPC]*pcResp.components_[iPC,chunkID['respIndStart'][iChunk]:segmentBuffer]

            # Get raw spike data divided into two for coherency calculations
            psthdata = preprocSound.split_psth(row.rawSpikeIndicies, resp.shape[0])
            concatPsthHalf1 = np.concatenate((concatPsthHalf1, psthdata['psth_half1']))
            concatPsthHalf2 = np.concatenate((concatPsthHalf2, psthdata['psth_half2']))
            numTrials = len(row.rawSpikeIndicies)  # One needs to check that all numTrials are equal
            pairCountTrials[ix] = numTrials
            
            # Plot the spectrogram and the prediction
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=100, figsize = (8,4))

            # The spectrogram
            soundSpect = np.copy(tfrep)
            t = np.arange(0,row.sample_length)
            maxB = soundSpect.max()
            minB = maxB-DBNOISE
            soundSpect[soundSpect < minB] = minB
            minSpect = soundSpect.min()
            cmap = plt.get_cmap('SpectroColorMap')
            
            ax1.imshow(tfrep, extent=[t[0], t[-1], row.freq[0]*1e-3, row.freq[-1]*1e-3],
                        aspect='auto', interpolation='nearest', origin='lower', cmap=cmap, vmin=minSpect, vmax=maxB)
            ax1.set_ylim(0, 8)
            ax1.set_ylabel('Frequency (kHz)')
            # ax1.set_xlabel('Time (s)')

            # The prediction
            fac = row.sampleRate/numTrials   # The response is in sample rate (usually ms) and summed over trials. Here we correct to get spikes/s
            ax2.plot(t, respSmoothed*fac, 'k', label='Resp')
            ax2.plot(t, respPrediction*fac, 'r', label='Pred')
            ax2.set_ylabel('Spikes/s')
            ax2.set_xlabel('Time (s)')
            plt.legend()
            plt.savefig(os.path.join(figDir, 'stimRespPred%d.eps' % (ix)))
                    



