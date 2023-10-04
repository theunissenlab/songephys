
# region STRF Estimation
import numpy as np
import luigi
import matplotlib.pyplot as plt
import os
import pandas
import pickle
from scipy.io import wavfile

from strfpy import findDatasets, preprocSound, strfSetup, trnDirectFit, plotTfrep,calcAvg,calcAutoCorr,calcCrossCorr, calcStrf

from .postprocess_operant import SiteProjectConfig,GenStimInfoDB,GenPlaybackSpikesDB
from .pipeline_utils import SiteParameter


class GenerateSTRFRequirements(luigi.Task):
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    def requires(self):
        return GenPlaybackSpikesDB(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster['cluster_id'],"dfPreprocessedSTRFStimResponsePairs.pkl"))
    def run(self):
        playbacks_df = pandas.read_pickle(self.input().path)
        preprocOptions = {}
        preprocDir = '/tmp/preproc'
        srData = preprocSound.preprocess_sound_from_df(playbacks_df, preprocess_type='ft', stim_params = preprocOptions, output_dir = preprocDir)
        
        # Convert srData into a format that strflab understands
        dfStrfLab = strfSetup.srdata2strflab(srData)
        dfStrfLab.to_pickle(self.output().path)


class EstimateClusterSTRF(luigi.Task):
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    def requires(self):
        return GenerateSTRFRequirements(site_dir=self.site_dir,cluster=self.cluster)
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","data","Cluster_%d"%self.cluster['cluster_id'],"clusterBestSTRFwithCC.pkl"))
    def run(self):
        dfStrfLab = pickle.load(open(self.input().path, 'rb'))

        # Create default optimization options structure
        params, optOptions = trnDirectFit.trnDirectFit()
        optOptions['display'] = 1

        stim_avg, resp_avg, avg_firing_rate = calcAvg.calculateAverage(dfStrfLab)
        stim_avg_mo, resp_avg_mo = calcAvg.calculateLeaveOneOutAverages(dfStrfLab=dfStrfLab,apply_resp_smoothing=True)

        twindow=74
        # =========================================================
        # Now calculating stimulus AutoCorr.
        # =========================================================
        autocorr = calcAutoCorr.calculateAutoCorrelation(dfStrfLab, twindow)
        crosscorr, crosscorr_JN = calcCrossCorr.calculateCrossCorr(dfStrfLab, twindow)
        fstim, stim_respf, stim_resp_JNf = calcCrossCorr.fftSmoothCorrelations(autocorr,crosscorr, crosscorr_JN, twindow,nstd=.5)
        
        # compute some indices to use later
        strfLength = 75
        strfDelays = np.arange(strfLength)
        modelParams = strfSetup.linInit(stim_avg.shape[0], strfDelays)
        halfIndx = twindow + 1
        startIndx = halfIndx + round(np.min(modelParams['delays']))
        endIndx = halfIndx + round(np.max(modelParams['delays']))
        strfRng = range(startIndx-1, endIndx)

        bestInfoVal = -1
        bestStrf = -1
        bestTol = -1
        bestSparseness = -1

        for tol in optOptions['tolerances']:
            strfMean, strfsJN, strfsJN_std = calcStrf.calculateSTRF(fstim, stim_respf, stim_resp_JNf, tol=tol)
            strfStdMean = np.mean(strfsJN_std,axis=2)

            for sp in optOptions['sparsenesses']:
                smoothedMeanStrf = trnDirectFit.df_fast_filter_filter(strfMean, strfStdMean, sp)
                smoothedMeanStrfToUse = smoothedMeanStrf[:,strfRng]

                infoSum = 0
                numJNStrfs = strfsJN.shape[2]
                infoTBins = 0
                infos = np.zeros(numJNStrfs)
                infotbins = np.zeros(numJNStrfs)
                for p in range(numJNStrfs):
                    smoothedMeanStrfJN = trnDirectFit.df_fast_filter_filter(strfsJN[:, :, p], strfsJN_std[:, :, p], sp)
                    strfToUse = smoothedMeanStrfJN[:, strfRng]

                    stim = dfStrfLab.stim.iloc[p].T
                    rresp = dfStrfLab.psth.iloc[p]

                    gindx = np.ones((1,stim.shape[0]))

                    mresp = trnDirectFit.conv_strf(stim, modelParams['delays'], strfToUse, gindx)
                    mresp = mresp + dfStrfLab.psth_loo_avg.iloc[p][:mresp.shape[1]].T

                    # compute coherence and info across pairs
                    cStruct = trnDirectFit.compute_coherence_mean(mresp, rresp, optOptions['respSampleRate'], optOptions['infoFreqCutoff'], optOptions['infoWindowSize'] )

                    infos[p] = np.real(cStruct['info'])
                    infotbins[p] = len(rresp)
                # TODO not sure why there are some nans probably because no spikes in the window
                infotbins = infotbins[~np.isnan(infos)]
                infos = infos[~np.isnan(infos)]
                avgInfo = np.sum(infos*infotbins)/np.sum(infotbins)

                print(f"Tolerance={tol}, Sparseness={sp}, Avg. Prediction Info={avgInfo}")

                # did this sparseness do better?
                if avgInfo > bestInfoVal:
                    bestTol = tol
                    bestSparseness = sp
                    bestInfoVal = avgInfo
                    bestStrf = smoothedMeanStrfToUse

        print('Best STRF found at tol=%f, sparseness=%d, info=%f bits\n' % (bestTol, bestSparseness, bestInfoVal))
        trained_model = dict({
            "bestTol": bestTol,
            "bestSparseness": bestSparseness,
            "bestInfoVal": bestInfoVal,
            "bestStrf": bestStrf,
            "avg": resp_avg,
            'avg_fr': avg_firing_rate
        })
        pickle.dump(trained_model,open(self.output().path,'wb'))
        #np.save(self.output().path,bestStrf)

class PlotClusterSTRF(luigi.Task):
    site_dir = SiteParameter()
    cluster = luigi.DictParameter()
    def requires(self):
        return {
        "dfStrflab": GenerateSTRFRequirements(site_dir=self.site_dir,cluster=self.cluster),
        "bestSTRF": EstimateClusterSTRF(site_dir=self.site_dir,cluster=self.cluster)
        }
    def output(self):
        return luigi.LocalTarget(os.path.join(self.site_dir,"analysis","figures","clusters_%s"%self.cluster['group'],"Cluster_%d"%self.cluster['cluster_id'],"clusterBestSTRF.png"))
    def run(self):
        bestStrf = np.load(self.input()["bestSTRF"].path)
        dfStrfLab = pickle.load(open(self.input()["dfStrflab"].path, 'rb'))

        # display the best STRF
        plt.figure(figsize=(3,2), dpi=100)

        # Looks at first ds to examine frequencies
        strfFrequencies = dfStrfLab.iloc[0]['freq']
        strfLength = 75
        strfDelays = np.arange(strfLength)

        mySTRF = np.real(bestStrf)
        maxabs = np.max(np.abs(mySTRF))

        plt.imshow( mySTRF, origin='lower', cmap = 'jet', aspect='auto',
                extent=[strfDelays[0], strfDelays[-1], strfFrequencies[0]*1e-3, strfFrequencies[-1]*1e-3],
                vmin=-maxabs,
                vmax=maxabs)

        plt.xlabel('Delay (ms)')
        plt.ylabel('Frequency (kHz)')
        plt.title('Best STRF')
        plt.savefig(self.output().path)




# endregion



