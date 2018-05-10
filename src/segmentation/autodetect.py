from __future__ import print_function

import cPickle as pickle
from soundsig.signal import lowpass_filter, highpass_filter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from soundsig.sound import BioSound 
import pandas as pandas
from scipy.stats import zscore
from scipy.signal import decimate
import sys
from neo.io import NixIO, RHDIO



def get_biosound_properties_simple(soundIn, fs=30000.0, plotme = False, lowFc = 50, highFc = 4800, spectro_sample_rate=100): 
    """
    Returns biosound properties from a chunk of sound.
    First subtract mean (correct for DC offsets), and normalize by maximum amplitude (correct for birds singing towards/away from mic, etc).
    Typically only analyze frequencies between 50 and 4800 Hz because we have noise at 5kHz.
    Returns a dict of simple values (vocSelData), plus the spectrogram.
    
    spec_sample_rate: sampling rate for the output spectrogram in Hz (spectro), typically 100. This variable sets the overlap for the windows in the STFFT.

    """

    soundIn = soundIn - np.mean(soundIn,axis=0) # correct for DC offsets if they exist.
    #print 'Processing sound'
    maxAmp = np.asarray(np.abs(soundIn).max())      
    myBioSound = BioSound(soundWave=np.squeeze(np.asarray(soundIn).astype(float))/maxAmp, fs=fs, calltype = "") # normalize to maximum amplitude
                 
    # Create BioSound Object and store some values
    myBioSound.spectroCalc(spec_sample_rate=spectro_sample_rate,min_freq=lowFc,max_freq=highFc) # this creates the spectrogram, saved as myBioSound.spectro
    myBioSound.rms = myBioSound.sound.std()     # The rms   
    # Calculate the power spectrum
    myBioSound.spectrum(f_high=highFc)
    # Calculate fundamental and related values
    myBioSound.fundest(maxFund = 1500, minFund = 500, lowFc = lowFc, highFc = highFc, minSaliency = 0.5)   
#    myBioSound.fundest(maxFund, minFund, lowFc = 150, highFc = 6000, minSaliency = 0.5) FET CHANGES!!!!!!!!!!!!!!!1
    meanF1 = np.mean(myBioSound.F1[~np.isnan(myBioSound.F1)])
    meanF2 = np.mean(myBioSound.F2[~np.isnan(myBioSound.F2)])
    meanF3 = np.mean(myBioSound.F3[~np.isnan(myBioSound.F3)])

    spectro = myBioSound.spectro

    if myBioSound.fund is not None:
        if myBioSound.fund.size == 0:
            fund = None
            cvfund = None
        else:
            fund = np.float(myBioSound.fund)
            cvfund = np.float(myBioSound.cvfund)
    else:
        fund = None 
        cvfund = None   
    duration = len(soundIn)/fs
    
    vocSelData = {"fund": fund, 
                       "cvfund": cvfund, "F1": meanF1, "F2": meanF2, "F3" : meanF3,
                       "sal": np.float(myBioSound.sal), "rms": np.float(myBioSound.rms), 
                       "meanS": np.float(myBioSound.meanspect), "stdS": np.float(myBioSound.stdspect), 
                       "duration": duration, "skewspect":np.float(myBioSound.skewspect),
                       "kurtosisspect": np.float(myBioSound.kurtosisspect), "entropyspect": np.float(myBioSound.entropyspect)}

    return vocSelData, spectro
    
    
def fill_biosound_properties_empties():
    duration = np.nan
    
    vocSelData = {"fund": np.nan, 
                       "cvfund": np.nan, "F1": np.nan, "F2": np.nan, "F3" : np.nan,
                       "sal": np.nan, "rms": np.nan, 
                       "meanS": np.nan, "stdS": np.nan, 
                       "duration": np.nan, "skewspect":np.nan,
                       "kurtosisspect": np.nan, "entropyspect": np.nan}
    spectro = np.nan# np.zeros([1,1])
    return vocSelData, spectro

def get_biosound_properties(soundIn, fs=30000, plotme = False, lowFc = 50, highFc = 4800, spectro_sample_rate=100): 
    """
    Returns biosound properties from a chunk of sound.
    First subtract mean (correct for DC offsets), and normalize by maximum amplitude (correct for birds singing towards/away from mic, etc).
    Typically only analyze frequencies between 50 and 4900 Hz because we have noise at 5kHz.
    Returns a dict of simple values (vocSelData), plus PSD, frequency variances, spectrograms, and amplitude envelope.
    
    spec_sample_rate: sampling rate for the output spectrogram in Hz (spectro). This variable sets the overlap for the windows in the STFFT.

    """

    soundIn = soundIn - np.mean(soundIn,axis=0) # correct for DC offsets if they exist.
    #print 'Processing sound'
    maxAmp = np.asarray(np.abs(soundIn).max())      
    myBioSound = BioSound(soundWave=np.squeeze(np.asarray(soundIn).astype(float))/maxAmp, fs=fs, calltype = "") # normalize to maximum amplitude
                 
    # Create BioSound Object and store some values
    myBioSound.spectroCalc(spec_sample_rate=spectro_sample_rate,min_freq=lowFc,max_freq=highFc) # this creates the spectrogram, saved as myBioSound.spectro
    myBioSound.rms = myBioSound.sound.std()     # The rms  
    # Calculate amplitude enveloppe
    myBioSound.ampenv()  

    # Calculate the power spectrum
    myBioSound.spectrum(f_high=highFc)
    # Calculate fundamental and related values
    myBioSound.fundest(maxFund = 1500, minFund = 500, lowFc = lowFc, highFc = highFc, minSaliency = 0.5)   
#    myBioSound.fundest(maxFund, minFund, lowFc = 150, highFc = 6000, minSaliency = 0.5) FET CHANGES!!!!!!!!!!!!!!!1
    meanF1 = np.mean(myBioSound.F1[~np.isnan(myBioSound.F1)])
    meanF2 = np.mean(myBioSound.F2[~np.isnan(myBioSound.F2)])
    meanF3 = np.mean(myBioSound.F3[~np.isnan(myBioSound.F3)])

    if myBioSound.fund is not None:
        if myBioSound.fund.size == 0:
            fund = None
            cvfund = None
        else:
            fund = np.float(myBioSound.fund)
            cvfund = np.float(myBioSound.cvfund)
    else:
        fund = None 
        cvfund = None   
    
    vocSelData = {"fund": fund, 
                       "cvfund": cvfund, "F1": meanF1, "F2": meanF2, "F3" : meanF3,
                       "sal": np.float(myBioSound.sal), "rms": np.float(myBioSound.rms), 
                       "meanS": np.float(myBioSound.meanspect), "stdS": np.float(myBioSound.stdspect), 
                       "q1": np.float(myBioSound.q1), "q2": np.float(myBioSound.q2), "q3": np.float(myBioSound.q3),
                       "timestd": np.float(myBioSound.stdtime), "skewspect":np.float(myBioSound.skewspect),
                       "kurtosisspect": np.float(myBioSound.kurtosisspect), "entropyspect": np.float(myBioSound.entropyspect)}

    if plotme:  
        myBioSound.plot()  
        myBioSound.play()
        waitstr = raw_input("Press Enter to continue...")
        plt.close('all')

    fpsd = myBioSound.fpsd
    psd = myBioSound.psd
    freqvar = np.var(myBioSound.spectro, axis = 1) # variance of frequencies in the spectrogram
    f_freqvar = myBioSound.fo
    spectro = myBioSound.spectro
    ampenv = myBioSound.amp # biosound downsamples to 1000Hz always
    return vocSelData, fpsd, psd, freqvar, f_freqvar, spectro, ampenv


def test_random_forest(classifier_dataset = "", fs=25000, save_it = False, pca_model = ""):
    
    if classifier_dataset == "":
        classifier_dataset = "/auto/tdrive/billewood/intan data/LbY6074/LbY6074__161215_145633_categorized_final.pkl"
        
        # Import the classifier dataset and format if necessary
    with open(classifier_dataset,"r") as f:
        classifier_df = pickle.load(f)

    features = classifier_df.columns[range(classifier_df.shape[1]-2)]
    classifier_df['is_train'] = np.random.uniform(0, 1, len(classifier_df)) <= .5
    train, test = classifier_df[classifier_df['is_train']==True], classifier_df[classifier_df['is_train']==False] 
   
    target_names = np.unique(classifier_df['vocal_category'])
    y = train['vocal_category'].cat.codes

    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train[features], y)

#    non_pca_cols = range(data_df.shape[1]-classifier_features.shape[1])
    predictions = target_names[clf.predict(test[features])]

    pandas.crosstab(test['vocal_category'], predictions, rownames=['actual'], colnames=['preds'])


    return test, predictions    
    

def classify_vocalizations_random_forest(test_df, classifier_model_name = "/auto/tdrive/billewood/intan data/autodetection files/vocalization_sorting_classifier.pkl"):
    """
    Sorts the vocalizations with a presaved classifier from a conglomeration of files.
    def classify_vocalizations_random_forest(test_df, classifier_model_name = "/auto/tdrive/billewood/intan data/autodetection files/saved_classifier_3_2018.pkl"):
    The returned dataframe has NANs replaced with column means, also returns the predictions.
    """
    if type(classifier_model_name) == str:
        if classifier_model_name == "":
            classifier_model_name = "/auto/tdrive/billewood/intan data/autodetection files/vocalization_sorting_classifier.pkl"
        else:
            if not(os.path.isfile(classifier_model_name)):
                print("classifier file: {} is missing.".format(classifier_model_name))
                raise Exception
        with open(classifier_model_name,"r") as f:
            classifier_model = pickle.load(f)
    elif type(classifier_model_name) == RandomForestClassifier:
        pass

    test_df = test_df.fillna(test_df.mean()) # Can't have NANs
    classifier_features = ['F1', 'F2', 'F3', 'cvfund', 'duration', 'entropyspect', 'fund',
       'kurtosisspect', 'meanS', 'rms', 'sal', 'skewspect', 'stdS',
       'PCA column 0', 'PCA column 1', 'PCA column 2', 'PCA column 3',
       'PCA column 4', 'PCA column 5', 'PCA column 6', 'PCA column 7',
       'PCA column 8', 'PCA column 9', 'mic_diff']    
    target_names = np.asarray(['Adult Songs', 'Cage Noise', 'Distance Calls', 'Juvenile Songs', 'Tets', 'Unknown Calls', 'Unknown Vocalization'])
    predictions = target_names[classifier_model.predict(test_df[classifier_features])]
    test_df['vocal category'] = pandas.Series(predictions, dtype = "category")

    return test_df, predictions
    
def align_spectrograms(spectro_list, freq_bins = [], t_before = 30, t_after = 30):
    """
        Aligns a list of spectrograms to the maximum amplitude and zero pads around it.
        t_before is in data points, as is t_after. freq_bins restricts it to certain frequencies
        Not using this as of 2018
        
    """
    if freq_bins == []:
        freq_bins = [0,spectro_list[0].shape[0]]
    num_freq_bins = freq_bins[1] - freq_bins[0]
    shifted_spect_array = list()# np.array(np.zeros((spect_len,len(spectro_list))))
    for i, spect in enumerate(spectro_list):
        max_amp_ind = np.argmax(np.sum(spect[freq_bins[0]:freq_bins[1],:], axis = 0)) # find point of maximum amplitude
        # shift and pad array
        start_time = max_amp_ind - t_before
        end_time = max_amp_ind + t_after
        pre_pad = 0
        post_pad = end_time
        if max_amp_ind < t_before:
            start_time = 0
            pre_pad = t_before - max_amp_ind
        if end_time > spect.shape[1]:
            end_time = spect.shape[1]
            post_pad = post_pad - (end_time - spect.shape[1])
        shifted_spect = np.zeros([num_freq_bins, (t_before + t_after)])
        #print("i: %d  start_time: %d  end_time: %d  pre_pad: %d  post_pad: %d  max_amp: %d"%(i, start_time, end_time, pre_pad, post_pad, max_amp_ind))
        shifted_spect[:, pre_pad:pre_pad + (end_time-start_time)] = spect[freq_bins[0]:freq_bins[1],start_time:end_time]
        shifted_spect_array.append(shifted_spect)

    print('returning shifted spectogram array')
    return shifted_spect_array

def return_df_for_sorting_no_pca(audio_streams, vocal_periods, syllable_times, fs=30000.0,minfreq=100,maxfreq=4800,bird_name="generic id"):
    """
    Takes a long audio stream (or two), vocal periods, and syllable timings. Returns a pandas dataframe of biosound and spectral properties from the first two syllables, 
    or one if there is only one syllable found. 
    NOW it calculates the whole syllable, and if the syllable is shorter than ideal_len (80 ms) 
    then I grab more sound from the file. 
    The use of this function is to generate a PCA so that it can subsequently be used on all other files (in return_df_for_sorting)
    The current PCA model this was used to create is: /auto/tdrive/billewood/intan data/autodetection files/PCA_spectrogram_model.pkl
    4-6-18 Bill.
    """


    ideal_len = np.round(.08*fs) # 80 ms in data points
    signal = np.squeeze(np.asarray(audio_streams[:,0])) # tutor mic not implemented yet, just grabbing a mic channel
    new_df_data_syll_one = list() #will transform into a pandas dataframe eventually
    new_df_data_syll_two = list() #will transform into a pandas dataframe eventually
    spectro_list_syll_one = list() # contains the spectrograms for plotting later if desired
    spectro_list_syll_two = list()
    print("Calculating biosound properties and populating a database, this may take a while. autodetect.return_df_for_sorting.")
    syll1_durs = list()
    syll2_durs = list()

    for i, pt in enumerate(vocal_periods):
        if i == 0:
            print("Working on sound ", i+1, " out of ", len(vocal_periods), ". ")
        else: sys.stdout.write(','+str(i))
        l,h = pt
        sound = signal[l:h] - np.mean(signal[l:h],axis=0)
        sound = lowpass_filter(sound.T, sample_rate=fs, cutoff_freq=maxfreq)
        sound = highpass_filter(sound.T, fs, minfreq)
        if len(syllable_times[i])==0:#(syllable_times[i][0]==0).all(): # if there are no syllables at all
            new_biosound, spectro = fill_biosound_properties_empties()
            new_df_data_syll_one.append(new_biosound)
            spectro_list_syll_one.append(spectro)
            new_df_data_syll_two.append(new_biosound)
            spectro_list_syll_two.append(spectro)    
            syll1_durs.append(0)
            syll2_durs.append(0)        
        else: # at least one syllable identified
            syllable = sound[syllable_times[i][0,0]:syllable_times[i][0,1]]#+offset_in_ms]
            syll1_durs.append(len(syllable)/fs) # save the original duration
            if len(syllable)<ideal_len: # biosound fundestimator doesn't accept sounds shorter than 1024, and it's really best to have them all the same length
                pad = ideal_len-len(syllable)
                syll_start = syllable_times[i][0,0]-np.ceil(pad/2) # divide the pad equally between the start and the end, rounding up for the start
                extra_pad = 0
                if syll_start < 0: # if padding makes the syllable start before 0, take instead from the end
                    # print("syll:1. syll_start <0, syll_start={}".format(syll_start))
                    extra_pad = -syll_start 
                    # print("syll 1 start < 0 and i=0.")
                    # print("extrapad{}".format(extra_pad))
                    syll_start = 0
                syll_end =  syllable_times[i][0,1]+np.floor(pad/2) + extra_pad 
                if syll_end > len(sound):
                    # print("syll 1 end too long")
                    extra_pad = syll_end - len(sound)
                    syll_start = syll_start - extra_pad
                    syll_end = len(sound)
                syllable = sound[syll_start:syll_end]
                # print("vocal period #{} (syll 1) being pa
                # print("LENGHT SYLLABLE 1 <ideal, final length: {}".format(len(syllable)))
            else:
                syllable=syllable[0:ideal_len]
                # print("LENGHT SYLLABLE 1 plenty long: {}".format(len(syllable)))
            new_biosound, spectro = get_biosound_properties_simple(syllable, fs=fs)  
            new_df_data_syll_one.append(new_biosound)
            spectro_list_syll_one.append(spectro)
            if len(syllable_times[i]) > 1: # if there are two syllables
                syllable = sound[syllable_times[i][1,0]:syllable_times[i][1,1]]
                syll2_durs.append(len(syllable)/fs)#original syllable duration
                if len(syllable)<ideal_len: # biosound fundestimator doesn't accept sounds shorter than 1024... 
                    pad = ideal_len-len(syllable)
                    syll_start = syllable_times[i][1,0]-np.ceil(pad/2)
                    extra_pad = 0
                    if syll_start < 0:
                        extra_pad = -syll_start # add more to end instead of beginning if there's no sound remaining in the beginning
                        # print("syll 2 start <0 and i==0.")
                        syll_start = 0
                    syll_end =  syllable_times[i][1,1]+np.floor(pad/2) + extra_pad 
                    if syll_end > len(sound):
                        # print("syll2 end too long")
                        extra_pad = syll_end - len(sound)
                        syll_start = syll_start - extra_pad
                        syll_end = len(sound)
                    syllable = sound[syll_start:syll_end]
                    # print("syll 2 <ideal, new syll length:{}".format(len(syllable)))
                else:
                    syllable=syllable[0:ideal_len]
                    # print("syllable 2 plenty long: {}".format(len(syllable)))
                # print("LENGHT SYLLABLE: {}".format(len(syllable)))
                new_biosound, spectro = get_biosound_properties_simple(syllable, fs=fs) # 
                new_df_data_syll_two.append(new_biosound)
                spectro_list_syll_two.append(spectro)
            else:
                # print("no second syllable")
                new_biosound, spectro = fill_biosound_properties_empties()
                new_df_data_syll_two.append(new_biosound)
                spectro_list_syll_two.append(spectro)              
                syll2_durs.append(0)
    print("Finished most calculations, populating pandas database.")
    # This is a pretty rought spectro_list method, just stuck one after the other because I don't really care what the order is
    spectro_list = list()
    for i in range(len(spectro_list_syll_one)):
        if np.isnan(spectro_list_syll_one[i]).all():
            spectro_list.append(np.zeros((91,3))-80)
        elif np.isnan(spectro_list_syll_two[i]).all():
            spectro_list.append(np.hstack((spectro_list_syll_one[i],np.zeros((91,2))-80)))
        else:
            spectro_list.append(np.hstack((spectro_list_syll_one[i],np.zeros((91,1))-80,spectro_list_syll_two[i])))

    # Combine the two syllables into one df
    df = pandas.DataFrame(new_df_data_syll_one)  
    df2 = pandas.DataFrame(new_df_data_syll_two)
    df2.rename(columns=lambda x: x+" syll2", inplace=True)
    df = pandas.concat([df,df2],axis=1)

    if audio_streams.shape[1] > 1:
        annotated_pts, pt_diffs, std_idx = separate_song_sources(audio_streams, vocal_periods, fs)
        df['mic_diff'] = pt_diffs
        df['segment start and stop'] = list(vocal_periods)
        df['vocalizer'] = "tutor"
        df.loc[std_idx,'vocalizer'] = "student"   
    else:
        print("Only one microphone channel found, can not determine mic_diff or vocalizer.")
        df['mic_diff'] = [0 for jj in range(len(vocal_periods))]
        df['segment start and stop'] = list(vocal_periods)
        df['vocalizer'] = ["unknown" for jj in range(len(vocal_periods))]

    df["bird id"] = [bird_name for ii in range(len(df))]

    return df, spectro_list

def get_full_spectrograms(signal,vocal_periods,fs=30000):
    """Returns a list of spectrograms given a audio file and a series of timestamps
    """
    spectro_list = list()
    for period in vocal_periods:
        l,h = period
        sound = signal[l:h]
        maxAmp = np.asarray(np.abs(sound).max())      
        myBioSound = BioSound(soundWave=np.squeeze(np.asarray(sound).astype(float))/maxAmp, fs=fs, calltype = "") # normalize to maximum amplitude
        # this creates the spectrogram, saved as myBioSound.spectro
        myBioSound.spectroCalc(spec_sample_rate=100,min_freq=500,max_freq=10000) # this creates the spectrogram, saved as myBioSound.spectro
        spectro_list.append(myBioSound.spectro)
    return spectro_list

def get_partial_spectrograms(signal,vocal_periods,fs=30000, spect_length=50, spect_sample_rate=500):
    """Returns a np.array of flattened spectrograms given a audio file and a series of timestamps
    """
    points = fs/1000*spect_length # len is in ms, convert to data points
    min_freq=500
    max_freq=10000
    freq_spacing = 50
    num_freq_bins = ((max_freq-min_freq)/freq_spacing)-8 # The 8 is because you have less frequency bins then expected...
    spectro_flat_array = np.zeros([len(vocal_periods), np.int((spect_length/1000.0*spect_sample_rate)*num_freq_bins)])
    spectro_list = list()
    for i,period in enumerate(vocal_periods):   
        l,h = period
        h = l+points
        sound = signal[l:h]
        maxAmp = np.asarray(np.abs(sound).max())      
        myBioSound = BioSound(soundWave=np.squeeze(np.asarray(sound).astype(float))/maxAmp, fs=fs, calltype = "") # normalize to maximum amplitude
        # this creates the spectrogram, saved as myBioSound.spectro
        myBioSound.spectroCalc(spec_sample_rate=spect_sample_rate,freq_spacing=freq_spacing,min_freq=min_freq,max_freq=max_freq) # this creates the spectrogram, saved as myBioSound.spectro
        spectro_flat_array[i,:] = np.ravel(myBioSound.spectro)
        spectro_list.append(myBioSound.spectro)
    return spectro_flat_array, spectro_list

def get_spectrogram(sound,fs=30000):
    myBioSound = BioSound(soundWave=np.squeeze(np.asarray(sound).astype(float))/(np.asarray(np.abs(sound).max())), fs=fs, calltype = "") # normalize to maximum amplitude
    myBioSound.spectroCalc(spec_sample_rate=500,freq_spacing=50,min_freq=50,max_freq=10000) # this creates the spectrogram, saved as myBioSound.spectro
    return myBioSound.spectro

def return_df_for_sorting(audio_streams, vocal_periods, syllable_times, fs=30000.0,minfreq=100,maxfreq=4800,spect_pca_filename="/auto/tdrive/billewood/intan data/autodetection files/PCA_spectrogram_model_4_17.pkl",bird_name="generic id"):
    """
    Takes a long audio stream (or two), vocal periods, and syllable timings. Returns a pandas dataframe of biosound and spectral properties from the first two syllables, 
    or one if there is only one syllable found.
    It calculates features over the whole syllable, and if the syllable is shorter than ideal_len (80 ms), 
    then it grabs more sound from the file. 
    In progress 4/6/18 Bill
    """
    ideal_len_s = .08
    ideal_len = np.round(ideal_len_s*fs) # 80 ms in data points, for padding short syllables before PCAing the spectrogram.
    max_len = np.round(ideal_len_s*100) # number of time bins in spectrograms- 10 ms bins. Is maybe unnecessary now that my padding is better.
    signal = np.squeeze(np.asarray(audio_streams[:,0])) # tutor mic not implemented yet, just grabbing a mic channel
    new_df_data_syll_one = list() #will transform into a pandas dataframe eventually
    new_df_data_syll_two = list() #will transform into a pandas dataframe eventually
    spectro_list_syll_one = list() # contains the spectrograms for plotting later if desired
    spectro_list_syll_two = list()
    print("Calculating biosound properties and populating a database, this may take a while. autodetect.return_df_for_sorting.")
    # go through all vocal periods one by one



    syll1_durs = list() # save the original syllable durations for entering into df later
    syll2_durs = list()
    for i, pt in enumerate(vocal_periods):
        if i == 0:
            print("Working on sound ", i+1, " out of ", len(vocal_periods), ". ")
        else: sys.stdout.write(','+str(i))
        l,h = pt
        sound = signal[l:h] - np.mean(signal[l:h],axis=0)
        sound = lowpass_filter(sound.T, sample_rate=fs, cutoff_freq=maxfreq)
        sound = highpass_filter(sound.T, fs, minfreq)
        if len(syllable_times[i])==0:#(syllable_times[i][0]==0).all(): # if there are no syllables at all
            new_biosound, spectro = fill_biosound_properties_empties()
            new_df_data_syll_one.append(new_biosound)
            spectro_list_syll_one.append(spectro)
            new_df_data_syll_two.append(new_biosound)
            spectro_list_syll_two.append(spectro)    
            syll1_durs.append(0)
            syll2_durs.append(0)        
        else: # at least one syllable identified
            syllable = sound[syllable_times[i][0,0]:syllable_times[i][0,1]]#+offset_in_ms]
            syll1_durs.append(len(syllable)/fs) # save the original duration
            if len(syllable)<ideal_len: # biosound fundestimator doesn't accept sounds shorter than 1024, and it's really best to have them all the same length
                pad = ideal_len-len(syllable)
                syll_start = syllable_times[i][0,0]-np.ceil(pad/2) # divide the pad equally between the start and the end, rounding up for the start
                extra_pad = 0
                if syll_start < 0: # if padding makes the syllable start before 0, take instead from the end
                    # print("syll:1. syll_start <0, syll_start={}".format(syll_start))
                    extra_pad = -syll_start 
                    # print("syll 1 start < 0 and i=0.")
                    # print("extrapad{}".format(extra_pad))
                    syll_start = 0
                syll_end =  syllable_times[i][0,1]+np.floor(pad/2) + extra_pad 
                if syll_end > len(sound):
                    # print("syll 1 end too long")
                    extra_pad = syll_end - len(sound)
                    syll_start = syll_start - extra_pad
                    syll_end = len(sound)
                syllable = sound[syll_start:syll_end]
                # print("vocal period #{} (syll 1) being pa
                # print("LENGHT SYLLABLE 1 <ideal, final length: {}".format(len(syllable)))
            else:
                syllable=syllable[0:ideal_len]
                # print("LENGHT SYLLABLE 1 plenty long: {}".format(len(syllable)))
            new_biosound, spectro = get_biosound_properties_simple(syllable, fs=fs)  
            new_df_data_syll_one.append(new_biosound)
            spectro_list_syll_one.append(spectro)
            if len(syllable_times[i]) > 1: # if there are two syllables
                syllable = sound[syllable_times[i][1,0]:syllable_times[i][1,1]]
                syll2_durs.append(len(syllable)/fs)#original syllable duration
                if len(syllable)<ideal_len: # biosound fundestimator doesn't accept sounds shorter than 1024... 
                    pad = ideal_len-len(syllable)
                    syll_start = syllable_times[i][1,0]-np.ceil(pad/2)
                    extra_pad = 0
                    if syll_start < 0:
                        extra_pad = -syll_start # add more to end instead of beginning if there's no sound remaining in the beginning
                        # print("syll 2 start <0 and i==0.")
                        syll_start = 0
                    syll_end =  syllable_times[i][1,1]+np.floor(pad/2) + extra_pad 
                    if syll_end > len(sound):
                        # print("syll2 end too long")
                        extra_pad = syll_end - len(sound)
                        syll_start = syll_start - extra_pad
                        syll_end = len(sound)
                    syllable = sound[syll_start:syll_end]
                    # print("syll 2 <ideal, new syll length:{}".format(len(syllable)))
                else:
                    syllable=syllable[0:ideal_len]
                    # print("syllable 2 plenty long: {}".format(len(syllable)))
                # print("LENGHT SYLLABLE: {}".format(len(syllable)))
                new_biosound, spectro = get_biosound_properties_simple(syllable, fs=fs) # 
                new_df_data_syll_two.append(new_biosound)
                spectro_list_syll_two.append(spectro)
            else:
                # print("no second syllable")
                new_biosound, spectro = fill_biosound_properties_empties()
                new_df_data_syll_two.append(new_biosound)
                spectro_list_syll_two.append(spectro)              
                syll2_durs.append(0)
    # spectro_list is for visualizing with spikes-vis later, if wanted. For this we just want each syllable's spectrogram stuck next to the other syllable's.
    # but note that this isn't used for the actual sorting DF later
    spectro_list = list()
    for i in range(len(spectro_list_syll_one)):
        if np.isnan(spectro_list_syll_one[i]).all():
            spectro_list.append(np.zeros((91,3))-80)
        elif np.isnan(spectro_list_syll_two[i]).all():
            spectro_list.append(np.hstack((spectro_list_syll_one[i],np.zeros((91,2))-80)))
        else:
            spectro_list.append(np.hstack((spectro_list_syll_one[i],np.zeros((91,1))-80,spectro_list_syll_two[i])))

    # Load pca here
    with open(spect_pca_filename,"r") as file:
        spect_pca = pickle.load(file)

    # flatten the spectro_list and make a pca
    # Pads any spectrograms with zeros that are too short. Really shouldn't even happen anymore since I now try to pad accurately in the beginning.
    for j,syll_list in enumerate([spectro_list_syll_one, spectro_list_syll_two]):
        flat_spect_array = np.zeros([len(syll_list), 91 * max_len]) # 91 is the number of frequency bands 
        for i, spect in enumerate(syll_list):
            if all(np.isnan(np.ravel(spect))): # if there was no syllable, just fill with zeros i guess
                flat_spect_array[i,:] = np.zeros([91*max_len])
            else:
                if max_len > spect.shape[1]:
                    post_pad = np.zeros([spect.shape[0], max_len-spect.shape[1]])
                    spect = np.concatenate((spect[:,:], post_pad), axis=1)
                else:
                    spect = spect[:,0:max_len]
                flat_spect = zscore(np.ravel(spect))
                flat_spect_array[i,:] = flat_spect
            print("testing flat spect :{}".format(i))
        if j==0:
            pca_array_syll_one = spect_pca.transform(flat_spect_array)
        else:
            pca_array_syll_two = spect_pca.transform(flat_spect_array)

    # Combine the two syllables into one df
    df = pandas.DataFrame(new_df_data_syll_one)  
    df2 = pandas.DataFrame(new_df_data_syll_two)
    df2.rename(columns=lambda x: x+" syll2", inplace=True)
    df = pandas.concat([df,df2],axis=1)

    # combine pca vals into the DF
    for i in range(pca_array_syll_one.shape[1]):
        name = "PCA syll one col %d" %i
        df[name] = pca_array_syll_one[:,i]
    for i in range(pca_array_syll_two.shape[1]):
        name = "PCA syll two col %d" %i
        df[name] = pca_array_syll_two[:,i]

    # Replace the duration columns with durations taken from syllable_times, plus add the duration of the whole segment
    df["duration"] = syll1_durs
    df["duration syll2"] = syll2_durs

    seg_durs = [(voc[1]-voc[0])/fs for voc in vocal_periods]
    df["segment duration"] = seg_durs

    if audio_streams.shape[1] > 1:
        annotated_pts, pt_diffs, std_idx = separate_song_sources(audio_streams, vocal_periods, fs)
        df['mic_diff'] = pt_diffs
        df['segment start and stop'] = list(vocal_periods)
        df['vocalizer'] = "tutor"
        df.loc[std_idx,'vocalizer'] = "student"   
    else:
        print("Only one microphone channel found, can not determine mic_diff or vocalizer.")
        df['mic_diff'] = [0 for jj in range(len(vocal_periods))]
        df['segment start and stop'] = list(vocal_periods)
        df['vocalizer'] = ["unknown" for jj in range(len(vocal_periods))]

    df["bird id"] = [bird_name for ii in range(len(df))]

    return df, spectro_list

def return_df_for_sorting_old(audio_streams, vocal_periods,fs=25000.0,minfreq=100,maxfreq=4800,spect_pca_filename="/auto/tdrive/billewood/intan data/autodetection files/PCA_model_for_reducing_spectrogram_dimensions.pkl",bird_name="generic id",onset_in_ms = 190,offset_in_ms = 300):
    """
    TO be deleted as soon as I finish rewriting with syllables taken into account (above)
    onset_in_ms = 190 # .2 s added before onset, so start .19 into file
    offset_in_ms = 300 # taking the lenght of a short syllable, approximately 110 ms
    """
    signal = np.squeeze(np.asarray(audio_streams[:,0])) # tutor mic not implemented yet, just grabbing a mic channel
    new_df_data = list() #will transform into a pandas dataframe eventually
    spectro_list = list() # contains the spectrograms for plotting later
    print("Calculating biosound properties and populating a database, this may take a while.")
# go through all vocal periods one by one
    for i, pt in enumerate(vocal_periods):
        if i == 0:
            print("Working on sound ", i+1, " out of ", len(vocal_periods), ". ")
        # elif np.mod(i,40) == 0:
        #     print(',',i)
        else: sys.stdout.write(','+str(i))

        l,h = pt

        sound = signal[l:h] - np.mean(signal[l:h],axis=0)
        sound = lowpass_filter(sound.T, sample_rate=fs, cutoff_freq=maxfreq)
        sound = highpass_filter(sound.T, fs, minfreq)
        
        new_biosound, spectro = get_biosound_properties_simple(sound, fs=fs)
        new_df_data.append(new_biosound)
        spectro_list.append(spectro)
        
    print("Finished most calculations, populating pandas database."    )

    # load the pca and sort the spectrograms, I need to make this a function
    with open(spect_pca_filename,"r") as file:
        spect_pca = pickle.load(file)

    onset,offset = np.int(np.round(onset_in_ms / 10)), np.int(np.round(offset_in_ms / 10)) # 10 ms bins in spectrograms
    flat_spect_array = np.zeros([len(spectro_list), spectro_list[0].shape[0] * (offset-onset)])
    for i, spect in enumerate(spectro_list):
        if offset > spect.shape[1]:
            post_pad = np.zeros([spect.shape[0], offset-spect.shape[1]])
            spect = np.concatenate((spect[:,onset:], post_pad), axis=1)
        else:
            spect = spect[:,onset:offset]
        flat_spect = zscore(np.ravel(spect))
        flat_spect_array[i,:] = flat_spect
    pca_array = spect_pca.transform(flat_spect_array)

    # Combine the flattened PCAs with the biosound values
    df = pandas.DataFrame(new_df_data) # the " * 1" makes it not a pointer
    for i in range(pca_array.shape[1]):
        name = "PCA column %d" %i
        df[name] = pca_array[:,i]

    if audio_streams.shape[1] > 1:
        annotated_pts, pt_diffs, std_idx = separate_song_sources(audio_streams, vocal_periods, fs)
        df['mic_diff'] = pt_diffs
        df['segment start and stop'] = list(vocal_periods)
        df['vocalizer'] = "tutor"
        df.loc[std_idx,'vocalizer'] = "student"   
    else:
        print("Only one microphone channel found, can not determine mic_diff or vocalizer.")
        df['mic_diff'] = [0 for jj in range(len(vocal_periods))]
        df['segment start and stop'] = list(vocal_periods)
        df['vocalizer'] = ["unknown" for jj in range(len(vocal_periods))]

    df["bird id"] = [bird_name for ii in range(len(df))]
    """
     I believe the columns should be 
     [u'F1', u'F2', u'F3', u'cvfund', u'duration', u'entropyspect', u'fund',
       u'kurtosisspect', u'meanS', u'rms', u'sal', u'skewspect', u'stdS',
       u'PCA column 0', u'PCA column 1', u'PCA column 2', u'PCA column 3',
       u'PCA column 4', u'PCA column 5', u'PCA column 6', u'PCA column 7',
       u'PCA column 8', u'PCA column 9', u'mic_diff']
       Nah, now (3/20/18) I'm feeling like just duration plus 40 PCs...
       """
    return df, spectro_list



def populate_dataframe(audio_streams, points, fs=30000.0, minfreq=150, maxfreq=4800, return_big_df=True, spect_pca_filename = None, df_pca_filename=None, tutor_mic=0):
    """                 
        Given a long audio file and a set of periods of vocalizations.
        Returns a pandas database of sound properties useful for sorting and analyzing vocalizations.
        Most features come from the Theunissen lab Biosound object:
            sound.py, https://github.com/theunissenlab/soundsig.git)        
        Most features are self-evident, but:
            -Sound is lowpassed (maxfreq) and highpassed (minfreq) before any sound features are calculated.
                Extraneous frequency bands are also discarded according to these variables in some spectral measures.
            -psds and freqvars are cut to only include frequencies between minfreq and maxfreq.
            -freqvars is the variance of each frequency bin in the spectrogram.
            -The pandas df that is returned is pretty unwieldy but is in a good format 
            for PCA and t-sne analysis/visualization (amongst others)
        About dimensionality reduction:
        Spectrograms are PCAd to reduce dimensions with spect_pca_filename.
        The PCAd spectrograms, along with the PSDs, freqvars, and amplitude envelope, are PCAd to reduce dimensions with pca_model.
        This is what is returned in the PCA columns of df.
        Values with NaNs are replaced with column averages (without warnings).
    """
    signal = np.squeeze(np.asarray(audio_streams[:,tutor_mic])) # tutor mic not implemented yet, just grabbing a mic channel
    new_df_data = list() #will transform into a pandas dataframe eventually
    freqvars = []
    psds = []
    ampenvs = []
    spectro_list = list() # contains the spectrograms for plotting later
    
    # make a pandas DataFrame with biousound properties of all syllables/vocalizations
    print("Calculating biosound properties and populating a database, this may take a while.")
    #annotated_pts, pt_diffs, std_idx = separate_song_sources(audio_streams, points, fs, tutor_mic)

    for i, pt in enumerate(points):
        #print "points: ", points
        if i == 0:
            print("Working on sound ", i+1, " out of ", len(points), ". ")
        elif np.mod(i,40) == 0:
            print('..',i)
        else: sys.stdout.write('..'+str(i))
        l,h = pt
        
        #if i in annotated_pts['tutor']: # 2/18 I don't understand why this line exists commenting it out- Bill
        sound = lowpass_filter(signal[l:h].T, sample_rate=fs, cutoff_freq=maxfreq)
        sound = highpass_filter(sound.T, fs, minfreq)
        
        new_biosound, fpsd, psd, freqvar, f_freqvar, spectro, ampenv = get_biosound_properties(sound, fs=fs)
        new_df_data.append(new_biosound)
        spectro_list.append(spectro)
        ampenvs.append(ampenv)
        # cut out frequencies we aren't interested in for both the psd and the freqvar
        if minfreq is not None and maxfreq is not None:
            psds.append(psd[(fpsd >= minfreq) & (fpsd <= maxfreq)])
            fpsd = fpsd[(fpsd >= minfreq) & (fpsd <= maxfreq)]
            freqvars.append(freqvar[(f_freqvar >= minfreq) & (f_freqvar <= maxfreq)])
            f_freqvar = f_freqvar[(f_freqvar >= minfreq) & (f_freqvar <= maxfreq)]
    print("Finished most calculations, populating pandas database."    )

    # This section is ugly because I'm making a column for each entry instead of putting
    # arrays in cells. Makes classification more straightforward later.    
    
    df = pandas.DataFrame(new_df_data)
    durations = np.expand_dims(map(lambda x: x[1] - x[0], points), 1)
    df['duration'] = durations # add a duration column
    psds = np.asarray(psds)
    freqvars = np.asarray(freqvars)
    ampenvs = np.asarray(ampenvs)
    #print "Adding fpsd columns and values"    
    for i in range(len(fpsd)): #add a column for every frequency bin of the PSD
        name = 'PSDf=%d' %fpsd[i]
        df[name] = psds[:,i]
    #print "Adding variance across frequency columns and values"
    for i in range(len(f_freqvar)): #add a column for every frequency bin of freqvar
        name = 'VARf=%d' %f_freqvar[i]
        df[name] = freqvars[:,i]
 
   # cut and downsample ampenvs, saves a lot of space
    maxl = max(len(l) for l in ampenvs) # longest ampenv    
    if maxl > 1000: maxl = 1000 # never save more than a second of ampenvs
    ampenvs = np.array([xi[0:maxl] for xi in ampenvs]) # cut long arrays
    ampenvs = np.array([np.pad(xi,[0,(maxl-len(xi))],"constant") for xi in ampenvs]) # pad short arrays
    ampenvs = np.array([decimate(xi,5) for xi in ampenvs]) #downsample ampenvs to 200 Hz
    print("Adding amplitude envelope values")
    for i in range(ampenvs.shape[1]): # add a column for every time bin of the amplitue envelope
        name = 'AMPENV t=%d' %i
        df[name] = ampenvs[:,i]

# TODO I have reservations about how these spectrograms are being manipulated. make t_before into seconds

    shifted_spectro_list = align_spectrograms(spectro_list, freq_bins = [], t_before = 30, t_after = 30)    

    flat_spect_array = np.zeros([len(shifted_spectro_list), shifted_spectro_list[0].shape[0] * shifted_spectro_list[0].shape[1]])
    for i, spect in enumerate(shifted_spectro_list):
        flat_spect = np.ravel(spect)
        flat_spect_array[i,:] = flat_spect

    if spect_pca_filename is not None:
        print("Calculating PCAs of aligned spectrograms and saving in dataframe.")
        print("This involves projecting onto a precomputed PCA space contained at: {}.".format(spect_pca_filename))
        with open(spect_pca_filename,"r") as f:
            spect_pca = pickle.load(f)
    else:
        print("Calculating PCAs of aligned spectrograms and saving in dataframe.")
        print("Creating a generic PCA space")
        spect_pca = PCA(n_components=40)
        spect_pca.fit(flat_spect_array)

    pca_array = spect_pca.transform(flat_spect_array) # PSD, freqVAR, and ampenv freatures. 

    for i in range(pca_array.shape[1]):
        name = 'spectro pca #%d' %i
        df[name] = pca_array[:,i]
      
    print("Checking for NANS and replacing with column averages")
    print("dataframe shape: ", df.shape)
    
    # check it all for NANs and replace with column averages
    df = df.fillna(df.mean())
      
    #print "Finished calculating biosound properties, calculating vocalizer and mic_diff."      
    # Add a column with the amplitude ratio of the mics and a putative identify of caller.
    df['segment start and stop'] = list(points)
    # df['mic_diff'] = pt_diffs
    # df['vocalizer'] = "tutor"
    # df.loc[std_idx,'vocalizer'] = "student"              
    # add categorical columns for sorting later.
    df['vocal_category'] = pandas.Series('Unknown Vocalization', dtype = "category") 
    df['vocal_category'] = "Unknown Vocalization"

    if return_big_df:
        big_df = df
    else:
        big_df = pandas.DataFrame()
        
    pca_df, df_pca = reduce_dimensions_pca(df, fs, df_pca_filename) # pca_df is df after applying the pca, df_pca is the PCA that was applied.
    
    return pca_df, spectro_list, big_df, spect_pca, df_pca #spectrogram list, for plotting spectrograms


def test_find_vocal_periods(audio_stream_in, fs=30000, fuse_duration=0.01, shortest_duration=0.01, test_mode=True, amp_thresh=None, normalize_sound=False, normalize_thresh=None, lowpass = 2000, highpass = 5000, analog_signals = None):
    """ 
    This basically just implements a test mode for find_vocal_periods such that you can plot some thresholds that are being automatically generated. Generally just use find_vocal_periods
    Bill 2/2018
    
        fuse_duration -         Gaps shorter than this will be fused into single syllables
        shortest_duration -     Syllables shosrter than this will be ignored
        test_mode -             Plots a bunch of things for helping determine ideal parameter values- in work
        amp_thresh -            The amplitude threshold that you're going to be changing the most to get this all to work right
        normalize_sound -       Divides the sound by the mean amplitude to normalize it, recommended to be True
        normalize_thresh -      When normalize_sound=True all values below normalize_thresh first get set to 0- keeps the noise floor from being amplified.
        lowpass_thresh and 
        highpass_thresh -       It can be surprisingly useful to only search over a very limited bandwidth. Don't be afraid to try 6kHz to 9kHz or something
        analog_signals -        Send a series of signals if you want your histograms to look any good
    """
    if test_mode:
        # try lowpass filtering it to throw away "noise" in fast frequency correlations
        if analog_signals:
            
            plt.figure()
            plt.plot(amp_env[0])
            plt.title("Raw Smoothed Amplitude Envelope- one of many")
            plt.show()
        else:
            signal = np.copy(audio_stream_in)        
            signal = highpass_filter(signal, fs, highpass)
            signal = np.abs(lowpass_filter(signal, fs, lowpass))
            amp_env = lowpass_filter(signal, fs, 100)
            plt.figure()
            plt.plot(amp_env)
            plt.title("Raw Smoothed Amplitude Envelope")
            plt.show()
            
            # normalize amplitude to near 1 in case this matters... make really low values just equal zero first to fix stretching        
            low_vals = np.where(amp_env<normalize_thresh)
            amp_env[low_vals] = 0
            amp_env = (1/np.mean(amp_env)) * amp_env
            plt.figure()
            plt.plot(amp_env)
            plt.title("normalized amplitude envelope")
    
            # for figuring out amp_thresh, or enter it in hardcoded on next cell
            a = analogsignals[0]
            plt.figure()
            plt.hist([a for a in analogsignals[0] if a > 0], bins=100)
            n, bins = np.histogram([a for a in analogsignals[0] if a > 0], bins=100)
            amp_thresh = bins[np.argmax(n)+2] 
            print("Amplitude threshold is: {}".format(amp_thresh))

    syll_times, _ = find_vocal_periods(
        audio_stream_in=audio_stream_in,
        fs=fs,
        fuse_duration=fuse_duration,
        shortest_duration=shortest_duration,
        amp_thresh=amp_thresh,
        normalize_sound=normalize_sound,
        normalize_thresh=normalize_thresh,
        lowpass=lowpass,
        highpass=highpass)

    return syll_times, amp_thresh

def get_amp_thresh(audio_stream_in, fs=None, lowpass = 5000, highpass = 2000):
    """
    Calculate amplitude waveform over a large recording and calculate the mean. That's all.
    I use to calculate an amplitude threshold for a whole recording period, and then pass it to each individual segment.
    """
    if fs is None:
        print("Warning, no sampling rate passed, fs set to 30kHz")
        fs = 30000
    audio_stream = np.squeeze(audio_stream_in)

    # First we try to find the amplitude envelope. We only care about vocal sounds,
    # so we high-pass filter higher than normal.
    streams_filtered_env = np.zeros(np.shape(audio_stream))
    streams_filtered_env = highpass_filter(audio_stream, fs, highpass)
    streams_filtered_env = np.abs(lowpass_filter(streams_filtered_env, fs, lowpass)) # Rectify and lowpass filter
    streams_filtered_env = lowpass_filter(streams_filtered_env, fs, 100) # We essentially have an amplitude trace now # was 100

    audio_stream = streams_filtered_env
    # Simply threshold against the mean of the signal. This works surprisingly well.
    # If you think about it, all of the noises that get made during a session drag
    # the mean to be slightly above background noise. 
    amp_thresh = np.mean(audio_stream) 

    return amp_thresh

def return_syllable_times(audio_streams, vocal_periods, fs, amp_thresh=None):
    """
    return_syllable_times(audio_streams, vocal_periods, fs, amp_thresh=None):

    A helper function that just calls find_vocal_periods with particular paramaters to return syllable times for all the different sound segments.
    Also returns vocal_periods of only those segments that had syllables found (others are likely trash)
    Also removes any dc offset if it exists:     audio_streams = audio_streams-np.mean(audio_streams)

    syllables, thresh = find_vocal_periods(sound, fs=fs, amp_thresh=amp_thresh, fuse_duration=.005, shortest_duration=.02,verbose=False)

     """


    # It's also a good idea to make sure there is no DC offset in the sound files
    audio_streams = audio_streams-np.mean(audio_streams)
    if amp_thresh==None:
        amp_thresh = get_amp_thresh(audio_streams, fs=fs)
    good_index = list()
    syllable_list = list()
    # try sorting syllables
    for index, period in enumerate(vocal_periods):
        l,h = period
        sound = audio_streams[l:h]
        syllables, thresh = find_vocal_periods(sound, fs=fs, amp_thresh=amp_thresh, fuse_duration=.005, shortest_duration=.02,verbose=False)
        if len(syllables)>0:
            syllable_list.append(syllables)
            good_index.append(index)
    vocal_periods = vocal_periods[good_index]
    return syllable_list, vocal_periods


def find_vocal_periods(audio_stream_in, fs=None, fuse_duration=0.2, shortest_duration=0.08, verbose=True, amp_thresh=None, normalize_sound=False, normalize_thresh=0.006, lowpass = 5000, highpass = 2000):
    """
    Given an audio stream, attempts to detect all segments of noise that might be bird calls of
    some sort.

    Parameters
    ----------
    audio_stream_in: np.array of floats.
        The audio stream. Will be immediately squeezed.
    fs: float.
        The sampling rate of audio_stream, in Hz.
    fuse_duration: float.
        A threshold duration (in seconds). Any vocal segments that are closer than this together
        will get fused into a single segment.
    verbose: bool.
        Specifies whether or not to print error messages.
    amp_thresh: Threshold for amplitude.
        If None uses the mean, which works pretty well.
    Returns
    -------
    np.array of (int, int).
        Each tuple represents a (low_index:high_index) interval in the original array.
    """
    if fs is None:
        print("Warning, no sampling rate passed, fs set to 30kHz")
        fs = 30000
    if verbose:
        if fuse_duration != 0.2:
            print("Warning, vocalization sorting optimized for fuse_duration of 0.2 s, you are using {}, which MIGHT cause problems, but probably is ok.".format(fuse_duration))
        if shortest_duration != 0.08:
            print("Warning, vocalization sorting optimized for shortest_duration of 0.08 s, you are using {}, which MIGHT cause problems, but probably is ok.".format(shortest_duration))


    # audio_stream = np.squeeze(audio_stream_in)
    # First we try to find the amplitude envelope. We only care about vocal sounds,
    # so we high-pass filter higher than normal.
    streams_filtered_env = np.zeros(np.shape(audio_stream_in))
    streams_filtered_env = highpass_filter(audio_stream_in, fs, highpass)
    streams_filtered_env = np.abs(lowpass_filter(streams_filtered_env, fs, lowpass)) # Rectify and lowpass filter
    streams_filtered_env = lowpass_filter(streams_filtered_env, fs, 100) # We essentially have an amplitude trace now # was 100
    # normalize amplitude to near 1 in case this matters... make really low values just equal zero first to fix stretching
    if normalize_sound:
        if normalize_thresh is None: # autocalculate a threshold based on histogram, a little sketchy- should be checked if using this
            n, bins = np.histogram([a for a in streams_filtered_env if a > 0], bins=100)
            normalize_thresh = bins[np.argmax(n)+1]
        print("setting values below {} to 0.".format(normalize_thresh))
        low_vals = np.where(streams_filtered_env < normalize_thresh)
        streams_filtered_env[low_vals] = 0
        streams_filtered_env = (1.0/np.mean(streams_filtered_env)) * streams_filtered_env

    # Simply threshold against the mean of the signal. This works surprisingly well.
    # If you think about it, all of the noises that get made during a session drag
    # the mean to be slightly above background noise. 
    if amp_thresh is None:
        amp_thresh = np.mean(streams_filtered_env) * 1
    # sound onsets: 1 is an onset, -1 is an offset
    vd_crosses = np.zeros(len(streams_filtered_env))
    for i in xrange(len(streams_filtered_env)):
        if streams_filtered_env[i] > amp_thresh:
            vd_crosses[i] = 1
    sound_onsets = vd_crosses[1:len(vd_crosses)] - vd_crosses[0:len(vd_crosses)-1]
    # Aligns crosses to proper index:
    sound_onsets = np.concatenate([np.array([0]), sound_onsets])

    # this just checks that the first sound onset wasn't obscured by the beginning of the file
    for i in range(len(sound_onsets)):
        if sound_onsets[i] == -1:
            sound_onsets[0] = 1  # the case where sound onset happens on the first data point
            if verbose:
                print("Warning, sound appears to onset before file start, first "
                      + "time point being set to an onset")
        if sound_onsets[i] == 1:
            break
    # check for missing offset (sound continues past data file)
    for i in range(len(sound_onsets)):
        if sound_onsets[len(sound_onsets)-i-1] == 1:
            sound_onsets[-1] = -1
            if verbose:
                print("Warning, sound appears to continue past file end, "
                      + "last time point being set to an offset")
        if sound_onsets[len(sound_onsets)-i-1] == -1:
            break

    # Fuse all of the nearby offset/onsets
    onsets = np.where(sound_onsets == 1)[0]
    offsets = np.where(sound_onsets == -1)[0]

    diffs = onsets[1:]-offsets[:-1]
    good = np.where(diffs>fuse_duration*fs)[0]
    if len(good)>0: 
        good_offsets = np.hstack([offsets[good],offsets[-1]])
        good = good+1
        good_onsets = np.hstack([onsets[0],onsets[good]])
        if good_onsets[0] < 0:
            good_onsets[0] = 0
            if verbose:
                print("First onset near beginning of file, rounding to account for fuse duration forced it to before file onset, "
                    + "first onset set to 0")
        vocal_periods = np.array(zip(good_onsets,good_offsets))
        lengths = np.apply_along_axis(lambda p: p[1] - p[0], 1, vocal_periods)
        vocal_periods = vocal_periods[lengths > shortest_duration * fs]
        if verbose:
            print("After fusing and cutting shortest, {} vocal periods remain".format(len(vocal_periods)))
    else:
        vocal_periods = []
    return vocal_periods, amp_thresh
    


def extract_birdsong_features(signal, points, fs=25000, minfreq=1000, maxfreq=8000,
                              pca_model=None):
    """
    This function is given a signal and a set of intervals, and extracts the frequency
    characteristics of the interval. It then projects them onto a pretrained set of principle
    components. Finally, it appends the length of the interval. All features
    are intended to be used with a classifer to classify vocal segment type.

    Parameters
    ----------
    signal: np.array of float.
        A time series representing an audio recoding.
    points: np.array of (int, int).
        Each tuple represents a (low_index:high_index) interval in the original array.
    fs: float.
        The sampling rate of the signal in Hz.
    minfreq: float.
        The minimum frequency to be considered for the feature set.
    maxfreq: float.
        The maximum frequency to be considered for the feature set.
    pca_model: string.
        A string pointing to a .pkl file specifying the trained PC-space. Note: the minfreq and
        maxfreq parameters must match whatever dataset the PC space was fit on. May set None
        to not reduce the dimensionality of spectral features.

    Returns
    -------
    np.array of float. Shape (n_points, n_features)
        A numpy array of the resulting features for each point.
    """
    if pca_model is not None:
        pca_file = os.path.join(os.path.dirname(__file__), pca_model)
        pca_space = pickle.load(open(pca_file, "rb"))
    feats = []
    minfreq = 1000
    maxfreq = 8000

    # all_lengths = scipy.stats.zscore(np.expand_dims(map(lambda x: x[1] - x[0], points), 1))
    all_lengths = np.expand_dims(map(lambda x: x[1] - x[0], points), 1)
    for pt in points:
        l, h = pt
        Pxx, freqs, _ = mlab.specgram(signal[l:h], 256, Fs=fs,
                                      detrend=mlab.detrend_none,
                                      window=mlab.window_hanning)
        if minfreq is not None and maxfreq is not None:
            Pxx = Pxx[(freqs >= minfreq) & (freqs <= maxfreq)]
        Z = 10. * np.log10(Pxx)
        Z = np.flipud(Z)
        Z = np.maximum(Z, np.max(Z) - 50)
        feats.append(np.mean(np.square(Z.T), axis=0))

    feats = np.array(feats)
    proj_feats = pca_space.transform(feats) if pca_model is not None else feats
    features = np.concatenate([proj_feats, all_lengths], axis=1)
    return features


def classify_birdsong(signal_in, points, fs=25000.0,
                      model='models/songvnoise_classifier_noz_big_nopca.pkl'):
    """
    Given a signal and a set of points, classifies which of the points are birdsong (versus other
    sorts of noise, such as calls, tets, and flapping/hopping).

    Parameters
    ----------
    signal_in: np.array of float.
        A time series representing an audio recoding.
    points: np.array of (int, int).
        Each tuple represents a (low_index:high_index) interval in the original array.
    fs: float.
        The sampling rate of the signal in Hz.
    model: string.
        A string pointing to a .pkl file specifying the trained SVM. Note: PCA model used by
        extract_birdsong_features() must have been trained with this model.

    Returns
    -------
    An array of labels, specified by the model. The default model labels as either 'birdsong' or
    'notbirdsong'.
    """
    signal = np.squeeze(signal_in)
    svm_file = os.path.join(os.path.dirname(__file__), model)
    svm = pickle.load(open(svm_file, "rb"))

    features = extract_birdsong_features(signal, points, fs=fs)
    return svm.predict(features)


def separate_song_sources(audio_streams, points, fs=25000, tutor_mic=0):
    """
    Uses the two mic streams to separate the student and the tutor.

    Parameters
    ----------
    audio_stream: np.array of floats. Shape (n_samples, 2)
        The audio streams, one from each mic.
    points: np.array of (int, int).
        Each tuple represents a (low_index:high_index) interval in the original array.
    fs: float.
        The sampling rate of the signal in Hz.
    tutor_mic: 0 or 1.
        The index of the mic which was closer to the student.

    Returns
    -------
    A dictionary with two lists of points, of keys 'student' and 'tutor'.
    """
    print("...Filtering Arrays to get Amp Envelope for calculating vocalizer")
    streams_filtered_env = np.zeros(np.shape(audio_streams))
    streams_filtered_env[:, 0] = highpass_filter(np.squeeze(audio_streams[:, 0]), fs, 2000)
    streams_filtered_env[:, 1] = highpass_filter(np.squeeze(audio_streams[:, 1]), fs, 2000)
    streams_filtered_env[:, 0] = np.abs(lowpass_filter(streams_filtered_env[:, 0], fs, 4800))
    streams_filtered_env[:, 1] = np.abs(lowpass_filter(streams_filtered_env[:, 1], fs, 4800))
    streams_filtered_env[:, 0] = lowpass_filter(streams_filtered_env[:, 0], fs, 100)
    streams_filtered_env[:, 1] = lowpass_filter(streams_filtered_env[:, 1], fs, 100)

    def get_amp_envelope(coordinates, swindow=1000):
        low, high = coordinates
        return streams_filtered_env[low:high, :]
    diff = lambda c: (np.mean(get_amp_envelope(c), axis=0)[0]
                      - np.mean(get_amp_envelope(c), axis=0)[1])
    #print np.shape(points)
    pt_diffs = np.expand_dims(map(diff, points), axis=1)
    #print pt_diffs
    #print np.shape(pt_diffs)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pt_diffs)
    if tutor_mic == 0:
        student_label = np.argmin(kmeans.cluster_centers_)
    else:
        student_label = np.argmax(kmeans.cluster_centers_)
    student_idx = np.where(kmeans.labels_ == student_label)[0]
    tutor_idx = np.where(kmeans.labels_ != student_label)[0]

    annotated_pts = {}
    annotated_pts['student'] = points[student_idx]
    annotated_pts['tutor'] = points[tutor_idx]
    return annotated_pts, pt_diffs, student_idx

def read_audio_from_nix(nix_filename):
    """ 
    Just a semi-stupid audio importer. It does normalize the sound by dividing by the max, and subtracts any dc offset
    """
    h5_filename = nix_filename   
    if not(os.path.isfile(h5_filename)):
        print("Error:{} is missing".format(h5_filename))
        raise Exception
    h5_exporter = NixIO(h5_filename)
    block = h5_exporter.read_block()
    sound_recording = block.segments[0].analogsignals[1].magnitude
    fs = np.double(np.round(block.segments[0].analogsignals[1].sampling_rate))
    sound_recording = block.segments[0].analogsignals[1].magnitude
    norm_sound = np.zeros(sound_recording.shape)
    for mic in range(sound_recording.shape[1]):
        norm_sound[:,mic] = sound_recording[:,mic] - np.mean(sound_recording[:,mic])
    for mic in range(norm_sound.shape[1]):
        norm_sound[:,mic] = norm_sound[:,mic]/np.max(norm_sound[:,mic])
    h5_exporter.close()

    return norm_sound, fs

    
def normalize_sound_by_the_second(audio_streams, fs):
    """
    Fixes DC offsets that vary over time. Really hopefully this doesn't happen to the data, but was present in some of the
    earliest recordings, such as RedGre7705M__171214_093157
    """
    segments = range(int(np.ceil(len(audio_streams[:,0])/fs)))
    segments = [seg*int(fs) for seg in segments]
    segments.append(len(audio_streams[:,0]))
    segments = zip(segments[0:-1],segments[1:])
    for i in range(audio_streams.shape[1]):
        for seg in (segments):
            m,n = seg
            audio_streams[m:n,i] = audio_streams[m:n,i] - np.mean(audio_streams[m:n,i])
    return audio_streams

