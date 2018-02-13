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

    
def get_biosound_properties(soundIn, fs=30000, plotme = False, lowFc = 50, highFc = 4900, spec_sample_rate=100): # FET CHANGES!!!!!!!!, minFund = 500, maxFund = 3000):
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
    myBioSound.spectroCalc(spec_sample_rate=spec_sample_rate)
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
    

def reduce_dimensions_pca(df, fs, pca_model="", pca_columns="", zscore_columns=""):
    """
        Takes a pandas dataframe created with autodetect.populate_dataframe.
        Reduces the dimensionality by PCAing most of the columns,
        Zscores the other columns, returns a np array ready to be sorted.
    """        
        
    if zscore_columns == "":
        zscore_columns = df.columns[range(16) + [-3]]
        print("columns to be zscored: ", zscore_columns)
    else:
        print("Warning changing the default columns might still be buggy...")
        
    if pca_columns == "":
        pca_columns = df.columns[range(17,len(df.columns)-5)+[-3]] #avoid last few, too
        print("PCA columns are: %s"%pca_columns)
    else:
        print("Warning changing the default columns is probably buggy..."    )
    
    if pca_model == "":
        print("No PCA model (pkl file) passed for determining pca dimensionality reduction, attempting to use hard-coded one")
        pca_model = '/auto/tdrive/billewood/intan data/LbY6074/LbY6074__161215_145633_PCA.pkl'
        print("Using pca model at {}".format(pca_model))
    
    with open(pca_model,"r") as f:
        pca = pickle.load(f)    
        
    zscore_array = np.array(df[zscore_columns].apply(zscore))

    pca_df = pandas.DataFrame()

    for col in range(len(zscore_columns)):
        pca_df["%s zscore"%zscore_columns[col]] = zscore_array[:,col]
    
    pca_df["duration"] = [np.diff(seg)/fs for seg in df["segment start and stop"]]

    
    pca = PCA(n_components=40) #
    pre_pca_array = df[pca_columns]
    pca_array = pca.fit_transform(pre_pca_array) # PSD, freqVAR, and ampenv freatures.
    

    for col in range(pca_array.shape[1]):
        pca_df["PCA %i"%col] = pca_array[:,col]
        
        # Then add in the categorical/descriptive data
    for col in df.columns[[-4]+[-2]+[-1]]:
        pca_df["%s"%col] = df[col]    

    PCAd_array = pca.transform(df[pca_columns]) # PSD, freqVAR, and ampenv freatures.
    zscore_array = np.array(df[zscore_columns].apply(zscore))
    
    for col in range(PCAd_array.shape[1]):
        pca_df["PCA %i"%col] = PCAd_array[:,col]

    
    pca_df['vocal_category'] = df['vocal_category']

    return pca_df


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
    
    
def classify_vocalizations_random_forest(data_df, classifier_dataset = "", fs=25000, save_it = False, pca_model = ""):
    
    if type(classifier_dataset) == str:
        if classifier_dataset == "":
            classifier_dataset = "/auto/tdrive/billewood/intan data/LbY6074/LbY6074__161215_145633_categorized_final.pkl"
        with open(classifier_dataset,"r") as f:
            classifier_df = pickle.load(f)
    elif type(classifier_dataset) == pandas.DataFrame:
        classifier_df = classifier_dataset
    else:
        print("classifier_dataset not recognized")


#    last_feature = min(classifier_df.shape[0], data_df.shape[0])        
#    classifier_features = classifier_df.columns[range(last_feature) + [-3]]
#    print classifier_features

    classifier_features = classifier_df.columns[0:-3]    
    print("Classifier Features are: %s" %classifier_features)
    #Shouldn't need to train it every time TODO save the actual trained matrix whatchamacallit, working as is    
    train = classifier_df

    target_names = np.unique(classifier_df['vocal_category'])
    y = train['vocal_category'].cat.codes

    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train[classifier_features], y)
#    non_pca_cols = range(data_df.shape[1]-classifier_features.shape[1])

    predictions = target_names[clf.predict(data_df[classifier_features])]

    data_df['vocal_category'] = pandas.Series(predictions, dtype = "category")
#    df['vocal_category'] = pandas.Series('UV', dtype = "category") 

    return data_df, predictions

def align_spectrograms(spectro_list, freq_bins = [], t_before = 30, t_after = 30):
    """
        Aligns a list of spectrograms to the maximum amplitude and zero pads around it.
        t_before is in data points, as is t_after. freq_bins restricts it to certain frequencies
        
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

def populate_dataframe(audio_streams, points, fs=30000.0, minfreq=400, maxfreq=7000, num_pcs_to_save=50, spect_pca_filename = "/auto/tdrive/billewood/intan data/LbY6074/spect_pca_combined.pkl", return_big_df=0, pca_model="", lowpass_thresh=4800, tutor_mic=0):
    """
        Given a long audio file and a set of periods of vocalizations.
        Returns a pandas database of sound properties useful for sorting and analyzing vocalizations.
        Most features come from the Theunissen lab Biosound object:
            sound.py, https://github.com/theunissenlab/soundsig.git)        
        Most features are self-evident, but:
            -Spectrogram derived data is only calculated for frequency bins
            between minfreq and maxfreq.
            -freqvars is the variance of each frequency bin in the spectrogram.
            -The pandas df that is returned is pretty unwieldy but is in a good format 
            for PCA and t-sne analysis/visualization (amongst others)
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
    annotated_pts, pt_diffs, std_idx = separate_song_sources(audio_streams, points, fs, tutor_mic)

    for i, pt in enumerate(points):
        #print "points: ", points
        if i == 0:
            print("Working on sound ", i+1, " out of ", len(points), ". ")
        elif np.mod(i,40) == 0:
            print('..',i)
        else: sys.stdout.write('..'+str(i))
        l,h = pt
        
        #if i in annotated_pts['tutor']: # 2/18 I don't understand why this line exists commenting it out- Bill
        sound = lowpass_filter(signal[l:h].T, sample_rate=fs, cutoff_freq=lowpass_thresh)
        sound = highpass_filter(sound.T, fs, 300)
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


    print("Calculating PCAs of aligned spectrograms and saving in dataframe.")
    print("This involves projecting onto a precomputed PCA space contained at: {}.".format(spect_pca_filename))
    with open(spect_pca_filename,"r") as f:
        pca = pickle.load(f)

# TODO I have great reservations about how these spectrograms are being manipulated.

    shifted_spectro_list = align_spectrograms(spectro_list, freq_bins = [10,90], t_before = 30, t_after = 30)    

    flat_spect_array = np.zeros([len(shifted_spectro_list), shifted_spectro_list[0].shape[0] * shifted_spectro_list[0].shape[1]])
    for i, spect in enumerate(shifted_spectro_list):
        flat_spect = np.ravel(spect)
        flat_spect_array[i,:] = flat_spect

    pca_array = pca.transform(flat_spect_array) # PSD, freqVAR, and ampenv freatures.   
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
    df['mic_diff'] = pt_diffs
    df['vocalizer'] = "tutor"
    df.loc[std_idx,'vocalizer'] = "student"              
    # add categorical columns for sorting later.
    df['vocal_category'] = pandas.Series('UV', dtype = "category") 
    df['vocal_category'] = "UV"

    if return_big_df:
        big_df = df
    else:
        big_df = pandas.DataFrame()
        
    pca_df = reduce_dimensions_pca(df, fs, pca_model)
    
    return pca_df, spectro_list, big_df #spectrogram list, for plotting spectrograms


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

def find_vocal_periods(audio_stream_in, fs=30000, fuse_duration=0.2, shortest_duration=0.6, verbose=False, amp_thresh=None, normalize_sound=False, normalize_thresh=0.006, lowpass = 2000, highpass = 5000):
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

    Returns
    -------
    np.array of (int, int).
        Each tuple represents a (low_index:high_index) interval in the original array.
    """
    audio_stream = np.squeeze(audio_stream_in)
    # First we try to find the amplitude envelope. We only care about vocal sounds,
    # so we high-pass filter higher than normal.
    streams_filtered_env = np.zeros(np.shape(audio_stream))
    streams_filtered_env = highpass_filter(audio_stream, fs, highpass)
    streams_filtered_env = np.abs(lowpass_filter(streams_filtered_env, fs, lowpass)) # Rectify and lowpass filter
    streams_filtered_env = lowpass_filter(streams_filtered_env, fs, 100) # We essentially have an amplitude trace now
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
        amp_thresh = np.mean(streams_filtered_env) 
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
            print("Warning, sound appears to onset before file start, first "
                  + "time point being set to an onset")
        if sound_onsets[i] == 1:
            break
    # check for missing offset (sound continues past data file)
    for i in range(len(sound_onsets)):
        if sound_onsets[len(sound_onsets)-i-1] == 1:
            sound_onsets[-1] = -1
            print("Warning, sound appears to continue past file end, "
                  + "last time point being set to an offset")
        if sound_onsets[len(sound_onsets)-i-1] == -1:
            break

    # Fuse all of the nearby offset/onsets
    onsets = np.where(sound_onsets == 1)[0]
    offsets = np.where(sound_onsets == -1)[0]
    n_fused = 0
    if len(onsets) > 1:
        onset_length = len(onsets)
        print("number of vocal periods = {}".format(onset_length))
        last_onset = onsets[1]
        last_offset = offsets[0]
        assert last_offset < last_onset
        for i in xrange(last_onset, len(sound_onsets)):
            if sound_onsets[i] == 1:
                last_onset = i
            elif sound_onsets[i] == -1:
                if (last_onset - last_offset) < fuse_duration*fs:
                    sound_onsets[last_onset] = 0
                    sound_onsets[last_offset] = 0
                    n_fused += 1
                last_offset = i
    if verbose:
        print("num fused: " + str(n_fused))

    # convert to a list of just onsets and offsets
    sound_onset = np.where(sound_onsets == 1)[0]
    sound_offset = np.where(sound_onsets == -1)[0]
    try:
        len(sound_onset) == len(sound_offset)
    except:
        print("Number of vocal onsets and offsets not equal!")
        
    sound_onset -= np.int(np.round(fuse_duration*fs))
    sound_offset += np.int(np.round(fuse_duration*fs))

    if sound_onset[0] < 0:
        sound_onset[0] = 0
        if verbose:
            print("First onset near beginning of file, rounding to account for fuse duration forced it to before file onset, "
                + "first onset set to 0")
    vocal_periods = np.array(zip(sound_onset, sound_offset))
    lengths = np.apply_along_axis(lambda p: p[1] - p[0], 1, vocal_periods)
    vocal_periods = vocal_periods[lengths > shortest_duration * fs]
    if verbose:
        print("After fusing and cutting shortest, {} vocal periods remain".format(len(vocal_periods)))

    return vocal_periods, amp_thresh
    
def find_syllables_old(audio_stream_in, amp_thresh = '', fs=25000, fuse_duration=0.1,  syll_dur = 20.0, sil_dur = 5.0, verbose=False):
 
    """
    This function is not yet very functional (Bill, 9/2017)- in fact as you can see it's "Old"
    For segmenting into syllables. The basic idea is implementing multiple amplitude threshholds, each of which has it's own duration threshold...
    I now think you could make the duration threshold dependent on the amplitude, such that the quieter it is, the shorter the gap needs to be to call it a new sylable.
    This is really just here for posterity.
    Bill 2/2018 
    """
    audio_stream = np.squeeze(audio_stream_in)
    # First we try to find the amplitude envelope. We only care about vocal sounds,
    # so we high-pass filter higher than normal.
    streams_filtered_env = np.zeros(np.shape(audio_stream))
    streams_filtered_env = highpass_filter(audio_stream, fs, 2000)
    streams_filtered_env = np.abs(lowpass_filter(streams_filtered_env, fs, 5000)) # Rectify and lowpass filter
    env = lowpass_filter(streams_filtered_env, fs, 100)

    if amp_thresh == '':
        amp_thresh = np.mean(env)
    # Now iterate through amplitude envelop and look for syllables. It would be more pythonic to do this in a 
    # linear algebraish way, but I am going to just try to get it to work for now.
    sound_on = 0 # keep count of how many points had sound
    sil_on = 0 #  " " silence
    hold = 0 # hold the value of sound_on in case of short silences
    syllables = list()
    for i in range(len(env)):
        if i < syll_dur + sil_dur: # When we havn't had enough time to gather a syllable yet
            if env[i] > amp_thresh:  
                sound_on += 1
                if sil_on < sil_dur: # if the silence was transient act like it didn't exist
                    sound_on = sound_on + hold + sil_on # add the transient silence time to the syllable time
                sil_on = 0
                hold = 0
            else:
                sil_on += 1
                if sound_on > 0: # previous point had sound
                    hold = sound_on # prepare to look if the silence is brief
                    sound_on = 0                
        else: # the main loop
            if env[i] > amp_thresh:  
                sound_on += 1
                if sil_on < sil_dur: # if the silence was transient act like it didn't exist
                    sound_on = sound_on + hold + sil_on # add the transient silence time to the syllable time
                sil_on = 0
                hold = 0
            else:
                sil_on += 1
                if sound_on > 0: # previous point had sound
                    hold = sound_on # prepare to look if the silence is brief
                    sound_on = 0                    
                elif sil_on == sil_dur: # if it's been silent long enough to trigger a syllable end check if there was a syllable
                    if hold > syll_dur: # Indicating there was a long enough period of sound to save as a syllable before
                        syll_end = i - sil_dur
                        syll_start = syll_end - hold
                        syllables.append([syll_start,syll_end])
                        
    if verbose:
        plt.figure()
        plt.plot(env)
        for i in range(len(syllables)):
            plt.plot(syllables[i][0], min(env),'*', markersize= 12)
            plt.plot(syllables[i][1], min(env),'x', markersize= 12)
#            plt.plot(range(len(env)), np.ones(len(env)) * lower_thresh,'b')
            plt.plot(range(len(env)), np.ones(len(env)) * amp_thresh,'r')
            
    # TODO turn syllables into seconds.
            
    return syllables

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
