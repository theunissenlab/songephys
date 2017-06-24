import cPickle as pickle
from soundsig.signal import lowpass_filter, highpass_filter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from soundsig.sound import BioSound 
import pandas as pandas
from sklearn.ensemble import RandomForestClassifier


def find_syllables(audio_stream_in, amp_thresh = '', fs=25000, fuse_duration=0.1,  syll_dur = 20.0, sil_dur = 5.0, verbose=False):
 
    """
    For segmenting into syllables.
    You should pass amp_thresh, which should come from find_vocal_periods

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

def getBioSoundProperties(soundIn, syllables = 0, plotme = False):
    ## Returns biosoudn properties from a chunk of sound.

    #print 'Processing sound'
    maxAmp = np.asarray(np.abs(soundIn).max())      
    myBioSound = BioSound(soundWave=np.squeeze(np.asarray(soundIn).astype(float))/maxAmp, fs=float(soundIn.sampling_rate), calltype = "")
                 
    # Create BioSound Object and store some values
    myBioSound.spectroCalc(spec_sample_rate=100)
    myBioSound.rms = myBioSound.sound.std()     # The rms  
    # Calculate amplitude enveloppe
    myBioSound.ampenv()  
    # Calculate the power spectrum
    myBioSound.spectrum(f_high=10000)
    # Calculate fundamental and related values
    myBioSound.fundest(maxFund = 1500, minFund = 100, lowFc = 50, highFc = 6000, minSaliency = 0.5)
    meanF1 = np.mean(myBioSound.F1[~np.isnan(myBioSound.F1)])
    meanF2 = np.mean(myBioSound.F2[~np.isnan(myBioSound.F2)])
    meanF3 = np.mean(myBioSound.F3[~np.isnan(myBioSound.F3)])

    if myBioSound.fund is not None:
        fund = np.float(myBioSound.fund)
        cvfund = np.float(myBioSound.cvfund)
    else:
        fund = None 
        cvfund = None   
    
    vocSelData ={"fund": fund, 
                       "cvfund": cvfund, "F1": meanF1, "F2": meanF2, "F3":meanF3,
                       "sal": np.float(myBioSound.sal), "rms": np.float(myBioSound.rms), 
                       "meanS": np.float(myBioSound.meanspect), "stdS": np.float(myBioSound.stdspect), 
                       "q1": np.float(myBioSound.q1), "q2": np.float(myBioSound.q2), "q3": np.float(myBioSound.q3),
                       "timestd": np.float(myBioSound.stdtime)}
    if plotme:  
        myBioSound.plot()  
        myBioSound.play()
        waitstr = raw_input("Press Enter to continue...")
        plt.close('all')
    # Make a data frame with all the data and return it                    
    #vocSelTable = pandas.DataFrame(vocSelData)
    #return vocSelTable
    return vocSelData
    
def classify_vocalizations_random_forest(signal, points, classifier_dataset = '/auto/tdrive/billewood/intan data/song_sorting/from_FET_Julie/vocSelTable.h5', fs=25000):
    #This is very slow, probably because I don't properly know how to allocate pandas df's.

    # Import the classifier dataset and format if necessary
    classifier_df = pandas.read_hdf(classifier_dataset)
    if classifier_dataset == '/auto/tdrive/billewood/intan data/song_sorting/from_FET_Julie/vocSelTable.h5':
        features = classifier_df.columns[[2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
        classifier_df['calltype'] = pandas.Series(classifier_df['calltype'], dtype="category") # make calltype categorica
        
    classifier_df = make_nans_col_means(classifier_df,features)  
    new_df_data = list()
    # make a pandas DataFrame with biousound properties of all syllables/vocalizations
    for i, pt in enumerate(points):
        l,h = pt
        new_df_data.append(getBioSoundProperties(signal[l:h,0]))
    df = pandas.DataFrame(new_df_data)
    
    # check it for NANs
    df = make_nans_col_means(df, features)  
    
    # add the durations as a column
    durations = np.expand_dims(map(lambda x: x[1] - x[0], points), 1)
    df['duration'] = durations
    df['vocal_category'] = pandas.Series('unknown', dtype = "category") 
    
    # and mabe sort the columns to look teh same as classifier_df...?
   # df = df['sort columns here??'] #TODO
    
    # Make a vocalization type category column here TODO

    #Shouldn't need to train it every time TODO save the actual trained matrix whatchamacallit    
    train = classifier_df

    target_names = np.unique(classifier_df['calltype'])
    y = train['calltype'].cat.codes

    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train[features], y)
    test = df
    preds = target_names[clf.predict(test[features])]
    return preds
#    pandas.crosstab(test['calltype'], preds, rownames=['actual'], colnames=['preds'])





def make_nans_col_means(df, features = ""):
    """
        Goes through a pandas dataframe and replaces NANs with column averages.
    """    
    if features.all == "":
        features = df.columns
    nans = list()
    for i in range(len(df)):
        for feature in features:
            if np.isnan(df[feature][i]):
                df.loc[i,(feature)] = df[feature].mean()
                nans.append([feature,i])
    nan_ind = np.unique([number for name,number in nans])
    nan_feats = np.unique([feature for feature, number in nans])
    if len(nan_ind) > 0:
        print "Warning, %d entries were replaced by column averages" %len(nan_ind)
        print "Features containing NaNs are: ", nan_feats
    return df

def find_vocal_periods(audio_stream_in, fs=25000, fuse_duration=0.1, verbose=False):
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
    streams_filtered_env = highpass_filter(audio_stream, fs, 2000)
    streams_filtered_env = np.abs(lowpass_filter(streams_filtered_env, fs, 5000)) # Rectify and lowpass filter
    streams_filtered_env = lowpass_filter(streams_filtered_env, fs, 100) # We essentially have an amplitude trace now

    # Simply threshold against the mean of the signal. This works surprisingly well.
    # If you think about it, all of the noises that get made during a session drag
    # the mean to be slightly above background noise.
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
            sound_onsets[len(sound_onsets)] = -1
            print("Warning, sound appears to continue past file end, "
                  + "last time point being set to an offset")
        if sound_onsets[len(sound_onsets)-i-1] == -1:
            break

    # Fuse all of the nearby offset/onsets
    onsets = np.where(sound_onsets == 1)[0]
    offsets = np.where(sound_onsets == -1)[0]
    n_fused = 0
    if len(onsets) > 1:
        print "len onsets > 1"
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
        print "num fused: " + str(n_fused)

    # convert to a list of just onsets and offsets
    sound_onset = np.where(sound_onsets == 1)[0]
    sound_offset = np.where(sound_onsets == -1)[0]
    try:
        len(sound_onset) == len(sound_offset)
    except:
        print("Number of vocal onsets and offsets not equal!")

    vocal_periods = np.array(zip(sound_onset, sound_offset))
    lengths = np.apply_along_axis(lambda p: p[1] - p[0], 1, vocal_periods)
    vocal_periods = vocal_periods[lengths > fs/10]
    return vocal_periods, amp_thresh
    


def extract_birdsong_features(signal, points, fs=25000, minfreq=1000, maxfreq=8000,
                              pca_model=None):
    """
    This function is given a signal and a set of intervals, and extracts the frequency
    characteristics of the interval. It then projects them onto a pretrained set of principle
    components. Finally, it appends the length of the interval. All features are z-scored, and
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
    print "...Filtering Arrays to get Amp Envelope"
    streams_filtered_env = np.zeros(np.shape(audio_streams))
    streams_filtered_env[:, 0] = highpass_filter(np.squeeze(audio_streams[:, 0]), fs, 2000)
    streams_filtered_env[:, 1] = highpass_filter(np.squeeze(audio_streams[:, 1]), fs, 2000)
    streams_filtered_env[:, 0] = np.abs(lowpass_filter(streams_filtered_env[:, 0], fs, 5000))
    streams_filtered_env[:, 1] = np.abs(lowpass_filter(streams_filtered_env[:, 1], fs, 5000))
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
    return annotated_pts
