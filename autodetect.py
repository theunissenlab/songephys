import cPickle as pickle
from lasp.signal import lowpass_filter, highpass_filter
import matplotlib.mlab as mlab
import numpy as np
import scipy.stats


def find_vocal_periods(audio_stream, fs=25000, fuse_duration=0.5, verbose=False):
    """
    Given an audio stream, attempts to detect all segments of noise that might be bird calls of
    some sort.

    Parameters
    ----------
    audio_stream: np.array of floats.
        The audio stream.
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
    # First we try to find the amplitude envelope. We only care about vocal sounds,
    # so we high-pass filter higher than normal.
    streams_filtered_env = np.zeros(np.shape(audio_stream))
    streams_filtered_env = highpass_filter(audio_stream, fs, 2000)
    streams_filtered_env = np.abs(lowpass_filter(streams_filtered_env, fs, 5000))
    streams_filtered_env = lowpass_filter(streams_filtered_env, fs, 100)

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
    return sound_onset, sound_offset


def extract_birdsong_features(signal, points, fs=25000, minfreq=1000, maxfreq=8000,
                              pca_model='pca_projection.pkl'):
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
        maxfreq parameters must match whatever dataset the PC space was fit on.

    Returns
    -------
    np.array of float. Shape (n_points, n_features)
        A numpy array of the resulting features for each point.
    """
    pca_space = pickle.load(open(pca_model, "rb"))
    feats = []
    minfreq = 1000
    maxfreq = 8000

    all_lengths = scipy.stats.zscore(np.expand_dims(map(lambda x: x[1] - x[0], points), 1))
    for pt in points:
        l, h = pt
        Pxx, _, _ = mlab.specgram(signal[l:h], 256, Fs=fs,
                                  detrend=mlab.detrend_none,
                                  window=mlab.window_hanning)
        if minfreq is not None and maxfreq is not None:
            Pxx = Pxx[(freqs >= minfreq) & (freqs <= maxfreq)]
        Z = 10. * np.log10(Pxx)
        Z = np.flipud(Z)
        Z = np.maximum(Z, np.max(Z) - 50)
        specs.append(Z)
        feats.append(np.mean(np.square(Z.T), axis=0))

    feats = np.array(feats)
    proj_feats = pca.transform(feats)
    features = np.concatenate([proj_feats, all_lengths], axis=1)
    return features


def classify_birdsong(signal, points, fs=25000.0, model='songvnoise_classifier.pkl'):
    """
    Given a signal and a set of points, classifies which of the points are birdsong (versus other
    sorts of noise, such as calls, tets, and flapping/hopping).

    Parameters
    ----------
    signal: np.array of float.
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
    svm = pickle.load(open(model, "rb"))

    features = extract_birdsong_features(signal, points, fs=fs)
    return svm.predict(features)


def separate_song_sources():
