# Dependencies 
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
from functools import partial
from scipy.stats import t

# Data Processing Stuff
from sklearn.neighbors import KernelDensity


rootPath = '/auto/fhome/kevin/Projects/songephys/'
dataPath = 'data/birds/'

birds = [
    os.path.basename(bird)
    for bird in sorted(glob.glob(os.path.join(rootPath, dataPath, "*")), key=os.path.getmtime, reverse=True)
    if (os.path.isdir(bird))]


df = pd.read_pickle("good_units_v2.pkl")

pecking_data = pd.read_csv("TrialData.csv")


def lookup_reward(subject, stim_file):
    splitted = stim_file.split("_")
    call_type = splitted[0]
    vocalizer = splitted[3]
    trials = pecking_data[
        (pecking_data["Subject"] == subject) &
        (pecking_data["Stimulus Vocalizer"] == vocalizer) &
        (pecking_data["Stimulus Call Type"] == call_type.upper())
    ]
    if len(trials["Stimulus Class"].unique()) > 1:
        raise Exception("Multiple stimulus classes for one stim file for this subject")
    
    return trials["Stimulus Class"].unique()[0]# random_index_value.observe(set_random_index)


def _get_spikes_in_range(tmin, tmax, row):
    good_spike_times = np.array(row["spike_times"])[
        (np.array(row["good_trials"])) &
        (row["prestim_silence"] >= 0.5)
    ]
    return [len(trial[(trial > tmin) & (trial < tmax)]) for trial in good_spike_times]


def _get_spikes_per_trial(row):
    good_spike_times = np.array(row["spike_times"])[np.array(row["good_trials"])]
    return [len(trial) for trial in good_spike_times]


def create_relation_columns(subject, df):
    def _get_relation(row):
        basename, ext = os.path.splitext(row["file"])
        parts = basename.split("_")
        last_part = parts[-1]
        
        if last_part == "Familiar":
            return "Familiar"
        
        elif last_part == "Unfamiliar":
            return "Unfamiliar"
        
        elif last_part == "SHUFFLED":
            rew = lookup_reward(subject, row["file"])
            return "Shuffled {}".format(rew)
        
        elif last_part == "fs30k":
            return "Ripple"
        
        else:
            rew = lookup_reward(subject, row["file"])
            return "Test {}".format(rew)
        
    df["relation"] = df.apply(_get_relation, axis=1)
    df["call_type"] = df["call_type"]


'''
def load_dataframes(file):
    with open(file, "rb") as picklefile:
        unitsInfo = pk.load(picklefile)
        dfAbsTime = pk.load(picklefile)
        dfRelTime = pk.load(picklefile)
        
    return {
        "unitsInfo": unitsInfo,
        "dfAbsTime": dfAbsTime,
        "df": dfRelTime        
    }
'''

def load_unit(site_name, elec=None, cluster=None):
    """Lists all (electrode, cluster) pairs if elec or cluster is not specified, otherwise load the cluster"""
    bird_name = site_name.split("_")[0]
    
    # path = os.path.join(rootPath, dataPath, bird_name, "sites", site_name, "AUTOPlaybackPkl")
    path = os.path.join(rootPath, dataPath, bird_name, "sites", site_name, "packaged_units")
    detected_files = glob.glob(os.path.join(path, "*"))
    detected_file = os.path.join(path, "packaged-e{}u{}.npy".format(elec, cluster))
    return np.load(detected_file, allow_pickle=True)[()]


def save_unit(site_name, elec, cluster, datadict):
    bird_name = site_name.split("_")[0]
    path = os.path.join(rootPath, dataPath, bird_name, "sites", site_name, "packaged_units")
    detected_file = os.path.join(path, "packaged-e{}u{}.npy".format(elec, cluster))
    return np.save(detected_file, datadict)


def bin_spikes(spike_times, t_min, t_max, sampling_rate):
    n = int((t_max - t_min) * sampling_rate)
    edges = np.linspace(t_min, t_max, n + 1)
    return np.histogram(np.concatenate(spike_times), bins=edges)


def half_and_half_psths(unit_df, min_trials=8):
    all_trials = []
    for i in range(len(unit_df)):
        row = unit_df.iloc[i]
        spike_times = row["spike_times"]
        
        t = row["t_stim"]
        dt = row["t_stim"][1] - row["t_stim"][0]
        sr = 1 / dt
        tmin = t[0]
        tmax = t[-1] + dt
    
        if len(spike_times) < min_trials:
            continue
        
        even_trials = spike_times[::2][:min_trials//2]
        odd_trials = spike_times[1::2][:min_trials//2]
        
        even_psth = bin_spikes(even_trials, tmin, tmax, 1000)[0]
        odd_psth = bin_spikes(odd_trials, tmin, tmax, 1000)[0]
        
        all_trials.append(np.array([even_psth, odd_psth]))
        
    return all_trials

'''

def _seek(arr, expected_onsets=3, reverse=False):
    """Where does the signal turn on"""
    if reverse:
        arr = arr[::-1]

    consec_zeros = 1
    potential_onsets = []
    for i, val in enumerate(arr):
        if val == 0:
            consec_zeros += 1
        elif val != 0:
            if consec_zeros > 0:
                potential_onsets.append((consec_zeros, i))
            else:
                pass
            
            consec_zeros = 0
            
    best_picks = potential_onsets[0][0]
    
    rest = sorted(potential_onsets[1:], reverse=True)[:expected_onsets-1]
    
    result = [best_picks] + [x[1] for x in rest]
    
    if reverse:
        result = list(reversed([len(arr) - 1 - x for x in result]))
    
    return sorted(result)

def split_stims_by_vocalizations(wav, expected_onsets=3):
    onsets = _seek(wav, expected_onsets=expected_onsets)
    offsets = _seek(wav, expected_onsets=expected_onsets, reverse=True)
    
    result = list(zip(onsets, offsets))
#     plt.plot(sig.data)
#     for x, y in result:
#         plt.vlines([x, y], *plt.ylim())
#     plt.show()
#     plt.close(plt.gcf())

    return result


def divide_into_calls(row, pad=0.5):
    """
    
    The data currently looks like
    
       |DC1     DC2    DC3
       |  | || | |     | ||
       
    times are relative to trial onset.
    This function should cut the stim into multiple "trial" chunks
    
       |DC1
       |  | |
       |DC2
      || |
       |DC3
       || ||
    """
    new_rows = []
    
    basename = os.path.basename(row["file"])

    if basename.startswith("So"):
        expected_onsets = 3
    elif basename.startswith("DC"):
        expected_onsets = 3
    else:
        expected_onsets = 1
        
    onsets_and_offsets = split_stims_by_vocalizations(row["stimWav"], expected_onsets=expected_onsets)
    min_dur = np.min([off - on for on, off in onsets_and_offsets])
    
    for onset, offset in onsets_and_offsets:
        t_on = row["tStim"][onset]
        t_off = row["tStim"][onset + min_dur]
        dur = t_off - t_on
        
        new_tStim = row["tStim"][
            (row["tStim"] >= (t_on - pad)) &
            (row["tStim"] < (t_off + pad))
        ] - t_on
        new_stimWav = row["stimWav"][
            (row["tStim"] >= (t_on - pad)) &
            (row["tStim"] < (t_off + pad))
        ]
        new_spikeTimes = np.array([
            row[(row >= (t_on - pad)) & (row < (t_off + pad))] - t_on
            for row in row["spikeTimes"]
        ])
        new_tKDE = row["tKDE"][
            (row["tKDE"] >= (t_on - pad)) &
            (row["tKDE"] < (t_off + pad))
        ] - t_on
        new_spikeKDE = row["spikeKDE"][
            (row["tKDE"] >= (t_on - pad)) &
            (row["tKDE"] < (t_off + pad))
        ]
        if not len(new_tStim):
            continue
        new_rows.append({
            "tStim": new_tStim,
            "stimWav": new_stimWav,
            "spikeTimes": new_spikeTimes,
            "tKDE": new_tKDE,
            "spikeKDE": new_spikeKDE,
            "file": row["file"],
            "relation_short": row["relation_short"],
            "call_type": row["call_type"],
        })
    
    return new_rows
'''

def _split_stims(row, pad=0.5):
    onsets_and_offsets = row["substim_idx"]
    min_dur = np.min([off - on for on, off in onsets_and_offsets])
    
    new_rows = []
    for i, (onset, offset) in enumerate(onsets_and_offsets):
        t_on = row["t_stim"][onset]
        t_off = row["t_stim"][onset + min_dur]
        dur = t_off - t_on
        
        new_tStim = row["t_stim"][
            (row["t_stim"] >= (t_on - pad)) &
            (row["t_stim"] < (t_off + pad))
        ] - t_on
        new_stimWav = row["stim"][
            (row["t_stim"] >= (t_on - pad)) &
            (row["t_stim"] < (t_off + pad))
        ]
        new_spikeTimes = np.array([
            row[(row >= (t_on - pad)) & (row < (t_off + pad))] - t_on
            for row in row["spike_times"]
        ])
        new_tKDE = row["t_kde"][
            (row["t_kde"] >= (t_on - pad)) &
            (row["t_kde"] < (t_off + pad))
        ] - t_on
        new_spikeKDE = row["kde"][
            (row["t_kde"] >= (t_on - pad)) &
            (row["t_kde"] < (t_off + pad))
        ]

        if i == 0:
            prestim_silence = pad
        else:
            prestim_silence = t_on - row["t_stim"][onsets_and_offsets[i-1][1]]

        if not len(new_tStim):
            continue
        new_rows.append({
            "t_stim": new_tStim,
            "stim": new_stimWav,
            "spike_times": new_spikeTimes,
            "t_kde": new_tKDE,
            "kde": new_spikeKDE,
            "file": row["file"],
            "call_type": row["call_type"],
            "good_trials": row["good_trials"],
            "n_good_trials": row["n_good_trials"],
            "n_trials": row["n_trials"],
            "stim_duration": t_off - t_on,
            "substim_idx": i,
            "prestim_silence": prestim_silence, # How much silence precedes t=0.0 in this stim
        })
    
    return new_rows


def create_split_df(df):
    rows = []
    for i in range(len(df)):
        rows.append(_split_stims(df.iloc[i]))
    return pd.DataFrame(list(np.concatenate(rows)))


import numpy as np
import nitime.algorithms as ntalg
import scipy.stats
from scipy.signal import detrend


def cross_spectra(X):
    """For each pairwise combination of channels, compute cross spectra
    For each chunk, taper, and channel pair (X, Y), compute
        X.conj() * Y
    Parameters
    ==========
    X: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_TAPERS, N_FREQS)
    Returns
    =======
    output: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_CHANNELS, N_TAPERS, N_FREQS)
    """
    return (
        np.conj(X[:, :, np.newaxis, :, :]) *
        X[:, np.newaxis, :, :, :]
    )


def chunk(X, window_size=1024, overlap=0.5):
    """Break up a signal into multiple chunks of fixed size
    Parameters
    ==========
    X: np.ndarray
        shape (N_CHANNELS, N_SAMPLES)
        dtype float
        - Input signal to chunk
    window_size: int
        default 1024
        - Width of chunks in number of samples
    overlap: float
        default 0.5
        - How much adjacent chunks should overlap (as a fraction of window_size)
    Returns
    =======
    output: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, window_size)
    """
    n_overlap = int(window_size * overlap)
    n_channels, n_samples = X.shape
    window_step = int(window_size - n_overlap)
    
    # compute the length of each chunk
    n_chunks = (n_samples - window_size) // window_step
    # pad signal with zeros along time axis
    # so that last chunk will be of width window_size
    padded_size = (n_chunks + 1) * window_step + window_size
    X = np.pad(X, [(0, 0), (0, padded_size - X.shape[-1])], "constant")

    output = np.array([
        detrend(X[:, i * window_step:i * window_step + window_size]) for i in range(0, n_chunks)
    ])
    
    return output


def taper_segments(X, NW=3):
    """Apply taper functions to signal over all chunks
    Parameters
    ==========
    X: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_SAMPLES)
        dtype float
        - Chunked input signal
    NW: int
        default 3
        - Time bandwith parameter (copied from matlab function),
        the higher the bandwidth, the more tapers used.
    Returns
    =======
    output: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_TAPERS, N_SAMPLES)
        dtype float
        - Signal transformed by applying tapers. Uses N_TAPERS = 2 * NW - 1
    """
    _, _, window_size = X.shape
    n_tapers = 2 * NW - 1
   
    # Get the taper functions as a matrix
    tapers, _ = ntalg.dpss_windows(window_size, NW, n_tapers)
    
    # Apply the dpss functions elementwise to the input signal
    return X[:, :, np.newaxis, :] * tapers


def fft_segments(X):
    """Computes fft of chunked signals across multiple channels and tapers
    Parameters
    ==========
    X: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_TAPERS, N_SAMPLES)
    Returns
    =======
    output: np.ndarray
        shape (N_CHUNKS, N_CHANNELS, N_TAPERS, N_FREQS == N_SAMPLES)
    """
    return np.fft.fft(X, axis=3)

    
def coherency(cross_spectra):
    """Compute coherency from cross spectra matrix
    Parameters
    ==========
    cross_spectra: np.ndarray
        shape (N_CHANNELS, N_CHANNELS, N_FREQS)
        - each element cross_spectra[i, j] represents the cross spectra between
        the ith and jth channels
    Returns
    =======
    coherency: np.ndarray
        shape (N_CHANNELS, N_CHANNELS, N_FREQS)
        - the coherency function for each pair of channels
    """
    p_spec = np.abs(cross_spectra.diagonal().T)
    return (
        cross_spectra /
        np.sqrt(p_spec[np.newaxis, :, :] * p_spec[:, np.newaxis, :])
    )


def _replace_inf_with_0(x):
    """Replace NaN and inf elements with 0
    """
    x[np.isinf(x)] = 0
    x[np.isnan(x)] = 0
    return x




def multitapered_coherence(X, sampling_rate=1, chunk_size=1024, overlap=0.5, NW=3):
    """Compute multitapered coherence for signal data over potentially several trials
    Parameters
    ==========
    X: list of np.ndarray [X[0], X[1], ... , X[N_TRIALS - 1]]
        X[0]: np.ndarray
            shape (N_CHANNELS, N_SAMPLES[i])
            dtype float
        - Signal data across N_CHANNELS across N_TRIALS (each trial can
        have a different length). Each element of X represents data from
        one trial (X should be length one for continuous, unchunked data).
        Each element X[i] should have the same number of channels, and
        the second dimension of X[i] should be the duration of trial i
        in number of samples. (If all trials are equal length, can just
        input an array of shape (N_TRIALS, N_CHANNELS, N_SAMPLES)
    sampling_rate: float
        default 1
        - Sampling rate of signal in Hz
    chunk_size: int
        default 1024
        - Size of chunks to break up signal when estimating power spectra
        in number fo samples. This should be a power of 2 to make computation
        of fourier transforms most efficient. Corresponds to twice the number
        of frequency bins
    overlap: float
        default 0.5
        - Fraction of overlap between adjacent chunks when estimating power spectra
    NW: int
        Time bandwith parameter for tapering. n_tapers = 2 * NW - 1, num tapers goes
        up with NW. Defaults to 3
    Returns
    =======
    Dictionary with the following keys:
    t: np.ndarray
        shape (chunk_size,)
        - Time axis of output coherency in time domain
    freqs: np.ndarray
        shape (chunk_size,)
        - Frequency axis of output coherence and psds
    coherence: np.ndarray
        shape (chunk_size,)
        dtype float
        - Estimated coherence values as a function of frequency (corresponding to
        freqs array)
    coherence_bounds: np.ndarray
        shape (chunk_size, 2)
        dtype float
        - Upper and lower bounds of coherence values corresponding to 2 SEM
    coherency: np.ndarray
        shape (chunk_size,)
        dtype complex
        - Estimated coherency values as a function of frequency
    """
    # First validate that all X have the same number of channels
    if not np.all([X[i].shape[0] == X[0].shape[0] for i in range(len(X))]):
        raise ValueError("All trials must have the same number of channels")

    # FFT frequency axis
    freqs = np.fft.fftfreq(chunk_size, 1 / sampling_rate)

    # we could keep a mapping from chunk index to which trial it came from?
    # potentially z score within trials before chunking (maybe?)
    
    # X: (N_TRIALS, N_CHANNELS, N_SAMPLES[i]) ->
    # segments: (N_CHUNKS, N_CHANNELS, chunk_size)
    segments = np.concatenate(
        [chunk(segment, chunk_size, overlap) for segment in X]
    )
    n_chunks = segments.shape[0]

    # segments: (N_CHUNKS, N_CHANNELS, chunk_size) ->
    # segments_tapered: (N_CHUNKS, N_CHANNELS, N_TAPERS, chunk_size)
    segments_tapered = taper_segments(segments,NW=NW)

    # segments_tapered: (N_CHUNKS, N_CHANNELS, N_TAPERS, chunk_size) ->
    # segments_fft: (N_CHUNKS, N_CHANNELS, N_TAPERS, n_freqs == chunk_size)
    segments_fft = fft_segments(segments_tapered)
    
    # (adaptive weights here)
    
    # For each pairwise combination of channels, compute cross-spectrum
    # segments_fft: (N_CHUNKS, N_CHANNELS, N_TAPERS, n_freqs) ->
    # segments_cross_spectra: (N_CHUNKS, N_CHANNELS, N_CHANNELS, N_TAPERS, n_freqs)
    segments_cross_spectra = cross_spectra(segments_fft)
    
    # Average over tapers
    # segments_cross_spectra: (N_CHUNKS, N_CHANNELS, N_CHANNELS, n_freqs)
    segments_cross_spectra = segments_cross_spectra.mean(axis=3)  # JN in matlab

    # segments_cross_spectra: (N_CHUNKS, N_CHANNELS, N_CHANNELS, n_freqs) ->
    # summed_cross_spectra: (N_CHANNELS, N_CHANNELS, n_freqs)
    summed_cross_spectra = segments_cross_spectra.sum(axis=0)  # y in matlab

    ## Compute jackknife samples for coherence and coherency

    # Estimated values of coherency and coherence across all chunks
    est_coherency = coherency(summed_cross_spectra)
    est_sqrt_coherence = np.arctanh(np.abs(est_coherency))
    est_sqrt_coherence = _replace_inf_with_0(est_sqrt_coherence)

    # Compute estimates of coherency in sets of (n_chunks - 1) chunks
    cross_spectra_jn = summed_cross_spectra - segments_cross_spectra
    # Convert samples into pseudovalues
    est_coherency_jackknife = np.array([
        (n_chunks * est_coherency) - ((n_chunks - 1) * coherency(x))
        for x in cross_spectra_jn
    ])
    est_coherency_final = np.mean(est_coherency_jackknife, axis=0)
    
    # Compute estimates of coherence in sets of (n_chunks - 1) chunks
    est_sqrt_coherence_jackknife = np.array([
        (
            (n_chunks * est_sqrt_coherence) -
            ((n_chunks - 1) * (
                _replace_inf_with_0(np.arctanh(np.abs(coherency(x))))
            ))
        )
        for x in cross_spectra_jn
    ])

    # Convert samples into pseudovalues
    # not sure if this mean should be in the tanh, if the bounds are after a tanh
    # then the mean should be... but not sure if any of this is right.
    
    # Convert samples into pseudovalues
    est_sqrt_coherence_final = np.mean(est_sqrt_coherence_jackknife, axis=0)
    est_sqrt_coherence_var = (1 / n_chunks) * np.var(est_sqrt_coherence_jackknife, axis=0)
    
    sqrt_coherence_upper = est_sqrt_coherence_final + 2 * np.sqrt(est_sqrt_coherence_var)
    sqrt_coherence_lower = est_sqrt_coherence_final - 2 * np.sqrt(est_sqrt_coherence_var)
    
    # Mask for non-significant coherence values
    coherency_mask = sqrt_coherence_lower < 0
    
    # Make sure we don't project to pseudovalues of sqrt coherence less than 0
    est_sqrt_coherence_final[est_sqrt_coherence_final < 0] = 0
    sqrt_coherence_upper[sqrt_coherence_upper < 0] = 0
    sqrt_coherence_lower[sqrt_coherence_lower < 0] = 0
       
    est_coherence_final = np.tanh(est_sqrt_coherence_final) ** 2
    coherence_upper = np.tanh(sqrt_coherence_upper) ** 2
    coherence_lower = np.tanh(sqrt_coherence_lower) ** 2
    
    # mask the coherency values by significant coherence values
    tmp_coherency = est_coherency_final.copy()
    tmp_coherency[coherency_mask] = 0
    
    coherency_time_domain = np.fft.ifft(tmp_coherency, axis=2)
    npts = coherency_time_domain.shape[-1]
    t = np.arange(-npts/2 +1, npts/2+1)*1000.0/sampling_rate

    return {
        "t": t,
        "freqs": freqs,
        "coherency": est_coherency_final,
        "coherency_t": coherency_time_domain,
        "coherence": est_coherence_final,
        "coherence_bounds": np.array([coherence_lower, coherence_upper])
    }



def coherence_correction(coherence_dict, n_trials=10):
    def _bias_correction(x):
         return 1 / (((-n_trials + n_trials * np.sqrt(1 / x)) / 2) + 1)

    return {
        "freqs": coherence_dict["freqs"],
        "coherence": _bias_correction(coherence_dict["coherence"][0, 1]),
        "coherence_bounds": _bias_correction(coherence_dict["coherence_bounds"][:, 0, 1])
    }


def information(coherence_dict, min_freq=None, max_freq=None, significant_only=True):
    delta_f = coherence_dict["freqs"][1] - coherence_dict["freqs"][0]
    
    selector = coherence_dict["freqs"] >= 0
    if min_freq:
        selector = selector & (coherence_dict["freqs"] >= min_freq)
    if max_freq:
        selector = selector & (coherence_dict["freqs"] <= max_freq)
    if significant_only:
        selector = selector & (coherence_dict["coherence_bounds"][0] > 0)
    
    info = np.sum(-np.log2(1 - coherence_dict["coherence"])[selector]) * delta_f
    return info


if __name__ == "__main__":
    results = []
    N = 6  # number of trials minimum/maximum to use for coherence estimation

    for i in range(len(df)):
        print("Finished {}/{}".format(i, len(df)), end="\r")
        row = df.iloc[i]
        
        unit_data = load_unit(row["site"], row["electrode"], row["cluster"])

        undivided_df = unit_data["df"]

        if "divided_df" in unit_data:
            divided_df = unit_data["divided_df"]
        else:
            divided_df = create_split_df(unit_data["df"])
            create_relation_columns(unit_data["unit_info"]["subject"], undivided_df)
            create_relation_columns(unit_data["unit_info"]["subject"], divided_df)

            save_unit(row["site"], row["electrode"], row["cluster"], {
                "df": unit_data["df"],
                "divided_df": divided_df,
                "unit_info": unit_data["unit_info"]
            })

        # Save these...
        
        result = {}

        result["site"] = row["site"]
        result["electrode"] = row["electrode"]
        result["cluster"] = row["cluster"]
        for key in [
                "subject",
                "pvalue",
                "tscore",
                "zscore",
                "snr",
                "baseline_fr",
                "stim_fr",
                "isi_violation_pct",
                "avg_good_trials",
                "max_good_trials",
                "min_good_trials"
                ]:
            result[key] = row[key]

        # then do the computation for only rewarded and nonrewarded
        # Coherence to each category of songs and dcs, ripples, non-ripples, and all
        categories = {
            "So Test Rewarded": (divided_df["call_type"] == "So") & (divided_df["relation"] == "Test Rewarded"),
            "So Test Nonrewarded": (divided_df["call_type"] == "So") & (divided_df["relation"] == "Test Nonrewarded"),
            "So Shuffled Rewarded": (divided_df["call_type"] == "So") & (divided_df["relation"] == "Shuffled Rewarded"),
            "So Shuffled Nonrewarded": (divided_df["call_type"] == "So") & (divided_df["relation"] == "Shuffled Nonrewarded"),
            "So Familiar": (divided_df["call_type"] == "So") & (divided_df["relation"] == "Familiar"),
            "So Unfamiliar": (divided_df["call_type"] == "So") & (divided_df["relation"] == "Unfamiliar"),
            "DC Test Rewarded": (divided_df["call_type"] == "DC") & (divided_df["relation"] == "Test Rewarded"),
            "DC Test Nonrewarded": (divided_df["call_type"] == "DC") & (divided_df["relation"] == "Test Nonrewarded"),
            "DC Familiar": (divided_df["call_type"] == "DC") & (divided_df["relation"] == "Familiar"),
            "DC Unfamiliar": (divided_df["call_type"] == "DC") & (divided_df["relation"] == "Unfamiliar"),
            "DC All": (divided_df["call_type"] == "DC"),
            "Ripple": (divided_df["relation"] == "Ripple"),
            "So NonTest": (divided_df["call_type"] == "So") & ~(divided_df["relation"].isin(["Test Rewarded", "Test Nonrewarded"])),
            "So Test": (divided_df["call_type"] == "So") & (divided_df["relation"].isin(["Test Rewarded", "Test Nonrewarded"])),
            "DC NonTest": (divided_df["call_type"] == "DC") & ~(divided_df["relation"].isin(["Test Rewarded", "Test Nonrewarded"])),
            "DC Test": (divided_df["call_type"] == "DC") & (divided_df["relation"].isin(["Test Rewarded", "Test Nonrewarded"])),
            "So All": (divided_df["call_type"] == "So"),
            "NonRipple": ~(divided_df["relation"] == "Ripple"),
            "All": np.ones(len(divided_df)).astype(np.bool)
        }

        for category_name, category_selector in categories.items():
            subdf = divided_df[category_selector]
            psths = half_and_half_psths(subdf, min_trials=N)
            coherence_result = multitapered_coherence(psths, chunk_size=512, sampling_rate=1000)
            corrected_coherence = coherence_correction(coherence_result, n_trials=N)
        
            result["{} Coherence".format(category_name)] = corrected_coherence["coherence"]
            result["{} Coherence Bounds".format(category_name)] = corrected_coherence["coherence_bounds"]
            result["{} Info".format(category_name)] = information(corrected_coherence, significant_only=True)
            result["{} Info 0to20".format(category_name)] = information(corrected_coherence, max_freq=20, significant_only=True)
            result["{} Info 0to40".format(category_name)] = information(corrected_coherence, max_freq=40, significant_only=True)
            result["{} Info 20to40".format(category_name)] = information(corrected_coherence, min_freq=20, max_freq=40,
                    significant_only=True)
            prestim_spike_counts = np.concatenate(list(subdf.apply(partial(_get_spikes_in_range, -0.5, 0.0), axis=1))) / 0.5
            poststim_spike_counts = np.concatenate(list(subdf.apply(partial(_get_spikes_in_range, 0.0, 0.5), axis=1))) / 0.5
            spikes_per_trial = np.concatenate(list(subdf.apply(_get_spikes_per_trial, axis=1)))
        
            result["{} Baseline FR".format(category_name)] = np.mean(prestim_spike_counts)
            result["{} Stim FR".format(category_name)] = np.mean(poststim_spike_counts)
            result["{} Baseline FR Std".format(category_name)] = np.std(prestim_spike_counts)
            result["{} Stim FR Std".format(category_name)] = np.std(poststim_spike_counts)
            result["{} nTrials".format(category_name)] = len(prestim_spike_counts)
            result["{} spikesPerTrial".format(category_name)] = np.mean(spikes_per_trial)
        
        result["freqs"] = corrected_coherence["freqs"]
            
        results.append(result)
       
    pd.DataFrame(results).to_pickle("coherence_output_{}trials.pkl".format(N))
