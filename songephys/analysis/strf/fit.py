import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
from IPython.display import clear_output
import wavio

import suss.io

from mne.decoding import ReceptiveField, TimeDelayingRidge

from scipy.stats import multivariate_normal
from scipy.io import loadmat
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm

from ...explore.playback_responses import (
    fig_grid, 
    generate_full_comparison,
    full_responses,
    inject_spikes,
    plot_raw, 
    view_spike_waveforms_on_channel,
)

from soundsig.sound import plot_spectrogram, spectrogram
from soundsig.signal import highpass_filter, bandpass_filter

import wavio
import scipy


_site_unit_cache = {}
def load_site_unit_info(site_dir, mode=("live", "playback", "both")):
    if mode not in ("live", "playback", "both"):
        raise ValueError("Must specify mode to be one of 'live', 'playback', or 'both'")

    if (site_dir, mode) in _site_unit_cache:
        return _site_unit_cache[(site_dir, mode)]
        
    events_path = os.path.join(site_dir, "vocal_periods.npy")
    if not os.path.exists(events_path):
        raise Exception("The vocal_periods.npy for site {} does not seem to exist. "
            "Double check that you have run the pipeline for this site.".format(os.path.basename(site_dir)))
    
    events = np.load(events_path)[()]
    if mode == "both":
        events = sorted(events["live"] + events["playback"], key=lambda e: e["start_time"])
    else:
        events = events[mode]
    
    lazy_data_path = os.path.join(site_dir, "lazy_data.pkl")
    if not os.path.exists(lazy_data_path):
        raise Exception("The lazy_data.pkl for site {} does not seem to exist. "
            "Double check that you have run the pipeline for this site.".format(os.path.basename(site_dir)))
        
    loader = suss.io.read_pickle(lazy_data_path)
    
    df = full_responses(events)
    
    # Go through manually curated files to report units found.
    potential_files = glob.glob(os.path.join(site_dir, "manually_curated", "curated-e*.pkl"))
    print("Located {} curated spike files".format(len(potential_files)))
    
    labeled_nodes = []
    for ch in range(8, 24):
        spikes = suss.io.read_pickle(os.path.join(site_dir, "manually_curated", "curated-e{}.pkl".format(ch)))
        print("{} units sorted on ch {}".format(len(spikes.labeled_nodes), ch))
        for k, v in spikes.labeled_nodes:
            key = "e{} u{}".format(ch, k)
            labeled_nodes.append(key)
            df = inject_spikes(df, v.flatten().times, buffer=1.0, key=key)

    _site_unit_cache[(site_dir, mode)] = (df, labeled_nodes, loader)

    return df, labeled_nodes, loader


def get_data(df, unit_key, lazy_loader, mic=True, mic_channel=0, time_limits=None):
    X_wav = []
    X_mic = []
    y_spikes = []

    f_spec = None
    for _, stim in df.iterrows():
        start_time = stim["start_time"]
        
        if time_limits and not (time_limits[0] <= start_time < time_limits[1]):
            continue

        ## TODO (kevin): choose a good minimum firing rate? 2.0 Hz is totally arbitrary
        if len(stim[unit_key]) / stim["duration"] < 2.0:
            continue

        if mic:
            # Either load from the microphone (choose mic as parameter)
            fs = lazy_loader._signals["mic"][0].sampling_rate
            mic_data = lazy_loader.time_slice(start_time, start_time + stim["duration"], signal="mic")
            mic_data = bandpass_filter(mic_data.data.T, fs, 200, 10000).T
            sound_arr = mic_data[:, mic_channel]

            t_spec, f_spec, mic_spec, _ = spectrogram(sound_arr, fs, 500, 150, min_freq=250, max_freq=8000, cmplx=False)
        else:
            # Or load the stimulus wav file
            wav = wavio.read(os.path.join("..", stim["file_path"]))
            sound_arr = wav.data[:, 0]
            fs = wav.rate
            t_spec, f_spec, wav_spec, _ = spectrogram(sound_arr, fs, 500, 150, min_freq=250, max_freq=8000, cmplx=False)
        
        
        if mic:
            spec = mic_spec
        else:
            spec = wav_spec

        STIMDIR = ""
        dt = 1/fs
        edges = np.concatenate([[start_time], start_time + t_spec + .001])

        spike_bins = np.zeros_like(t_spec)

        spike_idx = 0
        for i in np.arange(len(t_spec)):
            while spike_idx < len(stim[unit_key]):
                if edges[i] <= stim[unit_key][spike_idx] < edges[i + 1]:
                    spike_bins[i] = 1
                    spike_idx += 1
                    break
                elif stim[unit_key][spike_idx] >= edges[i + 1]:
                    break
                spike_idx += 1

        X_mic.append(spec.T)
        y_spikes.append(spike_bins)

    if f_spec is None:
        raise ValueError("No trials")

    return X_mic, y_spikes, f_spec


def fit_site(
        site_dir,
        mode=("playback", "live", "both"),
        use_mic=True,
        mic_channel=0,
        test_size=0.2,
        reg_type=("ridge", "laplacian"),
        tmin=0,    # ms
        tmax=200,  # ms
        random_state=0
    ):
    if reg_type not in ("ridge", "laplacian"):
        raise ValueError("reg_type must be either 'ridge' or 'laplacian'")

    if mode not in ("playback", "live", "both"):
        raise ValueError("Mode must be either 'playback', 'live', or 'both'")

    df, units, lazy_loader = load_site_unit_info(site_dir, mode=mode)

    results = {}
    for unit in units:
        print("... Fitting {} ...".format(unit))
        fits, t_spec, f_spec = fit_unit(
                df,
                lazy_loader,
                unit,
                mode=mode,
                use_mic=use_mic,
                mic_channel=mic_channel,
                test_size=test_size,
                reg_type=reg_type,
                tmin=tmin,
                tmax=tmax,
                random_state=random_state)
        yield unit, fits, t_spec, f_spec
        # results[unit] = (fits, t_spec, f_spec)

    # return results


def fit_unit(
        df,
        lazy_loader,
        unit_str,
        mode=("playback", "live", "both"),
        use_mic=True,
        mic_channel=0,
        test_size=0.2,
        reg_type=("ridge", "laplacian"),
        tmin=0,    # ms
        tmax=200,  # ms
        random_state=0
    ):
    try:
        X_mic, y_spikes, f_spec = get_data(
            df,
            unit_str,
            lazy_loader,
            mic=use_mic,
            mic_channel=mic_channel,
            time_limits=None  # how to configure this?
        )
    except ValueError:
        return [], [], []

    X_train, X_test, y_train, y_test = train_test_split(
        X_mic,
        y_spikes,
        test_size=test_size,
        random_state=random_state
    )

    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)

    alphas = np.logspace(-2, 15, 18)

    scores = []
    models = []

    n_freq = X_train.shape[1]

    for ii, alpha in tqdm(enumerate(alphas)):
        estimator = TimeDelayingRidge(tmin, tmax, 1, reg_type=reg_type, alpha=alpha)
        rf = ReceptiveField(tmin, tmax, 1, f_spec, estimator=estimator)

        rf.fit(X_train, y_train)

        try:
            scores.append(rf.score(X_test, y_test)[0])
        except ValueError:
            print("Failed on {}".format(unit_str))
            print("Failed on scoring... shape was {}, {}".format(X_test.shape, y_test.shape))
            break

        times = rf.delays_ / float(rf.sfreq)

        models.append((times, f_spec, rf.coef_))

        # if the fit has decreased 3 straight fits, stop.
        if len(scores) > 3 and np.all(np.diff(scores)[-3:] < 0):
            break

    rf_score_pairs = list(zip(models, scores))

    if not len(scores):
        return [], [], []

    best_idx = np.argmax(scores)

    # Return the best score and the two after (since sometimes it seems that the ones after look better?
    return rf_score_pairs[best_idx:best_idx + 3], times, f_spec

