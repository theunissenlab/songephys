import os
import pickle

import numpy as np
import scipy.stats
import hdbscan
from units_to_use import units_to_use, channel_map

from soundsig.sound import plot_spectrogram
from sklearn import preprocessing
from segmentation.sound_sorting import find_syllables, Syllable
from segmentation import autodetect
from block_io import load_block
from segment_analysis import MultiSegment
from IPython.display import Audio, display
from matplotlib import cm
import umap
import config
from suss.sparse import train
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from segmentation.sound_sorting import (
    collect_syllable_objects, sort_sounds, remove_noise, plot_sort_results,
    collect_spectrograms, collect_spectrogram_diffs
)
from soundsig.sound import spectrogram, plot_spectrogram
from suss.io import read_pickle, save_pickle
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge

from sort import xcorr, isi_frac
from isosplit5 import isosplit5

from strf import fit_strf


def custom_correction(sound_events, site):
    for e in sound_events:
        if e["sound_type"] == "playback":
            if "_So" in e["playback_file"]:
                e["vocalization_type"] = "So"
                
        if e["sound_type"] == "live" and "GreYel3594M___" in site:
            if e["vocalizer"] == "tutor":
                e["vocalizer"] = "con_male_familiar"

        if e["sound_type"] == "playback" and "GreYel3594M___" in e["playback_file"]:
            pb_file = os.path.basename(e["playback_file"])
            if "MAT" in pb_file:
                e["vocalizer"] = "con_male_familiar"
            elif "MAF" in pb_file:
                e["vocalizer"] = "tutor"
            elif "MJF" in pb_file: 
                e["vocalizer"] = "juvenile"

        if e["sound_type"] == "playback" and "RedGre7705M__" in e["playback_file"]:
            pb_file = os.path.basename(e["playback_file"])
            if "MJP" in pb_file:
                e["vocalizer"] = "subject"
            elif e["vocalizer"] == "con_male":
                e["vocalizer"] = "con_male_familiar"

        if e["sound_type"] == "playback" and "BroGre4497M_" in e["playback_file"]:
            pb_file = os.path.basename(e["playback_file"])
            if "MJB" in pb_file:
                e["vocalizer"] = "juvenile"
            elif "MAF" in pb_file:
                e["vocalizer"] = "con_male_familiar"

    return sound_events


def filter_events(events, **kwargs):
    return [
        e for e in events
        if np.all([e.get(k) == v for k, v in kwargs.items()])
    ]




def compute_strfs(SITE, elec, label, t_ranges, sound_events, loader, site, sorted_data):
    computed_units = {}

    save_file = "strfs/{}_e{}u{}.npy".format(SITE, elec, label)
    if os.path.exists(save_file):
        print("skipping", save_file)
        return

    if label not in sorted_data.labels:
        print("ERROR, {} ELEC {} MISSING LABEL {}".format(SITE, elec, label))
        return
    node = dict(sorted_data.labeled_nodes)[label]
    event_filters = [
        ("song_playback", sound_events, lambda event: event["sound_type"] == "playback" and event["vocalization_type"] == "So"),
        ("song_live", sound_events, lambda event: event["sound_type"] == "live" and event["vocalization_type"] == "So"),
        ("bos_live", sound_events, lambda event: event["sound_type"] == "live" and event["vocalizer"] == "subject" and event["vocalization_type"] == "So"),
        ("tut_live", sound_events, lambda event: event["sound_type"] == "live" and event["vocalizer"] == "tutor" and event["vocalization_type"] == "So"),
        ("tut_playback", sound_events, lambda event: event["sound_type"] == "playback" and event["vocalizer"] == "tutor" and event["vocalization_type"] == "So"),
        ("fam_playback", sound_events, lambda event: event["sound_type"] == "playback" and event["vocalizer"] == "con_male_familiar" and event["vocalization_type"] == "So"),
        ("juv_playback", sound_events, lambda event: event["sound_type"] == "playback" and event["vocalizer"] == "juvenile" and event["vocalization_type"] == "So"),
        ("bos_playback", sound_events, lambda event: event["sound_type"] == "playback" and event["vocalizer"] == "subject" and event["vocalization_type"] == "So"),
        # ("all sounds", np.concatenate([sound_events["live"], sound_events["playback"]]), lambda event: (
        #    event["vocalizer"] in ["subject", "tutor"] and not (
        #        event["vocalization_type"] != "So" and event.get("playback_file")
        #    )
        # ))
    ]

    for group_name, events, filter_fn in event_filters:
        all_models = []
        all_scores = []
        all_widths = []
        for convolve_width in [2]:
            try:
                models, scores, freq, X_specs, Y_spikes = fit_strf(
                    node.flatten().times,
                    events,
                    loader,
                    event_filter=filter_fn,
                    channel_map=lambda event: channel_map[SITE],
                    spec_sample_rate=200,
                    freq_spacing=100,
                    strf_width=0.1,
                    time_ranges=t_ranges,
                    alphas=[0.1, 1.0, 5.0],
                    convolve_width=convolve_width
                )
            except ValueError:
                continue

            all_models = np.concatenate([all_models, models])
            all_scores = np.concatenate([all_scores, scores])
            all_widths = np.concatenate([all_widths, convolve_width * np.ones(len(models))])

        if not len(all_models):
            continue

        best_score_idx = np.argmax(all_scores)
        best_model = all_models[best_score_idx]
        best_width = all_widths[best_score_idx]
        best_R2 = all_scores[best_score_idx]
        
        computed_units[group_name] = (best_model, best_width, best_R2)
        
        print("e{}u{}, {}\nbest width: {}, best R2: {:.1e}".format(elec, label, group_name, best_width, best_R2))

    np.save(save_file, computed_units)


import time
from multiprocessing import Process, Pool, Lock, Manager


start_time = time.time()
with Pool(8) as pool:
    results = []
    for SITE in units_to_use:
        site_dir = os.path.join(
            ("/auto/fhome/kevin/Projects/songephys/data/birds/"),
            SITE.split("_")[0],
            "sites")
        site = os.path.join(site_dir, SITE)
        loader = read_pickle(os.path.join(site, "lazy_data.pkl"))
        sound_events = np.load(os.path.join(site, "disjoint_vocal_periods.npy"))[()]
        sound_events = custom_correction(sound_events, SITE)

        prev_elec = None
        for PICK in np.arange(len(units_to_use[SITE])):
            elec, label, t_ranges = units_to_use[SITE][PICK]
            if elec != prev_elec:
                sorted_data = read_pickle(os.path.join(site, "manually_curated", "curated-e{}.pkl".format(elec)))
                prev_elec = elec
            
            results.append(pool.apply_async(compute_strfs,
                (SITE, elec, label, t_ranges, sound_events, loader, site, sorted_data)
            ))
    print([result.get(timeout=60.0 * 60.0) for result in results])

print("Finished: {:.2f}".format(time.time() - start_time))

