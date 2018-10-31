##
## Decode syllables
## for every site
##

from units_to_use import units_to_use, channel_map, putative_brain_regions
from decoding.groups import syllable_groups
from decoding.psth import sample_psth, get_raster, get_counts, mean_spec, peak_align, decode
import numpy as np
import os
from suss.io import read_pickle
import matplotlib.pyplot as plt
from segmentation.audio_segment import merge_segments
from soundsig.sound import spectrogram, plot_spectrogram
from sklearn.neighbors.kde import KernelDensity
from visualize.general import border
import config
from spikes.utils import bin_spikes
from spikes.plotting import raster
from spikes.psth import conv, gaussian_convolver, exponential_convolver
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
from sklearn.metrics import confusion_matrix
import pandas as pd

all_results = []
result_headings = [
        "site",
        "elec",
        "unit",
        "syllable_category",
        "decoding_in",
        "acc_mean",
        "acc_std",
        "chance_mean",
        "chance_std",
        "mi_mean",
        "mi_std"
]


CONDITION = "tutor"
for SITE, syllable_group in syllable_groups.items():
    bird = SITE.split("_")[0]
    site = os.path.join(config.DATADIR, "birds", bird, "sites", SITE)
    loader = read_pickle(os.path.join(site, "lazy_data.pkl"))

    fs = loader._signals["mic"][0].sampling_rate

    # Get all units at the site
    all_units = []
    for site_, d in units_to_use.items():
        for key in d:
            if site_ == SITE:
                all_units.append((site_, key[0], key[1]))

    spike_files = {}
    for elec in config.ELECTRODES:
        spike_files[elec] = read_pickle(
            os.path.join(
                site,
                "manually_curated",
                "curated-e{}.pkl".format(elec))
        ).flatten()

    syl_dataframe_path = os.path.join(site, "syllables_dataframe.pkl")
    if not os.path.exists(syl_dataframe_path):
        print("Havent converted syllables for {} into dataframe yet. "
                "{}".format(SITE))

    orig_syllables = pd.read_pickle(syl_dataframe_path)
    unique_syllables = get_counts(orig_syllables)
    unique_syllables["syllable_category"] = -1
    unique_syllables["syllable_category"] = unique_syllables["syllable_category"].astype(int)

    syllable_id = 0
    unique_syllables["syllable_id"] = -1
    for idx, group in enumerate(syllable_group):
        group_id = idx
        for vocalizer, sound_type, label in group["classes"]:
            unique_syllables["syllable_category"][
                (unique_syllables["label"] == label) &
                (unique_syllables["sound_type"] == sound_type) &
                (unique_syllables["vocalizer"] == vocalizer)
            ] = group_id
            unique_syllables["syllable_id"][
                (unique_syllables["label"] == label) &
                (unique_syllables["sound_type"] == sound_type) &
                (unique_syllables["vocalizer"] == vocalizer)
            ] = syllable_id
            syllable_id += 1

    syllables = orig_syllables.merge(unique_syllables, on=["label", "sound_type", "vocalizer"])

    # prepare display strings and colors
    colors = {}
    disp = {}
    for _, row in get_counts(syllables).iterrows():
        if row["syllable_id"] != -1:
            colors[row["syllable_id"]] = {
                ("live", "subject"): "#4d9221",
                ("playback", "subject"): "#e6f5d0",
                ("playback", "juvenile"): "#b8e186",
                ("live", "tutor"): "#d01c8b",
                ("playback", "tutor"): "#f1b6da",
                ("live", "con_male_familiar"): "#0571b0",
                ("playback", "con_male_familiar"): "#92c5de",
            }[(row["sound_type"], row["vocalizer"])]
            disp[row["syllable_id"]] = "{} {}".format(
                {
                    "subject": "BOS",
                    "tutor": "TUT",
                    "con_male_familiar": "CON",
                    "juvenile": "JUV",
                }[row["vocalizer"]],
                row["sound_type"]
            )

    for syl_cat, syl_lookup in enumerate(syllable_group):
        events = syllables[
            (syllables.syllable_category == syl_cat) &
        #     (syllables.sound_type == "playback")
            (syllables.vocalizer == CONDITION)
        ]

        # recast as int 
        events.syllable_category = events.syllable_category.astype(int)
        events.syllable_id = events.syllable_id.astype(int)
        dur = syl_lookup["duration"]
        if dur < 0.2:
            continue

        for (SITE_STR, ELEC, UNIT) in all_units:
            before = 0.5
            after = 0.5

            for e, u, time_ranges in units_to_use[SITE_STR]:
                if (ELEC, UNIT) == (e, u):
                    break

            selected_events = events.where(
                events["t_start"].apply(lambda t: np.any([
                        t0 < t < t1
                        for t0, t1 in time_ranges
                ]))
            ).dropna()
            
            labels = selected_events.syllable_id.astype(np.int)
            
            spikes = spike_files[ELEC]
            spikes = spikes.times[spikes.labels == UNIT]
            t, rast, rast_times = get_raster(spikes, selected_events["t_start"], before, after + dur)
            if rast is None:
                continue
                
            usable = np.array([np.sum(x) > 0 for x in rast.T])
            rast = rast[:, usable]
            rast = rast[(t > before) & (t < before + after + dur)]
            labels = labels[usable]
            selected_events = selected_events.loc[usable]
            
            print("Running", SITE_STR, ELEC, UNIT, putative_brain_regions[SITE_STR][(ELEC, UNIT)])
            print(get_counts(selected_events))
            
            if len([row for _, row in get_counts(selected_events).iterrows() if row["count"] >= 10]) < 2:
                continue

            rast = conv(rast.T, gaussian_convolver, 10).T
            results = decode(rast.T, labels)
            shuff_results = decode(rast.T, labels, shuffle=True)
            
            accuracy = (
                np.mean([np.mean(result[0] == result[1]) for result in results]),
                np.std([np.mean(result[0] == result[1]) for result in results])
            )  

            accuracy_chance = (
                np.mean([np.mean(result[0] == result[1]) for result in shuff_results]),
                np.std([np.mean(result[0] == result[1]) for result in shuff_results])
            )  
            
            mi = (
                np.mean([adjusted_mutual_info_score(result[0], result[1]) for result in results]),
                np.std([adjusted_mutual_info_score(result[0], result[1]) for result in results])
            )

            result_row = [
                SITE_STR,
                ELEC,
                UNIT,
                syl_cat,
                CONDITION,
                accuracy[0],
                accuracy[1],
                accuracy_chance[0],
                accuracy_chance[1],
                mi[0],
                mi[1]
            ]

            all_results.append(result_row)


CONDITION = "bos_vs_tut"
for SITE, syllable_group in syllable_groups.items():
    bird = SITE.split("_")[0]
    site = os.path.join(config.DATADIR, "birds", bird, "sites", SITE)
    loader = read_pickle(os.path.join(site, "lazy_data.pkl"))

    fs = loader._signals["mic"][0].sampling_rate

    # Get all units at the site
    all_units = []
    for site_, d in units_to_use.items():
        for key in d:
            if site_ == SITE:
                all_units.append((site_, key[0], key[1]))

    spike_files = {}
    for elec in config.ELECTRODES:
        spike_files[elec] = read_pickle(
            os.path.join(
                site,
                "manually_curated",
                "curated-e{}.pkl".format(elec))
        ).flatten()

    syl_dataframe_path = os.path.join(site, "syllables_dataframe.pkl")
    if not os.path.exists(syl_dataframe_path):
        print("Havent converted syllables for {} into dataframe yet. "
                "{}".format(SITE))

    orig_syllables = pd.read_pickle(syl_dataframe_path)
    unique_syllables = get_counts(orig_syllables)
    unique_syllables["syllable_category"] = -1
    unique_syllables["syllable_category"] = unique_syllables["syllable_category"].astype(int)

    syllable_id = 0
    unique_syllables["syllable_id"] = -1
    for idx, group in enumerate(syllable_group):
        group_id = idx
        for vocalizer, sound_type, label in group["classes"]:
            unique_syllables["syllable_category"][
                (unique_syllables["label"] == label) &
                (unique_syllables["sound_type"] == sound_type) &
                (unique_syllables["vocalizer"] == vocalizer)
            ] = group_id
            unique_syllables["syllable_id"][
                (unique_syllables["label"] == label) &
                (unique_syllables["sound_type"] == sound_type) &
                (unique_syllables["vocalizer"] == vocalizer)
            ] = syllable_id
            syllable_id += 1

    syllables = orig_syllables.merge(unique_syllables, on=["label", "sound_type", "vocalizer"])

    # prepare display strings and colors
    colors = {}
    disp = {}
    for _, row in get_counts(syllables).iterrows():
        if row["syllable_id"] != -1:
            colors[row["syllable_id"]] = {
                ("live", "subject"): "DarkBlue",
                ("playback", "subject"): "Blue",
                ("playback", "juvenile"): "Indigo",
                ("live", "tutor"): "Red",
                ("playback", "tutor"): "OrangeRed",
                ("live", "con_male_familiar"): "#1FDA52",
                ("playback", "con_male_familiar"): "#17700E",
            }[(row["sound_type"], row["vocalizer"])]
            disp[row["syllable_id"]] = "{} {}".format(
                {
                    "subject": "BOS",
                    "tutor": "TUT",
                    "con_male_familiar": "CON",
                    "juvenile": "JUV",
                }[row["vocalizer"]],
                row["sound_type"]
            )

    # Now we iterate over syllables:
    for syl_cat, syl_lookup in enumerate(syllable_group):
        events = syllables[
            (syllables.syllable_category == syl_cat) &
            (syllables.sound_type == "live") &
            (np.isin(syllables.vocalizer, ["tutor", "subject"]))
            # (syllables.vocalizer == CONDITION)
        ]

        # recast as int 
        events.syllable_category = events.syllable_category.astype(int)
        events.syllable_id = events.syllable_id.astype(int)
        dur = syl_lookup["duration"]
        if dur < 0.2:
            continue

        for (SITE_STR, ELEC, UNIT) in all_units:
            before = 0.5
            after = 0.5

            for e, u, time_ranges in units_to_use[SITE_STR]:
                if (ELEC, UNIT) == (e, u):
                    break

            selected_events = events.where(
                events["t_start"].apply(lambda t: np.any([
                        t0 < t < t1
                        for t0, t1 in time_ranges
                ]))
            ).dropna()
            
            labels = selected_events.syllable_id.astype(np.int)
            
            spikes = spike_files[ELEC]
            spikes = spikes.times[spikes.labels == UNIT]
            t, rast, rast_times = get_raster(spikes, selected_events["t_start"], before, after + dur)
            if rast is None:
                continue
                
            usable = np.array([np.sum(x) > 0 for x in rast.T])
            rast = rast[:, usable]
            rast = rast[(t > before) & (t < before + after + dur)]
            labels = labels[usable]
            selected_events = selected_events.loc[usable]
            
            print("Running", SITE_STR, ELEC, UNIT, putative_brain_regions[SITE_STR][(ELEC, UNIT)])
            print(get_counts(selected_events))
            
            if len([row for _, row in get_counts(selected_events).iterrows() if row["count"] >= 10]) < 2:
                continue

            rast = conv(rast.T, gaussian_convolver, 10).T
            results = decode(rast.T, labels)
            shuff_results = decode(rast.T, labels, shuffle=True)
            
            accuracy = (
                np.mean([np.mean(result[0] == result[1]) for result in results]),
                np.std([np.mean(result[0] == result[1]) for result in results])
            )  

            accuracy_chance = (
                np.mean([np.mean(result[0] == result[1]) for result in shuff_results]),
                np.std([np.mean(result[0] == result[1]) for result in shuff_results])
            )  
            
            mi = (
                np.mean([adjusted_mutual_info_score(result[0], result[1]) for result in results]),
                np.std([adjusted_mutual_info_score(result[0], result[1]) for result in results])
            )

            result_row = [
                SITE_STR,
                ELEC,
                UNIT,
                syl_cat,
                CONDITION,
                accuracy[0],
                accuracy[1],
                accuracy_chance[0],
                accuracy_chance[1],
                mi[0],
                mi[1]
            ]

            all_results.append(result_row)

df = pd.DataFrame(all_results, columns=result_headings)
df.to_pickle("DECODE_RESULT.pkl")
