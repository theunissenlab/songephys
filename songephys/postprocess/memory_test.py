import os
from functools import partial

import numpy as np
import pandas as pd
import scipy

from .coherence import multitapered_coherence


def lookup_reward(subject, stim_file, pecking_trial_data):
    """For a given subject and stim_file, lookup if that stim file was rewarded or not rewarded"""
    splitted = stim_file.split("_")
    call_type = splitted[0]
    vocalizer = splitted[3]
    trials = pecking_trial_data[
        (pecking_trial_data["Subject"] == subject) &
        (pecking_trial_data["Stimulus Vocalizer"] == vocalizer) &
        (pecking_trial_data["Stimulus Call Type"] == call_type.upper())
    ]
    if len(trials["Stimulus Class"].unique()) > 1:
        raise Exception("Multiple stimulus classes for one stim file for this subject")
    
    return trials["Stimulus Class"].unique()[0]


def _get_spikes_in_range(tmin, tmax, row, require_silence=0.0):
    """Get spikes in the range tmin-tmax for the given stim row (there are 0-10 trials per stim)

    But there can be more repetitions since each stim can be split up into multiple calls
    (e.g. 3 dcs per stim file)

    The require_silence option can be used to restrict the rows to only the renditions that are preceded
    by require_silence seconds of silence (this can be used to rejected the 2nd/3rd rendition of a song/DC
    in a file if it comes to closely after the preceding rendition)
    """
    good_spike_times = np.array(row["spike_times"])[
        (np.array(row["good_trials"])) &
        (row["prestim_silence"] >= require_silence)
    ]
    return [len(trial[(trial > tmin) & (trial < tmax)]) for trial in good_spike_times]


def _get_spikes_per_trial(row):
    good_spike_times = np.array(row["spike_times"])[np.array(row["good_trials"])]
    return [len(trial) for trial in good_spike_times]


def create_relation_columns(subject, df, pecking_trial_data):
    def _get_relation(row):
        basename, ext = os.path.splitext(row["file"])
        parts = basename.split("_")
        last_part = parts[-1]
        
        if last_part == "Familiar":
            return "Familiar"
        
        elif last_part == "Unfamiliar":
            return "Unfamiliar"
        
        elif last_part == "SHUFFLED":
            rew = lookup_reward(subject, row["file"], pecking_trial_data)
            return "Shuffled {}".format(rew)
        
        elif last_part == "fs30k":
            return "Ripple"
        
        else:
            rew = lookup_reward(subject, row["file"], pecking_trial_data)
            return "Test {}".format(rew)
        
    df["relation"] = df.apply(_get_relation, axis=1)
    df["call_type"] = df["call_type"]


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
    """Split trials into two chunks of min_trials/2"""
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


def get_auditory_stats(df):
    """Compute firing rate before and after stims, """  
    results = []
    
    def _get_spikes_in_range(tmin, tmax, row):
        good_spike_times = np.array(row["spike_times"])[np.array(row["good_trials"])]
        return [len(trial[(trial > tmin) & (trial < tmax)]) for trial in good_spike_times]
    
    prestim_spike_counts = np.concatenate(
        df.apply(partial(_get_spikes_in_range, -0.5, 0.0), axis=1)
    ) / 0.5
    poststim_spike_counts = np.concatenate(
        df.apply(partial(_get_spikes_in_range, 0.0, 0.5), axis=1)
    ) / 0.5

    if not len(prestim_spike_counts):
        result_dict = {
            "baseline_fr": None,
            "stim_fr": None,
            "tscore": None,
            "zscore": None,
            "pvalue": None,
        }
        return result_dict

    firing_rate_pre = np.mean(prestim_spike_counts)
    firing_rate_pre_std = np.std(prestim_spike_counts)
    firing_rate_post = np.mean(poststim_spike_counts)
    firing_rate_post_std = np.std(poststim_spike_counts)
    firing_rate_differences = (
        (poststim_spike_counts - prestim_spike_counts)
    )
    z = np.mean(firing_rate_differences) / np.std(firing_rate_differences)

    test = scipy.stats.ttest_rel(
        prestim_spike_counts, 
        poststim_spike_counts
    )
    
    result_dict = {
        "baseline_fr": firing_rate_pre,
        "stim_fr": firing_rate_post,
        "tscore": test.statistic,
        "zscore": z,
        "pvalue": test.pvalue,
    }

    return result_dict


def get_categories(divided_df):
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

    return categories


def compute_coherences(unit_data, pecking_test_data, n_trials=8):
    if "divided_df" in unit_data:
        divided_df = unit_data["divided_df"]
    else:
        divided_df = create_split_df(unit_data["df"])
        create_relation_columns(unit_data["unit_info"]["subject"], divided_df, pecking_test_data)
        create_relation_columns(unit_data["unit_info"]["subject"], unit_data["df"], pecking_test_data)
        unit_data["divided_df"] = divided_df

    result = {}

    result["site"] = unit_data["unit_info"]["site"]
    result["electrode"] = unit_data["unit_info"]["electrode"]
    result["cluster"] = unit_data["unit_info"]["cluster"]
    result["subject"] = unit_data["unit_info"]["subject"]
    result["snr"] = unit_data["unit_info"]["snr"]
    result["isi_violation_pct"] = unit_data["unit_info"]["isi_violations"]
    result["avg_good_trials"] = np.mean(unit_data["df"]["n_good_trials"])

    # Compute rough firing rate stats and # trials
    result.update(get_auditory_stats(divided_df))

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
        psths = half_and_half_psths(subdf, min_trials=n_trials)
        coherence_result = multitapered_coherence(psths, chunk_size=512, sampling_rate=1000)
        corrected_coherence = coherence_correction(coherence_result, n_trials=n_trials)


    
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

    return result, unit_data




