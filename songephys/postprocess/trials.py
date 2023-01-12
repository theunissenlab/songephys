"""
This is code for consolidating data from multiple sources of the preprocessing pipeline
(e.g. vocal periods, manually sorted spikes, etc, into tables for analysis
"""
# Core packages (Needs songephys environment)
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import time

# Data Processing Stuff
from sklearn.neighbors import KernelDensity
from scipy.io import wavfile

# Display and GUI
import ipywidgets as widgets

# Theunissen Lab Code
import suss.io

from ..preprocess.sound.playback_utils import _parse_ladder_filename_data
from ..analysis.playbacks.categories import create_stimulus_dataframe, inject_spikes


root_dir = "/auto/fhome/kevin/Projects/songephys/"
data_dir = os.path.join(root_dir, "data")
bird_dir = os.path.join(data_dir, "birds")


def choose_rate_threshold(dens):
    return np.mean(dens) / 2


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
    """Return indexes of wav where the sound onsets occur"""
    onsets = _seek(wav, expected_onsets=expected_onsets)
    offsets = _seek(wav, expected_onsets=expected_onsets, reverse=True)
    
    result = list(zip(onsets, offsets))

    return result


def to_unit_id(electrode, cluster, site):
    return "e{}u{}s{}".format(electrode, cluster, site)


def from_unit_id(unit_id):
    u_marker = unit_id.index("u")
    s_marker = unit_id.index("s")
    elec = int(unit_id[1:u_marker])
    unit = int(unit_id[u_marker + 1:s_marker])
    site = unit_id[s_marker + 1:]

    return {
        "electrode": elec,
        "unit": unit,
        "site": site
    }


def package_unit_lightweight(site, electrode, cluster, ignore_isi_above=0.1, good_isi_cutoff=0.02):
    """
    Create summary information for a single unit, split into separate data structures
    so it does not have to all be aligned at once

    id: keys that determine the identity of the unit
    summary: keys that summarize the unit (e.g. snr, isi information)
    waveforms: time and waveform arrays for all spike snippets
    responses: dataframe of spike timing responses grouped by stimulus
    """
    outputs = {}

    subject = site.split("_")[0]

    # Load cluster
    spike_file = suss.io.read_pickle(os.path.join(bird_dir, subject, "sites", site, "manually_curated", "curated-e{}.pkl".format(electrode)))
    labeled_nodes_dict = dict(spike_file.labeled_nodes)
    cluster_node = labeled_nodes_dict[cluster]

    unit_id = to_unit_id(electrode, cluster, site)
    outputs["id"] = dict(
        unit_id=unit_id,
        site=site,
        electrode=electrode,
        unit=cluster
    )
    spike_times = cluster_node.flatten().times
    spike_waveforms = cluster_node.flatten().waveforms

    ## Waveforms
    outputs["waveforms"] = dict(
        spike_times=spike_times,
        spike_waveforms=spike_waveforms,
    )

    ## Summary Stats
    waveform_mean = np.mean(spike_waveforms, axis=0)
    waveform_std = np.std(spike_waveforms, axis=0)
    waveform_snr = (np.max(waveform_mean) - np.min(waveform_mean)) / np.mean(waveform_std)

    isi_violation_pct = np.sum(np.diff(spike_times) < 0.001) / len(np.diff(spike_times))

    if isi_violation_pct > ignore_isi_above:
        return None
    else:
        isi_distribution = np.histogram(np.diff(spike_times), bins=200, range=(0, 0.2))
        outputs["summary"] = dict(
                snr=waveform_snr,
                isi_violation_pct=isi_violation_pct,
                isi_histogram=isi_distribution
        )

    # Select "good trials by thresholds"
    events = np.load(os.path.join(bird_dir, subject, "sites", site, "vocal_periods.npy"), allow_pickle=True)[()]

    # Should shoule make it use different file parser
    # TODO: use the correct file parser
    stim_df = create_stimulus_dataframe(events["playback"], parser=_parse_ladder_filename_data)
    stim_df = inject_spikes(stim_df, spike_times, t_buffer=10)

    # Get time range, choose a bandwidth based on the max spike rate??
    tmin, tmax = stim_df["start_time"].iloc[0], stim_df["start_time"].iloc[-1]
    kde = KernelDensity(kernel="gaussian", bandwidth=10.0, rtol=1e-2).fit(spike_times[:, None])
    t_start = np.fix(tmin) - 2
    t_stop = np.fix(tmax) + 2
    kde_time = np.linspace(t_start, t_stop, int(t_stop - t_start) + 1)
    kde_density = np.exp(kde.score_samples(kde_time[:, None]))
    kde_density = kde_density * len(spike_times)

    # Find time periods where isi violations are low
    isi_violations = spike_times[:-1][np.diff(spike_times, axis=0) < 0.001]
    if not len(isi_violations):
        isi_violation_density = np.zeros(len(kde_time))
    else:
        isi_violation_kde = KernelDensity(kernel="gaussian", bandwidth=10.0, rtol=1e-2).fit(isi_violations[:, None])
        isi_violation_density = np.exp(isi_violation_kde.score_samples(kde_time[:, None]))
        isi_violation_density = isi_violation_density * len(isi_violations)
    isi_violation_rate = isi_violation_density / kde_density
    below_threshold_filter = isi_violation_rate < good_isi_cutoff

    # TODO Choose lower kde bandwidth for lower rate units?
    rate_threshold = choose_rate_threshold(kde_density)
    above_threshold_filter = kde_density > rate_threshold
    good_times = kde_time[below_threshold_filter & above_threshold_filter]

    # Create an output dataframe of only trials with kde_above_threshold
    stim_df["KeepTrial"] = stim_df.apply(lambda row: np.fix(row["start_time"]) in good_times, axis=1)

    prestim_buffer = 1.0
    duration = 8.0  # (includes the prestim_buffer)

    threshold_metadata = {
        "t_arr": kde_time,
        "spike_rate": kde_density,
        "isi_violation_rate": isi_violation_density,
        "isi_violations_per_spike": isi_violation_rate,
        "spike_rate_threshold": rate_threshold,
        "isi_violations_per_spike_threshold": good_isi_cutoff,
    }

    # lazy_data = np.load(os.path.join(bird_dir, subject, "sites", site, "lazy.npy"), allow_pickle=True)[()]

    ## Get Responses
    output_rows = []
    for stim_file, subdf in stim_df.groupby(["file"]):
        first_row = subdf.iloc[0]

        local_spikes = []
        local_good_trials = []

        for i in range(len(subdf)):
            row = subdf.iloc[i]
            t0 = row["start_time"] - prestim_buffer

            # Spikes
            local_spikes.append(spike_times[
                (spike_times >= t0) & (spike_times < (t0 + duration))
            ] - t0 - prestim_buffer)

            local_good_trials.append(row["KeepTrial"])

        good_spikes = [local_spikes[j] for j in range(len(local_spikes)) if local_good_trials[j]]
        output_rows.append({
            "file": stim_file,
            "relation": first_row["relation_short"],
            "self": subject[:-1] in stim_file,
            "exposure": "familiar" if first_row["relation_short"] == "test" else first_row["relation_short"],  # this column just maps all test stimuli to familiar
            "call_type": first_row["call_type"],  # This was bugged before... so swapping it here
            "stim_duration": first_row["duration"],

            "n_trials": len(local_spikes),
            # "n_trials_good": int(np.sum(local_good_trials)),

            "good_trials": local_good_trials,
            "spike_times": local_spikes
            # "spike_times_good": local_spikes
        })
        output_rows[-1].update(outputs["summary"])
        output_rows[-1].update(outputs["id"])

    outputs["responses"] = pd.DataFrame(output_rows)

    threshold_metadata["unit_id"] = outputs["id"]["unit_id"]

    return {
        "id": outputs["id"]["unit_id"],
        "waveforms": outputs["waveforms"],
        "responses": outputs["responses"],
        "threshold_metadata": threshold_metadata,
    }


def package_stims_and_renditions(stim_files):
    output_rows = []
    for stim_file in stim_files:
        basename = os.path.basename(stim_file)

        # This is specific to the design of our pecking test
        if basename.startswith("So"):
            expected_onsets = 3
        elif basename.startswith("DC"):
            expected_onsets = 3
        else:
            expected_onsets = 1

        fs, sound_data = wavfile.read(stim_file)
        t_sound = np.arange(len(sound_data)) / fs
        output_rows.append({
            "stimulus_file": basename,
            "stim": sound_data,
            "t_stim": t_sound,
            "sampling_rate": fs,
            "duration": len(sound_data) / fs,
            "rendition_indexes": split_stims_by_vocalizations(sound_data, expected_onsets=expected_onsets)
        })

    return pd.DataFrame(output_rows)
        
