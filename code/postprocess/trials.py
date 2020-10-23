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
from preprocess.sound.playback_utils import _parse_ladder_filename_data
from analysis.playbacks.categories import create_stimulus_dataframe, inject_spikes, relative_spike_times
import suss.io


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
        
    onsets_and_offsets = split_stims_by_vocalizations(row["stim"], expected_onsets=expected_onsets)
    min_dur = np.min([off - on for on, off in onsets_and_offsets])

    substims = []
    for idx, (onset, offset) in enumerate(onsets_and_offsets):
        substims.append([onset, offset])
    return substims


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

    unit_id = "e{}u{}s{}".format(electrode, cluster, site)
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
    kde_density = kde_density * len(spike_times) / np.sum(kde_density)

    # Find time periods where isi violations are low
    isi_violations = spike_times[:-1][np.diff(spike_times, axis=0) < 0.001]
    if not len(isi_violations):
        isi_violation_density = np.zeros(len(kde_time))
    else:
        isi_violation_kde = KernelDensity(kernel="gaussian", bandwidth=10.0, rtol=1e-2).fit(isi_violations[:, None])
        isi_violation_density = np.exp(isi_violation_kde.score_samples(kde_time[:, None]))
        isi_violation_density = isi_violation_density * len(isi_violations) / np.sum(isi_violation_density)
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

            # Trials with no spikes get lumped in with "bad trials"
            if len(local_spikes[-1]):
                local_good_trials.append(row["KeepTrial"])
            else:
                local_good_trials.append(False)

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

    return {
        "id": outputs["id"]["unit_id"],
        "waveforms": outputs["waveforms"],
        "responses": outputs["responses"]
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
        

def package_units(site, electrode, cluster=None, ignore_isi_above=0.1):
    """Package a unit up into a nice dataframe(s)
    """
    subject = site.split("_")[0]
    # Load cluster
    spike_file = suss.io.read_pickle(os.path.join(bird_dir, subject, "sites", site, "manually_curated", "curated-e{}.pkl".format(electrode)))
    labeled_nodes_dict = dict(spike_file.labeled_nodes)
    cluster_node = labeled_nodes_dict[cluster]
    spike_times = cluster_node.flatten().times
    spike_waveforms = cluster_node.flatten().waveforms

    # Cluster stats
    waveform_mean = np.mean(spike_waveforms, axis=0)
    waveform_std = np.std(spike_waveforms, axis=0)
    waveform_snr = (np.max(waveform_mean) - np.min(waveform_mean)) / np.mean(waveform_std)

    isi_violation_pct = np.sum(np.diff(spike_times) < 0.001) / len(np.diff(spike_times))
    if isi_violation_pct > ignore_isi_above:
        return None

    isi_distribution = np.histogram(np.diff(spike_times), bins=200, range=(0, 0.2))

    # Load events
    events = np.load(os.path.join(bird_dir, subject, "sites", site, "vocal_periods.npy"), allow_pickle=True)[()]

    # Should shoule make it use different file parser
    stim_df = create_stimulus_dataframe(events["playback"])
    stim_df = inject_spikes(stim_df, spike_times, t_buffer=10)


    # Get time range, choose a bandwidth based on the max spike rate
    tmin, tmax = stim_df["start_time"].iloc[0], stim_df["start_time"].iloc[-1]
    kde = KernelDensity(kernel="gaussian", bandwidth=10.0, rtol=1e-2).fit(spike_times[:, None])
    t_start = np.fix(tmin) - 2
    t_stop = np.fix(tmax) + 2
    kde_time = np.linspace(t_start, t_stop, int(t_stop - t_start) + 1)
    kde_density = np.exp(kde.score_samples(kde_time[:, None]))
    kde_density = kde_density * len(spike_times) / np.sum(kde_density)


    # Find time periods where isi violations are low
    isi_violations = spike_times[:-1][np.diff(spike_times, axis=0) < 0.001]
    if not len(isi_violations):
        isi_violation_density = np.zeros(len(kde_time))
    else:
        isi_violation_kde = KernelDensity(kernel="gaussian", bandwidth=10.0, rtol=1e-2).fit(isi_violations[:, None])
        isi_violation_density = np.exp(isi_violation_kde.score_samples(kde_time[:, None]))
        isi_violation_density = isi_violation_density * len(isi_violations) / np.sum(isi_violation_density)
    isi_violation_rate = isi_violation_density / kde_density
    below_threshold_filter = isi_violation_rate < 0.02
    # isi_violations_below_threshold = kde_time[below_threshold_filter]


    # Choose lower kde bandwidth for lower rate units?
    rate_threshold = choose_rate_threshold(kde_density)
    above_threshold_filter = kde_density > rate_threshold
    # kde_time_above_threshold = kde_time[above_threshold_filter]

    good_times = kde_time[below_threshold_filter & above_threshold_filter]

    unit_info = {
        "subject": subject,
        "site": site,
        "electrode": electrode,
        "cluster": cluster,
        "rate_threshold": rate_threshold,
        "isi_violation_threshold": 0.02,
        "kde_bandwidth": 10.0,
        "spike_waveform_mean": waveform_mean,
        "spike_waveform_std": waveform_std,
        "snr": waveform_snr,
        "isi_violations": isi_violation_pct,
        "isi_histogram": isi_distribution,
        "t_rate_kde": kde_time,
        "rate_kde": kde_density,
        "t_rate_isi_violations": kde_time,
        "rate_isi_violations": isi_violation_density,
    }

    # Create an output dataframe of only trials with kde_above_threshold
    stim_df["KeepTrial"] = stim_df.apply(lambda row: np.fix(row["start_time"]) in good_times, axis=1)

    prestim_buffer = 0.5
    psth_duration = 7.0  # (includes the prestim_buffer)
    psth_sampling_rate = 1000.0
    kwidth = 0.03

    output_rows = []

    lazy_data = np.load(os.path.join(bird_dir, subject, "sites", site, "lazy.npy"), allow_pickle=True)[()]

    # for stim_file, subdf in stim_df[stim_df["KeepTrial"] == True].groupby(["file"]):
    for stim_file, subdf in stim_df.groupby(["file"]):
        local_spikes = []
        local_mics = []
        mics = []
        local_good_trials = []

        for i in range(len(subdf)):
            row = subdf.iloc[i]
            t0 = row["start_time"] - prestim_buffer

            # Spikes
            local_spikes.append(spike_times[
                (spike_times >= t0) & (spike_times < (t0 + psth_duration))
            ] - t0 - prestim_buffer)

            # Mics
            '''
            mic = lazy_data.time_slice(t0, t0 + psth_duration, signal="mic")
            mic_sig = mic.data[:, 0] - mic.data[:, 0].mean()
            local_mics.append(mic_sig)
            '''

            if len(local_spikes[-1]):
                local_good_trials.append(row["KeepTrial"])
            else:
                local_good_trials.append(False)

        good_spikes = [local_spikes[j] for j in range(len(local_spikes)) if local_good_trials[j]]

        # t_mic = mic.t - t0 - prestim_buffer

        # read the sound file and create a sound array
        # do this only once per stim
        first_row = subdf.iloc[0]
        audio_path = os.path.join(os.path.join(root_dir, first_row["file_path"]))

        fs, sound_data = wavfile.read(audio_path)
        pad_before = np.zeros(int(fs * prestim_buffer))
        n_pad_after = int(fs * psth_duration) - int(fs * prestim_buffer) - sound_data.size
        if n_pad_after > 0:
            pad_after = np.zeros(n_pad_after)
            sound_data = np.concatenate((pad_before, sound_data, pad_after))
        else:
            sound_data = sound_data[:n_pad_after - 1]
            sound_data = np.concatenate((pad_before, sound_data))
        t_sound = np.linspace(-prestim_buffer, psth_duration - prestim_buffer - 1 / fs, sound_data.size)

        # Generate a kde (once per stim)
        psth_pts = int(psth_duration * psth_sampling_rate) + 1
        t_kde = np.linspace(-prestim_buffer, psth_duration - prestim_buffer, psth_pts)
        if not len(good_spikes):
            all_spikes = np.zeros((0, 1))
        else:
            all_spikes = np.concatenate(good_spikes).reshape(-1, 1)

        if len(all_spikes) == 0:
            local_spike_density = np.zeros(t_kde.shape)
        else:
            local_kde = KernelDensity(kernel="gaussian", bandwidth=kwidth, rtol=1e-2).fit(all_spikes)
            local_spike_density = np.exp(local_kde.score_samples(t_kde[:, None]))
            # local_spike_density = local_spike_density * len(all_spikes) * psth_sampling_rate / (np.sum(local_spike_density) * psth_duration)
            local_spike_density = local_spike_density * len(all_spikes) * psth_sampling_rate / (np.sum(local_spike_density) * len(good_spikes))

        output_rows.append({
            "file": stim_file,
            "relation": first_row["relation_short"],
            "call_type": first_row["relation_short"],  # This was bugged before... so swapping it here
            "stim_duration": first_row["duration"],
            "n_trials": len(local_spikes),
            "n_good_trials": int(np.sum(local_good_trials)),
            "good_trials": local_good_trials,
            "t_stim": t_sound,
            "stim": sound_data,
            # "t_mic": t_mic,
            # "mic": local_mics,    # Excluding these so saving doens't take forever
            "t_kde": t_kde.ravel(),
            "kde": local_spike_density,
            "spike_times": local_spikes
        })

    df_out = pd.DataFrame(output_rows)
    df_out["substim_idx"] = df_out.apply(divide_into_calls, axis=1)

    return {
        "unit_info": unit_info,
        "df": df_out
    }

            

    # First, we get all the trials
    # then we compute the rate smoothed kde, threshold, and select the good trials based on that
    # Load vocal_periods.npy and curated_eX.pkl

    # then we export the responses and stim arrays for those trials

    # we also want to include in the data info about the unit itself
    # This includes the spike shape and std, but we should also include the isi violations

    # We save it out into a trials dataframe + mic, a stim dataframe, and a unit info dict

