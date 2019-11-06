"""Functions for assessing the quality of clusters
"""
import os
import numpy as np

import scipy.stats
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import suss.io

import matplotlib.pyplot as plt

from util import border


def _compute_overlap(neighbors, labels, A, B):
    a_or_b = (labels == A) | (labels == B)

    size = np.sum(a_or_b)

    total = []

    sample_count = np.min([500, np.sum(labels==A), np.sum(labels==B)])
    if sample_count < 20:
        return 1.0
    for cluster in [A, B]:
        points = np.random.choice(np.arange(np.sum(labels == cluster)), size=sample_count, replace=False)

        for pt in points:
            k_nearest_labels = labels[neighbors[labels == cluster][pt]]
            total.append(np.mean(k_nearest_labels == cluster))

    return np.mean(total)


def get_t_binned(times, t_max=None, dt=60.0):
    if t_max is None:
        t_max = np.max(times)
    steps = int(np.ceil(t_max / dt))
    t_binned = np.arange(steps) * dt
    return t_binned


def rate_over_time(times, t_max=None, dt=60.0):
    if t_max is None:
        t_max = np.max(times)
    steps = int(np.ceil(t_max / dt))
    rate_binned = np.zeros(steps)
    
    for t in times:
        bin_idx = np.int(t / dt)
        rate_binned[bin_idx] += 1
        
    rate_binned /= dt

    return rate_binned

def firing_rate_over_time(times, labels, t_max=None, dt=30.0):
    """
    Computer the firing rate over t0 second windows for all the data and for each label
    """
    if t_max is None:
        t_max = np.max(times)
    results = {}
    for label in np.unique(labels):
        rate = rate_over_time(times[labels == label], t_max=t_max, dt=dt)
        results[label] = rate
    results["all"] = rate_over_time(times, t_max=t_max, dt=dt)

    return results

def isi_violations_over_time(times, labels, t_max=None, dt=30.0):
    if t_max is None:
        t_max = np.max(times)
    steps = int(np.ceil(t_max / dt))
    
    t_binned = np.arange(steps) * dt
    results = {}
    
    for label in np.unique(labels):
        # Bins
        rate_binned = np.zeros(steps)
        bin_counts = np.zeros(steps)
        
        t = times[labels == label]
        
        for time_of_spike in t:
            bin_idx = int(time_of_spike / dt)
            bin_counts[bin_idx] += 1

        for violation_idx in np.where(np.diff(t) < 0.001)[0]:
            time_of_violation = t[violation_idx]
            bin_idx = int(time_of_violation / dt)
            rate_binned[bin_idx] += 1
        
        results[label] = rate_binned / bin_counts
    
    return results


def isolation(waveforms, labels, k=10):
    unique_labels = np.unique(labels)
    overlaps = np.zeros((len(unique_labels), len(unique_labels)))

    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=2).fit(waveforms)
    _, neighbors = knn.kneighbors(waveforms)

    for i, A in enumerate(unique_labels):
        for j, B in enumerate(unique_labels):
            if i == j:
                overlaps[i, j] = 1.0
                continue
            overlaps[i, j] = _compute_overlap(neighbors, labels, A, B)

    mins = np.min(overlaps, axis=0)

    result = {}
    for i, label in enumerate(unique_labels):
        result[label] = mins[i]

    return result

def cluster_isolation_over_time(times, waveforms, labels, t_max=None, dt=30.0):
    if t_max is None:
        t_max = np.max(times)
    steps = int(np.ceil(t_max / dt))
    
    t_binned = np.arange(steps) * dt
    
    results = dict(
        (label, np.zeros(steps)) for label in np.unique(labels)
    )
    for bin_idx, t_0 in enumerate(t_binned):
        print("{}/{}".format(t_0, t_max), end="\r")
        t_1 = t_0 + dt
        
        selector = np.where((times > t_0) & (times <= t_1))
        
        if not len(selector[0]):
            for label in np.unique(labels):
                results[label][bin_idx] = 0.0
            continue

        bin_wfs = waveforms[selector]
        bin_labels = labels[selector]
        iso_dict = isolation(bin_wfs, bin_labels)
        
        for label in np.unique(labels):
            results[label][bin_idx] = iso_dict.get(label, 0.0)
    
    return results

def recombine_with_deleted_wfs(sorted_data, original_data):
    if not len(sorted_data):
        return np.array([]), np.array([]), np.array([])
    
    sorted_data = sorted_data.flatten()
    original_data = original_data.flatten()
    
    total_times = []
    total_wfs = []
    total_labels = []
    
    pointer = 0
    
    bg_label = np.min(sorted_data.labels) - 1
    for t, wf in zip(original_data.times, original_data.waveforms):
        total_times.append(t)
        total_wfs.append(wf)

        if sorted_data.times[pointer] != t:
            total_labels.append(bg_label)
        else:
            total_labels.append(sorted_data.labels[pointer])
            pointer += 1
            if pointer >= len(sorted_data):
                break
    
    assert pointer == len(sorted_data.times)  # Make sure we accounted for all the sorted_data points
    return np.array(total_times), np.array(total_wfs), np.array(total_labels)


def generate_plots(site, elec=None, unit=None):
    save_path = os.path.join(site, "unit_summaries")

    sound_events = np.load(os.path.join(site, "vocal_periods.npy"), allow_pickle=True)[()]

    try:
        os.makedirs(save_path)
    except:
        pass
    
    if elec is None:
        elecs = config.ELECTRODES
    else:
        elecs = [elec]
        
    for elec in elecs:    
        try:
            os.makedirs(os.path.join(save_path, str(elec)))
        except:
            pass

        bg_data = suss.io.read_pickle(os.path.join(site, "spikes", "spike_waveforms-e{}.pkl".format(elec)))
        sorted_file = os.path.join(site, "manually_curated", "curated-e{}.pkl".format(elec))
        if os.path.exists(sorted_file):
            sorted_data = suss.io.read_pickle(sorted_file)
        else:
            sorted_data = suss.io.read_pickle(
                os.path.join(site, "manually_curated", "curated-e{}.pkl".format(elec)))
        
        t_arr, wfs, labels = recombine_with_deleted_wfs(sorted_data, bg_data)
        if not len(t_arr):
            continue

        dt = 5 * 60.0
        iso_binned = cluster_isolation_over_time(t_arr, wfs, labels, dt=dt)

        isi_binned = isi_violations_over_time(t_arr, labels, dt=dt)
        fr_binned = firing_rate_over_time(t_arr, labels, dt=dt)
        t_binned = get_t_binned(t_arr, dt=dt)
        
        for label in np.unique(labels):
            if unit is not None and label != unit:
                continue
                
#             t_ticks = np.arange(0, t_binned[-1], 3600)[1:]
#             t_labels = ["{:.1f} hr".format(t / 3600) for t in t_ticks]

            fig = plt.figure(figsize=(10, 8))
            ax1 = plt.subplot2grid((5, 3), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((5, 3), (2, 0), rowspan=1)
            ax7 = plt.subplot2grid((5, 3), (3, 0), rowspan=1)

            ax3 = plt.subplot2grid((5, 3), (3, 1), colspan=2)
            ax4 = plt.subplot2grid((5, 3), (2, 1), colspan=2)
            ax5 = plt.subplot2grid((5, 3), (1, 1), colspan=2)
            ax6 = plt.subplot2grid((5, 3), (0, 1), colspan=2)
            ax8 = plt.subplot2grid((5, 3), (4, 1), colspan=2)

            ax8.vlines([event["start_time"] for event in sound_events["playback"]], 0, 1, linestyle="-")

            ax8.vlines([event["start_time"] for event in sound_events["live"]], 1, 2, linestyle="-")
            '''
            ax8.vlines(
                [event["start_time"] for event in sound_events if event["sound_type"] == "live" and event.get("bird") == "tutor"],
                1, 2, linestyle="-")

            ax8.vlines(
                [event["start_time"] for event in sound_events if event["sound_type"] == "live" and event.get("bird") == "subject"],
                2, 3, linestyle="-")
            '''

            ax8.set_xlim(t_binned[0], t_binned[-1])

            border(ax8)
            ax8.set_xticks([])
            ax8.set_yticks([0.5, 1.5, 2.5])
            ax8.set_yticklabels(["playback", "tutor", "subject"])

            border(ax1, 1)
            border(ax2, 0, 0, 0, 1)

            ax1.plot(wfs[labels == label][::100].T, color="Gray", alpha=0.2)
            ax1.set_title("{}\nElec {} Unit {}; (n={})".format(
                os.path.basename(site),
                elec,
                label,
                int(np.sum(labels == label))))
            ax1.set_ylim(-500, 300)
            ax1.set_xticks([])

            peaks = wfs[labels == label][:, 20]
            mean_peak = np.mean(peaks)
            skew = np.mean((peaks - mean_peak) ** 3 / np.std(peaks) ** 3)
            ax2.hist(peaks, bins=100, range=(-300, 0), color="Black")
            ax2.set_title("skew = {:.1f}".format(skew))
            ax2.set_yticks([])
            
            isi = np.diff(t_arr[labels == label])
            ax7.hist(isi, range=(0, 0.1), bins=50, density=True, color="Black")
            ax7.set_title("isi")
            ax7.set_yticks([])
            border(ax7, 0, 0, 0, 1)

            ax3.fill_between(t_binned, 0, iso_binned[label], color="TEAL")
            ax3.set_ylim(0, 1)
            ax3.set_xlim(t_binned[0], t_binned[-1])
#             ax3.set_xticks(t_ticks)
#             ax3.set_xticklabels(t_labels)
            ax3.set_title("Cluster Isolation", loc="left")
            ax3.set_yticks(np.arange(11) / 10, minor=True)
            ax3.grid(which="minor", axis="y")
            ax3.set_axisbelow(True)

            border(ax3, 0, 0, 0, 1)

            ax4.fill_between(t_binned, 0, 100 * isi_binned[label], color="MAROON")
            ax4.set_xlim(t_binned[0], t_binned[-1])
            ax4.set_ylim(0, 10)
            ax4.hlines([2], *ax4.get_xlim(), color="Black", linestyle="--")
            ax4.set_xticks([])
            ax4.set_yticks([2, 5, 10])
            ax4.set_yticklabels([2, 5, 10])
            ax4.set_ylabel("% < 1ms")
            ax4.set_title("ISI Violations", loc="left")
            border(ax4, 1, 0, 0, 1)

        #     ax5.fill_between(t_binned, 0, fr_binned["all"], alpha=0.1, color="#7A9CE4")
            ax5.fill_between(t_binned, 0, fr_binned[label], color="Black")
            ax5.set_title("Firing Rate", loc="left")
            ax5.set_xlim(t_binned[0], t_binned[-1])
            ax5.set_xticks([])
            ax5.set_ylabel("Hz")
            ax5.grid(which="minor", axis="y")
            border(ax5, 1)

            lda = LDA(n_components=1).fit(wfs, labels == label)
            for _label in np.unique(labels):
                if _label == np.min(labels):
                    continue
                ax6.scatter(t_arr[labels == _label], lda.transform(wfs[labels == _label]).T[0],
                                 s=10 if _label == label else 2, 
                                 alpha=0.3, 
                                 zorder=3 if _label == label else 2,
                                 color="Black" if _label == label else "Gray")
            ax6.set_xlim(t_binned[0], t_binned[-1])
            border(ax6)
            ax6.set_xticks([])
            ax6.set_yticks([])

            plt.tight_layout()
            fig.savefig(os.path.join(save_path, str(elec), "Unit {}.png".format(label)))
            plt.show()
            plt.close(fig)

