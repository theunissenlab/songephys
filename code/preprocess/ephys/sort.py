import time

import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import umap
from MulticoreTSNE import MulticoreTSNE as TSNE
from isosplit5 import isosplit5
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor

from suss.io import read_pickle, save_pickle


PCS = 12


def coarse_cluster(data, k=3, m=4):    
    outer_labels = isosplit5(data.T)
    
    for outer_label in np.unique(outer_labels):
        inner_data = data[outer_labels == outer_label]
        if len(inner_data) <= 2 * k:
            continue
        gmm = KMeans(n_clusters=k, max_iter=300).fit(inner_data)
        labels = gmm.predict(inner_data)
        for label in np.unique(labels):
            if np.sum(labels == label) < 6:
                continue
            try:
                gmm = BayesianGaussianMixture(n_components=m, max_iter=500).fit(inner_data[labels == label])
            except:
                continue
            sublabels = gmm.predict(inner_data[labels == label])
            prob = np.exp(GaussianMixture(n_components=1).fit(inner_data[labels == label, :1]).score_samples(inner_data[labels == label, :1]))
            labels[labels == label] = sublabels + np.max(labels) + 1
        outer_labels[outer_labels == outer_label] = labels + np.max(outer_labels) + 1

    return outer_labels


def snr(waveforms, labels, label):
    all_stds = np.mean(np.std(waveforms, axis=1))
    mean_wf = np.mean(waveforms[labels == label], axis=0)
    peak = mean_wf[waveforms.shape[1] // 2]
    if peak < 0:
        peak = min(np.max(mean_wf), 0) - peak
    elif peak > 0:
        peak = peak - max(np.min(mean_wf), 0)

    if np.sum(labels == label) == 1:
        return np.mean(np.std(waveforms[labels == label]) / all_stds)
    if np.sum(labels == label) < 5:
        return peak / np.mean(np.std(waveforms[labels == label], axis=0))

    std = np.mean(np.std(waveforms[labels == label], axis=0))
    return peak / std


def xcorr(t1, t2, dt=3000, bins=31):
    times = np.concatenate([(t2[(t2 >= (t - dt)) & (t2 <= (t + dt))] - t) for t in t1])
    return times[(np.abs(times) <= dt) & (times != 0)]


class Waveforms(object):
    def __init__(self, spike_ind, spike_wf, index):
        self.ind = spike_ind
        self.wf = spike_wf
        self.index = index

        self.spike_width = self.wf.shape[1] // 2
        self.peaks = self.wf[:, self.spike_width]

    def __len__(self):
        return len(self.ind)

    def select(self, selector):
        return Waveforms(self.ind[selector], self.wf[selector], self.index[selector])
        
    def pcaed(self, pcs):
        return PCA(n_components=pcs).fit_transform(self.wf)


def remove_isi_violations(wf, labels):
    keep_overall = np.ones_like(labels).astype(np.bool)
    for label in np.unique(labels):
        if np.sum(labels == label) < 2:
            continue

        isi_violations = np.diff(wf.ind[labels == label]) < 30
        err_rate = np.mean(isi_violations)

        keep_in_label = np.ones(int(np.sum(labels == label))).astype(np.bool)
        if err_rate > 0.05:
            isi_violation_points = np.logical_or(
                np.concatenate([[False], isi_violations]),
                np.concatenate([isi_violations, [False]])
            )
            keep_in_label[isi_violation_points] = False
            keep_overall[labels == label] = keep_in_label  # Nested because second slice is not writable

    return keep_overall


def isi_frac(isi_violations, labels, label):
    isi = len(np.where(
	isi_violations &
	(
	    (labels[:-1] == label) |
	    (labels[1:] == label)
	)
    )[0])
    possible = len(np.where(
	    (labels[:-1] == label) |
	    (labels[1:] == label)
    )[0])
    return isi / possible


def get_unique_labels(labels):
    return sorted(np.unique(labels), key=lambda l: np.sum(labels == l), reverse=True)


def generate_similarity_matrix(wf, labels, unique_labels):
    similarity = np.zeros((len(unique_labels), len(unique_labels)))
    snrs = np.zeros((len(unique_labels), len(unique_labels)))

    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            snrs[i][j] = snrs[j][i] = max(
                snr(wf.wf, labels, label_i),
                snr(wf.wf, labels, label_j)
            )

            if label_i == label_j:
                similarity[i][j] = 1
                continue
            elif label_i >= label_j:
                continue

            # cross_corr = xcorr(wf.ind[labels == label_i], wf.ind[labels == label_j], dt=6000)
            # if len(cross_corr) >= 10:
            #     ks, p = scipy.stats.ks_2samp(cross_corr[cross_corr > 0], -cross_corr[cross_corr < 0])

            selector = np.isin(labels, [label_i, label_j])
            data = wf.wf[selector]
            # isi_violations = np.mean(wf.ind[selector] < 30)

            if np.sum(labels == label_i) < 20:
                std_i = np.std(wf.wf, axis=0)
            else:
                std_i = np.std(wf.wf[labels == label_i], axis=0)

            if np.sum(labels == label_j) < 20:
                std_j = np.std(wf.wf, axis=0)
            else:
                std_j = np.std(wf.wf[labels == label_j], axis=0)

            std = np.min([std_i, std_j], axis=0)

            errs = np.mean(np.abs(
                np.mean(wf.wf[labels == label_i], axis=0) -
                np.mean(wf.wf[labels == label_j], axis=0)
            ) > std)
            similarity[i][j] = similarity[j][i] = 1 - errs

            '''
            print("errs", label_i, label_j, errs)
            y = labels[selector]
            classified = LDA(n_components=3).fit(data, y).predict(data)
            # print(classified, y, np.mean(classified != y))
            similarity[i][j] = similarity[j][i] = 1 - 2 * np.abs(np.mean(classified != y))
            '''

    return similarity, snrs

def combined_disposed(disposed):
    if not len(disposed):
        return Waveforms(
                np.array([]),
                np.array([]),
                np.array([])
        )
    return Waveforms(
            np.concatenate([d.ind for d in disposed]),
            np.vstack([d.wf for d in disposed]),
            np.concatenate([d.index for d in disposed])
    )

def sort(spike_ind, spike_wf, polarity=-1):
    disposed = []

    wf = Waveforms(spike_ind, spike_wf, np.arange(len(spike_ind)))

    up_or_down = polarity * wf.peaks > 0
    wf = wf.select(up_or_down)
    if len(wf) < PCS:
        return wf, np.zeros(len(wf.wf)), combined_disposed(disposed)

    pcaed = wf.pcaed(PCS)

    # (1) do coarse clustering
    labels = coarse_cluster(pcaed, k=5, m=4)

    # (2) DETECT ISI VIOLATIONS and recluster
    keep_points = remove_isi_violations(wf, labels)
    if (
                len(np.unique(keep_points)) == 2 and
                np.sum(keep_points) > 10 and
                np.sum(np.logical_not(keep_points)) > 10
            ):
        lda = LDA(n_components=2).fit(wf.wf, keep_points)
        keep_points = lda.predict(wf.wf)
    
    # print("(2) losing", np.sum(keep_points == False))
    disposed.append(wf.select(np.logical_not(keep_points)))
    wf = wf.select(keep_points)
    labels = labels[keep_points]

    if len(wf) < PCS:
        return wf, labels, combined_disposed(disposed)
    pcaed = wf.pcaed(PCS)

    # labels = labels[keep_points]
    labels = coarse_cluster(pcaed, k=4, m=4)

    # print("(2) losing", np.sum(keep_points == False))

    # (2.5) REMOVE SMALL CLUSTERS
    '''
    normal_clusters = [label for label in labels if np.sum(labels == label) > 2]
    normal_labels = np.isin(labels, normal_clusters)
    
    # if len(np.unique(normal_labels)) == 2:
    #     stds = np.std(wf.wf, axis=1)[:, None]
    #     lda = LDA(n_components=1).fit(stds, normal_labels)
    #     normal_labels = lda.predict(stds)
    
    print("(2.5) losing", np.sum(normal_labels == False))
    disposed.append(wf.select(np.logical_not(normal_labels)))
    wf = wf.select(normal_labels)

    if len(wf) < PCS:
        return wf, labels
    pcaed = wf.pcaed(PCS)
    labels = coarse_cluster(pcaed, k=4, m=4)

    '''

    # (3) ELIMINATE LOW SNR clusters
    isi_violations = np.diff(wf.ind) < 30
    low_snr_labels_set = np.unique([
        label for label in np.unique(labels)
        if (
            (snr(wf.wf, labels, label) < 3 and isi_frac(isi_violations, labels, label) > 0.5) or
            # (snr(wf.wf, labels, label) < 4 and isi_frac(isi_violations, labels, label) > 0.5) or
            (np.sum(labels == label) <= 6 and isi_frac(isi_violations, labels, label) > 0.5)
        )
    ])

    keep_points = np.logical_not(np.isin(labels, low_snr_labels_set))
    # print("(3) losing", np.sum(keep_points == False))
    disposed.append(wf.select(np.logical_not(keep_points)))
    wf = wf.select(keep_points)
    labels = labels[keep_points]

    if len(wf) < PCS:
        return wf, labels, combined_disposed(disposed)
    pcaed = wf.pcaed(PCS)
    labels = coarse_cluster(pcaed, k=4, m=3)

    # (4) MERGE similar clusters by generating similarity matrix
    unique_labels = get_unique_labels(labels)
    similarity, snr_matrix = generate_similarity_matrix(wf, labels, unique_labels)
    merge_okay = (
        ((similarity > 0.6) & (snr_matrix < 2)) |
        ((similarity > 0.7) & (snr_matrix < 3)) |
        ((similarity > 0.8) & (snr_matrix < 4)) |
        # ((similarity > 0.9) & (snr_matrix < 6)) |
        (similarity > 0.9)
    )

    already_merged = {}
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            label_i = already_merged.get(label_i, label_i)
            label_j = already_merged.get(label_j, label_j)
            if label_i < label_j and merge_okay[i][j]:
                labels[labels == label_j] = label_i
                already_merged[label_i] = label_i
                already_merged[label_j] = label_i
            elif label_i < label_j:
                selector = np.isin(labels, [label_i, label_j])
                data = pcaed[selector]
                if np.sum(labels == label_i) > 3 and np.sum(labels == label_j) > 3:
                    new_labels = LDA(n_components=3).fit(data, labels[selector]).predict(data)
                    labels[selector] = new_labels

    # (5) ELIMINATE LOW SNR clusters A SECOND TIME, including BIG PEAKS
    isi_violations = np.diff(wf.ind) < 30
    low_snr_labels_set = np.unique([
        label for label in np.unique(labels)
        if (
            (snr(wf.wf, labels, label) < 3 and isi_frac(isi_violations, labels, label) > 0.3)
            # (snr(wf.wf, labels, label) < 4 and isi_frac(isi_violations, labels, label) > 0.4) or
            # (np.sum(labels == label) < 5 and isi_frac(isi_violations, labels, label) > 0.2)
        )
    ])

    peaks = np.abs(wf.wf[:, wf.wf.shape[1] // 2])
    pre_peak = np.abs(np.min(polarity * wf.wf[:, :wf.wf.shape[1] // 2 - 3], axis=1))
    post_peak = np.abs(np.min(polarity * wf.wf[:, wf.wf.shape[1] // 2 + 4:], axis=1))

    datapoints_with_huge_peaks = np.zeros_like(labels).astype(np.bool)
    # Get points with large peak in the middle and large peaks on either side
    for label in np.unique(labels):
        cluster_mean_peak = np.mean(peaks[labels == label])  # Mean peak of cluster
        normal_max_value = np.abs(np.min(polarity * np.mean(wf.wf[labels == label], axis=0)))
        std = np.std(peaks[labels == label])  # Std of wf center within cluster
        datapoints_with_huge_peaks[labels == label] = (
                ((peaks[labels == label] - cluster_mean_peak) > (2 * std)) &
                (pre_peak[labels == label] - normal_max_value > (2 * std)) &
                (post_peak[labels == label] - normal_max_value > (2 * std))
        )

    keep_points = np.logical_not(np.isin(labels, low_snr_labels_set) | datapoints_with_huge_peaks)
    # print("(5) losing", np.sum(keep_points == False))
    disposed.append(wf.select(np.logical_not(keep_points)))
    wf = wf.select(keep_points)

    labels = labels[keep_points]

    disposed = combined_disposed(disposed)

    return wf, labels, disposed


def find_excessive_peaks(wfs, labels):
    """
    Assuming that clusters are nice so far, identify datapoints with waveforms that
    exceed 4 std deviations on either side of the center point
    """
    peaks = np.abs(wfs[:, wfs.shape[1] // 2])
    
    excessive_peaks = np.zeros_like(labels).astype(np.bool)
    for label in np.unique(labels):
        mean = np.mean(wfs[labels == label], axis=0)
        std = np.std(wfs[labels == label], axis=0)
        exceeds_4_std = np.abs(wfs[labels == label] - mean) > 3 * std
        
        exceeds_prepeak = np.any(exceeds_4_std[:, :exceeds_4_std.shape[1] // 2], axis=1)
        exceeds_postpeak = np.any(exceeds_4_std[:, exceeds_4_std.shape[1] // 2 + 1:], axis=1)
        exceeds_atpeak = exceeds_4_std[:, exceeds_4_std.shape[1] // 2]
        
        excessive_peaks[labels == label] = (
            (exceeds_prepeak & exceeds_postpeak & exceeds_atpeak) |
            (np.mean(exceeds_4_std, axis=1) > 0.5)
        )
        
    return excessive_peaks


def split_up_by_time(times, labels, sizes=None):
    new_labels = np.zeros_like(labels)
    times = times[:, None]
    
    if sizes is None:
        sizes = np.ones(len(labels))
    
    for label in np.unique(labels):
        if np.sum(labels == label) > 4:
            new_labels[labels == label] = (
                KMeans(n_clusters=4).fit_predict(
                    times[labels == label],
                    sample_weight=sizes[labels == label]
                )
                + 1 
                + np.max(new_labels)
            )
        else:
            new_labels[labels == label] = np.arange(np.sum(labels == label))
    return preprocessing.LabelEncoder().fit_transform(new_labels)


def _sort_set(dataset, polarity=-1, dpoints=1000, sr=30000.0):
    outputs = []
    disposed_all = []

    start = time.time()

    for i, (pt, _, window) in enumerate(dataset.windows(dpoints=dpoints)):
        ind = (window.times * sr).astype(np.int)
        wf = window.waveforms
        output, labels, disposed = sort(ind, wf, polarity=polarity)
        outputs.append((output, labels))
        print("Running {}/{} in {:.2f} min".format(
            pt, len(dataset), (time.time() - start) / 60), end="\r")

        disposed_all.append(disposed)

    links = []
    overall_labels = np.copy(outputs[0][1])
    overall_wfs = np.copy(outputs[0][0].wf)
    overall_ind = np.copy(outputs[0][0].ind)
    overall_index = np.copy(outputs[0][0].index)

    for i in range(len(outputs) - 1):
        links.append([])
        wf_1, labels_1 = outputs[i]
        wf_2, labels_2 = outputs[i + 1]

        print(overall_labels.shape, labels_2.shape)
        print(overall_index.shape, wf_2.index.shape)
        overall_labels = np.concatenate([overall_labels, labels_2 + 1 + np.max(overall_labels)])
        overall_wfs = np.vstack([overall_wfs, wf_2.wf])
        overall_ind = np.concatenate([overall_ind, wf_2.ind])
        overall_index = np.concatenate([overall_index, wf_2.index + ((i + 1) * dpoints)])

    return overall_wfs, overall_ind, overall_labels, overall_index


def hdb_fit_near(data, n_clusters=10, try_range=(2,)):
    good_labels_list = []
    for min_cluster_size in try_range:
        good_labels_list.append(hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size)
        ).fit_predict(data))

    cluster_count = np.array([len(np.unique(good_labels)) for good_labels in good_labels_list])
    good_labels = good_labels_list[np.argmin(np.abs(cluster_count - n_clusters))]
    print(cluster_count, good_labels_list)

    if np.sum(data[good_labels != -1]) < 6:
        return KNeighborsClassifier(n_neighbors=5).fit(
                data,
                good_labels
        ).predict(data)
    else:
        return KNeighborsClassifier(n_neighbors=5).fit(
                data[good_labels != -1],
                good_labels[good_labels != -1]
        ).predict(data)


def sort_set(spike_dataset):
    for polarity in [-1, 1]:
        wf, ind, labels, index = _sort_set(
                spike_dataset,
                polarity=polarity,
                dpoints=2000,
                sr=spike_dataset.sample_rate)
        dataset = spike_dataset.select(index)
        clustered = dataset.cluster(labels)

        wf, ind, labels, index = _sort_set(
                clustered,
                polarity=polarity,
                dpoints=500,
                sr=spike_dataset.sample_rate)

        projected = umap.UMAP(n_components=6, repulsion_strength=2.0).fit_transform(wf)
        projected = umap.UMAP(n_components=2, repulsion_strength=2.0).fit_transform(projected)
        labels = hdb_fit_near(projected, n_clusters=16, try_range=np.arange(10, 40, 1))

        reclustered = clustered.select(index).cluster(labels)

        flat = reclustered.flatten()
        outer_labels = reclustered.flatten().labels
        inner_labels = reclustered.flatten(1).flatten().labels

        keep_points = np.logical_not(find_excessive_peaks(flat.waveforms, outer_labels))
        fixed = flat.select(keep_points).cluster(inner_labels[keep_points])
  
        # TODO if it doesnt work here just end it
        projected = umap.UMAP(n_components=2, repulsion_strength=2.0).fit_transform(fixed.waveforms)
        labels = hdb_fit_near(projected, n_clusters=20, try_range=np.arange(10, 40, 1))
        good_labels = isosplit5(projected.T)
        good_labels = KNeighborsClassifier(n_neighbors=5).fit(projected, good_labels).predict(projected)

        near_final_result = fixed.cluster(good_labels)
        flat = near_final_result.flatten(1)

        new_labels = split_up_by_time(
             flat.times,
             flat.labels, 
             sizes=np.array([len(n) for n in flat.nodes]))
        final_result = flat.cluster(new_labels)

        stds = np.array([
            np.mean(np.std(cluster.flatten().waveforms, axis=1))
            for cluster in final_result.nodes
        ])

        final_result = final_result.select(np.where(stds < (4 * np.median(stds))), child=False)

        return final_result # cuz i dont want to deal with upgoing also yet


def mymainmansort_set(spike_dataset):
    labels = -np.ones(len(spike_dataset.times))
    for polarity in [-1, 1]:
        peaks = spike_dataset.waveforms[:, 20]
        dataset = spike_dataset.select(peaks > 0)
        selector = LocalOutlierFactor().fit_predict(PCA(n_components=6).fit_transform(dataset.waveforms)) == 1
        dataset = dataset.select(selector)

        umap_model = umap.UMAP(n_components=6, min_dist=0.1).fit(dataset.waveforms)

        step = 2000
        waveform_windows = [dataset.waveforms[i:i+step] for i in range(0, len(dataset.times), step)]

        all_labels = []
        for window_idx, window_wfs in enumerate(waveform_windows):
            all_labels.append(cluster_window(window_wfs, umap_model.transform(window_wfs)))

        all_labels, new_labels_flat, G = connect_windows(waveforms_windows, all_labels)
        all_labels, new_labels_flat, G = connect_windows(waveforms_windows, all_labels, reassign=False)

        # Now cluster all the points that weren't included 

        unclustered = new_labels_flat == -1
        waveforms2 = waveforms[unclustered]
        windows2 = [waveforms2[i:i+step] for i in range(0, len(waveforms2), step)]

        all_labels2 = []
        for window_idx, window_wfs in enumerate(windows2):
            all_labels2.append(cluster_window(windows2, umap_model.transform(window_wfs)))

        all_labels2, new_labels_flat2, G2 = connect_windows(windows2, all_labels2)
        all_labels2, new_labels_flat2, G2 = connect_windows(windows2, all_labels2, reassign=False)

        all_new_labels_flat = np.copy(new_labels_flat)
        selector = np.where(unclustered)
        for label in np.unqiue(new_labels_flat2):
            if label == -1:
                continue
            selector_ = np.zeros_like(new_labels_flat)
            for k in selector[new_labels_flat2 == label]:
                selector_[k] = 1

            np.place(all_new_labels_flat, selector_, np.max(all_new_labels_flat) + 1)





        wf, ind, labels, index = _sort_set(
                spike_dataset,
                polarity=polarity,
                dpoints=2000,
                sr=spike_dataset.sample_rate)
        dataset = spike_dataset.select(index)
        clustered = dataset.cluster(labels)

        wf, ind, labels, index = _sort_set(
                clustered,
                polarity=polarity,
                dpoints=500,
                sr=spike_dataset.sample_rate)

        projected = umap.UMAP(n_components=6, repulsion_strength=2.0).fit_transform(wf)
        projected = umap.UMAP(n_components=2, repulsion_strength=2.0).fit_transform(projected)
        labels = hdb_fit_near(projected, n_clusters=16, try_range=np.arange(10, 40, 1))

