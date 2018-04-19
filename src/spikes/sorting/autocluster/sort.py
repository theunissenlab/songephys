"""
Autoclustering functions

1. Feature selection
2. Data filtering (taking out noisy data heuristically)
3. Cluster initialization
4. Iterative isosplit algorithm (see mountainsort)
"""
import numpy as np

from .merging import cluster_fns, merge, reassign_fns, Projector
from .isosplit import IsoSplit, Merger, Tracker, compute_means


# TODO: let you use tsne as features?
def sort_debugger(
        waveforms,
        feature_transform=None,
        data_filter=None,
        initial_cluster_type="gmm",
        initial_cluster_count=20,
        cluster_filter=None,
        ordered=True,
        merge_mode="linear",
        merge_initial_threshold=0.6,
        merge_final_threshold=0.1,
        reassign_mode="kmeans",
        reassign_dims=1,
        max_peak_size=None,
        min_peak_size=None,
        peak_idx=None,
        debug=False,
    ):
    """
    Sorting function
    """
    # apply data filter here... and keep track of what was filtered out
    # or do that beforehand?

    data = feature_transform(waveforms) if feature_transform else waveforms

    initial_cluster_fn = cluster_fns[initial_cluster_type]

    keep = np.ones(len(data))
    if max_peak_size:
        if not peak_idx:
            peak_idx = data.shape[1] / 2
        peaks = data[:, peak_idx]
        keep[np.abs(peaks) > max_peak_size] = 0

    if min_peak_size:
        if not peak_idx:
            peak_idx = data.shape[1] / 2
        peaks = data[:, peak_idx]
        keep[np.abs(peaks) < min_peak_size] = 0

    if len(data) >= initial_cluster_count:
        clusters = initial_cluster_fn(data, n=initial_cluster_count)
    else:
        clusters = -1 * np.ones(len(data))
    final_labels = np.ones_like(clusters) * -1  # The official labels

    # REMOVING "bad" initial clusters here. will put those data points back in with label -1 at end
    # apply cluster filter here... keep track of which points we arent sorting
    keep = np.ones_like(clusters)
    for clust, count in zip(*np.unique(clusters, return_counts=True)):
        if count < 10:
            keep[clusters == clust] = 0
    keep = keep.astype(bool)

    # waveforms = waveforms[keep]
    data = data[keep]
    clusters = clusters[keep]

    print("Clustering this much data:", data.shape)

    tracker = Tracker(np.unique(clusters))
    iso = IsoSplit(scale=4.0)
    merger = Merger(clusters, merge_initial_threshold, merge_final_threshold, mode=merge_mode)

    last_cluster_count = initial_cluster_count
    iters_on_same_cluster_count = 0

    # Clustering merging algorithm
    iters = 0
    while tracker.pairs():
        if iters % 100 == 0:
            print("{} cluster pairs left after {} iterations".format(len(tracker.pairs(), iters)))
        iters += 1

        if ordered:
            means = compute_means(data, clusters)
            i, j = tuple(tracker.pairs(sorter=means)[0])
        else:
            samp = np.random.choice(np.arange(len(tracker.pairs())))
            i, j = tracker.pairs()[samp]

        proj = Projector()
        proj.fit_transform(data, clusters, i, j)
        projected_data = proj.projected

        k, _optimal_cut, _bins, _unimodal_fit, _actual = iso.isocut(projected_data)

        if debug is True:
            final_labels[keep] = clusters
            yield dict(
                i=i,
                j=j,
                k=k,
                optimal_cut=_optimal_cut,
                bins=_bins,
                unimodal_fit=_unimodal_fit,
                actual=_actual,
                proj=proj,
                data_1=data[clusters == i],
                data_2=data[clusters == j],
                clusters=final_labels,
                waveforms=waveforms,
            )

        merger.update(len(np.unique(clusters)))
        if merger.is_unimodal(k):
            clusters = merge(clusters, i, j)
            tracker.touch(i)
            tracker.remove(j)
            tracker.check([i, j])
        elif not merger.is_unimodal(k):
            reassign_fn = reassign_fns[reassign_mode]
            if reassign_dims == 1:
                clusters = reassign_fn(clusters, proj.project(data), i, j)
            else:
                clusters = reassign_fn(clusters, data, i, j)

            tracker.touch(i)
            tracker.touch(j)
            tracker.check([i, j])
        else:
            raise Exception("Debug: Shouldnt be here")
        
        new_cluster_count = len(np.unique(clusters))

        if new_cluster_count == last_cluster_count:
            if iters_on_same_cluster_count > new_cluster_count ** 2:
                break
            else:
                iters_on_same_cluster_count += 1
        else:
            last_cluster_count = new_cluster_count
            iters_on_same_cluster_count = 0

    # TODO: if we filter anything out, we need to include it either as a cluster or include those
    # indices in the return
    final_labels[keep] = clusters
    yield final_labels, waveforms


def sort(*args, **kwargs):
    kwargs["debug"] = False
    for subresult in sort_debugger(*args, **kwargs):
        pass

    return subresult
