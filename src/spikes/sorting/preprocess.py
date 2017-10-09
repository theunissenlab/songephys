import numpy as np

import config


def collect_waveforms(spiketrains, threshold=0, force_single_peak=False):
    """Combines several neo spiketrains into a single data structure

    Params
    ======
    spiketrains (list of neo.core.spiketrain.SpikeTrain)
        List of neo spiketrain objects (i.e. for each segment)
    threshold (float)
        Exclude waveforms that with midpoints greater than this threshold.
        This is just a convenient way to add electrode-specific thresholds
        after the initial thresholding has been done already.
    force_single_peak (bool, default=False)
        Excludes waveforms whose minimum point is not the midpoint.

    Returns
    =======
    waveforms (np.array; (N_spikes, width))
    times (np.array; (N_spikes,))
        The length of these arrays should equal the total number of time steps
        covered by the sum of all segments in spiketrains
    """
    waveforms = []
    times = []

    for spiketrain in spiketrains:
        if len(spiketrain.waveforms):
            wf_peak_idx = (spiketrain.waveforms.shape[1] // 2) - config.WF_START
            if force_single_peak:
                # array that selects waveforms whose peak value is at the array center
                mins = np.argmin(spiketrain.waveforms[:, config.WF_SLICE, 0], axis=1) == wf_peak_idx
            else:
                mins = np.ones(len(spiketrain.waveforms)).astype(bool)

            # this lets us apply a secondary threshold
            # FIXME: it seems like there are some cases where spiketrain.waveforms is 2D?
            midpoints = spiketrain.waveforms[:, config.WF_SLICE, 0][:, wf_peak_idx]

            waveforms += list(spiketrain.waveforms[(midpoints < threshold) & (mins)])
            times += list(spiketrain.times[(midpoints < threshold) & (mins)])

    return np.array(waveforms)[:, :, 0] if len(waveforms) else np.array(waveforms), np.array(times)


def sort_spikes_on_unit(unsorted_unit, threshold, model, unit_clusters):
    """Bin spikes by their clusters

    Params
    unsorted_unit (neo.core.Unit)
        Unit that has references to a spiketrain for each segment. The spiketrain
        contains all waveforms for potential spikes found on one electrode
    threshold (float)
        Exclude waveforms with peaks that do not cross this threshold
    clusterer (model)
        KMeans model
    unit_clusters
        list of lists, each inner list is a list of cluster ids to group as one unit

    Returns
    spike_times (list)
    """
    overall_spike_times = []
    overall_spike_waveforms = []

    for i, st in enumerate(unsorted_unit.spiketrains):
        spike_times = []
        spike_waveforms = []
        waveforms, times = collect_waveforms([st], threshold=threshold, force_single_peak=True)
        if not len(waveforms):
            for _ in unit_clusters:
                spike_times.append([])
                spike_waveforms.append([])
        else:
            waveforms = waveforms[:, config.WF_SLICE]
            labels = model.predict(waveforms)
            for unit_group in unit_clusters:
                mapped_labels = np.array([True if l in unit_group else False for l in labels])
                spike_times.append(times[mapped_labels])
                spike_waveforms.append(waveforms[mapped_labels])

        overall_spike_times.append(spike_times)
        overall_spike_waveforms.append(spike_waveforms)

    return overall_spike_times, overall_spike_waveforms
