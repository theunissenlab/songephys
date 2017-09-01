import numpy as np
import scipy.sparse as sparse

from config import config

from spikes.utils import bin_spikes


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
            if force_single_peak:
                mins = np.argmin(spiketrain.waveforms.squeeze()[:, 8:40], axis=1) == 16
            else:
                mins = np.ones(len(spiketrain.waveforms)).astype(bool)
            midpoints = spiketrain.waveforms.squeeze()[:, 8:40][:, 16]
            waveforms += list(spiketrain.waveforms[(midpoints < threshold) & (mins)])
            times += list(spiketrain.times[(midpoints < threshold) & (mins)])

    return np.array(waveforms)[:, :, 0] if len(waveforms) else np.array([]), np.array(times)


def make_V(block, electrode_idx):
    """Concatenate voltages of electrode across all segments in block
    """
    return np.vstack([
        # TODO: should make this more robust than taking analogsignals[-1]
        segment.analogsignals[-1].as_array()[:, electrode_idx:electrode_idx+1]
        for segment in block.segments
    ])


def binned_spikes(unsorted_unit, threshold, clusterer):
    """Bin spikes by their clusters

    Params
    unsorted_unit (neo.core.Unit)
        Unit that has references to a spiketrain for each segment. The spiketrain
        contains all waveforms for potential spikes found on one electrode
    threshold (float)
        Exclude waveforms with peaks that do not cross this threshold
    clusterer (spikes.sorting.clustering.GaussianMixtureClustererPruned)
        Clustering object that has already been fitted

    Returns
    t_arrs (list)
        list of time arrays for each unit
    X (np.ndarray; (n_bins, n_units))
        Full binary spike data over all segments concatenated together
    """
    X_data = []
    t_arrs = []

    for i, st in enumerate(unsorted_unit.spiketrains):
        spike_times = []
        waveforms, times = collect_waveforms([st], threshold=threshold)
        if not len(waveforms):
            for label in clusterer.good_labels:
                spike_times.append([])
        else:
            waveforms = waveforms[:, 8:40]  # TODO hardcoded slice again...
            labels, p = clusterer.predict(waveforms)
            for label in clusterer.good_labels:
                spike_times.append(times[labels == label])

        V = st.segment.analogsignals[-1]
        t_arr, X = bin_spikes(spike_times,  # TODO include this in spikes lib?
                min_time=st.t_start.item(),
                dt=V.sampling_period.item(),
                n_bins=len(V))

        X_data.append(X)
        t_arrs.append(t_arr)
    X_data = np.vstack(X_data)

    return t_arrs, sparse.csr_matrix(X_data)
