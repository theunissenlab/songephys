"""Find the periods of a recording during which a unit is active (held)."""
import numpy as np
import scipy.stats as stats


def compute_unit_valid_intervals(spike_times, tot_time, t_bin=3, window_s=30, step_s=15,
                                 merge_gap_s=180, min_duration_s=300):
    """Valid (active) intervals for one unit.

    Spikes are binned in t_bin second bins; a sliding window of window_s seconds
    (step step_s) is called active when the counts are significantly nonzero
    (one sample t-test, p < 0.2) and its log rate is not far below the unit's
    typical active rate. Active periods closer than merge_gap_s are merged and
    periods shorter than min_duration_s are dropped.

    Params
    ======
    spike_times (array): spike times in seconds
    tot_time (float): total recording duration in seconds

    Returns
    =======
    list of (start, stop) tuples in seconds
    """
    tot_time = int(tot_time)
    bins = np.arange(0, tot_time, t_bin)
    spike_counts, _ = np.histogram(spike_times, bins=bins)

    # take a sliding window of 30 seconds and determine if the rate is significantly nonzero using a t test
    window = window_s // t_bin
    step = step_s // t_bin
    active = []
    active_rates = []
    active_bins = [0]
    for i in range(0, len(spike_counts) - window, step):
        t, p = stats.ttest_1samp(spike_counts[i:i+window], 0)
        active_rates.append(np.mean(spike_counts[i:i+window]))
        active_bins.append(i*t_bin)
        active.append(p < 0.2)
    if len(active) == 0:
        return []
    active = np.array(active, dtype=int)
    with np.errstate(divide='ignore'):
        active_rates = np.log10(np.array(active_rates))
    if not np.any(active == 1):
        return []
    active_mean = np.mean(active_rates[active == 1])
    active_std = np.std(active_rates[active == 1])
    # reject active bins that are well below the typical active rate
    active = active * (active_rates > active_mean - 2.5*active_std)

    # calculate contiguous active periods
    if np.all(active == 1):
        active_periods = [(active_bins[0], active_bins[len(active)])]
    else:
        active_periods = []
        start = 0
        for i in range(1, len(active)):
            if active[i] == 1 and active[i-1] == 0:
                start = i
            if active[i] == 0 and active[i-1] == 1:
                active_periods.append((active_bins[start], active_bins[i+1]))
        if active[-1] == 1:
            active_periods.append((active_bins[start], active_bins[-1]))
    if len(active_periods) == 0:
        return []
    # merge active periods separated by short inactive periods
    agglomerated_active_periods = [active_periods[0]]
    for i in range(1, len(active_periods)):
        if active_periods[i][0] - agglomerated_active_periods[-1][1] < merge_gap_s:
            agglomerated_active_periods[-1] = (agglomerated_active_periods[-1][0], active_periods[i][1])
        else:
            agglomerated_active_periods.append(active_periods[i])
    # now remove any active periods that are too short
    return [x for x in agglomerated_active_periods if x[1] - x[0] > min_duration_s]


def recording_duration(nwb):
    """Total duration (s) of an NWB file, from LFP if present, else audio, else spikes."""
    for name in ('LFP', 'audio'):
        if name in nwb.acquisition:
            ts = nwb.acquisition[name]
            return ts.data.shape[0] / ts.rate
    return float(np.max(nwb.units['spike_times'].data[:]))
