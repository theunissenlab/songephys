"""
Perform vector (auto)regression to remove correlations within and across electrodes

>>> filter_segment(segment)

"""
from __future__ import print_function
import time

import numpy as np
import quantities as pq
import theano
import theano.tensor as T
from neo.core import AnalogSignal, Epoch
from sklearn.cluster import KMeans
import quantities as pq

from preproephys import get_mask, extract_spikes_from_signal
from soundsig.signal import lowpass_filter, highpass_filter

import config
from .timeseries import filter_data_by_blocks, target_slice
from util import float_x, convert_mins, estimate_time_remaining


class Filterer(object):
    def __init__(self, block_width):
        X = T.fmatrix("input_batch")
        n_lags = T.iscalar("n_lags")
        original_signal, predicted_signal = filter_data_by_blocks(X, n_lags, block_width)

        self._predict = theano.function([X, n_lags], [original_signal, predicted_signal])

    def filter(self, signal, n_lags):
        new_signal = np.zeros_like(signal)
        predicted_signal = np.zeros_like(signal)

        original, predicted = self._predict(float_x(signal), n_lags)

        new_signal[target_slice(n_lags)] = original - predicted
        predicted_signal[target_slice(n_lags)] = predicted
        return new_signal, predicted_signal


filterer = Filterer(block_width=800)


def filter_segment(seg):
    block = seg.block 

    # Get the signal that has the microphone. We use this to clean up the spikes. Maybe.
    # sound_signal = next(
    #     sig for sig in seg.analogsignals
    #     if sig.name.startswith("Board ADC")
    # )

    elec_signal = next(
        sig for sig in seg.analogsignals
        if sig.name.startswith("Amplifier")
    )
    # Load the elec signal lazily for the segment using the time bounds

    # Generate high pass
    # this line is erroring for LblYel
    sig_highpass = highpass_filter(
            elec_signal.T,
            elec_signal.sampling_rate,
            config.HIGHPASS_CUTOFF * pq.Hz).T

    signal_highpass_clean, predicted_signal = filterer.filter(
            sig_highpass,
            n_lags=9)
    signal_highpass_clean = signal_highpass_clean.astype(np.float32)
    predicted_signal = predicted_signal.astype(np.float32)

    new_analogsignal = AnalogSignal(
           signal_highpass_clean,
           name="{}_HPC".format(elec_signal.name),
           units=elec_signal.units,
           t_start=elec_signal.t_start,
           sampling_rate=elec_signal.sampling_rate,
           file_origin=elec_signal.file_origin,
           copy=False
    )

    predicted_signal = AnalogSignal(
           predicted_signal,
           name="{}_pred".format(elec_signal.name),
           units=elec_signal.units,
           t_start=elec_signal.t_start,
           sampling_rate=elec_signal.sampling_rate,
           file_origin=elec_signal.file_origin,
           copy=False
    )

    new_analogsignal.segment = seg
    predicted_signal.segment = seg

    seg.analogsignals.append(new_analogsignal)

    # Clean up the old signal and add the new ones
    for channel_index in block.channel_indexes:
        if channel_index.name == "Amplifier":
            channel_index.analogsignals.remove(elec_signal)
            channel_index.analogsignals.append(new_analogsignal)
            channel_index.create_relationship(force=True)

    seg.analogsignals.remove(elec_signal)
    seg.create_relationship(force=True)
    return new_analogsignal, predicted_signal


def compute_correlation_levels(signal, fs, t_window=1000):
    # t_window is a window width in seconds
    window = int(t_window * fs)
    signal = lowpass_filter(signal.T, fs, 3000.0).T
    signal = highpass_filter(signal.T, fs, 1000.0).T
    signal = signal - np.mean(signal, axis=0)
    var = np.var(signal, axis=0)
    var = np.outer(var, var)

    results = []
    t_arr = []
    for t_start in np.arange(0, len(signal), window):
        window_slice = slice(t_start, t_start + window)
        sig = signal[window_slice]
        corr = (
            np.sum(sig[:, :, None] * sig[:, None, :], axis=0) ** 2 /
            (var * ((window - 1) ** 2))
        )
        np.fill_diagonal(corr, 0)
        results.append(corr)
        t_arr.append(t_start / fs)

    return np.array(t_arr), np.array(results)


def create_epochs(t_arr, corr, window, threshold):
    epochs_by_unit = []
    for unit in range(corr.shape[1]):
        epochs_by_unit.append([])

        curr_window = None
        for t, val in zip(t_arr, np.mean(corr, axis=1)[:, unit]):
            # print(val, threshold)
            if val < threshold and curr_window is not None:
                epochs_by_unit[-1].append(curr_window)
                curr_window = None
            elif val < threshold and curr_window is None:
                pass
            elif val >= threshold and curr_window is not None:
                curr_window[1] += window
            else:
                curr_window = [t, t + window]

    return epochs_by_unit


def threshold_crossing_windows(t_arr, signal, threshold, width):
    """Create list of time windows during which threshold is exceeded

    Params
    t_arr (np.ndarray): time in seconds
    signal (np.ndarray)
    threshold (float)
    width (fl

    """
    all_windows = []
    curr_window = None
    dt = t_arr[1] - t_arr[0]
    for t, s in zip(t_arr, signal):
        next_start = np.max([t - width, t])
        use_previous = (
                curr_window is None and             # We are not in a window
                len(all_windows) > 0 and            # We have a previous window
                all_windows[-1][-1] >= next_start   # We are still within the previous windows time range
        )

        if use_previous:
            curr_window = all_windows.pop()

        if s >= threshold and curr_window is None:
            curr_window = (next_start, t + dt + width)
        elif s >= threshold and curr_window is not None:
            curr_window = (curr_window[0], t + dt + width)
        elif s < threshold and curr_window is not None:
            all_windows.append(curr_window)
            curr_window = None

    if curr_window is not None:
        all_windows.append(curr_window)

    return all_windows


def get_noise_floor(signal):
    sig = np.abs(signal)
    kmeans = KMeans(n_clusters=2, n_init=3, max_iter=100)

    # where the derivative changes sign to identify extrema
    maximas = np.sign(np.diff(signal)[:-1]) != np.sign(np.diff(signal)[1:])

    # the magnitude of the derivative
    magnitude = np.abs((np.diff(signal)[:-1] - np.diff(signal)[1:]))[:, None]
    peaks = sig[1:-1][maximas]
    magnitude = magnitude[maximas]
    
    if peaks.ndim == 1:
        peaks = peaks[:, None]
        
    # cluster extrema on their magnitude and the magnitude of the first derivative
    labels = None
    tries = 0
    # FIXME (kevin): I'm not sure the best way to deal with the fact
    # that this sometimes errors but since an error
    # wastes up to 10 hours of processing just retry it over and over
    # until it works...
    while tries < 10:
        try:
            labels = kmeans.fit_predict(np.hstack([np.abs(peaks), magnitude]))
        except:
            tries += 1
        else:
            break

    # choose the baseline noise cluster as the one with the smaller max amplitude
    base_cluster_idx = np.argmin([np.max(peaks[labels == label]) for label in np.unique(labels)])
    base_cluster = peaks[labels == base_cluster_idx]

    return np.median(base_cluster)


def detect_noisy_epochs(clean_signal, predicted_signal, threshold=3):
    t_arr = np.array(predicted_signal.times)
    sig = predicted_signal.as_array().copy()

    sig[sig < 0] = 0.0  # 1. Rectify signal
    noise_level = lowpass_filter(
        sig.T,
        predicted_signal.sampling_rate,
        50.0
    ).T

    epochs_by_channel = []
    stds = []
    for channel in range(16):
        std = get_noise_floor(clean_signal.as_array()[:, channel])
        stds.append(std)
        windows = threshold_crossing_windows(t_arr, noise_level[:, channel], threshold * std, 0.05)

        full_noise_mask = get_mask(clean_signal, windows)
        base_std = clean_signal.as_array()[np.invert(full_noise_mask), channel].std()
        base_max = clean_signal.as_array()[np.invert(full_noise_mask), channel].max()
        base_min = clean_signal.as_array()[np.invert(full_noise_mask), channel].min()

        remove = []
        for i, window in enumerate(windows):
            mask = get_mask(clean_signal, [window])
            if (clean_signal.as_array()[mask, channel].max() > base_max or
                    clean_signal.as_array()[mask, channel].min() < base_min):
                remove.append(i)

        windows = np.array(windows)[remove]

        if len(windows):
            epochs_by_channel.append(windows)

    return np.array(epochs_by_channel)


def convert_to_epoch_obj(noise_epochs):
    times = []
    durations = []
    labels = []
    for elec_idx, epochs in enumerate(noise_epochs):
        times += [t_start for t_start, _ in epochs]
        durations += [t_stop - t_start for t_start, t_stop in epochs]
        labels += [elec_idx for _ in epochs]

    return Epoch(
        times=np.array(times) * pq.s,
        durations=np.array(durations) * pq.s,
        labels=np.array(labels))


filterer = Filterer(block_width=config.FILTER_BLOCK)

config.SPIKE_THRESHOLD_STD = 4
config.REFILTER_EVENTS_RATE = 10


def spike_correlation(
	signals,
        channel_index,
        spike_index,
        spike_width):
    """Compute correlation in spike window over all channels

    Params:
        signals: (N x C) array, N samples, C channels
        channel_idx: integer index of channel with the spike on it
        spike_index: integer between 1 and N corresponding to
            index of spike sample
        spike_width: width (radius) of spike in samples
    
    Returns correlation between channel_index and other channels
    """
    snippet = signals[spike_index - spike_width: spike_index + spike_width]
    corr = np.corrcoef(snippet.T)
    np.fill_diagonal(corr, 0)
    corr = corr[:, channel_index:channel_index + 1]
    return corr


def consolidate_intervals(intervals):
    """Combine overlapping intervals into non-overlapping intervals

    Ex. [(0, 4), (2, 7), (9, 11)] -> [(0, 7), (9, 11)]
    """
    intervals = sorted(intervals)
    new_intervals = []
    last_start = None
    for idx, (i1, i2) in enumerate(intervals):
        if last_start is None:
            last_start = i1
        
        if idx + 1 >= len(intervals):
            new_intervals.append((last_start, i2))
            break
        elif intervals[idx + 1][0] >= i2:
            new_intervals.append((last_start, i2))
            last_start = None

    return new_intervals


def filter_and_detect_spikes(
        signal_loader,
        output_signal,
        sampling_rate,
        batch_size):
    """Jointly filter electrode signals and detect spikes
    
    Runs an algorithm that filters an electrode signal and
    collects spikes that cross a threshold value. Invokes a simple filtering
    method (highpass and mean subtraction) as well as complex filtering
    (time-lagged filter across all channels). The complex version is much
    slower and so is only invoked when spikes with high correlation values
    across channels are detected.
    """
    _spike_count = 0
    _start_time = time.time()

    all_spike_ind = []
    all_spike_wf = []
    all_noise_events = []
    spike_width = int(config.SPIKE_WIDTH * sampling_rate)
    
    for batch_idx, chunk in signal_loader.batches(batch_size):
        # (1) Highpass filter
        base_signal = highpass_filter(
            chunk.T,
            sampling_rate,
            config.HIGHPASS_CUTOFF
        ).T

        # (2) Subtract mean across channels
        #     this should work for most cases!
        highpass_signals = base_signal - np.mean(base_signal, axis=1)[:, None]

        # (3) First round of spike detections
        # Store times + channels of spikes with high correlations as noise,
        # and and non-noisy events as spike indices
        noise_intervals = []
        noise_events = np.array([])
        spike_ind = []
        for ch_idx, sig in enumerate(highpass_signals.swapaxes(0, 1)):
            _, spike_indexes, spike_wfs = extract_spikes_from_signal(
                sig, 
                threshold_std=0.75 * config.SPIKE_THRESHOLD_STD,
                spike_width=spike_width,
                upward_gain=1
            )
           
            # Count how many channels have a correlation over threshold
            # If there are enough over that threshold, call it a noise event
            is_noise_event = np.array([
                len(np.where(
                    np.abs(
                        spike_correlation(highpass_signals, ch_idx, ind, spike_width)
                    ) > 0.8
                )[0]) > 6
                for ind in spike_indexes
            ]).astype(np.bool)
            
            if len(spike_indexes):
                noise_events = np.concatenate([noise_events, spike_indexes[is_noise_event]])
            spike_ind.append(spike_indexes[~is_noise_event])

        if len(noise_events):
            # (4) Turn the noise events into blocks to filter
            noise_intervals = []
            buffer = 5 * spike_width
            for event in noise_events:
                i1 = int(max(0, event - buffer))
                i2 = int(event + buffer)
                noise_intervals.append((i1, i2))

            noise_intervals = consolidate_intervals(noise_intervals)
            
            # (5) Build up array of noise intervals to learn filter on
            noisy_snippets = None
            offsets = []
            for i1, i2 in noise_intervals:
                if noisy_snippets is None:
                    noisy_snippets = base_signal[i1:i2]
                else:
                    noisy_snippets = np.concatenate([noisy_snippets, base_signal[i1:i2]])
                offsets.append((
                    len(noisy_snippets) - len(base_signal[i1:i2]),
                    len(noisy_snippets)
                ))

            if len(noisy_snippets) > (config.FILTER_BLOCK + config.N_LAGS):
                cut_size = len(noisy_snippets) // 2
                noisy_snippets = np.concatenate([
                    noisy_snippets[-cut_size:],
                    noisy_snippets,
                    noisy_snippets[:cut_size]
                ])

                # (6) Fit the time-lagged filter to the noise periods
                # TODO (kevin): i don't know why, but the filterer errors when this condition matches
                if noisy_snippets.shape[0] % config.FILTER_BLOCK == config.N_LAGS:
                    noisy_snippets = np.vstack([noisy_snippets, noisy_snippets[:1]])

                clean_signal, _ = filterer.filter(noisy_snippets, config.N_LAGS)
                clean_signal = clean_signal[cut_size:-cut_size]

                for (i1, i2), (j1, j2) in zip(noise_intervals, offsets):
                    for idx, clean_sig in enumerate(clean_signal[j1:j2].T):
                        if np.std(clean_sig) > np.std(highpass_signals[i1:i2, idx]):
                            continue

                        baseline_std = np.std(
                            highpass_signals[max(0, i1 - 100): i2 + 100, idx]
                        )
                        if np.std(clean_sig) > baseline_std * 2:
                            highpass_signals[i1:i2, idx] = 0.0
                            continue

                        highpass_signals[i1:i2, idx] = clean_sig

        # (7) Re-identify spikes in the window with new parameters
        spike_ind = []
        spike_wf = []
        for ch_idx, sig in enumerate(highpass_signals.swapaxes(0, 1)):
            _, spike_indexes, spike_wfs = extract_spikes_from_signal(
                sig,
                limit_std=config.Z_SPIKE_LIMIT,
                threshold_std=config.Z_THRESHOLD,
                spike_width=spike_width,
                upward_gain=1
            )

            # (8) # Don't forget to correct the index to the batch offset!
            spike_ind.append(spike_indexes + batch_idx)
            if len(spike_wfs):
                spike_wf.append(spike_wfs.astype(np.float32))
            else:
                spike_wf.append(np.zeros((0, 2 * spike_width), dtype=np.float32))

        output_signal.write(highpass_signals, highpass_signals.shape[0], batch_idx)

        all_spike_ind.append(spike_ind)
        all_spike_wf.append(spike_wf)
        all_noise_events += [(i1 + batch_idx, i2 + batch_idx) for i1, i2 in noise_intervals]

        _spike_count += np.sum([len(x) for x in spike_ind])

        elapsed, remaining = estimate_time_remaining(_start_time, batch_idx / signal_loader.N)
        print(
            "Filtering {}/{} in {}... "
            "Est. remaining: {}. {} spikes so far.".format(
                batch_idx,
                signal_loader.N,
                convert_mins(elapsed),
                convert_mins(remaining),
                _spike_count
            ),
            end="\r"
        )

    all_spike_ind = np.array([np.concatenate(x) for x in list(zip(*all_spike_ind))])
    all_spike_wf = np.array([np.concatenate(x) for x in list(zip(*all_spike_wf))])
    
    return all_spike_ind, all_spike_wf, all_noise_events

