"""
Perform vector (auto)regression to remove correlations within and across electrodes

>>> filter_segment(segment)

"""
from __future__ import print_function

import numpy as np
import quantities as pq
import theano
import theano.tensor as T
from neo.core import AnalogSignal, Epoch
from sklearn.cluster import KMeans
import quantities as pq


from soundsig.signal import lowpass_filter, highpass_filter

import config
from .timeseries import filter_data_by_blocks, target_slice
from util import float_x


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

    # Generate high pass
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
    all_windows = []
    curr_window = None
    dt = t_arr[1] - t_arr[0]
    for t, s in zip(t_arr, signal):
        next_start = t - width
        use_previous = (
                curr_window is None and
                len(all_windows) > 0 and
                all_windows[-1][-1] >= next_start
        )

        if use_previous:
            curr_window = all_windows.pop()

        if s >= threshold and curr_window is None:
            curr_window = (t - width, t + dt + width)
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
    labels = kmeans.fit_predict(np.hstack([np.abs(peaks), magnitude]))

    # choose the baseline noise cluster as the one with the smaller max amplitude
    base_cluster_idx = np.argmin([np.max(peaks[labels == label]) for label in np.unique(labels)])
    base_cluster = peaks[labels == base_cluster_idx]

    return np.median(base_cluster)


def get_mask(signal, ranges):
    total = np.zeros(len(signal.times)).astype(np.bool)
    for t_start, t_stop in ranges:
        total = total | ((signal.times > t_start) & (signal.times < t_stop))
    return total


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
