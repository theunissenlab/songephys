"""
Filtering for data
"""
import logging
import time

import numpy as np
from soundsig.signal import highpass_filter, lowpass_filter

from preprocess.ephys.detect import (
    detect_spike_times,
    get_spike_snippets
)
from preprocess.ephys._timelagged_filter import TimelaggedFilterer
from util import convert_mins, estimate_time_remaining

import config


timelagged_filterer = TimelaggedFilterer(block_width=config.FILTER_BLOCK)
from statsmodels.tsa.api import VAR

DELAYVAR = 4 #config.N_LAGS


def var_filter(x):
    X = np.asarray(x)
    modelVAR = VAR(np.asarray(X))
    resultsVAR = modelVAR.fit(DELAYVAR)

    #  To prevent overfitting (in the sense of spike removal) we delete info from own channel
    paramsDEL = resultsVAR.params[:,:]

    nelec = paramsDEL.shape[1]   # This is the number of electrodes also equal to sig.shape[1]
    nlags = (paramsDEL.shape[0]-1)//nelec  # This is also equal to DELAYS
    for i in range(nelec):
        for j in range(nlags):
            paramsDEL[j*nelec+1+i, i] = 0;

    # Generate predictions but adjust their gain using OLS
    predX = modelVAR.predict(paramsDEL, lags = nlags)
    for i in range(nelec):
        meanX = np.mean(X[nlags:, i])   # Should be close to zero since from high-pass
        meanpredX = np.mean(predX[:,i]) # Should also be close to zero
        gain = np.sum((predX[:,i]-meanpredX)*(X[nlags:,i]-meanX))/np.sum((predX[:,i]-meanpredX)*(predX[:,i]-meanpredX))
        predX[:,i] = gain*predX[:,i]


    X[DELAYVAR:] = predX
    return X

def context_based_filtering(
        signal_loader,
        output_signal,
        sampling_rate,
        batch_size,
    ):
    """Filter ephys signals (in batches) and extract spikes

    Iterate over signal loader in batches, and write to an output_signal

    Args
    ----
    signal_loader : format.lazy.LazySignal
        Input signal to read unfiltered data from
    output_signal : format.lazy.LazySignal
        Output signal to write filtered data to
    sampling_rate : int or float
        Sampling rate of signal in Hz
    batch_size : int
        Size of windows to filter in samples
    
    Returns a tuple of length two:
        0: a LazySignal output signal
        1: a list of intervals which were labeled as noise due to correlated events.
    """
    _start_time = time.time()

    spike_radius = int(config.SPIKE_WIDTH * sampling_rate)

    all_noise_intervals = []
    for batch_idx, chunk in signal_loader.batches(batch_size):
        base_signal = basic_filter(chunk, sampling_rate)
        highpass_signals = common_average_reference(base_signal)

        noise_intervals = []

        '''
        events, snippets, corrs = find_correlated_events(
            highpass_signals,
            spike_radius
        )

        # Check for correlations that meet criteria
        # Criterion: 20% of electrode pairs have abs(correlation) >= 0.4
        is_noisy = np.array([
            np.mean(np.abs(corr) > 0.5) > 0.2 for corr in corrs
        ]) | np.array([np.mean(np.abs(corr) > 0.2) > 0.6 for corr in corrs])
        noise_intervals = create_noise_intervals(
            events[is_noisy],
            5 * spike_radius
        )
        noise_snippets = [
            highpass_signals[i1:i2] for i1, i2 in noise_intervals
        ]

        if not len(noise_snippets):
            clean_snippets = []
        else:
            # FIXME: The filtering is not working - basically zero-ing out

            # Do a time-lagged filter to remove cross-channel correlations
            clean_snippets = filter_snippets(
                noise_snippets,
                filter_fn=lambda x: timelagged_filterer.filter(x, config.N_LAGS)
                #filter_fn=var_filter
            )

        if clean_snippets is not None:
            # Compare to clean to see if we want to replace or zero out
            for old_snippet, new_snippet, (i1, i2) in zip(
                        noise_snippets,
                        clean_snippets,
                        noise_intervals
                    ):
                for channel_idx, clean_sig in enumerate(new_snippet.T):
                    # Check 1: Std of signal is not increased by new filtering
                    #if np.std(clean_sig) > np.std(old_snippet[:, channel_idx]):
                    #    logging.warning("erro1")
                        # Variance of new signal too high
                    #    continue

                    # Check 2: Std of signal is not more than twice the
                    # surrounding signal
                    baseline_std = np.std(highpass_signals[
                        max(0, i1 - 5 * spike_radius):i2 + 5 * spike_radius,
                        channel_idx
                    ])
                    # if np.std(clean_sig) > baseline_std * 2:
                    #    logging.warning("erro2")

                    #    highpass_signals[i1:i2, channel_idx] = 0.0
                    #    continue

                    highpass_signals[i1:i2, channel_idx] = clean_sig
        '''

        if output_signal is None:
            logging.info("Output signal is None. Not saving anywhere.")
            # yield highpass_signals, base_signal, noise_intervals         
        else:
            output_signal.write(
                highpass_signals,
                highpass_signals.shape[0],
                batch_idx)
        if len(noise_intervals):
            all_noise_intervals.append([
                (i + batch_idx, j + batch_idx)
                for i, j in noise_intervals
            ])

        elapsed, remaining = estimate_time_remaining(
            _start_time, batch_idx / signal_loader.N)
        logging.warning(
            "Filtering {}/{} in {}... "
            "Est. remaining: {}".format(
                batch_idx,
                signal_loader.N,
                convert_mins(elapsed),
                convert_mins(remaining)
            )
        )

    if len(all_noise_intervals):
        all_noise_intervals = np.concatenate(all_noise_intervals)

    return output_signal, all_noise_intervals


'''
def context_based_filtering(
        signal_loader,
        output_signal,
        sampling_rate,
        batch_size,
    ):
    """Filter ephys signals (in batches) and extract spikes

    Iterate over signal loader in batches, and write to an output_signal

    Args
    ----
    signal_loader : format.lazy.LazySignal
        Input signal to read unfiltered data from
    output_signal : format.lazy.LazySignal
        Output signal to write filtered data to
    sampling_rate : int or float
        Sampling rate of signal in Hz
    batch_size : int
        Size of windows to filter in samples
    
    Returns a tuple of length two:
        0: a LazySignal output signal
        1: a list of intervals which were labeled as noise due to correlated events.
    """
    _start_time = time.time()

    spike_radius = int(config.SPIKE_WIDTH * sampling_rate)
    # all_noise_snippets = []
    all_noise_intervals = []
    for batch_idx, chunk in signal_loader.batches(batch_size):
        base_signal = basic_filter(chunk, sampling_rate)
        highpass_signals = common_average_reference(base_signal)
        events, snippets, corrs = find_correlated_events(
            highpass_signals,
            spike_radius
        )

        # Check for correlations that meet criteria
        # Criterion: 20% of electrode pairs have abs(correlation) >= 0.4
        is_noisy = np.array([
            np.mean(np.abs(corr) > 0.5) > 0.2 for corr in corrs
        ]) | np.array([np.mean(np.abs(corr) > 0.2) > 0.6 for corr in corrs])
        noise_intervals = create_noise_intervals(
            events[is_noisy],
            5 * spike_radius
        )
        noise_snippets = [
            highpass_signals[i1:i2] for i1, i2 in noise_intervals
        ]

        if len(noise_intervals):
            all_noise_intervals.append([
                (i + batch_idx, j + batch_idx)
                for i, j in noise_intervals
            ])
            # all_noise_snippets.append(
            #     noise_snippets
            # )

        elapsed, remaining = estimate_time_remaining(
            _start_time, batch_idx / signal_loader.N)
        logging.warning(
            "Filtering {}/{} in {}... "
            "Est. remaining: {}".format(
                batch_idx,
                signal_loader.N,
                convert_mins(elapsed),
                convert_mins(remaining)
            )
        )
        logging.warning("{:.2f}".format(np.sum([
            np.sum(np.diff(x)) for x in all_noise_intervals
        ]) / sampling_rate / 60.0))

    if len(all_noise_intervals):
        all_noise_intervals = np.concatenate(all_noise_intervals)
        # all_noise_snippets = np.concatenate(all_noise_snippets)


    if not len(noise_snippets):
        clean_snippets = []
    else:
        # Do a time-lagged filter to remove cross-channel correlations
        clean_snippets = filter_snippets(
            noise_snippets,
            filter_fn=lambda x: timelagged_filterer.filter(x, config.N_LAGS)
            #filter_fn=var_filter
        )

    if clean_snippets is not None:
        # Compare to clean to see if we want to replace or zero out
        for old_snippet, new_snippet, (i1, i2) in zip(
                    noise_snippets,
                    clean_snippets,
                    noise_intervals
                ):
            yield new_snippet, old_snippet, (i1, i2)

                # for channel_idx, clean_sig in enumerate(new_snippet.T):
                #     # Check 1: Std of signal is not increased by new filtering
                #     #if np.std(clean_sig) > np.std(old_snippet[:, channel_idx]):
                #     #    logging.warning("erro1")
                #         # Variance of new signal too high
                #     #    continue

                #     # Check 2: Std of signal is not more than twice the
                #     # surrounding signal
                #     baseline_std = np.std(highpass_signals[
                #         max(0, i1 - 5 * spike_radius):i2 + 5 * spike_radius,
                #         channel_idx
                #     ])
                #     # if np.std(clean_sig) > baseline_std * 2:
                #     #    logging.warning("erro2")

                #     #    highpass_signals[i1:i2, channel_idx] = 0.0
                #     #    continue

                #     highpass_signals[i1:i2, channel_idx] = clean_sig

    return output_signal, all_noise_intervals
'''


def basic_filter(unfiltered_signal, sampling_rate):
    """Simple highpass"""
    base_signal = highpass_filter(
        unfiltered_signal.T,
        sampling_rate,
        config.HIGHPASS_CUTOFF
    ).T

    return base_signal


def common_average_reference(signal):
    """Subtract the common mean across channels"""
    return signal - np.mean(signal, axis=1)[:, None]


def find_correlated_events(signal, event_radius):
    """Detect threshold crossings that are correlated across channels

    Return the indices of the events, the snippets of the signal during
    those events, and the correlation matrix across electrodes during
    those events.
    """
    all_spike_indices = set()
    for elec_idx, sig in enumerate(signal.swapaxes(0, 1)):
        spike_indices = detect_spike_times(
            sig,
            threshold_std=0.75 * config.SPIKE_THRESHOLD_STD,
            spike_radius=event_radius,
        )
        all_spike_indices = all_spike_indices.union(set(spike_indices))

    all_spike_indices = np.array(sorted(list(all_spike_indices)))

    snippets = get_spike_snippets(
        signal, 
        all_spike_indices,
        event_radius)

    correlation_matrices = []
    for idx, snippet in zip(all_spike_indices, snippets):
        corr = np.corrcoef(snippet.T)
        np.fill_diagonal(corr, 0)
        correlation_matrices.append(corr)

    return all_spike_indices, snippets, np.array(correlation_matrices)


def _consolidate_intervals(intervals):
    """Combine overlapping intervals into non-overlapping intervals

    Ex. [(0, 4), (2, 7), (9, 11)] -> [(0, 7), (9, 11)]

    Args
    ----
    intervals : list
        List of tuples, where each tuple is a start and stop index
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
        elif intervals[idx + 1][0] > i2:
            new_intervals.append((last_start, i2))
            last_start = None

    return new_intervals


def create_noise_intervals(event_indices, event_radius):
    """Convert events into non-overlapping windows of specified width

    Args
    ----
    event_indices : np.ndarray (int)
        Array of integer indices of noise events
    event_radius : int
        Specify width of noise event on either side of the peak

    Returns an (N x 2) array where N is the number of noise events
    and the second axis is the start index and stop index of the
    corresponding intervals
    """
    noise_intervals = []
    for idx in event_indices:
        noise_intervals.append((
            int(max(0, idx - event_radius)),
            int(idx + event_radius)
        ))

    noise_intervals = _consolidate_intervals(noise_intervals)
    return np.array(noise_intervals).astype(np.int)


def filter_snippets(noise_snippets, filter_fn):
    """Filter in specified intervals
    """
    interval_sizes = np.array([snippet.shape[0] for snippet in noise_snippets])
    interval_endpoints = np.cumsum([interval_sizes])
    interval_startpoints = interval_endpoints - interval_sizes

    # Double the snippet array to eliminate edges
    noise_snippets = np.concatenate(noise_snippets)
    cut_size = len(noise_snippets)
    noise_snippets = np.concatenate([
        noise_snippets[-cut_size:],
        noise_snippets,
        noise_snippets[:cut_size]
    ])

    if len(noise_snippets) < config.FILTER_BLOCK + config.N_LAGS:
        return

    if noise_snippets.shape[0] % config.FILTER_BLOCK == config.N_LAGS:
        noise_snippets = np.vstack([noise_snippets, noise_snippets[:1]])

    # FILTER HERE
    clean_signal = filter_fn(noise_snippets)
    clean_signal = clean_signal[cut_size:-cut_size]

    result = []
    for i1, i2 in zip(interval_startpoints, interval_endpoints):
        result.append(clean_signal[i1:i2])

    return result
