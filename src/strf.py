"""
Functions for fitting STRF to specific segments of a site
"""
import numpy as np


from soundsig.sound import plot_spectrogram, spectrogram
from soundsig.signal import bandpass_filter

from sklearn.model_selection import train_test_split

import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
from spikes.psth import conv, exponential_convolver, gaussian_convolver

import config


def bin_spikes(spike_times, t_start, dt, n_bins, binarized=False, convolve_width=1):
    binned = np.zeros(n_bins)
    for spike in spike_times:
        bin_idx = int((spike - t_start) // dt)
        if 0 <= bin_idx < len(binned):
            binned[bin_idx] += 1
    
    if binarized:
        binned = (binned > 0).astype(np.int)

    convolver = exponential_convolver(convolve_width)
    return convolver(binned)


def prepare_data(
        events,
        spike_times,
        data_loader,
        spec_sample_rate,
        freq_spacing,
        channel_map=None,
        pad_before=0.1,
        sampling_rate=config.ELEC_SAMPLE_RATE,
        convolve_width=1
    ):
    spike_bins = []
    spec_bins = []
    using_events = []

    for event in events:
        t1 = event["start_time"].item() + pad_before
        t2 = (
            event["start_time"].item() + 
            event["duration"].item()
        )
        times = spike_times[(spike_times >= t1) & (spike_times <= t2)]
        if not len(times):
            continue

        channel = 0 if not channel_map else channel_map(event)
        mic_data = data_loader.time_slice(t1, t2, signal="mic").data[:, channel]
        mic_data = mic_data - np.mean(mic_data)
        mic_data = bandpass_filter(mic_data, 30000, 500, 8000)

        t, freq, timefreq, rms = spectrogram(
                mic_data,
                sampling_rate,
                spec_sample_rate, 
                freq_spacing,
                min_freq=500,
                max_freq=8000,
                cmplx=False)

        binned_spikes = bin_spikes(times, t1, dt=1/spec_sample_rate, n_bins=len(t), convolve_width=convolve_width)
        using_events.append(event)
        spike_bins.append(binned_spikes)
        spec_bins.append(timefreq)

    if not len(spike_bins):
        raise ValueError("No events")

    return freq, spike_bins, spec_bins, using_events


def concat(spec_bins, spike_bins):
    return np.hstack(spec_bins).swapaxes(0, 1), np.concatenate(spike_bins)


def collect_data_for_strf(
        spike_times,
        events,
        data_loader,
        event_filter=None,
        channel_map=None,
        time_ranges=None,
        spec_sample_rate=200,
        freq_spacing=100,
        epoch_size=500,
        strf_width=0.2,  # seconds
        pad_before=0.1,
        convolve_width=1,
        alphas=(0.1, 1.0, 10.0)
    ):
    """
    Fit a strf to spikes during certain events

    Parameters
    ----------
        spike_times: 1d array of floats for spike times
        events: list of dictionaries for vocal periods with metadata
        data_loader: instance of lazy.LazySite object to load audio
        event_filter (default=None): function that takes an event dict
            from events and returns True if it should be used to compute
            the strf. Always use event if event_filter is None.
        channel_map (default=None): function that takes an event dict
            from events and returns with mic channel to use. Always use
            channel with higher average amplitude if None
        time_ranges (default=None): list of tuples (start_time, stop_time)
            indicating what times the strf should be fit over
        spec_sample_rate (default=200): sample rate of strfs and
            spike bins
        freq_spacing (default=100): frequency spacing of spectrograms
        epoch_size (default=500)
        strf_width (default=0.2): time in seconds for width of strf
            window (prior to spike)
        alphas (default=(0.1, 1.0, 10.0)): list of regularization
            parameters to use in 
    """

    def _filter_fn(event):
        check_time = np.any([
            (t[0] <= event["start_time"] <= t[1])
            for t in time_ranges
        ])

        check_filter = (event_filter is None) or event_filter(event)

        return (check_time and check_filter)

    freq_arr, Y_spikes, X_specs, _ = prepare_data(
        filter(_filter_fn, events),
        spike_times,
        data_loader,
        spec_sample_rate,
        freq_spacing,
        channel_map=channel_map,
        convolve_width=convolve_width,
        pad_before=pad_before
    )

    return freq_arr, Y_spikes, X_specs


def fit_strf(
        spike_times,
        events,
        data_loader,
        event_filter=None,
        channel_map=None,
        time_ranges=None,
        spec_sample_rate=200,
        freq_spacing=100,
        epoch_size=500,
        strf_width=0.2,  # seconds
        pad_before=0.1,
        convolve_width=1,
        repeat=5,
        alphas=(0.1, 1.0, 10.0)
    ):
    """
    Fit a strf to spikes during certain events

    Parameters
    ----------
        spike_times: 1d array of floats for spike times
        events: list of dictionaries for vocal periods with metadata
        data_loader: instance of lazy.LazySite object to load audio
        event_filter (default=None): function that takes an event dict
            from events and returns True if it should be used to compute
            the strf. Always use event if event_filter is None.
        channel_map (default=None): function that takes an event dict
            from events and returns with mic channel to use. Always use
            channel with higher average amplitude if None
        time_ranges (default=None): list of tuples (start_time, stop_time)
            indicating what times the strf should be fit over
        spec_sample_rate (default=200): sample rate of strfs and
            spike bins
        freq_spacing (default=100): frequency spacing of spectrograms
        epoch_size (default=500)
        strf_width (default=0.2): time in seconds for width of strf
            window (prior to spike)
        alphas (default=(0.1, 1.0, 10.0)): list of regularization
            parameters to use in 
    """

    def _filter_fn(event):
        check_time = np.any([
            (t[0] <= event["start_time"] <= t[1])
            for t in time_ranges
        ])

        check_filter = (event_filter is None) or event_filter(event)

        return (check_time and check_filter)

    freq_arr, Y_spikes, X_specs, _ = prepare_data(
        filter(_filter_fn, events),
        spike_times,
        data_loader,
        spec_sample_rate,
        freq_spacing,
        channel_map=channel_map,
        convolve_width=convolve_width,
        pad_before=pad_before
    )
    # freq_arr = freq_arr[:len(freq_arr) // 2]
    # X_specs = [x[:, :len(freq_arr) // 2] for x in X_specs]

    scores = np.zeros_like(alphas)
    models = []
    for alpha_idx, alpha in enumerate(alphas):
        # FIXME: train on different epochs
        alpha_scores = []
        for _ in range(repeat):
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_specs,
                Y_spikes,
                test_size=0.3
            )

            estimator = TimeDelayingRidge(0.0, strf_width, spec_sample_rate,
                    reg_type="ridge", alpha=alpha)
            rf = ReceptiveField(0.0, strf_width, spec_sample_rate, estimator=estimator)
            uh = concat(X_train, Y_train)
            uh = concat(X_test, Y_test)

            rf.fit(*concat(X_train, Y_train))
            score = rf.score(*concat(X_test, Y_test))
            alpha_scores.append(score)
        score = np.mean(alpha_scores)

        scores[alpha_idx] = score
        models.append(rf)

    return models, scores, freq_arr, X_specs, Y_spikes


