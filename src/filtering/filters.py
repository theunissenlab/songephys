"""
Preform vector (auto)regression to remove correlations within and across electrodes

>>> filtered_sig = vector_autoregression_windowed(
...     signal_highpassed,
...     window_size=1000,
...     update_slice=slice(100, 900),
...     n_lags=4,
...     causal=False,
...     downsample=1,
...     self_buffer=5)

"""

import numpy as np
import quantities as pq
import scipy.stats

from neo.core import AnalogSignal, SpikeTrain, Unit, ChannelIndex

from soundsig.signal import lowpass_filter, highpass_filter

from timeseries import TimeSeriesRegression, PredictTimeseries

import config


def filter_segment(seg):
    block = seg.block 

    # Get the signal that has the microphone. We use this to clean up the spikes. Maybe.
    sound_signal = (
        sig for sig in seg.analogsignals
        if sig.name.startswith("Board ADC")
    ).next()

    elec_signal = (
        sig for sig in seg.analogsignals
        if sig.name.startswith("Amplifier")
    ).next()

    # Generate high pass
    signal_highpass_clean = clean_signal(elec_signal)

    new_analogsignal = AnalogSignal(
           signal_highpass_clean,
           name="{}_HPC".format(elec_signal.name),
           units=elec_signal.units,
           t_start=elec_signal.t_start,
           sampling_rate=elec_signal.sampling_rate,
           file_origin=elec_signal.file_origin,
           copy=False
    )
    new_analogsignal.segment = seg

    seg.analogsignals.append(new_analogsignal)
    
    # Clean up the old signal and add the new ones
    for channel_index in block.channel_indexes:
        if channel_index.name == "Amplifier":
            channel_index.analogsignals.remove(elec_signal)
            channel_index.analogsignals.append(new_analogsignal)
            channel_index.create_relationship(force=True)

    seg.analogsignals.remove(elec_signal)
    seg.create_relationship(force=True)


def clean_signal(signal):
    sig_highpass = highpass_filter(
            signal.T,
            signal.sampling_rate,
            config.HIGHPASS_CUTOFF * pq.Hz).T
    return vector_autoregression(sig_highpass)


def vector_autoregression(signal, **kwargs):
    """Remove correlations between electrodes"""
    new_signal = np.zeros_like(signal)

    full_model = PredictTimeseries(**kwargs)
    full_model.fit(signal)

    _, residual = full_model.predict(signal)

    new_signal[full_model._valid_slice(signal)] = residual

    return new_signal


def vector_autoregression_windowed(signal, window_size=1000, update_slice=slice(100, 900), **kwargs):
    new_signal = np.zeros_like(signal)
    model = PredictTimeseries(**kwargs)

    step = update_slice.stop - update_slice.start

    for window_start in range(0, signal.shape[0], step):
        subsignal = signal[window_start:window_start + window_size]

        model.fit(subsignal)

        filtered = np.zeros_like(subsignal)
        filtered[model._valid_slice(filtered)] = model.predict(subsignal)[1]

        if subsignal.shape[0] < window_size:
            filtered = filtered[update_slice.start:]
            new_signal[window_start:][update_slice.start:update_slice.start + filtered.shape[0]] = filtered
        else:
            new_signal[window_start:window_start + window_size][update_slice] = filtered[update_slice]

    return new_signal
