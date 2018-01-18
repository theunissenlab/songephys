"""
Perform vector (auto)regression to remove correlations within and across electrodes

>>> filter_segment(segment)

"""
from __future__ import print_function

import numpy as np
import quantities as pq
import theano
import theano.tensor as T
from neo.core import AnalogSignal

from soundsig.signal import highpass_filter

import config
from timeseries import filter_data_by_blocks, target_slice
from util import float_x


class Filterer(object):
    def __init__(self, block_width):
        X = T.fmatrix("input_batch")
        n_lags = T.iscalar("n_lags")
        original_signal, predicted_signal = filter_data_by_blocks(X, n_lags, block_width)

        self._predict = theano.function([X, n_lags], [original_signal, predicted_signal])

    def filter(self, signal, n_lags):
        new_signal = np.zeros_like(signal)

        original, predicted = self._predict(float_x(signal), n_lags)
        new_signal[target_slice(n_lags)] = original - predicted
        return new_signal


filterer = Filterer(block_width=800)


def filter_segment(seg):
    block = seg.block 

    # Get the signal that has the microphone. We use this to clean up the spikes. Maybe.
    # sound_signal = (
    #     sig for sig in seg.analogsignals
    #     if sig.name.startswith("Board ADC")
    # ).next()

    elec_signal = (
        sig for sig in seg.analogsignals
        if sig.name.startswith("Amplifier")
    ).next()

    # Generate high pass
    sig_highpass = highpass_filter(
            elec_signal.T,
            elec_signal.sampling_rate,
            config.HIGHPASS_CUTOFF * pq.Hz).T

    signal_highpass_clean = filterer.filter(
            sig_highpass,
            n_lags=9)

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

