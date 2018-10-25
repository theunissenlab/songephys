#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:45:48 2017

Preprocessing routines to clean up ephys signal and perform first spike thresholding
@author: frederictheunissen
"""
from __future__ import division, print_function

# Depencies
# Needs these packages,functions
import numpy as np
import quantities as pq
from statsmodels.tsa.api import VAR
import scipy.signal as signal
import matplotlib.pyplot as plt

# Theunissen Lab dependencies.
from neo.core import AnalogSignal, SpikeTrain, Unit, ChannelIndex
from soundsig.signal import lowpass_filter, highpass_filter, resample_signal, linear_filter1D

import config


def get_mask(signal, ranges):
    total = np.zeros(len(signal.times)).astype(np.bool)
    for t_start, t_stop in ranges:
        total = total | ((signal.times > t_start) & (signal.times < t_stop))
    return total


# Function to divide segment signals into low pass and high pass signals.
def high_low_filter(seg, keep_lfp=False, keep_unfiltered=False, debug = 0):
    """ Replaces the signals that starts with Amplifier (the signal from the electrodes)
    with Low Pass and High Pass version of those signals.
    The low pass version is downsampled to get LFPs
    The high pass version is filtered.  Two filtering techniques are used:
    1. An Auto-regressive Vector model to remove
    within electrod correlations (due to motion) and
    2. A linear deconvolution to remove sound bleeding into electrodes."""

    # Parameters for this procedure - could be arguments.
    CFL = 300.0*pq.Hz   # The cutoff frequency for LFP
    CFH = 300.0*pq.Hz   # The cutoff frequency for Spikes
    DELAYVAR = 4        # A delay of 4 sampls for removing within electrode array noise - this number was verified using the partial autocorrelation
    DELAYSOUND = 40     # A delay of 40 samples is used for the sound to electrode because these are not synched as well. This was obtained by cross-coherence analysis
    COHTHRES = 0.1      # The coherence threshold for sound filteringe

    # First check is this has already been done.  If so do nothing.
    for sig in seg.analogsignals:
        if sig.name.endswith('LFP'):
            print('Warning: segment already has filtered signals')
            return

    # Get the signal that has the microphone. We use this to clean up the spikes.
    for soundsig in seg.analogsignals:
        if soundsig.name.startswith('Board ADC'):
            break

    # Get the parent block
    parentblk = seg.block

    # Now process signals that have Amplifier in name: the electrode signals
    siglist = seg.analogsignals[:]  # I make a copy becuase seg.analogsignals will change dynamically
    for sig in siglist:
        if sig.name.startswith('Amplifier'):
            # print('Processing', sig.name)

            # Genererate low-pass and high-pass
            sr = sig.sampling_rate
            rawSignal = np.asarray(sig)

            # For the LFP low-pass filter and resample
            if keep_lfp:
                lowSignal = []
                for channel in rawSignal.T:
                    lowchannel = lowpass_filter(channel, sr, CFL)
                    tr, lowchannel = resample_signal(lowchannel, sr, 2*CFL)
                    lowSignal.append(lowchannel)

                lowSignal = np.vstack(lowSignal).T
                lowSR = (1.0/(tr[1]-tr[0]))   # Has the correct units.

            # For the high-pass signal filter and get clean version
            highSignal = np.vstack([highpass_filter(channel, sr, CFH) for channel in rawSignal.T]).T

            # Filter out noise by removing correlated noise across electrodes and sound bleeding
            highSignal_clean = np.zeros(highSignal.shape)
            nelec = sig.shape[1]   # This is the number of electrodes

            # Vector Auto Regression model
            X = np.asarray(highSignal)
            modelVAR = VAR(np.asarray(highSignal))
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

            # The first points are set to zero since we don't have predictions for those
            highSignal_clean[nlags:, :] = highSignal[nlags:, :] - predX

            # TODO Remove sound correlations - needs to be modified to work for 1 either 2 microphone inputs
            n_microphones = soundsig.shape[1]

            sound0 = highpass_filter(np.squeeze(soundsig[:,0]), sr, CFH)
            sound0 = sound0 - np.mean(sound0)

            if n_microphones == 2:
                sound1 = highpass_filter(np.squeeze(soundsig[:,1]), sr, CFH)
                sound1 = sound1 - np.mean(sound1)

            for ielec in range(nelec):
                # TODO (kevin): really lazy way to do 2 microphone implementation; clean this up!!!
                if n_microphones == 2:
                    h0, lags, g0, f0 = linear_filter1D(sound0, highSignal_clean[:, ielec], lag=DELAYSOUND, debug = 0)
                    h1, lags, g1, f1 = linear_filter1D(sound1, highSignal_clean[:, ielec], lag=DELAYSOUND, debug = 0)
                    maxg0 = g0.max()
                    maxg1 = g1.max()

                    predSound = np.zeros(sound0.shape)
                    if (maxg0 > maxg1 and maxg0 > COHTHRES):
                        predSound0 = signal.convolve(sound0, h0, mode = 'same' ) + np.mean(highSignal_clean[:, ielec])
                        predSound = predSound + predSound0

                        h1, lags, g1, f1 = linear_filter1D(sound1, highSignal_clean[:, ielec]-predSound, lag=DELAYSOUND, debug = 0)
                        maxg1 = g1.max()
                        if( maxg1 > COHTHRES):
                            predSound1 = signal.convolve(sound1, h1, mode = 'same' ) + np.mean(highSignal_clean[:, ielec])
                            predSound = predSound + predSound1
                    elif (maxg1 > maxg0 and maxg1 > COHTHRES):
                            predSound1 = signal.convolve(sound1, h1, mode = 'same' ) + np.mean(highSignal_clean[:, ielec])
                            predSound = predSound + predSound1

                            h0, lags, g0, f0 = linear_filter1D(sound0, highSignal_clean[:, ielec]-predSound, lag=DELAYSOUND, debug = 0)
                            maxg0 = g0.max()
                            if( maxg0 > COHTHRES):
                                predSound0 = signal.convolve(sound0, h0, mode = 'same' ) + np.mean(highSignal_clean[:, ielec])
                                predSound = predSound + predSound0
                elif n_microphones == 1:
                    h0, lags, g0, f0 = linear_filter1D(sound0, highSignal_clean[:, ielec], lag=DELAYSOUND, debug = 0)
                    maxg0 = g0.max()
                    predSound = signal.convolve(sound0, h0, mode = 'same' ) + np.mean(highSignal_clean[:, ielec])

                # Remove prediction from sound response
                # TODO (kevin): We should keep a record of the prediction that we are removing
                highSignal_clean[:, ielec] = highSignal_clean[:, ielec] - predSound

                # Plot the original signal, the two modelled sources of noise and the clean signal
                if (debug):
                    plt.figure()
                    plt.subplot(211)
                    # The first has a , because it returns multiple handles - I think..
                    rsh, = plt.plot(highSignal[:, ielec], 'k', label = 'Raw Signal')
                    cnh, = plt.plot(np.hstack((np.zeros(nlags), predX[:, ielec])), 'r', label = 'Corr Noise')
                    snh, = plt.plot(predSound, 'b', label='Sound Noise')
                    plt.legend(handles=[rsh, cnh, snh])
                    plt.title('Electrode %d' % ielec)

                    plt.subplot(212)
                    rsh, = plt.plot(highSignal[:, ielec], 'k', label='Raw')
                    fsh, = plt.plot(highSignal_clean[:, ielec], 'r', label='Filtered')
                    plt.legend(handles=[rsh, fsh])

            # TODO (kevin): this block optional (we can leave out the non-clean signal to save space)
            # Make three new neo signals
            if keep_lfp:
                lowSignalNeo = AnalogSignal(lowSignal,
                                   name=sig.name+'LFP',
                                   units=sig.units,
                                   t_start=sig.t_start,
                                   sampling_rate=lowSR,
                                   file_origin=sig.file_origin,
                                   copy=False)
                lowSignalNeo.segment = seg
                seg.analogsignals.append(lowSignalNeo)

            if keep_unfiltered:
                highSignalNeo = AnalogSignal(highSignal,
                                   name=sig.name+'HP',
                                   units=sig.units,
                                   t_start=sig.t_start,
                                   sampling_rate=sr,
                                   file_origin=sig.file_origin,
                                   copy=False)
                highSignalNeo.segment = seg
                seg.analogsignals.append(highSignalNeo)

            highCleanSignalNeo = AnalogSignal(highSignal_clean,
                               name=sig.name+'HPC',
                               units=sig.units,
                               t_start=sig.t_start,
                               sampling_rate=sr,
                               file_origin=sig.file_origin,
                               copy=False)
            highCleanSignalNeo.segment = seg
            seg.analogsignals.append(highCleanSignalNeo)

            # Delete full sig from channel index and add new ones
            for ci in parentblk.channel_indexes:
                if ci.name == 'Amplifier':
                    ci.analogsignals.remove(sig)
                    if keep_lfp:
                        ci.analogsignals.append(lowSignalNeo)
                    if keep_unfiltered:
                        ci.analogsignals.append(highSignalNeo)
                    ci.analogsignals.append(highCleanSignalNeo)
                    ci.create_relationship(force=True)

            # Delete sig from segment
            seg.analogsignals.remove(sig)
            seg.create_relationship(force=True)
    return


# Fixing the channels from one for the array to one per electrode - to be done BERFORE adding units
def fix_chIndex(block):
    """ Takes the channel index named Amplifier and makes into a channel Index per electrode called Amp 1, Amp 2, etc"""

    if len(block.channel_indexes) == 0:
        print("Block has no channels. Nothing to fix?")
        return

    # finds channel called Amplifier
    for chIndex in block.channel_indexes:
        if chIndex.name == 'Amplifier':
            break

    if chIndex.name != 'Amplifier':
        print("Block does not have a channel called Amplifer. Nothing to fix?")
        return

    nelec = len(chIndex.index)
    for ie in range(nelec):

        # Remove prior info on channel _ indexes
        if ie == 0:
            for sig in chIndex.analogsignals:
                sig.channel_index = None

        # FIXME (kevin): the channel indexes we create here should be numbered 8-23
        # not 0-15. however, the chIndex.index should be 0-15 to properly access the
        # individual channels in the analog signal
        chx = ChannelIndex(
                index = np.array([ie]),
                name = 'Amp Elec %d' % np.arange(8, 24)[ie],
                channel_names=np.array(['ch %d' % np.arange(8, 24)[ie]]))
        chx.block = block
        chx.analogsignals = chIndex.analogsignals
        # chx.create_relationship()  # This is commented out because it creates a sig.channel_index?

        block.channel_indexes.append(chx)

    block.channel_indexes.remove(chIndex)
    block.create_relationship(force=True, append = False, recursive=False)  # Just do one level so that signals are not linked to channels.


def drop_electrode_signals(block):
    """Drop electrode signals from block

    (Elec signals identified by analogsignals that start with
    the string "Amplifier_HP")
    """
    for channel_index in block.channel_indexes:
        electrode_signals = [
                sig for sig in channel_index.analogsignals
                if sig.name.startswith("Amplifier_HP")
        ]
        non_electrode_signals = [
                sig for sig in channel_index.analogsignals
                if not sig.name.startswith("Amplifier_HP")
        ]

        channel_index.analogsignals = non_electrode_signals

        for sig in electrode_signals:
            try:
                sig.segment.analogsignals.remove(sig)
            except ValueError:
                pass

    for segment in block.segments:
        electrode_signals = [
                sig for sig in segment.analogsignals
                if sig.name.startswith("Amplifier_HP")
        ]
        non_electrode_signals = [
                sig for sig in segment.analogsignals
                if not sig.name.startswith("Amplifier_HP")
        ]

        segment.analogsignals = non_electrode_signals

        for sig in electrode_signals:
            try:
                sig.channel_index.analogsignals.remove(sig)
            except ValueError:
                pass

    return block


def extract_spikes_from_signal(signal,
        threshold_std=4,
        limit_std=None,
        spike_width=20,
        upward_gain=2,
        min_spacing=1,
        noise_mask=None
        ):
    """Extract negative-going spikes from signal

    Exclude any spikes with peaks greater than limit_std as noise

    Exclude spikes that are too close to each other or are in a period with too much noise
    """
    noise_mask = noise_mask if noise_mask is not None else np.zeros(len(signal)).astype(np.bool)

    std = np.std(signal[np.invert(noise_mask)])
    threshold = threshold_std * std

    spike_indices = []
    spike_waveforms = []
    downward_mask = np.where(np.logical_and(
        signal <= -threshold,
        np.invert(noise_mask)
    ))[0]
    upward_mask = np.where(np.logical_and(
        signal >= 2 * threshold,
        np.invert(noise_mask)
    ))[0]
    if len(downward_mask):
        downward_mask = np.concatenate([downward_mask, [len(signal)]])
    if len(upward_mask):
        upward_mask = np.concatenate([upward_mask, [len(signal)]])

    downward_edges = np.where(np.diff(downward_mask) > min_spacing)[0]
    upward_edges = np.where(np.diff(upward_mask) > min_spacing)[0]

    for direction, edges in [("down", downward_edges), ("up", upward_edges)]:
        start_ind = 0
        for end_ind in edges:
            if direction is "down":
                mask = downward_mask
                spike_ind = np.argmin(signal[mask[start_ind]:mask[end_ind]+1])
            elif direction is "up":
                mask = upward_mask
                spike_ind = np.argmax(signal[mask[start_ind]:mask[end_ind]+1])

            spike_ind = mask[start_ind] + np.int(spike_ind)
            if limit_std and signal[spike_ind] < - limit_std * std:
                pass
            elif spike_ind - spike_width < 0:
                # don't include spikes that are too close to the beginning
                pass
            elif spike_ind + spike_width > len(signal):
                # don't include spikes that are too close to the end
                pass
            else:
                spike_indices.append(spike_ind)
            start_ind = end_ind + 1

    spike_indices = np.array(spike_indices).astype(np.int)

    for spike_idx in spike_indices:
        spike_wf = signal[spike_idx - spike_width:spike_idx + spike_width]
        spike_waveforms.append(spike_wf)
    spike_waveforms = np.array([np.array(x) for x in spike_waveforms])

    return spike_indices, spike_waveforms


def extract_spikes(block):
    """ Generates Units based on threshold crossing of the analog signals that end with HPC (High Pass Clean).

    A single Unit data structure (that contains multi units) is generated for each electrode.
    """
    # Following neo's phylosophy we look through the analogsignals from the channel index since we want to get spikes
    # across segments. We assume one channel per electrode.

    # Units are unique to electrodes in our data
    for channel_index in block.channel_indexes:
        # If channel index is not an electrode continue
        if not channel_index.name.startswith('Amp Elec'):
            continue;

        new_spiketrains = []
        elec_id = np.int(channel_index.index[0])    # Assume a single index
        print("Extracting spikes from channel {}.".format(elec_id))

        # HPC signals are clean electrode signals
        for sig in filter(lambda s: s.name.endswith("HPC"), channel_index.analogsignals):
            seg = sig.segment
            sig_magnitude = sig.magnitude[:, elec_id]

            ''' (skip the bad noise check)
            if len(seg.epochs):
                epoch = seg.epochs[0]
                t_starts = epoch.times[epoch.labels == elec_id]
                t_stops = t_starts + epoch.durations[epoch.labels == elec_id]
                noise_mask = get_mask(sig, zip(t_starts, t_stops))
                noise_indices = set(np.where(noise_mask)[0])
            else:
                noise_mask = np.zeros(len(sig_magnitude)).astype(np.bool)
                noise_indices = set()
            '''
            noise_mask = np.zeros(len(sig_magnitude)).astype(np.bool)
            noise_indices = set()

            # Get the threshold
            # Get the spike width in number of points
            spikewd = (sig.sampling_rate * 1 * pq.ms)
            spikewd = spikewd.rescale('dimensionless')
            spikewd = np.int(spikewd)

            spike_indices, spike_waveforms = extract_spikes_from_signal(
                    sig_magnitude,
                    threshold_std=config.Z_THRESHOLD,
                    limit_std=config.Z_SPIKE_LIMIT,
                    min_spacing=config.MIN_SPIKE_THRESHOLD_SPACING,
                    spike_width=spikewd,
                    noise_mask=noise_mask
            )

            # print("Extracted {} spikes from {}".format(len(spike_indices), elec_id))
            spike_times = sig.times[spike_indices]

            new_spiketrain = SpikeTrain(
                    spike_times,
                    t_start=sig.t_start,
                    t_stop=sig.t_stop,
                    name="{}_MU_{}".format(seg.name, elec_id)
            )
            new_spiketrain.segment = seg
            seg.spiketrains.append(new_spiketrain)
            seg.create_relationship(force=True, append=False, recursive=False)
            new_spiketrain.waveforms = spike_waveforms
            new_spiketrains.append(new_spiketrain)

        # We got all the spike trains for an unsorted unit
        new_unit = Unit(name="MU {}".format(elec_id))
        new_unit.spiketrains = new_spiketrains
        new_unit.create_relationship(force=True, append=False, recursive=False)
        channel_index.units.append(new_unit)
        new_unit.channel_index = channel_index

        # Doing the links for all the children does not work as it restores the sig-->channel_index link that we don't have
        # channel_index.create_relationship(force=True, append=False, recursive=False)


def unit_report(block):
    """ Makes a little report and plot of snippets for all the units in the block """

    # Numbers
    ntot = 0
    for chIndex in block.channel_indexes:
        nunits = len(chIndex.units)
        print('Channel', chIndex.name, 'has', nunits, 'units.')
        ntot += nunits
    print('Total number of units:', ntot)


    for chIndex in block.channel_indexes:
        nunits = len(chIndex.units)
        for unit in chIndex.units:
            print(unit.name, ':', unit.size['spiketrains'], 'spiketrains')
            totspikes = 0;
            for st in unit.spiketrains:
                if totspikes == 0:
                    print(st.segment)    # Debugging...
                totspikes += st.size
                if st.size != 0:
                    wavelen = st.waveforms.shape[1]
            print('\t Total Spikes (in all segments):', totspikes)

    # Make Plots of snippets
    for chIndex in block.channel_indexes:
        nunits = len(chIndex.units)
        for unit in chIndex.units:
            waveAvg = np.zeros(wavelen)
            plt.figure()
            totspikes = 0
            for st in unit.spiketrains:
                totspikes += st.size
                if st.size != 0:
                    plt.plot(st.waveforms[:,:,0].T, 'grey', linewidth = 0.25)
                    waveAvg += np.sum(np.squeeze(st.waveforms[:,:,0]), axis = 0)
            waveAvg /= totspikes
            maxplot = np.max([waveAvg.max(), np.abs(waveAvg.min())])
            plt.plot(waveAvg, 'k', linewidth = 2)
            plt.ylim([-1.5*maxplot, 1.5*maxplot])
            plt.title(unit.name)


################# OLD CODE. FET WANTS TO KEEP FOR NOW #######################
 ### THIS IS OLD CODE FROM THE seg_low_high program

#    for ielec in range(nelec):
 #               X = np.asarray(np.delete(highSignal, ielec, 1))
                # Uncomment the following line if you want sound correlations as well
                # X = np.hstack((np.asarray(soundsig[:,0]), X))
 #               Y = np.squeeze(np.asarray(highSignal[:,ielec]))

                # This is to look at sound correlations - they looked weak.
                # c0 = np.corrcoef(Y, np.squeeze(soundsig[:,0]))
                # c1 = np.corrcoef(Y, np.squeeze(soundsig[:,1]))
                # print '/t', ielec, ':', c0[0,1], c1[0,1]

#                w, res, rk, s = np.linalg.lstsq(X, Y)         # leastsquare fit y with x
 #               highSignal_clean[:, ielec] = Y - np.matmul(X,w) # residuals
                #

                # Generalized least square with auto-regressive model
                # modelAR = sm.GLSAR(Y, X, rho=2)
                # modelAR_results = modelAR.iterative_fit(10)
                # Ypred = modelAR.predict(modelAR_results.param, X)
                # Yres = Y-Ypred

