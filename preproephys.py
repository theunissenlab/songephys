#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:45:48 2017

Preprocessing routines to clean up ephys signal and perform first spike thresholding
@author: frederictheunissen
"""

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


# Function to divide segment signals into low pass and high pass signals.
def high_low_filter(seg, debug = 0):
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
            print 'Warning: segment already has filtered signals'
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
            
            print 'Processing', sig.name
            
            # Genererate low-pass and high-pass
            sr = sig.sampling_rate
            rawSignal = np.asarray(sig)
            
            # For the LFP low-pass filter and resample
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
            nlags = (paramsDEL.shape[0]-1)/nelec  # This is also equal to DELAYS 
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
                       
            # Remove sound correlations - needs to be modified to work for 1 either 2 microphone inputs
            sound0 = highpass_filter(np.squeeze(soundsig[:,0]), sr, CFH)
            sound0 = sound0 - np.mean(sound0)
            
            sound1 = highpass_filter(np.squeeze(soundsig[:,1]), sr, CFH)
            sound1 = sound1 - np.mean(sound1)
            
            for ielec in range(nelec):
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
                
                # Remove prediction from sound response
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
                    
                                             
            # Make three new neo signals
            lowSignalNeo = AnalogSignal(lowSignal,
                               name=sig.name+'LFP',
                               units=sig.units,
                               t_start=sig.t_start,
                               sampling_rate=lowSR,
                               file_origin=sig.file_origin,
                               copy=False)
            lowSignalNeo.segment = seg
            
            highSignalNeo = AnalogSignal(highSignal,
                               name=sig.name+'HP',
                               units=sig.units,
                               t_start=sig.t_start,
                               sampling_rate=sr,
                               file_origin=sig.file_origin,
                               copy=False)
            highSignalNeo.segment = seg
            
            highCleanSignalNeo = AnalogSignal(highSignal_clean,
                               name=sig.name+'HPC',
                               units=sig.units,
                               t_start=sig.t_start,
                               sampling_rate=sr,
                               file_origin=sig.file_origin,
                               copy=False)
            highCleanSignalNeo.segment = seg
            
            # Add signals to segment
            seg.analogsignals.append(lowSignalNeo)
            seg.analogsignals.append(highSignalNeo)
            seg.analogsignals.append(highCleanSignalNeo)
                    
            # Delete full sig from channel index and add new ones
            for ci in parentblk.channel_indexes:
                if ci.name == 'Amplifier':
                    ci.analogsignals.remove(sig)
                    ci.analogsignals.append(lowSignalNeo)
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
        print "Block has no channels. Nothing to fix?"
        return
    
    # finds channel called Amplifier
    for chIndex in block.channel_indexes:
        if chIndex.name == 'Amplifier':
            break
            
    if chIndex.name != 'Amplifier':
        print "Block does not have a channel called Amplifer. Nothing to fix?"
        return
    
    nelec = len(chIndex.index)
    for ie in range(nelec):
        
        # Remove prior info on channel _ indexes
        if ie == 0:
            for sig in chIndex.analogsignals:
                sig.channel_index = None
        chx = ChannelIndex(index = np.array([ie]), name = 'Amp Elec %d' % ie, channel_names=np.array(['ch %d'% chIndex.index[ie]]))
        chx.block = block
        chx.analogsignals = chIndex.analogsignals
        # chx.create_relationship()  # This is commented out because it creates a sig.channel_index?
                
        block.channel_indexes.append(chx)
                           
    block.channel_indexes.remove(chIndex)
    block.create_relationship(force=True, append = False, recursive=False)  # Just do one level so that signals are not linked to channels.
    
def extract_spikes(block):
    """ Generates Units based on threshold crossing of the analog signals that end with HPC (High Pass Clean). 
    A single Unit data structure (that contains multi units) is generated for each electrode."""

    # Parameters
    SPIKEWIDTH = 1*pq.ms    # Sets the number of data points to be saved in the spike snippet - on each side from peak
    NSTD = 2              # The level of the threshold in rms

    # Following neo's phylosophy we look through the analogsignals from the channel index since we want to get spikes
    # across segments. We assume one channel per electrode.


# Units are unique to electrodes in our data
    for chIndex in block.channel_indexes: 
        # If channel index is not an electrode continue
        if not chIndex.name.startswith('Amp Elec'):
            continue;
        st_list = []
        ie = np.int(chIndex.index[0])    # Assume a single index
        print ie
        for sig in chIndex.analogsignals:
            if sig.name.endswith('HPC'):       
                # Get the threshold
                th = NSTD*np.std(sig.magnitude[:,ie]) 
                
                # Get the spike width in number of points
                spikewd = (sig.sampling_rate*SPIKEWIDTH)
                spikewd = spikewd.rescale('dimensionless')
                si = np.int(spikewd)
                   
                # Make a mask where signal is below this value
                spike_mask = np.where(sig.magnitude[:,ie] < -th)[0]
                spike_mask_edge = np.diff(spike_mask)
                ind_edge = np.where(spike_mask_edge >1)[0]

                # Find minimum value and index of spike within intervals below threshold
                startSpIndInd=0
                spikeIndex = []
                for endSpIndInd in ind_edge:
                    startSpInd = spike_mask[startSpIndInd] # First point crossing the threshold
                    endSpInd = spike_mask[endSpIndInd]     # Last point crossing the threshold
                    spikeVal = np.min(sig.magnitude[startSpInd:endSpInd+1, ie])
                    spikeIndOff = np.where(sig.magnitude[startSpInd:endSpInd+1, ie] == spikeVal)[0]
                    #print 'Spike between', startSpInd, 'and', endSpInd,':', np.min(sig.magnitude[startSpInd:endSpInd+1, ie]), 'found at', startSpInd + spikeIndOff
                    spike_idx = startSpInd + np.int(spikeIndOff)
                    if spike_idx - si <0:
                        continue # this spike is too close to the beginning of the segment
                    if spike_idx + si > sig.shape[0]:
                        break # this spike is too close to the end of the segment.
                    spikeIndex.append(spike_idx)
                    startSpIndInd = endSpIndInd+1

                # create a spike train
                spikeTimes = sig.times[spikeIndex]
                seg = sig.segment
                st = SpikeTrain(spikeTimes, t_start=sig.t_start, t_stop=sig.t_stop, name= seg.name+'_MU_%d'%ie)
                st.segment = seg
                seg.spiketrains.append(st)
                seg.create_relationship(force=True, append=False, recursive=False)

                # remember the spike waveforms
                wf_list = []
                for spike_idx in spikeIndex:
                    wf_list.append(sig[spike_idx-si:spike_idx+si, ie])
                
                st.waveforms = np.array(wf_list)
                st_list.append(st)
        
        # We got all the spike trains for an unsorted unit
        u = Unit(name='MU %d'%ie)
        u.spiketrains = st_list
        u.create_relationship(force=True, append=False, recursive=False)
        chIndex.units.append(u)
        u.channel_index = chIndex
        
        # Doing the links for all the children does not work as it restores the sig-->channel_index link that we don't have
        # chIndex.create_relationship(force=True, append=False, recursive=False)
        
def unit_report(block):
    """ Makes a little report and plot of snippets for all the units in the block """
    
    # Numbers
    ntot = 0
    for chIndex in block.channel_indexes: 
        nunits = len(chIndex.units)
        print 'Channel', chIndex.name, 'has', nunits, 'units.'
        ntot += nunits
    print 'Total number of units:', ntot 


    for chIndex in block.channel_indexes: 
        nunits = len(chIndex.units)
        for unit in chIndex.units:
            print unit.name, ':', unit.size['spiketrains'], 'spiketrains'
            totspikes = 0;
            for st in unit.spiketrains:
                if totspikes == 0:
                    print st.segment    # Debugging...
                totspikes += st.size                
                if st.size != 0:
                    wavelen = st.waveforms.shape[1]
            print '\t Total Spikes (in all segments):', totspikes
 
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
