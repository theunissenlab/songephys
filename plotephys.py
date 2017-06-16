#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Plotting functions for our 16 channel neo files
Created on Thu Jun 15 15:36:08 2017

@author: frederictheunissen
"""

# Depebdencues
import numpy as np
import quantities as pq

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Theunissen Lab dependencies.
from soundsig.sound import spectrogram, plot_spectrogram
from soundsig.hinton import hinton
from soundsig.signal import lowpass_filter



def plot_segment_HPC_Check(seg): 
# Plot all segments to check out noise reduction 
    
    # Find the high pass electrode signals
    for sig in seg.analogsignals:
        if sig.name.endswith('HP'):
            break
            
    # Find the cleaned electrode signals       
    for sigclean in seg.analogsignals:
        if sigclean.name.endswith('HPC'):
            break
            
    # Find the microphone signal
    for soundsig in seg.analogsignals:
        if soundsig.name.startswith('Board ADC'):
            break
            
    # The amplifier signal with the spikes is a matrix time x electrode         
    high_amp = np.asarray(sig) 
    nelec = sig.shape[1]   # This is the number of electrodes
    maxabs = np.std(high_amp)*6.0
    
    # Print some correlations between the high amp channels 
    corr_table = np.corrcoef(high_amp.T)
    fig = plt.figure(1);
    fig = hinton(corr_table)
    
    # This is the clean signal
    high_amp_clean = np.asarray(sigclean)
                   
    # Now generate the plots
    # Find the loudest sound to plot   
    rms0 = np.std(soundsig[:, 0])
    rms1 = np.std(soundsig[:, 1])
    if rms1 > rms0:
        sound = np.asarray(soundsig[:, 1]).squeeze()
    else:
        sound = np.asarray(soundsig[:, 0]).squeeze()
            
    # Calculate envelope and spectrogram
    sound = sound - sound.mean()
    sound_env = lowpass_filter(np.abs(sound), float(soundsig.sampling_rate), 250.0)  # Sound Enveloppe
    to, fo, spect, rms = spectrogram(sound, float(soundsig.sampling_rate), 1000, 50)
              
    # Plot the neural data - Before noise Reduction.
    fig = plt.figure(2, figsize=(15,15))
    gs = gridspec.GridSpec(110, 2)
        
    for iplt in range(2):
        ax = fig.add_subplot(gs[:20,iplt])
        plot_spectrogram(to, fo, spect, ax=ax, 
                         ticks=True, fmin=250, fmax=8000, colormap=None, colorbar=False, log = True, dBNoise = 50)
        plt.ylabel('Frequency')
        plt.tick_params(labelbottom='off')

    for iplt in range(nelec):
        # make a 2 column plot
        if iplt < nelec/2:
            ax = fig.add_subplot(gs[20+10*iplt:20+10*iplt+9,0])
        else:
            isub = iplt%(nelec/2);
            ax = fig.add_subplot(gs[20+10*isub:20+10*isub+9,1])
        ax.plot(sig.times, high_amp[:, iplt], linewidth = 0.5, color = 'black')

        plt.ylim(-maxabs,maxabs)
        if iplt != ((nelec/2)-1):
            plt.tick_params(labelbottom='off', labelleft='off')
        else:
            plt.xlabel('Time (s)')    

            
    # Plot the neural data - After noise Reduction.
    corr_table = np.corrcoef(high_amp_clean.T)
    fig = plt.figure(4);
    fig = hinton(corr_table)
    
    fig = plt.figure(5, figsize=(15,15))
    gs = gridspec.GridSpec(100, 2)
        
    for iplt in range(2):
        ax = fig.add_subplot(gs[:20,iplt])
        plot_spectrogram(to, fo, spect, ax=ax, 
                         ticks=True, fmin=250, fmax=8000, colormap=None, colorbar=False, log = True, dBNoise = 50)
        plt.ylabel('Frequency')
        plt.tick_params(labelbottom='off')

    for iplt in range(nelec):
        # Make 2 column plot
        if iplt < nelec/2:
            ax = fig.add_subplot(gs[20+10*iplt:20+10*iplt+9,0])
        else:
            isub = iplt%(nelec/2);
            ax = fig.add_subplot(gs[20+10*isub:20+10*isub+9,1])
        
        th = 3.5*np.std(high_amp_clean[:,iplt])    
        ax.plot(sig.times, high_amp_clean[:, iplt], linewidth = 0.5, color = 'blue')
        ax.plot([sig.times[0], sig.times[-1]], [-th, -th], linewidth = 1, color = 'black')

        plt.ylim(-maxabs,maxabs)
        if iplt != ((nelec/2)-1):
            plt.tick_params(labelbottom='off', labelleft='off')
        else:
            plt.xlabel('Time (s)')     
                       
    plt.show()

