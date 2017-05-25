# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:17:07 2017

@author: billewood
"""
from lasp.sound import spectrogram, plot_spectrogram
import numpy as np
from lasp.signal import lowpass_filter, highpass_filter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def plotSoundSeg(fig, seg, corr_plot=None, ephys_plot = None):
    
    # fig pointer to figure
    # seg is the segment
    # if corr_plot isn't None it will plot correlation vectors, optional
    # ephys_plot should be an array of integers corresponding to up to 8 amplifier channels you want to plot
    # returns the filtered sound from the loudest of the two signals.
    
    # clear figure
    fig.clear()
    if ephys_plot == 0: # makes entering 0 intuitively set it not to plot
        ephys_plot = None
    if corr_plot == 0: #intuitive user commands
        corr_plot = None
        
    # The sound signal
    soundSnip = seg.analogsignals[0] # the microphones
    fs = soundSnip.sampling_rate
    tvals = soundSnip.times
    
    # The Artifact Vector (Correlation Vector)
    if corr_plot != None:
        corr_glob = seg.analogsignals[3] #various correlation signals
        corr_loc = seg.analogsignals[4]
        corr_glob_lfp = seg.analogsignals[5]
        corr_loc_lfp = seg.analogsignals[6]        
#        artifact = seg.analogsignals[corr_plot]
        a_tvals = corr_glob.times
        a_lfp_tvals = corr_glob_lfp.times
    
    # Calculate the rms of each mic
    rms0 = np.std(soundSnip[:, 0])
    rms1 = np.std(soundSnip[:, 1])
            
    # Choose the loudest to plot
    if rms1 > rms0:
        sound = np.asarray(soundSnip[:, 1]).squeeze()
    else:
        sound = np.asarray(soundSnip[:, 0]).squeeze()
            
    # Calculate envelope and spectrogram
    sound = sound - sound.mean()
    sound_env = lowpass_filter(np.abs(sound), float(fs), 250.0)  # Sound Enveloppe
    to, fo, spect, rms = spectrogram(sound, float(fs), 1000, 50)
           
    # Plot sonogram and spectrogram
    gs = gridspec.GridSpec(100, 1) # top subplot
    plot_y = 20
    ax = fig.add_subplot(gs[0:plot_y,0])
    ax.plot(tvals-tvals[0], sound/sound.max())
    plt.xlim(0.0, to[-1])

    if corr_plot == None: 
        ax.plot(tvals-tvals[0], sound_env/sound_env.max(), color="red", linewidth=2)
    else:# plot the artifacts
        ax.plot(a_tvals-a_tvals[0], np.asarray(corr_loc), color="k", linewidth=2)
        ax.plot(a_lfp_tvals-a_lfp_tvals[0], np.asarray(corr_loc_lfp), color="c", linewidth=2)
        plt.ylabel('Local\nCorrs')
        ymin = min(min(np.asarray(corr_loc)), min(np.asarray(corr_loc_lfp)))
        ymax = max(max(np.asarray(corr_loc)), max(np.asarray(corr_loc_lfp)))
        plt.ylim(ymin, ymax) 
        ax = fig.add_subplot(gs[plot_y:plot_y + 10,0])
        plot_y += 10
        ax.plot(tvals-tvals[0], sound/sound.max())
        plt.xlim(0.0, to[-1])
        ax.plot(a_tvals-a_tvals[0], np.asarray(corr_glob), color="k", linewidth=2)
        ax.plot(a_lfp_tvals-a_lfp_tvals[0], np.asarray(corr_glob_lfp), color="c", linewidth=2)
        ymin = min(min(np.asarray(corr_glob)), min(np.asarray(corr_glob_lfp)))
        ymax = max(max(np.asarray(corr_glob)), max(np.asarray(corr_glob_lfp)))
        plt.ylim(ymin, ymax) 
        print ymin, ymax
        plt.ylabel('Global\nCorrs')
    plt.title('%s %d' % (seg.name, seg.index))
    plt.xlim(0.0, to[-1])
    if ephys_plot == None:
        ax = fig.add_subplot(gs[plot_y:,0])
    else:
        ax = fig.add_subplot(gs[plot_y:plot_y + 20,0])
        plot_y += 20
    plot_spectrogram(to, fo, spect, ax=ax,
        ticks=True, fmin=250, fmax=8000, colormap=None, colorbar=False, log = True, dBNoise = 50)
    plt.ylabel('Freq.')
    plt.tick_params(labelbottom='off')
    plt.xlim(0.0, to[-1])
    if ephys_plot != None:
        y_size = (100 - plot_y) / len(ephys_plot)
        #maxabs = np.std(low_amp)*3.0
        amp = seg.analogsignals[1] # highpassed amplifier
        fs = amp.sampling_rate
        tvals = amp.times
        for iplt in ephys_plot:
            ax = fig.add_subplot(gs[plot_y:plot_y+y_size,0])
            plot_y = plot_y + y_size
#            subplot(9, 2, iplt+3)
            plt.plot(tvals-tvals[0], amp[:,iplt])
            plt.xlim(0.0, to[-1])
            plt.ylabel('%d' %iplt)
            #ylim(-maxabs, maxabs)
            #xlim(t_start_plot, t_end_plot)
        plt.xlabel('Time (s)')
    plt.show()
    return sound, fs
    
    
def filteredPlot(fig, seg, corr_plot=None, ephys_plot = [0]):
    
    # fig pointer to figure
    # seg is the segment
    # if corr_plot isn't None it will plot correlation vectors, optional
    # ephys_plot should be an array of integers corresponding to up to 8 amplifier channels you want to plot, defaults to the first channel
    # returns the filtered sound from the loudest of the two signals.
    
    # clear figure
    fig.clear()

    if corr_plot == 0: #intuitive user commands
        corr_plot = None
        
    # The sound signal
    soundSnip = seg.analogsignals[0]
    fs = soundSnip.sampling_rate
    tvals = soundSnip.times
    
    # The Artifact Vector (Correlation Vector)
    if corr_plot != None:
        corr_glob = seg.analogsignals[3]
        corr_loc = seg.analogsignals[4]
        corr_glob_lfp = seg.analogsignals[5]
        corr_loc_lfp = seg.analogsignals[6]        
#        artifact = seg.analogsignals[corr_plot]
        a_tvals = corr_glob.times
        a_lfp_tvals = corr_glob_lfp.times
    
    # Calculate the rms of each mic
    rms0 = np.std(soundSnip[:, 0])
    rms1 = np.std(soundSnip[:, 1])
            
    # Choose the loudest to plot
    if rms1 > rms0:
        sound = np.asarray(soundSnip[:, 1]).squeeze()
    else:
        sound = np.asarray(soundSnip[:, 0]).squeeze()
            
    # Calculate envelope and spectrogram
    sound = sound - sound.mean()
    sound_env = lowpass_filter(np.abs(sound), float(fs), 250.0)  # Sound Enveloppe
    to, fo, spect, rms = spectrogram(sound, float(fs), 1000, 50)
           
    # Plot sonogram and spectrogram
    gs = gridspec.GridSpec(100, 1)
    plot_y = 10
    ax = fig.add_subplot(gs[0:plot_y,0])
    ax.plot(tvals-tvals[0], sound/sound.max())
    plt.xlim(0.0, to[-1])

    if corr_plot == None: 
        ax.plot(tvals-tvals[0], sound_env/sound_env.max(), color="red", linewidth=2)
    else:# plot the artifacts
        ax.plot(a_tvals-a_tvals[0], np.asarray(corr_loc), color="k", linewidth=2)
        ax.plot(a_lfp_tvals-a_lfp_tvals[0], np.asarray(corr_loc_lfp), color="c", linewidth=2)
        ax = fig.add_subplot(gs[plot_y:plot_y + 10,0])
        plot_y += 10
        ax.plot(tvals-tvals[0], sound/sound.max())
        ax.plot(a_tvals-a_tvals[0], np.asarray(corr_glob), color="k", linewidth=2)
        ax.plot(a_lfp_tvals-a_lfp_tvals[0], np.asarray(corr_glob_lfp), color="c", linewidth=2)

    plt.title('%s %d' % (seg.name, seg.index))
    plt.xlim(0.0, to[-1])
    ax = fig.add_subplot(gs[plot_y:plot_y + 10,0])
    plot_y += 10
    plot_spectrogram(to, fo, spect, ax=ax,
        ticks=True, fmin=250, fmax=8000, colormap=None, colorbar=False, log = True, dBNoise = 50)
    plt.ylabel('Frequency')
    plt.tick_params(labelbottom='off')
    plt.xlim(0.0, to[-1])
    y_size = (100 - plot_y) / (len(ephys_plot) * 2)
    amp = seg.analogsignals[1]
    filt_amp = seg.analogsignals[7]
    fs = amp.sampling_rate
    tvals = amp.times
    for iplt in ephys_plot:
        ax = fig.add_subplot(gs[plot_y:plot_y+y_size,0])
        plot_y = plot_y + y_size
        plt.plot(tvals-tvals[0], amp[:,iplt])
        plt.xlim(0.0, to[-1])
        ax = fig.add_subplot(gs[plot_y:plot_y+y_size,0])
        plot_y = plot_y + y_size
        plt.plot(tvals-tvals[0], filt_amp[:,iplt])
        plt.xlim(0.0, to[-1])

        plt.xlabel('Time (s)')
    plt.show()
    return sound, fs
    
    
def windowFilteredPlot(fig, seg, corr_plot=None, ephys_plot = [0]):
    
    # fig pointer to figure
    # seg is the segment
    # if corr_plot isn't None it will plot correlation vectors, optional
    # ephys_plot should be an array of integers corresponding to up to 8 amplifier channels you want to plot, defaults to the first channel
    # returns the filtered sound from the loudest of the two signals.
    
    # clear figure
    fig.clear()

    if corr_plot == 0: #intuitive user commands
        corr_plot = None
        
    # The sound signal
    soundSnip = seg.analogsignals[0]
    fs = soundSnip.sampling_rate
    tvals = soundSnip.times
    
    # The Artifact Vector (Correlation Vector)
    if corr_plot != None:
        corr_glob = seg.analogsignals[3]
        corr_loc = seg.analogsignals[4]
        corr_glob_lfp = seg.analogsignals[5]
        corr_loc_lfp = seg.analogsignals[6]        
#        artifact = seg.analogsignals[corr_plot]
        a_tvals = corr_glob.times
    
    # Calculate the rms of each mic
    rms0 = np.std(soundSnip[:, 0])
    rms1 = np.std(soundSnip[:, 1])
            
    # Choose the loudest to plot
    if rms1 > rms0:
        sound = np.asarray(soundSnip[:, 1]).squeeze()
    else:
        sound = np.asarray(soundSnip[:, 0]).squeeze()
            
    # Calculate envelope and spectrogram
    sound = sound - sound.mean()
    sound_env = lowpass_filter(np.abs(sound), float(fs), 250.0)  # Sound Enveloppe
    to, fo, spect, rms = spectrogram(sound, float(fs), 1000, 50)
           
    # Plot sonogram and spectrogram
    gs = gridspec.GridSpec(100, 1)
    plot_y = 10
    ax = fig.add_subplot(gs[0:plot_y,0])
    ax.plot(tvals-tvals[0], sound/sound.max())
    plt.xlim(0.0, to[-1])

    if corr_plot == None: 
        ax.plot(tvals-tvals[0], sound_env/sound_env.max(), color="red", linewidth=2)
    else:# plot the artifacts
        ax.plot(a_tvals-tvals[0], np.asarray(corr_loc), color="k", linewidth=2)
        ax.plot(a_tvals-tvals[0], np.asarray(corr_loc_lfp), color="c", linewidth=2)
        ax = fig.add_subplot(gs[plot_y:plot_y + 10,0])
        plot_y += 10
        ax.plot(tvals-tvals[0], sound/sound.max())
        ax.plot(a_tvals-tvals[0], np.asarray(corr_glob), color="k", linewidth=2)
        ax.plot(a_tvals-tvals[0], np.asarray(corr_glob_lfp), color="c", linewidth=2)

    plt.title('%s %d' % (seg.name, seg.index))
    plt.xlim(0.0, to[-1])
    ax = fig.add_subplot(gs[plot_y:plot_y + 10,0])
    plot_y += 10
    plot_spectrogram(to, fo, spect, ax=ax,
        ticks=True, fmin=250, fmax=8000, colormap=None, colorbar=False, log = True, dBNoise = 50)
    plt.ylabel('Frequency')
    plt.tick_params(labelbottom='off')
    plt.xlim(0.0, to[-1])
    y_size = (100 - plot_y) / (len(ephys_plot) * 2)
    amp = seg.analogsignals[7]
    filt_amp = seg.analogsignals[8]
    fs = amp.sampling_rate
    tvals = amp.times
    for iplt in ephys_plot:
        ax = fig.add_subplot(gs[plot_y:plot_y+y_size,0])
        plot_y = plot_y + y_size
        plt.plot(tvals-tvals[0], amp[:,iplt])
        plt.xlim(0.0, to[-1])
        ax = fig.add_subplot(gs[plot_y:plot_y+y_size,0])
        plot_y = plot_y + y_size
        plt.plot(tvals-tvals[0], filt_amp[:,iplt])
        plt.xlim(0.0, to[-1])

        plt.xlabel('Time (s)')
    plt.show()
    return sound, fs