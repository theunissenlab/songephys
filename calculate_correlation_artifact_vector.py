# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:20:29 2017

@author: billewood
Also highpass and lowpass filters the neural data and saves that if it wasn't already done.
Should be done on all Intan generated pickle files, I believe.

Calculates teh corr coeff over the highpassed and lowpassed data (separately), and calculates it using both a local variance and a global variance,
meaning it produces 4 new analogsignal arrays, in addition to the highpassed and lowpassed signals, for potentially 6 new analogsignal arrays.

So, in summary, creates 4 vectors which are the squared correlation coefficient averaged for all pairs of electrodes over time. Default corr_window is 250 data points, 
which is 10 ms at our 25kHz sampling rate- 2 of the vectors are calculated on LFP data so the sampling window on those is 25 data points.

Also deletes the unfiltered amplifier analogsignals to save space
"""
import os
import numpy as np
import quantities as pq
try:
   import cPickle as pickle
except:
   import pickle
from neo.core import AnalogSignal
from lasp.signal import lowpass_filter, highpass_filter, resample_signal # resample_signal is super slow and slows down this whole program a ton
import matplotlib.pyplot as plt
from NIXandPickleFunctions import plotSoundSeg

               
# Specify the input file again. The segemented file is a pickle file that has s single block but
# multiple labelled segements corresponding to sections with song.  It was created with audiosegmentnix.py (in songephys).
high_thresh = 300.0
low_thresh = 200.0
corr_window = 250
corr_lfp_window = 25
experiment_dir = "/auto/tdrive/billewood/intan data/LbY6074"
pkl_file_in = "LbY6074__161215_104204_seg_c1.pkl"               
# Open the Inupt Pkl file 
pklFileIn = open(os.path.join(experiment_dir, 'NIX', pkl_file_in), 'rb')
# Read the data
block = pickle.load(pklFileIn)
pklFileIn.close()

# Print and display information on epochs, check for highpass and lowpass filtered signals, if not present create them.
print 'Found', len(block.segments), 'segments:'
nstudent = 0
ntutor = 0
for i,seg in enumerate(block.segments):
    if seg.name.startswith('student'):
        nstudent +=1
    elif seg.name.startswith('tutor'):
        ntutor += 1
print '\t', nstudent, 'student songs and', ntutor, 'tutor songs.'
print '\t Segments appear to have ', len(seg.analogsignals), ' analog signals' 

print '\t Creating high passed (%dHz) and low passed (%dHZ) analog signals' %(high_thresh, low_thresh)
# Highpass filter and store as a new analogsignal array on every segment 
# Lowpass filter and store as a new " " "                          
for seg in block.segments:
    channel = seg.analogsignals[0]
    high_channel = list()
    low_channel = list()    
    for elec in range(seg.analogsignals[0].shape[1]): # number of electrodes
        h_filtered = np.float32(highpass_filter(np.asarray(channel)[:,elec], np.round(float(channel.sampling_rate)), high_thresh))
        high_channel.append(h_filtered)
        l_filtered = np.float32(lowpass_filter(np.asarray(channel)[:,elec], np.round(float(channel.sampling_rate)), low_thresh))
        t_rs, l_filtered = resample_signal(l_filtered, round(channel.sampling_rate), 2*low_thresh)    # Note this is resample from LASP, not scipy (although they are very related)
        low_channel.append(l_filtered)
    high_channel_T = map(list,map(None,*high_channel)) #transpose channels to make AnalogSignal call work right, which seems intuitively backwards to me
    high_sigs = AnalogSignal(signal = high_channel_T, name = 'Highpass Amplifier', t_start = channel.t_start, units = channel.units, sampling_rate=channel.sampling_rate, description = "Highpass filtered at %dHz" %high_thresh)    
    seg.analogsignals.append(high_sigs)
    low_channel_T = map(list,map(None,*low_channel)) #transpose low channels too
    low_sigs = AnalogSignal(signal = low_channel_T, name = 'Lowpass Amplifier', t_start = channel.t_start, units = channel.units, sampling_rate=2*low_thresh*pq.Hz, description = 'Lowpass filtered at %dHz' %low_thresh)    
    seg.analogsignals.append(low_sigs)
del high_channel_T, high_sigs, low_sigs, high_channel, low_channel, l_filtered, h_filtered

    
#Calculate the global variance of each electrode (variance over all segments). Do for highpassed and lowpassed data.
# To be used in calculations of correlation coefficients later.
global_vari_high = np.zeros(seg.analogsignals[0].shape[1]) #number of electrodes
global_vari_low = np.zeros(seg.analogsignals[0].shape[1])
flatten = lambda l: [item for sublist in l for item in sublist]
for elec in range(seg.analogsignals[0].shape[1]):
    full_length_amp_high = list()
    full_length_amp_low = list()
    for seg in block.segments:
        channel = seg.analogsignals[2]
        full_length_amp_high += [np.asarray(channel[elec])] 
        channel = seg.analogsignals[3]
        full_length_amp_low += [np.asarray(channel[elec])] 
    global_vari_high[elec] = np.var(flatten(full_length_amp_high)) #calculate variance
    global_vari_low[elec] = np.var(flatten(full_length_amp_low))
del full_length_amp_high, full_length_amp_low

# Now compute a correlation number, which aims to represent how correlated the channels are with each other at moments in time, which
# shoudl represent periods with large movement artifacts.
# let x and y be two channels, for each time window: sum( (x(t)-mean(x)) * (y(t)-mean(y))]**2 and divide by overall var of channel x and y multiplied
# which is the correlation coefficient squared 
# then average for each pair of channels, giving one number per window
# Do this for each segment using linear algebra b/c it's 1000x faster than loops even though it's tough to read
# Do this with the highpassed data:               
print '\n \t Creating squared correlation coefficient vectors (x4)'                                      
corr_coeff_squared = list()
corr_coeff_squared_localvar = list()
for seg in block.segments:
    channel = seg.analogsignals[2] # the highpassed channel
    global_corr = list()
    local_corr = list()
    j = 0
    t = 0
    while t < len(channel) - corr_window:  
        C = np.asarray(channel[t:t+corr_window])
        C -= np.mean(C, axis=0)
        local_vari = np.var(C, axis=0)
        local_vari = np.outer(local_vari,local_vari)
        VAR = np.outer(global_vari_high,global_vari_high) #product of all the variances 16 x 16 
        temp_global_corr = np.sum(C[:, :, None] * C[:, None, :], axis=0) ** 2 / (VAR*((corr_window-1)**2))
        temp_local_corr = np.sum(C[:, :, None] * C[:, None, :], axis=0) ** 2 / (local_vari*((corr_window-1)**2))
        np.fill_diagonal(temp_global_corr,0)
        np.fill_diagonal(temp_local_corr,0)
        global_corr.append(np.mean(temp_global_corr))
        local_corr.append(np.mean(temp_local_corr))
        t = t + corr_window
        j += 1
    corr_coeff_squared.append(global_corr)
    corr_coeff_squared_localvar.append(local_corr)
# Now save the Noise Artifacts Correlation singals as analogsignals.    
for i, seg in enumerate(block.segments):
    channel = seg.analogsignals[2]
    sigarr = AnalogSignal(np.array(corr_coeff_squared[i]), name = 'Square Corr Coeff over time, global variance', t_start = channel.t_start + (corr_window/channel.sampling_rate)/2, units = pq.dimensionless, sampling_rate=channel.sampling_rate / corr_window, description = 'Calculated from highpass filtered data, global variance')    
    seg.analogsignals.append(sigarr)
# Now save the Noise Artifacts Correlation singals as analogsignals.    
for i, seg in enumerate(block.segments):
    channel = seg.analogsignals[2]
    sigarr = AnalogSignal(np.array(corr_coeff_squared_localvar[i]), name = 'Square Corr Coeff over time, local variance', t_start = channel.t_start + (corr_window/channel.sampling_rate)/2, units = pq.dimensionless, sampling_rate=channel.sampling_rate / corr_window, description = 'Calculated from highpass filtered data, local variance')    
    seg.analogsignals.append(sigarr)
# Now delete the orignal unfiltered data
for i, seg in enumerate(block.segments):
    seg.analogsignals.remove(seg.analogsignals[0])


# And now just do it all again with the lowpassed data:

corr_coeff_squared = list()
corr_coeff_squared_localvar = list()
for seg in block.segments:
    channel = seg.analogsignals[2] # the lowpassed channel- note a channel has been deleted so the index is still 2
    global_corr = list()
    local_corr = list()
    j = 0
    t = 0
    while t < len(channel) - corr_lfp_window:  
        C = np.asarray(channel[t:t+corr_lfp_window])
        C -= np.mean(C, axis=0)
        local_vari = np.var(C, axis=0)
        local_vari = np.outer(local_vari,local_vari)
        VAR = np.outer(global_vari_low,global_vari_low) #product of all the variances 16 x 16 
        temp_global_corr = np.sum(C[:, :, None] * C[:, None, :], axis=0) ** 2 / (VAR*((corr_lfp_window-1)**2))
        temp_local_corr = np.sum(C[:, :, None] * C[:, None, :], axis=0) ** 2 / (local_vari*((corr_lfp_window-1)**2))
        np.fill_diagonal(temp_global_corr,0)
        np.fill_diagonal(temp_local_corr,0)
        global_corr.append(np.mean(temp_global_corr))
        local_corr.append(np.mean(temp_local_corr))
        t = t + corr_lfp_window
        j += 1
    corr_coeff_squared.append(global_corr)
    corr_coeff_squared_localvar.append(local_corr)
# Now save the Noise Artifacts Correlation singals as analogsignals.    
for i, seg in enumerate(block.segments):
    channel = seg.analogsignals[2]
    sigarr = AnalogSignal(np.array(corr_coeff_squared[i]), name = 'Square Corr Coeff over time, global variance, lfp', t_start = channel.t_start + (corr_window/channel.sampling_rate)/2, units = pq.dimensionless, sampling_rate=channel.sampling_rate / corr_window, description = 'Calculated from lowpass filtered data')    
    seg.analogsignals.append(sigarr)
# Now save the Noise Artifacts Correlation singals as analogsignals.    
for i, seg in enumerate(block.segments):
    channel = seg.analogsignals[2]
    sigarr = AnalogSignal(np.array(corr_coeff_squared_localvar[i]), name = 'Square Corr Coeff over time, local variance, lfp', t_start = channel.t_start + (corr_window/channel.sampling_rate)/2, units = pq.dimensionless, sampling_rate=channel.sampling_rate / corr_window, description = 'Calculated from lowpass filtered data')    
    seg.analogsignals.append(sigarr)

print 'Done!'
############################################3
# For plotting, not really part of this code, but useful for verification.

seg = block.segments[0]
fig = plt.figure()
plotSoundSeg(fig, seg, corr_plot = 1, ephys_plot = [0,1,2,3,4,5,6])


#For Saving, will overwrite the pickle file!!!
pkl_file_out = pkl_file_in[0:-4] + '_corr.pkl'
pklFileOut = open(os.path.join(experiment_dir, 'NIX', pkl_file_out), 'wb')
# Write the data
pickle.dump(block, pklFileOut)

pklFileOut.close()
## This is like the above matrix algebra based loop, but potentially easier to read but muuuuch slower.
## just leaving it for reference as personally it's quite hard for me to make sense of the linear algebra above.
#                                                      
#new_sigs = list()
#corr_window = 250
#for seg in block.segments[:1]:
#    seg_start_time = time.time()    
#    channel = seg.analogsignals[0]
#    simple_corr = list()
#    j = 0
#    t = 0
#    while t < len(channel) - corr_window:  
#        temp_corr = list()
##        start = time.time()
#        y_win_mean = {}
#        for x in range(channel.shape[1]):
#            win_mean = np.mean(channel[t:t+corr_window,x])
#            for y in range(x+1, channel.shape[1]):
#                if y not in y_win_mean:
#                    y_win_mean[y] = np.mean(channel[t:t+corr_window,y])
#                temp_corr.append((np.sum( (channel[t:t+corr_window,x]-win_mean) *  (channel[t:t+corr_window,y]-y_win_mean[y])) ** 2)/(vari[x]* vari[y]))
##        print time.time() - start
#        simple_corr.append(np.mean(temp_corr))
#        t = t + corr_window
#        j += 1
#    new_sigs.append(simple_corr/np.mean(simple_corr))
#    print "seg time", time.time() - seg_start_time
#    #save the scaled_corr as a new analog signal with fs in each segment
#print 'Making new analog signal in segment: ', seg.name
#
#

#
#
#simple_corr = list()
#corr_window = 250
#j = 0
#i = 0
#while i < high_amp.shape[0]-corr_window: 
#    temp_corr = list()
#    for x in range(high_amp.shape[1]):
#        for y in range(high_amp.shape[1]):
#            temp_corr.append((np.sum( (high_amp[i:i+corr_window,x]-np.mean(high_amp[i:i+corr_window,x])) *  (high_amp[i:i+corr_window,y]-np.mean(high_amp[i:i+corr_window,y]))) ** 2)/vari[x]* vari[y])
#    simple_corr.append(np.mean(temp_corr))
#    i = i+corr_window
#    j += 1
#    
#scaled_corr = simple_corr/np.mean(simple_corr)
