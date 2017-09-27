# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:44:17 2017

Script to segment nix files into sections where there has been automatic recognition of song or calls. 
Designed primarily as the first step in the Intan analysis pipeline, creating many segments from 
our Nix files and storing them in Neo format HDF5.
It also creates a few files for visualizing how the sorting went.

!!!! Currently only working for live song, need to update for auditory playback

Segments should have:

analog signals -> 16
epochs -> syllables? events? Not functional yet
spiketrain -> empty for now
index -> ascending
name -> SoundType_Vocalizer_Index_
desciption -> Created by thisfile.py + hashcode of this file
file_origin -> original rhd file, or nix maybe
sound_type -> Playback, Live; Silence
Vocalization_Type: So, Te, DC, UV (Song, Tet, Distance Call, Unknown Vocalization)
Vocalizer -> Self; Tutor; Mom; Sister; Con_Male; Con_Female
Playback_File -> 


Putative vocalization periods are found with autodetect.find_vocal_periods (room to improve here)
Sorted into songs, tets, and distance calls with a random forest classifier. The classifier is computed every time currently.

@author: billewood


"""
import os
import sys
import numpy as np
import quantities as pq
try:
   import cPickle as pickle
except:
   import pickle

from neo.core import Segment
from neo.io import NixIO
from segmentation import autodetect
import serialize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import hashlib



# Adds a little time before and after each !!!!!!!!!!!!!!!!!!1 Thsi is in progress
tbefore = 0.2*pq.s # Time before begining of event
tafter = 0.2*pq.s # Time after end of even
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Specify the Nix input file and two directories and the pickle file out. classifier dataset optional

nix_file = sys.argv[1] # nix_file = "/auto/tdrive/billewood/intan data/LbY6074/NIX/LbY6074__161216_104808.nix"
experiment_dir = sys.argv[2] # experiment_dir = "/auto/tdrive/billewood/intan data/LbY6074"
spikes_vis_core_dir = sys.argv[3] # spikes_vis_core_dir = "/auto/fhome/billewood/Code/spikes-vis/data/vocalizations"
pickle_file_out = sys.argv[4] # pickle_file_out = "/auto/tdrive/billewood/intan data/LbY6074/104808"
if sys.argv[5] :
    classifier_dataset = sys.argv[5]
else:
    classifier_dataset = ""    

if not(os.path.isfile(nix_file)):
    print 'Error: Input NIX file,', nix_file, ', does not exist'
    sys.exit()
    
if os.path.isdir(experiment_dir):
    print "Segmented data file will be stored in : " + experiment_dir
else:
    os.mkdir(experiment_dir)
    print "Segmented data file will be stored in : " + experiment_dir
    print "(directory made)" 

spikes_vis_dir = spikes_vis_core_dir + "/" + nix_file[:-4]    
if os.path.isdir(spikes_vis_dir):
    print "Spikes_vis will be stored in existing directory: " + spikes_vis_dir
else:
    os.mkdir(spikes_vis_dir)
    print "Spikes_vis files will be stored in " + spikes_vis_dir
    print "(directory made)" 

if os.path.isfile(pickle_file_out):
   print 'Error: Output pickle file,', pickle_file_out, ',exists'
   print 'Delete and run again if you want to overwrite'
   sys.exit()

save_df_name = os.path.join(experiment_dir, nix_file[:-4] + 'sound_df.pkl')

# Get the audio data- no need to load all the analog signals now as that's likely to cause more
# memory issues than it will save in loading time    
print "Loading audio data", nix_file
audio_streams = serialize.read_audio_from_nix(experiment_dir, nix_file)
print "Done."

# Find periods of vocal output, fuse_duration of .2 seems to work quite well for now.
vocal_periods, amp_thresh = autodetect.find_vocal_periods(audio_streams[:, 0], fuse_duration=0.2)
print "Found ", len(vocal_periods), " vocal periods"

#WARNING THIS STEP TAKES SOME HOURS!!!!!!!!!!!!!!!
print "Beginning sound analysis- lengthy process! Expect some hours to complete..."
df, spectro_list = autodetect.populate_dataframe(audio_streams, vocal_periods, fs=25000)
print "FINISHED."

print "Classifying Vocalizations."
df, predictions = autodetect.classify_vocalizations_random_forest(df, classifier_dataset)

# For the moment I discard BOS classifications- this could be optimized as we begin working on multiple birds but it's possible
# the mic amplitude step makes this obselete anyway.
# TODO
bos_trials = np.where(df['vocal_category'] == 'BOS')
df['vocal_category'][bos_trials[0]] = 'So'

# Save the Dataframe, 
print("Saving classified dataframe: ", os.path.join(experiment_dir, nix_file[:-4] + '.pkl'))

with open(os.path.join(experiment_dir, nix_file[:-4] + '.pkl'), "wb") as output_file:
    pickle.dump(df, output_file)

print("Saving spectrograms for spikes-vis: ", os.path.join(spikes_vis_dir , 'spectrograms.pkl'))
with open(os.path.join(spikes_vis_dir , 'spectrograms.pkl'),"wb") as output_file:
    pickle.dump(spectro_list, output_file)

print "Creating reduced dimension PCA"
PCAd_array = autodetect.reduce_dimensions_pca(df)
print "original df shape: ", df.shape
print "Reduced df shape: ", PCAd_array.shape
   
print "Creating t-sne visualization of reduced dimension dataframe with clusters color coded"
print("Saving to :", os.path.join(spikes_vis_dir, '2d.pkl'))
print "Note: run spikes-vis to really see how this worked"

#pca_array = np.concatenate((np.array(df[zscore_columns]), pca_array), axis = 1)
tsne = TSNE(n_components=2, perplexity = 50, n_iter_without_progress = 100)
Y = tsne.fit_transform(PCAd_array[PCAd_array.columns[:-1]])
plt.figure(figsize=(8, 8))
plt.scatter(Y.T[0], Y.T[1])

tsne_name = os.path.join(spikes_vis_dir, '2d.pkl') 
with open(tsne_name, "wb") as output_file:
    pickle.dump(Y, output_file)

# this is for spikes-vis, not crucial to the pipeline but still nice
clusters = np.asarray(df['vocal_category'].cat.codes + 1, dtype = int)
cluster_name = os.path.join(spikes_vis_dir, 'clusters.pkl') 
with open(cluster_name, "wb") as output_file:
    pickle.dump(clusters, output_file)


print 'Making new segments...Loading whole data file...'
del PCAd_array
del Y
del tsne


# Open the input nix file.  
nixInput = NixIO(nix_file)

# We are reading the entire block and all the data since lazy reading does not work
# This next line is slow
print 'Reading input file ', nix_file, ', will probably take a couple minutes...'
block = nixInput.read_block()
nixInput.close()
print 'Done.'
fullSeg = block.segments[0]


# Loop through channels and delete links to analog signals
for ci in block.channel_indexes:
    if ci.name == 'Amplifier':
        ci.analogsignals = []       
    if ci.name == 'ADC':
        ci.analogsignals = []

#analog signals -> 16
#epochs -> syllables? events? Not functional yet
#spiketrain -> empty for now

#name -> SoundType_Vocalizer_Index_
#desciption -> Created by thisfile.py + hashcode of this file
#file_origin -> original rhd file, or nix maybe
#index -> ascending
#sound_type -> Playback, Live; Silence
#Vocalization_Type: So, Te, DC, UV (Song, Tet, Distance Call, Unknown Vocalization)
#Vocalizer -> Self; Tutor; Mom; Sister; Con_Male; Con_Female
#Playback_File -> 


"""
hashtag = hashlib.md5(os(__file__))
ind = 0 #ascending index of segments
good_vocs = ['DC', 'So', 'Te']
for row in df.shape[0]:
    if df['vocal_category'][row] in good_vocs: 
        newSeg = Segment(name='%s_%s_%d' % (df['vocal_category'][row], df['vocalizer'][row], ind), 
                         description = 'Created by %s, hashtag: %s' % (os.path.basename(__file__), hashtag),
                         file_origin = nix_file, 
                         index = ind,
                         sound_type = 'Only live sounds being processed currently, stay tuned',
                         vocalization_type = df['vocal_category'][row],
                         vocalizer = df['vocalizer'][row],
                         #file_datetime=fullSeg.file_datetime, 
                         #rec_datetime=fullSeg.rec_datetime,      
                         playback_file = 'work in progress',                         
                         )


        newSeg.annotate(**fullSeg.annotations)
        ind += 1
        
        # Take a time slice of a segment
        start = timevals[0]
        duration = timevals[1] - timevals[0]
        
        for sig in fullSeg.analogsignals:
            startTime = start/sig.sampling_rate
            durationTime = duration/sig.sampling_rate
            sigslice = sig.time_slice(startTime-tbefore, startTime + durationTime + tafter)
            sigslice.name = '%s_%d' % (sigslice.name, j)
            # The code below is clunky but the names are a bit different - something to fix in rhdio and then here
            if sigslice.name.startswith('Amplifier') :
                for ci in block.channel_indexes:
                    if ci.name == 'Amplifier':
                        ci.analogsignals.append(sigslice)
            elif sigslice.name.startswith('Board ADC'):
                for ci in block.channel_indexes:
                    if ci.name == 'ADC':
                        ci.analogsignals.append(sigslice)
            newSeg.analogsignals.append(sigslice)
            
        newSeg.create_many_to_one_relationship()
        block.segments.append(newSeg)

# Deleting the first full segment
del block.segments[0]
block.create_many_to_one_relationship(force=True, recursive=True)
block.create_many_to_many_relationship()
print 'Done.'
  
# Our data only have one segment and the second analog signal is the audio from the two microphones. 
# The first is the 16 channel neural data
fullSeg = block.segments[0]
sound_recording = fullSeg.analogsignals[1]
norm_sound = sound_recording / np.max(sound_recording)
"""