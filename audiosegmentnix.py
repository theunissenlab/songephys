# Script to segment nix files into sections where there has been automatic recognition of song

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
import autodetect


# Adds a little time before and after each 
tbefore = 0.2*pq.s # Time before begining of event
tafter = 0.2*pq.s # Time after end of even
      
# Check to see if input Nix file exists.

nixFileIn = sys.argv[1]
# nixFileIn = "/Users/frederictheunissen/Documents/Data/Chronic V2/LbY6074/NIX/LbY6074__161215_145633.nix"
if not(os.path.isfile(nixFileIn)):
    print 'Error: Input NIX file,', nixFileIn, ', does not exist'
    sys.exit()
    
# Check to see if output picle file exists.
pklFileOut = sys.argv[2]
# pklFileOut = "/Users/frederictheunissen/Documents/Data/Chronic V2/LbY6074/NIX/LbY6074__161215_145633_seg.pkl"
if os.path.isfile(pklFileOut):
   print 'Error: Output pickle file,', pklFileOut, ',exists'
   print 'Delete and run again if you want to overwrite'
   sys.exit()
    
# Open the input nix file.  
nixInput = NixIO(nixFileIn)

# We are reading the entire block and all the data since lazy reading does not work
# This next line is slow
print 'Reading input file', nixFileIn, '...'
block = nixInput.read_block()
nixInput.close()
print 'Done.'


# Our data only have one segment and the second analog signal is the audio from the two microphones. 
# The first is the 16 channel neural data
fullSeg = block.segments[0]
sound_recording = fullSeg.analogsignals[1]
norm_sound = sound_recording / np.max(sound_recording)

# Find periods of vocal output
print 'Detecting periods of vocalization ...'
vocal_periods = autodetect.find_vocal_periods(norm_sound[:, 0])

# Find periods of song withing vocal periods
print 'Detecting periods of song...'
birdsong_points = autodetect.classify_birdsong(norm_sound[:, 0], vocal_periods)
song_points = vocal_periods[birdsong_points == 'birdsong']

# Classify songs into tutor/tutee based on amplitude.
print 'Classifying songs into near and far (tutee and tutor)...'
student_teacher_dict = autodetect.separate_song_sources(norm_sound, song_points)
print 'Done.'

print 'Making new segments...'
# Loop through channels and delete links to analog signals
for ci in block.channel_indexes:
    if ci.name == 'Amplifier':
        ci.analogsignals = []       
    if ci.name == 'ADC':
        ci.analogsignals = []

# Loop through all epochs
j = 0   # j will  be used as an index for trials/segments
for k in ['student', 'tutor']:
    intervals = student_teacher_dict[k]
    for i,timevals in enumerate(intervals):
        newSeg = Segment(name='%s_song_%d' % (k,i), description=k+'_song', file_origin=fullSeg.file_origin,
                 file_datetime=fullSeg.file_datetime, rec_datetime=fullSeg.rec_datetime, index=j)
        newSeg.annotate(**fullSeg.annotations)
        j += 1
        
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

# Open the Output pkl file.  
print 'Writing segmented Pickle file', pklFileOut
pkloutput = open(pklFileOut, 'wb')
pickle.dump(block, pkloutput)
pkloutput.close()

# Write out the new blowk with all the segments to NIX - too slow
# nixOutput = NIXIO(nixFileOut)
# nixOutput.write_block(block)
# nixOutput.close()
print 'Done'

