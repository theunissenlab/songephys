sys.path.append('/Users/frederictheunissen/Documents/Code/songephys')
sys.path.append('/Users/frederictheunissen/Documents/Code/LaSP')
sys.path.append('/Users/frederictheunissen/Documents/Code/zeebeez')
sys.path.append('/Users/frederictheunissen/Documents/Code/neosound')
sys.path.append('/Users/frederictheunissen/Documents/Code/theunissen-neo')
sys.path.append('/Users/frederictheunissen/Documents/Code/load_intan_rhd_format')

import os
import sys
import numpy as np
import quantities as pq



from neo.core import Epoch, Segment
from neo.io import RHDIO, NixIO
import autodetect


# Adds a little time before and after each 
tbefore = 0.2*pq.s # Time before begining of event
tafter = 0.2*pq.s # Time after end of even
      
# Check to see if input Nix file exists.
# nixFileIn = sys.argv[1]

nixFileIn = "/Users/frederictheunissen/Documents/Data/Chronic V2/LbY6074/NIX/LbY6074__161215_145633.nix"
if not(os.path.isfile(nixFileIn)):
    print 'Error: Input NIX file,', nixFileIn, ', does not exist'
    sys.exit()
    
# Check to see if output NIX file exists.
# nixFileOut = sys.argv[2]
nixFileOut = "/Users/frederictheunissen/Documents/Data/Chronic V2/LbY6074/NIX/LbY6074__161215_145633_seg.nix"
if os.path.isfile(nixFileOut):
   print 'Error: Output NIX file,', nixFileOut, ',exists'
   print 'Delete and run again if you want to overwrite'
   sys.exit()
    

# Open the input nix file.  
nixInput = NixIO(nixFileIn)

# We are reading the entire block and all the data since lazy reading does not work
# This next line is slow
block = nixInput.read_block()
nixInput.close()

# Our data only have one segment and the second analog signal is the audio from the two microphones. 
# The first is the 16 channel neural data
fullSeg = block.segments[0]
sound_recording = fullSeg.analogsignals[1]
norm_sound = sound_recording / np.max(sound_recording)

# Find periods of vocal output
vocal_periods = autodetect.find_vocal_periods(norm_sound[:, 0])

# Find periods of song withing vocal periods
birdsong_points = autodetect.classify_birdsong(norm_sound[:, 0], vocal_periods)
song_points = vocal_periods[birdsong_points == 'birdsong']

# Classify songs into tutor/tutee based on amplitude.
student_teacher_dict = autodetect.separate_song_sources(norm_sound, song_points)


# Loop through all epochs
for k in ['student', 'tutor']:
    intervals = student_teacher_dict[k]
    for i,timevals in enumerate(intervals):
        newSeg = Segment(name='%s_song_%d' % (k,i), description=k+'_song', file_origin=fullSeg.file_origin,
                 file_datetime=fullSeg.file_datetime, rec_datetime=fullSeg.rec_datetime, index=i)
        newSeg.annotate(**fullSeg.annotations)
        
        # Take a time slice of a segment
        start = timevals[0]
        duration = timevals[1] - timevals[0]
        for sig in fullSeg.analogsignals:
            startTime = start/sig.sampling_rate
            durationTime = duration/sig.sampling_rate
            sigslice = sig.time_slice(startTime-tbefore, startTime + durationTime + tafter)
            newSeg.analogsignals.append(sigslice)
        newSeg.create_many_to_one_relationship()
        block.segments.append(newSeg)
        block.create_many_to_one_relationship(force=True, recursive=True)

# Deleting the first full segment
del block.segments[0]
block.create_many_to_one_relationship(force=True, recursive=True)

# Open the Output nix file.  
nixOutput = NixIO(nixFileOut)

# Write out the new blowk with all the segments
nixOutput.write_block(block)
nixOutput.close()

