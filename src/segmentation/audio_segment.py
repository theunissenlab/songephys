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
from config import config


# Adds a little time before and after each 
tbefore = 0.2 * pq.s # Time before begining of event
tafter = 0.2 * pq.s # Time after end of even
      

def segment(block):
    """Convert a block of ephys data into segments of singing

    The data starts with one segment and two analog signals:
        * 16 channel neural data
        * 2 channel microphone data
    """
    fullSeg = block.segments[0]
    sound_recording = fullSeg.analogsignals[1]
    norm_sound = sound_recording / np.max(sound_recording)

    # Find periods of vocal output
    # Modify so that it does not include playback...

    sound_recording = fullSeg.analogsignals[1]
    norm_sound = sound_recording / np.max(sound_recording)
    vocal_periods, _ = autodetect.find_vocal_periods(norm_sound[:, 0], fuse_duration=0.2)
    vocal_periods = vocal_periods[1:]
    # FIXME: negative index possible on first one

    # Find periods of song withing vocal periods
    birdsong_points = autodetect.classify_birdsong(
            norm_sound[:, 0],
            vocal_periods,
            model=config.SONG_CLASSIFIER_PARAMS)
    song_points = vocal_periods[birdsong_points == 'birdsong']

    # Classify songs into tutor/tutee based on amplitude.
    student_teacher_dict = autodetect.separate_song_sources(norm_sound, song_points)[0]

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
        for i, timevals in enumerate(intervals):
            newSeg = Segment(
                    name='%s_song_%d' % (k,i),
                    description=k+'_song',
                    file_origin=fullSeg.file_origin,
                    file_datetime=fullSeg.file_datetime,
                    rec_datetime=fullSeg.rec_datetime,
                    index=j)
            newSeg.annotate(**fullSeg.annotations)
            j += 1
            
            # Take a time slice of a segment
            start = timevals[0]
            duration = timevals[1] - timevals[0]
            for sig in fullSeg.analogsignals:
                if sig.name not in ["Amplifier", "Board ADC"]:
                    continue

                startTime = start / sig.sampling_rate
                durationTime = duration/sig.sampling_rate

                print "startTime", startTime
                print "tbefore", config.PAD_SONG_BEFORE
                print "durationTime", durationTime
                print "tafter", config.PAD_SONG_AFTER
                print "sig", sig
                print "sigshape", sig.shape
                print "sample rate", sig.sampling_rate

                sigslice = sig.time_slice(
                        startTime - config.PAD_SONG_BEFORE * pq.s,
                        startTime + durationTime + config.PAD_SONG_AFTER * pq.s)
                sigslice.name = '%s_%d' % (sigslice.name, j)
                # The code below is clunky but the names are a bit different -
                # something to fix in rhdio and then here
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

    return block
