import cPickle as pickle
from lasp.signal import lowpass_filter, highpass_filter

from neo.io import NixIO
import numpy as np


class SongSamples:
    """
    SongSamples is a class used as an in-memory container for birdsong data processed
    by the analysis pipeline in this package. It is used because the RHD/NIX formats
    use an enormous amount of memory, and aren't suitable for quick analysis based on
    processed features. To use, simply instantiate, and then call read_from_nix()
    on the nix file of your choice. You can then access the various attributes
    as needed.
    All data will be buffered by a number of points self.buffer, so be aware of that.
    """
    def __init__(self):
        self.audio_windows = []
        self.amplitude_windows = []
        self.ephys_windows = []
        self.labels = []
        self.buffer = 5000

    def read_from_nix(self, fname):
        exporter = NixIO(fname)
        print "reading info from: " + str(fname)
        block = exporter.read_block()
        seg = block.segments[0]
        fs = seg.analogsignals[1].sampling_rate
        audio_streams = seg.analogsignals[1]
        streams_filtered_env = np.zeros(np.shape(audio_streams))
        streams_filtered_env[:, 0] = highpass_filter(np.squeeze(audio_streams[:, 0]), fs, 2000)
        streams_filtered_env[:, 1] = highpass_filter(np.squeeze(audio_streams[:, 1]), fs, 2000)
        streams_filtered_env[:, 0] = np.abs(lowpass_filter(streams_filtered_env[:, 0],
                                                           fs, 5000))
        streams_filtered_env[:, 1] = np.abs(lowpass_filter(streams_filtered_env[:, 1],
                                                           fs, 5000))
        streams_filtered_env[:, 0] = lowpass_filter(streams_filtered_env[:, 0], fs, 100)
        streams_filtered_env[:, 1] = lowpass_filter(streams_filtered_env[:, 1], fs, 100)

        for i in xrange(len(seg.epochs)):
            starts = map(int, seg.epochs[i].times * fs)
<<<<<<< Updated upstream
            ends = map(int, seg.epochs[i].durations * fs)
            periods = map(lambda se: (se[0], se[0]+se[1]), zip(starts, ends))
=======
            ends = starts + map(int, seg.epochs[i].durations * fs)
            periods = zip(starts, ends)
>>>>>>> Stashed changes

            for l, h in periods:
                lb = max(l - self.buffer, 0)
                hb = min(h + self.buffer, np.shape(streams_filtered_env)[0])
#                print "audio indices:"
#                print np.shape(streams_filtered_env)[0]
#                print "l: " + str(l) + ", h: " + str(h) + ", lb: " + str(lb) + ", hb: " + str(hb)
                self.amplitude_windows.append(np.copy(streams_filtered_env[lb:hb, :]))
                self.audio_windows.append(np.copy(audio_streams[lb:hb, :]))
                self.labels.append(seg.epochs[i].name)
                self.ephys_windows.append(seg.analogsignals[0][lb:hb, :])
        print "...read " + str(len(seg.epochs)) + " epochs"
        exporter.close()
        del audio_streams
        del streams_filtered_env

    def save(self, fname):
        pickle.dump(self, open(fname, "wb"))
        
        