import cPickle as pickle
from lasp.signal import lowpass_filter, highpass_filter

from neo.io import NixIO
import numpy as np


class SongSamples:
    def __init__(self):
        self.audio_windows = []
        self.amplitude_windows = []
        self.ephys_windows = []
        self.labels = []
        self.buffer = 25000

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
            
            ##  THIS CODE LOOKS WRONG?  ends shoudl be times+durations and it should be *fs?
            starts = map(int, seg.epochs[i].times / fs)
            ends = map(int, seg.epochs[i].durations / fs)
            periods = zip(starts, ends)

            for l, h in periods:
                lb = max(l - self.buffer, 0)
                hb = min(h + self.buffer, np.shape(streams_filtered_env)[0])
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
