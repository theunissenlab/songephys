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

    def read_from_nix(self, fname):
        exporter = NixIO(fname)
        block = exporter.read_block()
        seg = block.segments[0]
        for i in xrange(len(seg.epochs)):
            fs = seg.analogsignals[1].sampling_rate
            starts = map(int, seg.epochs[i].times / fs)
            ends = map(int, seg.epochs[i].durations / fs)
            periods = zip(starts, ends)

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
            for l, h in periods:
                self.amplitude_windows.append(streams_filtered_env[l:h, :])
                self.audio_windows.append(audio_streams[l:h, :])
                self.labels.append(seg.epochs[i].name)
                self.ephys_windows.append(seg.analogsignals[0][l:h, :])

    def save(self, fname):
        pickle.dump(self, open(fname, "wb"))
