import os

import matplotlib.pyplot as plt
import neo.core
import numpy as np
from scipy.io import wavfile

from soundsig.sound import spectrogram, plot_spectrogram

import config

from segment_analysis import MultiSegment
from spikes.utils import bin_spikes


class PlaybacksManager(object):
    """Helper class to deal with sessions involving sound playback

    Mainly used to get access to TrialsViewer objects
   
    Example:
    >>> pbm = PlaybacksManager(block.segments)
    >>> trials_viewer = pbm.get_playback_responses(<stim_id>)
    >>> trials_viewer.plot_psth(unit=1)
    """

    def __init__(self, segments):
        self.segments = filter(lambda seg: seg.annotations["sound_type"] == "playback", segments)
        # TODO fix this to access the file
        self.stim_ids = np.unique(list(self.get_id(seg) for seg in self.segments))

    def get_id(self, seg):
        """Return id of the playback file for the given segment"""
        # TODO: figure out if I should use the playback filename or stimulus id
        return seg.annotations["playback_file"].split("/")[-1]
        return seg["stim_id"]

    def filter_segments_by_stim_id(self, stim_id):
        """Return segments during which the specific playback file was played"""
        return filter(lambda seg: self.get_id(seg) == stim_id, self.segments)

    def get_playback_responses(self, stim_id):
        """Return TrialsViewer helper object for the given playback"""
        return MultiSegment(self.filter_segments_by_stim_id(stim_id))




class SegmentViewer(object):
    """Helper class for viewing Segment specific information, plots

    Features include:
    * Viewing the spectrogram of the playback corresponding to the segment
    * Viewing spectrogram of mic recordings during segment
    * Viewing all spike waveforms from all units from the segment
    * Binning spikes from all units during the segment
    
    Example:
    >>> seg_viewer = SegmentViewer(block.segments[0])
    >>> fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
    >>> seg_viewer.plot_spectrogram(source=0, ax=ax1)
    >>> seg_viewer.plot_spectrogram(source=1, ax=ax2)
    >>> seg_viewer.plot_spectrogram(source="playback", ax=ax2)
    >>> plt.show()
    """

    def __init__(self, segment):
        self.seg = segment

        if self.seg.annotations["sound_type"] == "playback":
            self.playback_rate, self.playback_sound = wavfile.read(
                    os.path.join(config.DATADIR, self.seg.annotations["playback_file"].split("data/")[1])
            )
            self.playback_sound = np.lib.pad(
                self.playback_sound,
                (
                    int(self.playback_rate * config.PAD_SONG_BEFORE),
                    int(self.playback_rate * config.PAD_SONG_AFTER)
                ),
                "constant",
                constant_values=(0, 0)
            )
        else:
            self.playback_rate = self.playback_sound = None

        self.sound = self.seg.analogsignals[0].as_array()
        self.sampling_rate = self.seg.analogsignals[0].sampling_rate.item()

        self.t_start = self.seg.t_start.item()
        self.t_stop = self.seg.t_stop.item()
        self.t = np.linspace(self.t_start, self.t_stop, len(self.sound))

        self._spectrogram_cache = {}

    def spectrogram(self, source="playback"):
        """Plot spectrogram of segment audio

        Args:
            source ("playback" or int):
                Source of audio for spectrogram. Specify either a mic channel index or "playback"
                for the raw playback file used
        """
        if source == "playback" and self.playback_sound is None:
            raise ValueError("This segment does not have playback; must specify source channel")
        if isinstance(source, int) and source >= self.sound.shape[1]:
            raise ValueError("This segment only has {} channels".format(self.sound.shape[1]))

        if source in self._spectrogram_cache:
            return self._spectrogram_cache[source]

        if source == "playback":
            sound_source = self.playback_sound
            sampling_rate = self.playback_rate
        else:
            sound_source = self.sound[:, source]
            sampling_rate = self.sampling_rate

        t_spec, f_spec, spec, rms = spectrogram(
                sound_source - np.mean(sound_source),
                sampling_rate,
                1000.0,
                50.0,
                min_freq=0,
                max_freq=10000,
                nstd=6)

        result = t_spec, f_spec, spec, rms
        self._spectrogram_cache[source] = result

        return result

    def plot_spectrogram(self, source="playback", ax=None, zero_point=0.0):
        t_spec, f_spec, spec, _ = self.spectrogram(source=source)
        t_spec = t_spec - zero_point
        plot_spectrogram(t_spec, f_spec, spec, colormap="gray_r", ax=ax, colorbar=False)

    def spikes(self):
        return self.seg.spiketrains

    def binned_spikes(self):
        t_arr, spikes = bin_spikes(
                self.spikes(),
                min_time=self.t_start,
                dt=0.001,
                n_bins=int((self.t_stop - self.t_start) / 0.001))

        return t_arr, spikes

