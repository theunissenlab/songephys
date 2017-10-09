import matplotlib.pyplot as plt
import numpy as np

from soundsig.sound import spectrogram, plot_spectrogram

from spikes.utils import bin_spikes
from spikes.psth import conv, gaussian_convolver


class SegmentViewer(object):

    def __init__(self, segment):
        self.seg = segment
        self.sound = self.seg.analogsignals[0].as_array()
        self.sampling_rate = self.seg.analogsignals[0].sampling_rate

        self.t_start = self.seg.t_start.item()
        self.t_stop = self.seg.t_stop.item()
        self.t = np.linspace(self.t_start, self.t_stop, len(self.sound))

    def spectrogram(self):
        # FIXME: just take sound from one mic?
        # TODO: load playback audio if its available?
        t_spec, f_spec, spec, rms = spectrogram(
                self.audio(channel=0),
                self.sampling_rate,
                1000.0,
                50.0,
                min_freq=0,
                max_freq=10000,
                nstd=6)

        return t_spec, f_spec, spec, rms

    def plot_spectrogram(self, ax=None):
        t_spec, f_spec, spec, _ = self.spectrogram()
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

    def audio(self, channel=0):
        return self.sound[:, channel] - self.sound[:, channel].mean()


class TrialsViewer(object):
    """Helper for grouping segments

    All segments must have same duration
    """
    def __init__(self, segments):
        self.segments = segments
        self.segment_viewers = [SegmentViewer(seg) for seg in self.segments]
        self.n_units = len(self.segments[0].spiketrains)
        self.units = range(self.n_units)

    def binned_spikes(self):
        """Collect spikes by unit
       
        Returns
            np.ndarray: (n_trials, n_units, n_bins)
                Binary spike array by units for all segments
        """
        return np.array([seg_view.binned_spikes()[1] for seg_view in self.segment_viewers])

    def t_arr(self):
        return self.segment_viewers[0].binned_spikes()[0]

    def psths(self, mean=True):
        convolved = conv(self.binned_spikes(), gaussian_convolver, 20.0)
        if not mean:
            return convolved
        else:
            return convolved.mean(axis=0)

    def plot_psth(self, unit):
        psths = self.psths(mean=False)[:, :, unit].T
        plt.plot(self.t_arr(), psths, alpha=0.3, color="Gray")
        plt.plot(self.t_arr(), np.mean(psths, axis=1), linewidth=2, color="Black")

    def waveforms(self, unit=None):
        wfs = []
        units = [unit] if unit is not None else range(self.n_units)

        for unit_idx in units:
            unit_wfs = []
            for seg in self.segments:
                if len(seg.spiketrains[unit_idx].waveforms):
                    unit_wfs += list(seg.spiketrains[unit_idx].waveforms)
            wfs.append(unit_wfs)

        return np.array(wfs)


class PlaybacksManager(object):

    def __init__(self, segments):
        self.segments = filter(lambda seg: seg.annotations["sound_type"] == "playback", segments)
        # TODO fix this to access the file
        self.stim_ids = np.unique(list(self.get_id(seg) for seg in self.segments))

    def get_id(self, seg):
        return seg.annotations["playback_file"].split("/")[-1]
        return seg["stim_id"]

    def filter_segments_by_stim_id(self, stim_id):
        return filter(lambda seg: self.get_id(seg) == stim_id, self.segments)

    def get_playback_responses(self, stim_id):
        return TrialsViewer(self.filter_segments_by_stim_id(stim_id))


    # use for combining segments that were in repsonse to same playback file
