import neo.core
import numpy as np
import matplotlib.pyplot as plt

from soundsig.sound import spectrogram, plot_spectrogram

from spikes.utils import bin_spikes
from spikes.psth import conv, exponential_convolver, gaussian_convolver


class Plotter(object):

    def __init__(self, multi_segment):
        self._segments = multi_segment

    def plot_psth(self, unit, ax=None, color="Black", t_start=None, alpha=0.1, align_to_segment=None, single_segment=None):
        if ax is None:
            ax = plt.gca()

        psths = self._segments.psths(mean=False)[:, :, unit].T
        t_arr = self._segments.t_arr(align_to_segment=align_to_segment)

        if single_segment is not None:
            if isinstance(single_segment, int):
                seg_idx = single_segment
            elif isinstance(single_segment, neo.core.Segment):
                seg_idx = self._segments.index_of(single_segment)
            else:
                raise Exception("single_segment must be int or neo.core.Segment")
            ax.plot(t_arr, psths[:, seg_idx], linewidth=2, color=color)
        else:
            ax.plot(t_arr, psths, alpha=alpha, color=color)
            ax.plot(t_arr, np.mean(psths, axis=1), linewidth=2, color=color)

    def plot_spectrogram(self, seg, signal=0, channel=0, ax=None, align_to_segment=None):
        t_spec, f_spec, spec, _ = self._segments.get_spectrogram(
                seg, signal=signal, channel=channel, align_to_segment=align_to_segment)
        plot_spectrogram(t_spec, f_spec, spec, colormap="gray_r", ax=ax, colorbar=False)


class MultiSegment(object):

    @classmethod
    def _read_annotations(cls, segments):
        original_annotations = dict(segments[0].annotations)
        for segment in segments:
            for key in segment.annotations:
                if key in original_annotations and segment.annotations[key] != original_annotations[key]:
                    original_annotations.pop(key)

        return original_annotations

    @property
    def plotter(self):
        return Plotter(self)

    def __str__(self):
        return "MultiSegment with {} Segments, {} Units, t_start={}s, t_stop={}s\n{}".format(
                len(self.segments),
                len(self.units),
                self.t_start,
                self.t_stop,
                self.annotations
        )

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return (x for x in self.segments)

    def __init__(self, segments, offsets=None, t_start=None, t_stop=None):
        self.segments = segments

        self.annotations = MultiSegment._read_annotations(segments)
        self.units = range(len(segments[0].spiketrains))
        self.offsets = np.array(offsets) if offsets is not None else np.zeros(len(segments))
        self._stops = np.array([seg.t_stop.item() for seg in segments])
        self._starts = np.array([seg.t_start.item() for seg in segments])
        self._durations = self._stops - self._starts

        # these keep track of if t_start was set as a value or just as default (None)
        self._t_start = t_start
        self._t_stop = t_stop
        self.set_t_start(self._t_start)
        self.set_t_stop(self._t_stop)

    def index_of(self, segment):
        return list(self.segments).index(segment)

    def filter(self, fn):
        """Will be a helper to narrow down MultiSegment"""
        pass

    def set_t_start(self, t_start=None):
        self._t_start = t_start
        self.t_start = t_start if t_start is not None else -np.max(self.offsets)
        if np.min(self.offsets) + self.t_start < 0:
            print("Warning: start of some segments may be padded with zeros.")

    def set_t_stop(self, t_stop=None):
        self._t_stop = t_stop
        max_t_stop = np.max(self._durations - self.offsets)
        self.t_stop = t_stop if t_stop is not None else max_t_stop
        if max_t_stop < self.t_stop:
            print("Warning: ending of some segments may be padded with zeros.")

    def set_offsets(self, offsets=None):
        self.offsets = np.array(offsets) if offsets is not None else np.zeros(len(segments))
        self.set_t_start(self._t_start)
        self.set_t_stop(self._t_stop)

    def spike_times(self, aligned=False):
        results = []
        for seg, offset in zip(self.segments, self.offsets):
            if aligned:
                total_offset = seg.t_start.item() + offset
            else:
                total_offset = seg.t_start.item()

            subresult = []
            for spiketrain in seg.spiketrains:
                subresult.append(spiketrain.as_array() - total_offset)
            results.append(subresult)

        return results

    # TODO: fix dt for all functions or make it a kwarg for all
    def binarized(self, dt=0.001):
        """
        Returns:
        binarized_array (n_segments, n_units, n_timesteps)
        """
        results = []
        for seg_spiketimes in self.spike_times(aligned=True):
            t_arr, spikes = bin_spikes(
                    [times[(times > self.t_start) & (times < self.t_stop)] for times in seg_spiketimes],
                    min_time=self.t_start,
                    dt=dt,
                    n_bins=int((self.t_stop - self.t_start) / dt))
            results.append(spikes.T)

        return t_arr, np.vstack(np.array(results)[None])

    def t_arr(self, align_to_segment=None):
        """Return a time bin array 
        """
        if align_to_segment is not None:
            if isinstance(align_to_segment, neo.core.Segment):
                align_to_segment = self.index_of(align_to_segment)
            # since binarized incorporates t_start
            return self.binarized()[0] + self.offsets[align_to_segment] + self._starts[align_to_segment] 
        else:
            return self.binarized()[0]

    def psths(self, mean=True, mode="gaussian"):
        """Compute psth spike data"""
        if mode == "gaussian":
            convolved = conv(self.binarized()[1].swapaxes(1, 2), gaussian_convolver, 10.0)
        elif mode == "exponential":
            convolved = conv(self.binarized()[1].swapaxes(1, 2), exponential_convolver, 20.0)
        elif mode == "kde":
            raise Exception("Not did this yet")

        if not mean:
            return convolved
        else:
            return convolved.mean(axis=0)

    def waveforms(self, unit=None):
        wfs = []
        units = [unit] if unit is not None else self.units

        for unit_idx in units:
            unit_wfs = []
            for seg in self.segments:
                if len(seg.spiketrains[unit_idx].waveforms):
                    unit_wfs += list(seg.spiketrains[unit_idx].waveforms)
            wfs.append(unit_wfs)

        return np.array(wfs)

    def get_analogsignal(self, seg_idx, signal=0, channel=None, original_times=False):
        """
        Return t_arr, timeseries, sampling_rate for an analogsignal
        """
        if isinstance(seg_idx, neo.core.Segment):
            seg_idx = self.index_of(seg_idx)
        seg = self.segments[seg_idx]

        analogsignal = seg.analogsignals[signal]
        t_arr = np.linspace(seg.t_start.item(), seg.t_stop.item(), len(analogsignal))
        if not original_times:
            t_arr -= self._starts[seg_idx] + self.offsets[seg_idx]

        result = analogsignal.as_array()
        if channel is not None:
            result = result[:, channel]

        return t_arr, result, analogsignal.sampling_rate.item()

    def get_spectrogram(self, seg_idx, signal=0, channel=0, align_to_segment=False):
        if isinstance(seg_idx, neo.core.Segment):
            seg_idx = self.index_of(seg_idx)

        if isinstance(align_to_segment, bool):
            original_times = True
        elif isinstance(align_to_segment, neo.core.Segment):
            align_seg_idx = self.index_of(align_to_segment)
            if align_seg_idx != seg_idx:
                raise Exception("Cannot align {} to {}".format(
                    self.segments[seg_idx],
                    self.segments[align_seg_idx]))
            original_times = True
        else:
            original_times = False

        _, signal_data, sampling_rate = self.get_analogsignal(seg_idx, signal=signal, channel=channel)

        t_spec, f_spec, spec, rms = spectrogram(
            signal_data - np.mean(signal_data),
            sampling_rate,
            1000.0,
            50.0,
            min_freq=0,
            max_freq=10000,
            nstd=6)

        if not original_times:
            t_spec -= self.offsets[seg_idx]
        else:
            t_spec += self._starts[seg_idx]

        return t_spec, f_spec, spec, rms
