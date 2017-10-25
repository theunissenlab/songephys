import numpy as np

from spikes.utils import bin_spikes
from spikes.psth import conv, exponential_convolver, gaussian_convolver



class MultiSegment(object):

    @classmethod
    def _read_annotations(cls, segments):
        original_annotations = dict(segments[0].annotations)
        for segment in segments:
            for key in segment.annotations:
                if key in original_annotations and segment.annotations[key] != original_annotations[key]:
                    original_annotations.pop(key)

        return original_annotations

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

    def t_arr(self):
        return self.binarized()[0]

    def psths(self, mean=True, mode="gaussian", *args):
        """Compute psth spike data"""
        if mode == "gaussian":
            convolved = conv(self.binarized()[1].swapaxes(1, 2), gaussian_convolver, *args)
        elif mode == "exponential":
            convolved = conv(self.binarized()[1].swapaxes(1, 2), exponential_convolver, *args)
        elif mode == "kde":
            raise Exception("Not did this yet")

        if not mean:
            return convolved
        else:
            return convolved.mean(axis=0)
