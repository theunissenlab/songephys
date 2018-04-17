from __future__ import print_function, unicode_literals

import os

import neo.core
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from scipy.io import wavfile

from soundsig.sound import spectrogram, plot_spectrogram

import config

from spikes.utils import bin_spikes
from spikes.psth import conv, exponential_convolver, gaussian_convolver
from spikes.plotting import raster


class MultiSegmentPlotter(object):
    """Plotting functions for MultiSegment class"""

    def __init__(self, multi_segment):
        self._segments = multi_segment

    def plot_psth(self,
            unit,
            ax=None,
            color="Black",
            alpha=0.1,
            align_to_segment=None,
            single_segment=None,
            mode="gaussian",
            convolve_params=(10,)):
        """Plot the estimated psth for a particular unit, overlaid on single trial estimates

        Params
        unit (int): unit index to plot psth for
        ax (matplotlib.pyplot.Axes): pyplot axes to plot on
        color (str): matplotlib color for psth line
        alpha (float, [0.0, 1.0]): opacity to plot single trial traces
        align_to_segment (neo.core.Segment or int):
            If not given, will use the MultiSegment's default alignment (aligned at t=0.0)
            If given, sets time axis to the original time axis of given segment
            (the index is relative to the segments in MultiSegment, not index of segment in the neo block)
        single_segment (neo.core.Segment, optional)
            If provided, will only plot estimated psth for a single trial/segment

        Returns
        None

        TODO
        Allow options of using different psth calculations (gaussian, exp, kde, ...)
        """
        if ax is None:
            ax = plt.gca()

        psths = self._segments.psths(mean=False, mode=mode, convolve_params=convolve_params)[:, :, unit].T
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
        """Plot the spectrogram of a given segment's analogsignal

        Params
        seg (neo.core.Segment or int):
            The segment or index of segment to plot spectrogram of
            (the index is relative to the segments in MultiSegment, not index of segment in the neo block)
        signal (int):
            The index of the analogsignal (in segment.analogsignals) to visualize
        channel (int):
            The channel of the analogsignal to visualize
        ax (matplotlib.pyplot.Axes): pyplot axes to plot on
        align_to_segment (neo.core.Segment or int):
            If not given, will use the MultiSegment's default alignment (aligned at t=0.0)
            If given, sets time axis to the original time axis of given segment
            (the index is relative to the segments in MultiSegment, not index of segment in the neo block)

        Returns
        None
        """
        t_spec, f_spec, spec, _ = self._segments.get_spectrogram(
                seg, signal=signal, channel=channel, align_to_segment=align_to_segment)
        plot_spectrogram(t_spec, f_spec, spec, colormap="gray_r", ax=ax, colorbar=False)

    def plot_raster(self, unit, ax=None, align_to_segment=None, mark_noise_epochs=False, elec_idx=None):
        """Plot the raster of the given unit over all trials in MultiSegment

        Params
        unit (int): unit index to plot psth for
        ax (matplotlib.pyplot.Axes): pyplot axes to plot on
        align_to_segment (neo.core.Segment or int):
            If not given, will use the MultiSegment's default alignment (aligned at t=0.0)
            If given, sets time axis to the original time axis of given segment
            (the index is relative to the segments in MultiSegment, not index of segment in the neo block)

        TODO
        Make align_to_segment do something useful? Right now is all aligned at 0.0, cant align anywhere else
        """
        # TODO: how to align to segment?
        if ax is None:
            ax = plt.gca()

        if mark_noise_epochs:
            epochs = [seg.epochs[0] for seg in self._segments]
            raster.plot_epochs(epochs, elec_idx=elec_idx, ax=ax)

        raster.plot(
            self._segments.spike_times(unit=unit, aligned=True),
            min_time=self._segments.t_start,
            max_time=self._segments.t_stop,
            ax=ax
        )


class MultiSegment(object):
    """Helper class to run analyses on multiple Segments/trials

    Primary feature is to align trials with segment-specific offsets
    """

    @classmethod
    def _read_annotations(cls, segments):
        """Helper function to compute which annotation values are shared across all segments"""
        original_annotations = dict(segments[0].annotations)
        for segment in segments:
            for key in segment.annotations:
                if key in original_annotations and segment.annotations[key] != original_annotations[key]:
                    original_annotations.pop(key)

        return original_annotations

    @property
    def plotter(self):
        """MultiSegmentPlotter object for this MultiSegment"""
        return MultiSegmentPlotter(self)

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
        """Instantiate a MultiSegment object with a list of segments

        Params
        segments (list): list of segments to analyze
        offsets (list, optional):
            List of the time in seconds at which you want to set to be 0.0 for each segment
            (list should be the same length as segments parameter)
            Offset must be computed relative to segment start time (segment.t_start)
            For example, defining the stimulus onset time for each segment to be zero.
        t_start (float):
            Set default start time relative to alignment point
            For example, t_start might be -0.5 to view half a second before stimulus onset
        t_stop (float):
            Set default end time relative to alignment point
            For example, t_start might be 2.0 to view the two secondsa after stimulus onset
        """
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
        """Return the index of a given segment in the MultiSegment

        Params
        segment (neo.core.Segment)

        Returns
        idx (int)
        """
        if list(self.segments).count(segment) > 1:
            raise Exception("Multiple instances of segment found in MultiSegment. Cannot "
                    "access by object. Pass an index instead")
        elif list(self.segments).count(segment) == 0:
            raise Exception("Segment not found in MultiSegment")

        return list(self.segments).index(segment)

    def filter(self, fn):
        """Will be a helper to narrow down MultiSegment, not implemented yet
       
        Params
        filter_fn (function):
            Function that takes a list of segments, and returns a list of segments,
            filtered by some condition

        Returns
        new_multisegment (segment_analysis.MultiSegment):
            Returns a new instance of MultiSegment with the filtered list of segments
        """
        return MultiSegment(list(filter(fn, self.segments)))

    def set_t_start(self, t_start=None):
        """Set a new t_start time"""
        self._t_start = t_start
        self.t_start = t_start if t_start is not None else -np.max(self.offsets)
        if np.min(self.offsets) + self.t_start < 0:
            print("Warning: start of some segments may be padded with zeros.")

    def set_t_stop(self, t_stop=None):
        """Set a new t_stop time"""
        self._t_stop = t_stop
        max_t_stop = np.max(self._durations - self.offsets)
        self.t_stop = t_stop if t_stop is not None else max_t_stop
        if max_t_stop < self.t_stop:
            print("Warning: ending of some segments may be padded with zeros.")

    def set_offsets(self, offsets=None):
        """Set new offset times for each segment

        Params
        offsets (list, optional):
            List of the time in seconds at which you want to set to be 0.0 for each segment
            Offset must be computed relative to segment start time (segment.t_start)
            For example, defining the stimulus onset time for each segment to be zero.
        """
        if len(offsets) != len(self.segments):
            raise Exception("offsets must be the same length as segments")

        self.offsets = np.array(offsets) if offsets is not None else np.zeros(len(self.segments))
        self.set_t_start(self._t_start)
        self.set_t_stop(self._t_stop)

    def spike_times(self, unit=None, aligned=False):
        """Return arrays of spike times for a specific unit or all units

        Params
        unit (int, optional):
            If provided, return array of spike times for each segment, for this unit.
            Otherwise, return arrays of spike times for each segment, for all units.
        aligned (bool):
            If True, spike times are realigned to 0.0 at the defined offset times.
            If False, spike times are returned with their original times in the session.

        Returns
            If unit is not given:
                spike_times_array (np.ndarray, (n_segments, n_units))
                    An array with dimensions n_segments, n_units.
                    The element [i, j] is an array of spike times for trial i and unit j
            If unit is given:
                spike_times_array (np.ndarray, (n_segments))
                    An array with dimensions n_segments.
                    The element i is an array of spike times for trial i
        """
        results = []
        for seg, offset in zip(self.segments, self.offsets):
            if aligned:
                total_offset = seg.t_start.item() + offset
            else:
                total_offset = seg.t_start.item()

            if unit is None:
                subresult = []
                for spiketrain in seg.spiketrains:
                    subresult.append(spiketrain.as_array() - total_offset)
                results.append(subresult)
            else:
                results.append(seg.spiketrains[unit].as_array() - total_offset)

        return np.array(results)

    # TODO: fix dt for all functions or make it a kwarg for all
    def binarized(self, dt=0.001):
        """Generate binarized spike data with ms bins

        Returns
        t_arr (np.ndarray, (n_timesteps)):
            Time axis for binarized array; ranges from the MultiSegment's t_start and t_stop,
            Corresponds to 0.0 aligned segments
        binarized_array (np.ndarray, (n_segments, n_units, n_timesteps)):
            Binary array, 1 indicates a spike, indicates no spike
        """
        results = []
        for seg_spiketimes in self.spike_times(aligned=True):
            t_arr, spikes = bin_spikes(
                    [times[(times > self.t_start) & (times < self.t_stop)] for times in seg_spiketimes],
                    min_time=self.t_start,
                    dt=dt,
                    n_bins=int((self.t_stop - self.t_start) / dt) + 1)
            results.append(spikes.T)

        return t_arr, np.vstack(np.array(results)[None])

    def t_arr(self, align_to_segment=None):
        """Return a time bin array for this segment corersponding to binarized data

        Params
        align_to_segment (neo.core.Segment or int):
            If not given, will use the MultiSegment's default alignment (aligned at t=0.0)
            If given, sets time axis to the original time axis of given segment
            (the index is relative to the segments in MultiSegment, not index of segment in the neo block)

        Returns
        t_arr (np.ndarray, (n_timesteps)):
            Time axis for binarized array; ranges from the MultiSegment's t_start and t_stop,
            Corresponds to 0.0 aligned segments
        """
        if align_to_segment is not None:
            if isinstance(align_to_segment, neo.core.Segment):
                align_to_segment = self.index_of(align_to_segment)
            # since binarized incorporates t_start
            return self.binarized()[0] + self.offsets[align_to_segment] + self._starts[align_to_segment] 
        else:
            return self.binarized()[0]

    def psths(self, mean=True, mode="gaussian", convolve_params=None):
        """Estimate psth for spike data over all trials for all units
       
        Params
        mean (bool, default=True):
            Only return the mean psth over all trials. Otherwise, return the estimated psth for every trial.
        mode (str):
            Options: gaussian, exponential

        Returns
        psths (np.ndarray, (n_segments, n_units, n_timesteps)):
        """
        if mode == "gaussian":
            if convolve_params is None:
                convolve_params = [10.0]
            convolved = conv(self.binarized()[1].swapaxes(1, 2), gaussian_convolver, *convolve_params)
        elif mode == "exponential":
            if convolve_params is None:
                convolve_params = [20.0]
            convolved = conv(self.binarized()[1].swapaxes(1, 2), exponential_convolver, *convolve_params)
        elif mode == "kde":
            raise Exception("Not did this yet")

        if not mean:
            return convolved
        else:
            return convolved.mean(axis=0)

    def waveforms(self, unit=None):
        """Return waveforms for all units, or a given unit"""
        wfs = []
        units = [unit] if unit is not None else self.units

        for unit_idx in units:
            unit_wfs = []
            for seg in self.segments:
                if len(seg.spiketrains[unit_idx].waveforms):
                    unit_wfs += list(seg.spiketrains[unit_idx].waveforms)
            wfs.append(np.array(unit_wfs))

        return np.array(wfs)

    def get_analogsignal(self, seg_idx, signal=0, channel=None, original_times=False):
        """Return timeseries data from a segment's analogsignals, and the time axis

        Params
        seg_idx (neo.core.Segment or int):
            The segment or index of segment to plot spectrogram of
            (the index is relative to the segments in MultiSegment, not index of segment in the neo block)
        signal (int):
            The index of the analogsignal (in segment.analogsignals)
        channel (int, default=None):
            The channel of the analogsignal to return, returns all channels if None
        original_times (bool, default=False)
            If True, the time array will return times corresponding to their time during the session.
            If False, the time array will align the defined offset times at 0.0.

        Returns
        t_arr (np.ndarray, (n_timestpes))
            Time axis for signal, either in original times in session or
            realigned so that 0.0 is at the specified time offset
        data (np.ndarray, (n_timesteps) or (n_timesteps, n_channels))
            The data of the analogsignal
        sampling_rate (float)
            Sampling rate of the signal
        """
        if isinstance(seg_idx, neo.core.Segment):
            seg_idx = self.index_of(seg_idx)
        seg = self.segments[seg_idx]

        # Python 2/3 compatibility hack for default string type
        if isinstance(signal, "".__class__):
            # we are probably looking at a request for playback
            if signal != "playback":
                raise Exception("signal must be either int or 'playback'")

            sampling_rate, playback_sound = wavfile.read(
                os.path.join(
                    config.DATADIR,
                    seg.annotations["playback_file"].split("data/")[1]
                )
            )
            analogsignal = np.lib.pad(
                playback_sound,
                (
                    int(sampling_rate * config.PAD_SONG_BEFORE),
                    int(sampling_rate * config.PAD_SONG_AFTER)
                ),
                "constant",
                constant_values=(0, 0)
            )
            result = analogsignal
        else:
            analogsignal = seg.analogsignals[signal]
            sampling_rate = analogsignal.sampling_rate.item()
            result = analogsignal.as_array()

        t_arr = np.linspace(seg.t_start.item(), seg.t_stop.item(), len(analogsignal))
        if not original_times:
            t_arr -= self._starts[seg_idx] + self.offsets[seg_idx]

        if channel is not None:
            result = result[:, channel]

        return t_arr, result, sampling_rate

    def get_spectrogram(self, seg_idx, signal=0, channel=0, align_to_segment=None):
        """Generate a spectrogram of a segment's signal

        Params
        seg_idx (neo.core.Segment or int):
            The segment or index of segment to plot spectrogram of
            (the index is relative to the segments in MultiSegment, not index of segment in the neo block)
        signal (int):
            The index of the analogsignal (in segment.analogsignals)
        channel (int)
            The channel of the analogsignal to generate spectrogram from
        align_to_segment (neo.core.Segment or int):
            If not given, will use the MultiSegment's default alignment (aligned at t=0.0)
            If given, sets time axis to the original time axis of given segment
            (the index is relative to the segments in MultiSegment, not index of segment in the neo block)

        Returns
        t_spec (np.ndarray, (n_timestpes)):
            Time axis for spectrogram, either in original times in session or
            realigned so that 0.0 is at the specified time offset
        f_spec (np.ndarray, (n_frequencies)):
            Frequency axis for spectrogram
        spec (np.ndarray, (n_timesteps, n_frequencies)):
            Spectrogram data
        rms (float)
            Probably is something
        """
        if isinstance(seg_idx, neo.core.Segment):
            seg_idx = self.index_of(seg_idx)

        if align_to_segment is True:
            original_times = True
        elif isinstance(align_to_segment, int) or isinstance(align_to_segment, neo.core.Segment):
            if isinstance(align_to_segment, neo.core.Segment):
                align_seg_idx = self.index_of(align_to_segment)
            else:
                align_seg_idx = align_to_segment
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

    def recut_segments(self, interval_array):
        """Recut segments using given time intervals

        Params
        interval_array (list)
            List should be the same length as segments parameter.
            Each sublist contains tuples of (start_time, end_time) for each segment.
            The start_time and end_time must be computed relative to the segment's t=0.0s,
                IGNORING ANY OFFSETS

        Returns
        updated_multiseg (segment_analysis.MultiSegment)
            Updated multiseg where each segment corresponds to one of the intervals specified
        """
        if len(interval_array) != len(self.segments):
            raise Exception("interval_array must be the same length as MultiSegment.segments")

        new_segments = []
        for seg_idx, (intervals, segment) in enumerate(zip(interval_array, self.segments)):
            for syllable_idx, (start_time, end_time) in enumerate(intervals):
                start_time = segment.t_start.item() + start_time
                end_time = segment.t_start.item() + end_time

                if start_time < segment.t_start.item() or end_time > segment.t_stop.item():
                    print("Warning: you passed an interval that extends outside the segment time boundaries. Skipping it.")
                    continue

                segment_params = dict(**segment.annotations)
                segment_params["name"] = "{}_Syllable_{}".format(segment.name, syllable_idx)
                segment_params["syllable_category"] = None
                new_segment = neo.core.Segment(**segment_params)

                sliced_signal = segment.analogsignals[0].time_slice(
                        start_time * pq.s,
                        end_time * pq.s
                )
                new_segment.analogsignals.append(sliced_signal)

                for st in segment.spiketrains:
                    sliced_st = st.time_slice(start_time, end_time)
                    new_segment.spiketrains.append(sliced_st)
                new_segments.append(new_segment)

        return MultiSegment(new_segments)

