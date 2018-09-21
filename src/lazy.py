import os
import pickle

import numpy as np
from neo.io import NixIO
from block_io import load_block, load_yaml
from segmentation.audio_segment import join_continuous_segments

import config


def time_to_index(x, sampling_rate):
    return int(round(x * sampling_rate))


class LoadedTimeSliceDict(dict):
    def __init__(self, t0, t1, *args, **kwargs):
        self._t0 = t0
        self._t1 = t1
        super().__init__(self, *args, **kwargs)

    def __repr__(self):
        return "LoadedTimeSlices(keys={}, t=[{:.2f}s, {:.2f}s])".format(
                list(self.keys()),
                self._t0,
                self._t1)


class LoadedTimeSlice(object):
    """Just for more convenient access to time and data arrays"""
    def __init__(self, t, data):
        self.t = t
        self.data = data

    def __repr__(self):
        return "LoadedTimeSlice(shape={}, t=[{:.2f}s, t1={:.2f}s])".format(
                self.data.shape,
                self.t[0],
                self.t[-1])


class LazySignal(object):
    """Access a signal with lazy loading by time slices

    Example:
    
    * Load the raw signal

        >>> sig = LazySignal(
        ...     N=162000000,
        ...     file="<path>",
        ...     sampling_rate=30000.0,
        ...     mode="r"
        ... )

        >>> print(sig)
        LazySignal(N=162000000 <5400.0>, file=/auto/fhome/kevin/Projects/songephys/data/birds/BroGre4497M/sessions/BroGre4497M_180610_114535/raw/filtered.raw.memmap, sampling_rate=30000.0, mode=r)

    * Plot the data on one electrode at t=30s to t=30.1s

        >>> t_arr, elec_data = sig.time_slice(30.0, 30.1)
        >>> plt.plot(t_arr, elec_data[:, 0])

    """

    def __init__(self, N, M, file, sampling_rate, mode="r"):
        """Initialize the LazySignal reader

        Params

            N: integer length of the array in number of samples
            file: string path to the memmap file
            sampling_rate: float number of samples per second
            mode (r, w, w+): read or write mode
        """

        self.N = N
        self.M = M
        self.file = os.path.abspath(file)
        self.sampling_rate = sampling_rate
        self.set_mode(mode)

    def __repr__(self):
        return "LazySignal(N={} <{:.1f}s>, file={}, sampling_rate={}, mode={})".format(
            self.N, self.N / self.sampling_rate, self.file, self.sampling_rate, self.mode)
        
    def _load_samples(self, n, offset):
        # n = self.N - (n + offset)
        return np.memmap(self.file,
            mode=self.mode,
            dtype="float32",
            shape=(n, self.M),
            offset=int(offset * self.M * (32 // 8))
        )

    def write(self, new_array, n=None, offset=0):
        """(Over)write a section of the numpy array at a particular location

        Parameters:

            new_array: The array of size n to write
            n (default=None): The length of the new array being written. This
                can be left as None if the entire array is being written.
            offset (default=0): The position at which the new array will be
                written.
        """
        if len(new_array) != self.N and n is None:
            raise ValueError("Cannot write to subset of array without specifying n")
        if offset + len(new_array) > self.N:
            raise ValueError("Array update of size {} at location {} "
                "would extend beyond original size {}".format(len(new_array), offset, self.N))
        if "w" not in self.mode:
            raise ValueError("Cannot write to array in mode {}".format(self.mode))

        arr = self._load_samples(new_array.shape[0], offset)
        arr[:] = new_array[:]

    def set_mode(self, mode):
        if mode not in ["r", "r+", "w", "w+"]:
            raise ValueError("Invalid mode to memmap")
        self.mode = mode

    def time_slice(self, t_start, t_stop):
        """Access a subset of the data array
        
        Parameters:

            t_start: time in seconds of the first sample, rounded down
            t_stop: time in seconds of the last sample, rounded down

        Returns a tuple of two arrays. The first array is an array of
            times in seconds corresponding to each sample returned. The second
            array is an array of (N x E) where N is the number of samples and
            E is the number of electrodes

        >>> signal = LazySignal(...)
        >>> t_arr, elec_data = signal.time_slice(30.0, 30.1)
        """
        start_idx = time_to_index(t_start, self.sampling_rate)
        stop_idx = time_to_index(t_stop, self.sampling_rate)
        n = stop_idx - start_idx
        t_arr = np.arange(start_idx, stop_idx) / self.sampling_rate

        return LoadedTimeSlice(t=t_arr, data=self._load_samples(n, start_idx))
    
    def batches(self, step):
        """Create a generator that iterates over the data

        Parameters:

            step (int): integer number of samples representing the width of
                each window being iterated over
        """
        for start_idx in range(0, self.N - 1, step):
            yield self._load_samples(step, start_idx)

    
class LazySite(object):
    """Access a signal with lazy loading by time slices

    Example:
    
    * Load the raw signals and associate with their relative start times

        >>> site = LazySite([ ... raw signals ...], signal_times=[0, 5400.0, 10800.0])
        >>> print(site)
        LazySite(3 signals, times=[   0. 5400. 10800.0])

        >>> plt.plot(t_arr, elec_data[:, 0])
    """

    def __init__(self, signal_times, mode="r", **lazy_signals):
        """
            signal_times: An float array of length K with the kth element
                the start time (in seconds) of the kth session in the site
            mode: 'r' or 'w'
            lazy_signals: a dictionary of signal names (i.e. "raw_elec", "mic")
                to a list of K LazySignal objects, one for each session
        """

        self._signals = lazy_signals
        self.signal_times = np.array(signal_times)
        self.set_mode(mode)

    def __repr__(self):
        return "LazySite(keys={}, {} signals, times={})".format(
            list(self._signals.keys()),
            len(self.signal_times),
            self.signal_times
        )
    
    def _get_lazy_signals(self, t_start):
        session_idx = np.where(self.signal_times <= t_start)[0][-1]

        signals = dict((key, val[session_idx]) for key, val in self._signals.items())
        return self.signal_times[session_idx], signals

    def add_signals(self, signal_name, signals):
        if len(signals) != len(self.signal_times):
            raise ValueError("Number of new signals ({}) does not match number of "
                "sessions in LazySite ({}, signal_times={})".format(
                    len(signals),
                    len(self.signal_times),
                    self.signal_times))

        if signal_name in self._signals:
            raise ValueError("Signal name {} already exists".format(signal_name))

        self._signals[signal_name] = signals

    def set_mode(self, mode):
        self.mode = mode
        for sigs in self._signals.values():
            for sig in sigs:
                sig.set_mode(self.mode)
        
    def time_slice(self, t_start, t_stop, signal=None):
        """Access a subset of the data array
        
        Parameters:

            t_start: time in seconds of the first sample, rounded down
            t_stop: time in seconds of the last sample, rounded down

        Returns a tuple of two arrays. The first array is an array of
            times in seconds corresponding to each sample returned. The second
            array is an array of (N x E) where N is the number of samples and
            E is the number of electrodes

        >>> site = LazySite(...)
        >>> t_arr, elec_data = site.time_slice(30.0, 30.1)
        """
        t_signal, lazy_signals = self._get_lazy_signals(t_start)

        if signal is not None:
            result = lazy_signals[signal].time_slice(
                    t_start - t_signal,
                    t_stop - t_signal)
            result.t += t_signal
            return result

        signals = LoadedTimeSliceDict(t_start, t_stop)
        for key, lazy_signal in lazy_signals.items():
            signals[key] = lazy_signal.time_slice(t_start - t_signal, t_stop - t_signal)
            signals[key].t += t_signal  # Offset by the start time of the session

        return signals
    

def create_lazy_signal_elec(session_dir):
    """Copy data from the neo block to a memmapped array for lazy loading"""
    print("Loading {}".format(session_dir))
    session_name = os.path.basename(session_dir)
    block = load_block(os.path.join(session_dir, "segmentation", "{}-filtered.pkl".format(session_name)))
    block = join_continuous_segments(block)
    segment = block.segments[0]
    for sig in segment.analogsignals:
        if sig.name.startswith("Amplifier"):
            break
    
    lazy_signal = LazySignal(
        sig.shape[0],
        sig.shape[1],
        os.path.join(session_dir, "raw", "filtered.raw.memmap"),
        sig.sampling_rate.item(),
        mode="w+"
    )
    print("Saving {} signal to {}".format(session_dir, lazy_signal.file))
    lazy_signal.write(sig)
    print("Saved.")
    lazy_signal.set_mode("r")
    return lazy_signal


def create_lazy_signal_mic(session_dir):
    session_name = os.path.basename(session_dir)
    nix = NixIO(os.path.join(session_dir, "segmentation", "{}.nix".format(session_name)))
    block = nix.read_block(lazy=True)
    segment = block.segments[0]

    for sig in segment.analogsignals:
        if sig.name.startswith("Board ADC"):
            break

    sig = nix.load_lazy_object(sig)

    lazy_signal = LazySignal(
        sig.shape[0],
        sig.shape[1],
        os.path.join(session_dir, "raw", "mic.raw.memmap"),
        sig.sampling_rate.item(),
        mode="w+"
    )
    print("Saving {} signal to {}".format(session_dir, lazy_signal.file))
    lazy_signal.write(sig)
    print("Saved.")
    lazy_signal.set_mode("r")
    return lazy_signal


def create_lazy_site(site_dir):
    """
    Prepare all signals for sessions in the site to be loaded by LazySignal and LazySite
    """
    site_config = load_yaml(os.path.join(site_dir, "sessions.yaml"))
    lazy_signals_elec = []
    lazy_signals_mic = []
    signal_times = []
    for session in site_config["sessions"]:
        print("Processing {}".format(session))
        session_dir = os.path.join(site_dir, "..", "..", "sessions", session["id"])
        # lazy_signals_elec.append(create_lazy_signal_elec(session_dir))
        lazy_signals_mic.append(create_lazy_signal_mic(session_dir))

        signal_times.append(session["t"])
        
    return LazySite(signal_times, mic=lazy_signals_mic)
