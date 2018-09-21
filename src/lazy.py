import os
import pickle

import numpy as np
from neo.io import NixIO
from block_io import load_block, load_yaml
from segmentation.audio_segment import join_continuous_segments

import config


def time_to_index(t, sampling_rate):
    """Convert a time to an index"""
    return int(round(t * sampling_rate))


class LoadedTimeSliceDict(dict):
    """Dictionary with convenient string representation for time slices

    Basically just operates like a dictionary; but you initialize
    it with parameters t0 and t1 which are displayed when printing
    out the object. In addition, the keys of the dictionary are
    printed when the object is printed.

    Example:

        >>> time_slice_dict = lazy_site.time_slice(0.0, 1.0)
        >>> print(time_slice_dict)
        LoadedTimeSliceDict(keys=['filtered', 'mic'], t in [0.00s, 1.00s])
        >>> print(time_slice_dict["filtered"])
        LoadedTimeSlice(shape=(30000, 16), t in [0.00s, t1=1.00s])
    """
    def __init__(self, t0, t1, *args, **kwargs):
        self._t0 = t0
        self._t1 = t1
        super().__init__(self, *args, **kwargs)

    def __repr__(self):
        return "LoadedTimeSliceDict(keys={}, t in [{:.2f}s, {:.2f}s])".format(
                list(self.keys()),
                self._t0,
                self._t1)


class LoadedTimeSlice(object):
    """Convenience object for holding data and its associated time axis

    Example:

        >>> time_slice = LoadedTimeSlice(...)
        >>> print(time_slice)
        LoadedTimeSlice(shape=(30000, 2), t in [0.00s, 1.00s])
        >>> print(time_slice.t)
        array([0.00000000e+00, 3.33333333e-05, 6.66666667e-05, ...,
               9.99900000e-01, 9.99933333e-01, 9.99966667e-01])
        >>> print(time_slice.data)
        memmap([[ ... ]])

    """
    def __init__(self, t, data):
        """Create a LoadedTimeSlice object

        Parameters

            t: 1d array of length N (number of samples) representing the
                time in seconds of the data
            data: 2d array of shape (N, M), with M = number of channels
        """
        self.t = t
        self.data = data

    def __repr__(self):
        return "LoadedTimeSlice(shape={}, t in [{:.2f}s, {:.2f}s])".format(
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
        LazySignal(N=162000000 <5400.0s>, file=<abs-path>, sampling_rate=30000.0, mode=r)

    * Plot the data on one electrode at t=30s to t=30.1s

        >>> electrode_data = sig.time_slice(30.0, 30.1)
        >>> plt.plot(electrode_data.t, electrode_data.data[:, 0])

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
        """Load subset of array into memory

        Loads a numpy memmmap object of 32-bit floating points numbers into memory
        by indexing into the saved data array. Can be used to read and write depending
        on the current mode.

        Parameters

            n: integer value of number of samples to load
            offset: integer value of sample to start at

        Returns a numpy.memmap object
        """
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

        Returns a LoadedTimeSlice object with two attributes:
            t: 1d array of length N (number of samples) with the time in
                seconds of each sample
            data: 2d array of shape (N, M), with M the number of channels

        Example:

            >>> signal = LazySignal(...)
            >>> elec = signal.time_slice(30.0, 30.1)
            >>> plt.plot(elec.t, elec.data[:, 0])

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
        """Find the LazySignal containing a certain time (for each signal)

        First it identifies which session should contain t_start, by finding the
        last session that starts before t_start. Then it grabs the corresponding
        LazySignal for that session, for each of the signal types (e.g.
        "filtered", "raw", "mic") that it knows about.

        Parameters
        
            t_start: start time in seconds to search for

        Returns a tuple of the start time of the session in seconds (float),
            and a dictionary mapping signal string names (e.g. "filtered") to
            the LazySignal object for that session.
        """
        session_idx = np.where(self.signal_times <= t_start)[0][-1]

        signals = dict((key, val[session_idx]) for key, val in self._signals.items())
        return self.signal_times[session_idx], signals

    def add_signals(self, signal_name, signals):
        """Add a new signal to the LazySite

        Parameters
        
            signal_name: string name of the new signal to add
            signals: list of LazySignal objects, one for each of the sessions
                in the site. They should correspond to the .start_times of the
                LazySite object.
        """
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
        """Toggle the read/write mode of all referenced LazySignals"""
        self.mode = mode
        for sigs in self._signals.values():
            for sig in sigs:
                sig.set_mode(self.mode)
        
    def time_slice(self, t_start, t_stop, signal=None):
        """Access a subset of the data array
        
        Parameters:

            t_start: time in seconds of the first sample, rounded down
            t_stop: time in seconds of the last sample, rounded down
            signal (default=None): string name of the specific signal
                to load (e.g. "filtered", "mic"). If not provided,
                loads all the signals. Specify this to prevent loading
                of unneeded data arrays.

        If signal is not provided, returns a LoadedTimeSliceDict whose keys
            are the signal names and values are LoadedTimeSlice objects.
            If signal is specified, directly return the LoadedTimeSlice object
            for that array.

        Example:

            >>> site = LazySite(...)
            >>> site.time_slice(0.0, 1.0)
            LoadedTimeSlices(keys=['filtered', 'mic'], t in [0.00s, 1.00s])
            >>> site.time_slice(0.0, 1.0, signal="mic")
            LoadedTimeSlice(shape=(30000, 2), t in [0.00s, t1=1.00s])

        """
        t_signal, lazy_signals = self._get_lazy_signals(t_start)

        # Floating point errors is worst thing in the world. Must convert

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


def create_lazy_signal_raw(session_dir):
    session_name = os.path.basename(session_dir)
    nix = NixIO(os.path.join(session_dir, "segmentation", "{}.nix".format(session_name)))
    block = nix.read_block(lazy=True)
    segment = block.segments[0]

    for sig in segment.analogsignals:
        if sig.name.startswith("Amplifier"):
            break

    sig = nix.load_lazy_object(sig)

    lazy_signal = LazySignal(
        sig.shape[0],
        sig.shape[1],
        os.path.join(session_dir, "raw", "raw.raw.memmap"),
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
    lazy_signals_raw = []
    signal_times = []
    for session in site_config["sessions"]:
        print("Processing {}".format(session))
        session_dir = os.path.join(site_dir, "..", "..", "sessions", session["id"])
        lazy_signals_elec.append(create_lazy_signal_elec(session_dir))
        lazy_signals_mic.append(create_lazy_signal_mic(session_dir))
        lazy_signals_raw.append(create_lazy_signal_raw(session_dir))

        signal_times.append(session["t"])
        
    return LazySite(signal_times,
            mic=lazy_signals_mic,
            filtered=lazy_signals_elec,
            raw=lazy_signals_raw)
