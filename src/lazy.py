import os
import pickle

import numpy as np
from block_io import load_block, load_yaml
from segmentation.audio_segment import join_continuous_segments

import config


class RawSignal(object):
    """Access a signal with lazy loading by time slices

    Example:
    
    * Load the raw signal

        >>> sig = RawSignal(
        ...     N=162000000,
        ...     file="<path>",
        ...     sampling_rate=30000.0,
        ...     mode="r"
        ... )

        >>> print(sig)
        RawSignal(N=162000000 <5400.0>, file=/auto/fhome/kevin/Projects/songephys/data/birds/BroGre4497M/sessions/BroGre4497M_180610_114535/raw/filtered.raw.memmap, sampling_rate=30000.0, mode=r)

    * Plot the data on one electrode at t=30s to t=30.1s

        >>> t_arr, elec_data = sig.time_slice(30.0, 30.1)
        >>> plt.plot(t_arr, elec_data[:, 0])

    """

    def __init__(self, N, file, sampling_rate, mode="r"):
        """Initialize the RawSignal reader

        Params

            N: integer length of the array in number of samples
            file: string path to the memmap file
            sampling_rate: float number of samples per second
            mode (r, w, w+): read or write mode
        """

        self.N = N
        self.file = file
        self.sampling_rate = sampling_rate
        self.set_mode(mode)

    def __repr__(self):
        return "RawSignal(N={} <{:.1f}s>, file={}, sampling_rate={}, mode={})".format(
            self.N, self.N / self.sampling_rate, self.file, self.sampling_rate, self.mode)
        
    def _load_samples(self, n, offset):
        # n = self.N - (n + offset)
        return np.memmap(self.file,
            mode=self.mode,
            dtype="float32",
            shape=(n, 16),
            offset=int(offset * 16 * (32 // 8))
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

        >>> signal = RawSignal(...)
        >>> t_arr, elec_data = signal.time_slice(30.0, 30.1)
        """
        start_idx = int(np.floor(t_start * self.sampling_rate))
        stop_idx = int(np.floor(t_stop * self.sampling_rate))
        n = stop_idx - start_idx
        t_arr = np.arange(start_idx, stop_idx) / self.sampling_rate
        return t_arr, self._load_samples(n, start_idx)
    
    def batches(self, step):
        """Create a generator that iterates over the data

        Parameters:

            step (int): integer number of samples representing the width of
                each window being iterated over
        """
        for start_idx in range(0, self.N - 1, step):
            yield self._load_samples(step, start_idx)

    
class RawSite(object):
    """Access a signal with lazy loading by time slices

    Example:
    
    * Load the raw signals and associate with their relative start times

        >>> site = RawSite([ ... raw signals ...], signal_times=[0, 5400.0, 10800.0])
        >>> print(site)
        RawSite(3 signals, times=[   0. 5400. 10800.0])

        >>> plt.plot(t_arr, elec_data[:, 0])
    """

    def __init__(self, raw_signals, signal_times, mode="r"):
        self.raw_signals = raw_signals
        self.signal_times = np.array(signal_times)
        self.mode = mode
        for raw_signal in self.raw_signals:
            raw_signal.set_mode(self.mode)

    def __repr__(self):
        return "RawSite({} signals, times={})".format(len(self.raw_signals), self.signal_times)
    
    def _get_raw_signal(self, t_start):
        session_idx = np.where(self.signal_times <= t_start)[0][-1]
        return self.signal_times[session_idx], self.raw_signals[session_idx]
        
    def time_slice(self, t_start, t_stop):
        """Access a subset of the data array
        
        Parameters:

            t_start: time in seconds of the first sample, rounded down
            t_stop: time in seconds of the last sample, rounded down

        Returns a tuple of two arrays. The first array is an array of
            times in seconds corresponding to each sample returned. The second
            array is an array of (N x E) where N is the number of samples and
            E is the number of electrodes

        >>> site = RawSite(...)
        >>> t_arr, elec_data = site.time_slice(30.0, 30.1)
        """
        t_signal, raw_signal = self._get_raw_signal(t_start)
        t_arr, signal = raw_signal.time_slice(t_start - t_signal, t_stop - t_signal)
        return t_arr + t_signal, signal
    

def create_raw_signal(session_dir):
    """Copy data from the neo block to a memmapped array for lazy loading"""
    print("Loading {}".format(session_dir))
    session_name = os.path.basename(session_dir)
    block = load_block(os.path.join(session_dir, "segmentation", "{}-filtered.pkl".format(session_name)))
    block = join_continuous_segments(block)
    segment = block.segments[0]
    for sig in segment.analogsignals:
        if sig.name.startswith("Amplifier"):
            break
    
    raw_signal = RawSignal(
        sig.shape[0],
        os.path.join(session_dir, "raw", "filtered.raw.memmap"),
        sig.sampling_rate.item(),
        mode="w+"
    )
    print("Saving {} signal to {}".format(session_dir, raw_signal.file))
    raw_signal.write(sig)
    print("Saved.")
    raw_signal.set_mode("r")
    return raw_signal


def create_raw_site(site_dir):
    """
    Prepare all signals for sessions in the site to be loaded by RawSignal and RawSite
    """
    site_config = load_yaml(os.path.join(site_dir, "sessions.yaml"))
    raw_signals = []
    signal_times = []
    for session in site_config["sessions"]:
        print("Processing {}".format(session))
        session_dir = os.path.join(site_dir, "..", "..", "sessions", session["id"])
        raw_signals.append(create_raw_signal(session_dir))
        signal_times.append(session["t"])
        
    return RawSite(raw_signals, signal_times)



