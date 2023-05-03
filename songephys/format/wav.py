import logging
import os

import numpy as np
import soundfile
from neo.io import NixIO, IntanIO, NWBIO
from neo.io.proxyobjects import AnalogSignalProxy

from .io import load_yaml


def time_to_index(t, sampling_rate):
    """Convert a time to an index"""
    return int(round(t * sampling_rate))

def get_raw_data_from_neo(neo_file, mapping, bad_channels=None):
    if neo_file.endswith(".nix"):
        neo = NixIO(neo_file)
    elif neo_file.endswith(".rhd"):
        neo = IntanIO(neo_file)
    elif neo_file.endswith(".nwb"):
        neo = NWBIO(neo_file)
    else:
        raise IOError("Requires nwb, rhd or nix file")
    block = neo.read_block(lazy=True)
    segment = block.segments[0]

    signals_created = {}

    neo_signals = {}
    neo_names = {}
    i = 0
    for sig in segment.analogsignals:
        if not sig.name:
            logging.warning(
                "Signal has no name. "
                "Using sig.annotations.nix_name instead: "
                "{nix_name}".format(**sig.annotations))
            sig.name = sig.annotations["nix_name"]
        #sig.channel_index.index = np.arange(sig.shape[1])
        neo_signals[sig.name] = sig
        if 'signal_channels' in segment.annotations:
            # for nix, this information gets put into the segment annotations
            neo_names[sig.name] = segment.annotations['signal_channels'][i:i+sig.shape[1]] #list([str(x) for x in sig.channel_index.channel_names])
        elif 'signal_channels' in neo.header:
            # for intan, thie information is within the header
            neo_names[sig.name] = [s[0] for s in neo.header['signal_channels'] if s[-1] == sig.annotations['stream_id']]
        else:
            raise IOError("Signal_channels information not found in header or annotations of %s"%neo_file)
        i+= sig.shape[1]

    def gather(config_key, default_sig_name, raw=True):
        """Return a tuple of original chnanel names, data array, sampling rate"""
        if config_key in mapping and mapping[config_key]:
            data = []
            channel_names = []
            for sig_name, channels in mapping[config_key]:
                if not raw and isinstance(neo_signals[sig_name],AnalogSignalProxy):
                    # load the signal if we need to
                    neo_signals[sig_name] = neo_signals[sig_name].load()
                if isinstance(channels, int):
                    channels = [channels]
                else:
                    channels = list(range(*channels))
                for channel in channels:
                    if bad_channels is not None and str(neo_names[sig_name][channel]) in bad_channels.get(config_key,dict()).get(sig_name,[]):
                        print("Skipping %s"%channel)
                    else:
                        #if bad_channels is not None:
                        #    print("%s not in %s"%(channel,bad_channels.get(config_key,dict()).get(sig_name,[])))
                        if raw:
                            data.append(np.array(neo._raw_data[str(neo_names[sig_name][channel])]).flatten())
                        else:
                            data.append(neo_signals[sig_name][:, channel])
                        channel_names.append([str(sig_name), str(neo_names[sig_name][channel])])
                sampling_rate = neo_signals[sig_name].sampling_rate
            if raw:
                data = np.ascontiguousarray(np.vstack(data).T)
                sampling_rate = int(round(sampling_rate.item()))
            else:
                data = np.hstack(data)
            #data.sampling_rate = sampling_rate
        else:
            print("OHNO")
            sampling_rate = None
            data = neo_signals.get(default_sig_name, None)
            channel_names = [[str(default_sig_name), str(name)] for name in neo_names.get(default_sig_name, [])]

        if data is None:
            return {"data": None, "sampling_rate": None, "channel_names": None }
        
        if sampling_rate is None:
            sampling_rate = int(round(neo_signals[default_sig_name].sampling_rate.item()))

        return {
            "channel_names": channel_names,
            "data": data,
            "sampling_rate": sampling_rate
        }

    ephys = gather("ephys.wav", "Amplifier")
    mic = gather("mic.wav", "Board ADC",raw=False)
    motion = gather("motion.wav", "AUX Input",raw=False)

    return {
        "ephys": ephys,
        "mic": mic,
        "motion": motion
    }

def get_wavfile_data_from_neo(neo_file, mapping):
    """Generate lazy-loadable signals from mic and elec data in a NIX file

    Identifies the mic signal by finding a signal named "Board ADC",
    and identifies electrode signal by finding name "Amplifier".

    Args
    ----
    nix_file : str
        Path to nix file
    output_mic : str
        Path to output memmap file for the mic data
    output_elec : str
        Path to output memmap file for the electrode data
    batch_size : int
        Size of chunks to write to disk at a time
    """
    if neo_file.endswith(".nix"):
        neo = NixIO(neo_file)
    elif neo_file.endswith(".rhd"):
        neo = IntanIO(neo_file)
    elif neo_file.endswith(".nwb"):
        neo = NWBIO(neo_file)
    else:
        raise IOError("Requires nwb, rhd or nix file")
    block = neo.read_block()
    segment = block.segments[0]

    signals_created = {}

    neo_signals = {}
    neo_names = {}
    i = 0
    for sig in segment.analogsignals:
        if not sig.name:
            logging.warning(
                "Signal has no name. "
                "Using sig.annotations.nix_name instead: "
                "{nix_name}".format(**sig.annotations))
            sig.name = sig.annotations["nix_name"]
        #sig.channel_index.index = np.arange(sig.shape[1])
        neo_signals[sig.name] = sig
        if 'signal_channels' in segment.annotations:
            # for nix, this information gets put into the segment annotations
            neo_names[sig.name] = segment.annotations['signal_channels'][i:i+sig.shape[1]] #list([str(x) for x in sig.channel_index.channel_names])
        elif 'signal_channels' in neo.header:
            # for intan, thie information is within the header
            neo_names[sig.name] = [s[0] for s in neo.header['signal_channels'] if s[-1] == sig.annotations['stream_id']]
        else:
            raise IOError("Signal_channels information not found in header or annotations of %s"%neo_file)
        i+= sig.shape[1]

    def gather(config_key, default_sig_name):
        """Return a tuple of original chnanel names, data array, sampling rate"""
        if config_key in mapping and mapping[config_key]:
            data = []
            channel_names = []
            for sig_name, channels in mapping[config_key]:
                if isinstance(channels, int):
                    channels = [channels]
                else:
                    channels = list(range(*channels))
                for channel in channels:
                    data.append(neo_signals[sig_name][:, channel])
                    channel_names.append([str(sig_name), str(neo_names[sig_name][channel])])
                sampling_rate = neo_signals[sig_name].sampling_rate
            data = np.hstack(data)
            data.sampling_rate = sampling_rate
            sampling_rate = int(round(sampling_rate.item()))
        else:
            sampling_rate = None
            data = neo_signals.get(default_sig_name, None)
            channel_names = [[str(default_sig_name), str(name)] for name in neo_names.get(default_sig_name, [])]

        if data is None:
            return {"data": None, "sampling_rate": None, "channel_names": None }
        
        if sampling_rate is None:
            sampling_rate = int(round(neo_signals[default_sig_name].sampling_rate.item()))

        return {
            "channel_names": channel_names,
            "data": data,
            "sampling_rate": sampling_rate
        }

    ephys = gather("ephys.wav", "Amplifier")
    mic = gather("mic.wav", "Board ADC")
    motion = gather("motion.wav", "AUX Input")

    return {
        "ephys": ephys,
        "mic": mic,
        "motion": motion
    }


class LoadedTimeSlice(object):
    """Hold data and its associated time axis

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

        :param t: Length N time array
        :type t:
        Args:
        t: 1d array of length N (number of samples)
        representing the time in seconds of the data
        data: 2d array of shape (N, M), with M = number of channels
        """
        self.t = t
        self.data = data

        def __repr__(self):
            return "LoadedTimeSlice(shape={}, t in [{:.2f}s, {:.2f}s])".format(
            self.data.shape,
            self.t[0],
            self.t[-1])


class AudioFile:
    """Container for a audio file on disk

    Uses the RF64 format instead of WAV becuase WAV is limited to 4GB on disk.
    Arguments
    ---------
    path : str
        Full path to the audio file on disk
    """

    @staticmethod
    def create(path, samplerate, channels):
        if os.path.exists(path):
            raise IOError("File at {} already exists".format(path))

        with soundfile.SoundFile(path, "w+", format="RF64", subtype="FLOAT", samplerate=samplerate, channels=channels) as f:
            pass

        return AudioFile(path)

    def __init__(self, path):
        self._path = path
        self._file = None
        self._is_readonly = True

        if not os.path.exists(path):
            raise IOError("File at {} does not exist. Maybe use AudioFile.create() instead?".format(path))

        with soundfile.SoundFile(path) as f:
            self._sampling_rate = f.samplerate
            self._channels = f.channels
            self._frames = f.frames

    def is_open(self):
        return self._file is not None and not self._file.closed

    def is_closed(self):
        return self._file is None or self._file.closed

    def is_readonly(self):
        return self._is_readonly

    def open(self, readonly=True, **kwargs):
        if self.is_open():
            if readonly != self.is_readonly():
                self.close()
            else:
                return

        if readonly:
            self._file = soundfile.SoundFile(self._path, "r", **kwargs)
            self._is_readonly = True
        else:
            self._file = soundfile.SoundFile(self._path, "w", **kwargs)
            self._is_readonly = False

    def close(self):
        if self.is_open():
            self._file.close()

    def __repr__(self):
        return "<AudioFile: {}; {} Hz; {} Ch; {} frames>".format(
            os.path.basename(self._path),
            self.sampling_rate,
            self.channels,
            self.frames
        )

    @property
    def path(self) -> str:
        """str: Full path to audio file"""
        return self._path

    @property
    def sampling_rate(self) -> int:
        """int: Sampling rate of the audio file"""
        return self._sampling_rate

    @property
    def frames(self) -> int:
        """int: Number of readable samples in audio file"""
        return self._frames

    @property
    def channels(self) -> int:
        """int: Number of channels in audio file"""
        return self._channels

    def batches(self, step: int) -> 'Iterator[int, np.ndarray]':
        """Create a generator that iterates over the data

        Yields the starting index of the batch and the data array
        for that batch.

        Args
        ----
        step : int
            Size in samples of each batch
        """
        if self.is_closed():
            self.open()

        for start_idx in range(0, self.frames, step):
            self._file.seek(start_idx)
            yield start_idx, self._file.read(min(step, self.frames - start_idx))

    def read(self, i0: int, i1: int) -> np.ndarray:
        """Read samples from i0 to i1 on channel
        Arguments
        ---------
        i0 : int
            Starting index to read from (inclusive)
        i1 : int
            Ending index to read until (exclusive)
        Returns
        -------
        data : ndarray
            A 2D array of shape (frames: int, channels: int) containing data from the requested channel.
            The first dimension is the sample index, the second dimension is the channel
            axis.
        """
        idx = np.arange(i0, i1)
        read_start = max(i0, 0)
        read_stop = min(i1, self.frames)

        if read_start >= self.frames or read_stop <= 0:
            return idx, np.zeros((i1 - i0, self.channels))

        if self.is_closed():
            self.open()

        pad_zeros_before = max(0 - i0, 0)
        pad_zeros_after = max(i1 - self.frames, 0)

        self._file.seek(read_start)
        data = self._file.read(read_stop - read_start, dtype=np.float32, always_2d=True)
        data = np.vstack([
            np.zeros((pad_zeros_before, data.shape[1])),
            data,
            np.zeros((pad_zeros_after, data.shape[1]))
        ])

        return idx, data

    def write(self, i0: int, data: np.ndarray):
        if self.is_closed() or self.is_readonly():
            self.open(readonly=False)

        self._file.seek(i0)
        self._file.write(data)
        self._frames = self._file.frames

    def time_slice(self, t0: float, t1: float):
        idx, data = self.read(time_to_index(t0, self.sampling_rate), time_to_index(t1, self.sampling_rate))
        return LoadedTimeSlice(t=idx / self.sampling_rate, data=data)


class MultiWavReader:
    """Data access to audio data in a series of Audio Files

    Arguments
    ---------
    start_times: List[float]
        An ordered list of start times for each audio file in ``files``. Although the relative
        times of each file can be inferred by the number of samples and sampling rate,
        this program uses the start_time as the ground truth. This probably will cause
        some small errors at the edges of files but is necessary when files are not stopped and started
        back-to-back (e.g. recording is paused briefly and then restarted between two sessions
        in one site)
    files : List[AudioFile]
        An ordered list of AudioFile to read in the MultiWavReader. The indices determine
        the order data will be read, so the MultiWavReader should be initialized with AudioFiles
        in the correct order (if relevant). All AudioFiles must have the same number of
        channels and sampling rate.
    """

    def __init__(self, start_times: 'List[float]', files: 'List[AudioFile]'):
        """Initialize the MultiWavReader
        """
        self.start_times = np.array(start_times)
        self._files = files

        rates = [f.sampling_rate for f in self._files]
        channels = [f.channels for f in self._files]

        if not all([r == rates[0] for r in rates]):
            raise ValueError("Cannot instantiate MultiWavReader with AudioFiles of different rates: {}".format(rates))

        if not all([c == channels[0] for c in channels]):
            raise ValueError("Cannot instantiate MultiWavReader with AudioFiles with different channel counts: {}".format(channels))

    def __repr__(self):
        return "<MultiWavReader: {}>".format(len(self.files))

    @property
    def channels(self) -> int:
        """int: Number of channels in all AudioFiles of this MultiWavReader"""
        return self._files[0].channels

    @property
    def sampling_rate(self) -> int:
        """int: Sampling rate of audio in this MultiWavReader"""
        return self._files[0].sampling_rate

    @property
    def frames(self) -> int:
        """int: Total number of samples in the entire MultiWavReader"""
        return int(np.sum([f.frames for f in self._files]))

    @property
    def files(self) -> 'List[AudioFile]':
        """List[AudioFile]: An ordered list of the AudioFiles in the MultiWavReader"""
        return self._files

    def iter_files(self, start=0) -> 'Iterable[Tuple[int, int], AudioFile]':
        """Iterate over files in the MultiWavReader

        Returns
        -------
        generator
            An iterator that yields tuples of the form ((start, stop), audiofile), where
            start and stop are ints instances ponting to the startpoint (inclusive)
            and endpoint (exclusive) of the AudioFile, audiofile.
        """
        start_frame = time_to_index(start, self.sampling_rate)
        for i, f in enumerate(self.files):
            frame = time_to_index(self.start_times[i], self.sampling_rate)
            if frame + f.frames >= start_frame:
                yield ((frame, frame + f.frames), f)

    def close_files(self):
        for f in self.files:
            f.close()

    def _get_session_by_time(self, t):
        """Get the first signal containing time t_start

        First, find the first session that contains t_start using
        self.signal_times. Then, grab the corresponding LazySignal
        for that session from each of the keys in self._signals.

        Args
        ----
        t_start : float
            Start time in seconds to search for

        Returns a tuple of the start time of the session in seconds (float),
            and a dictionary mapping signal string names (e.g. "filtered") to
            the LazySignal object for that session.
        """
        session_idx = np.where(self.start_times <= t)[0][-1]
        return self.start_times[session_idx], self.files[session_idx]

    def time_slice(
            self,
            start: float,
            stop: float,
        ) -> np.ndarray:
        """Reads data of a start->stop slice,

        Arguments
        ---------
        start : float
        stop : float

        Returns
        -------
        data : np.ndarray
            A 2D array of shape (stop - start, len(channels)) representing data between
            the start and stop indices.
        """
        stop_idx = time_to_index(stop, self.sampling_rate)
        start_idx = time_to_index(start, self.sampling_rate)
        n = stop_idx - start_idx
        # If you request past the end of the slice, just pad with zeros
        # The calling function should deal with completing the rest of the slice
        t_arr = np.arange(start_idx, stop_idx) / self.sampling_rate

        final_result = LoadedTimeSlice(t=t_arr, data=np.zeros((n, self.channels)))

        if start > self.frames:
            return final_result

        last_file_t, last_file_obj = self._get_session_by_time(stop)
        last_file_t /= self.sampling_rate

        for (file_t, _), file_obj in self.iter_files(start=start):
            file_t /= self.sampling_rate
            result = file_obj.time_slice(start - file_t, stop - file_t)
            result.t += file_t
            final_result.data[final_result.t >= file_t] = result.data[final_result.t >= file_t]

            if file_t == last_file_t:
                break

        return final_result


class SiteReader(object):
    """Read data from site as if it were a continuous recording

    Reads directly from the sessions stored in

    Attributes
    ----------
    signal_times : list
        Relative start times (in seconds) of each enclosed LazySignal
    _signals : list
        List of LazySignal objects to access

    Methods
    -------
    time_slice(t_start, t_stop, signal=None)
        Load data within a time range into memory
    max_time()
        Get the maximum time of the last signal
    add_signals(signal_name, signals)
        Add a new signal by key to the site

    Example: Load the raw signals and associate with their
        relative start times
    >>> site = LazySite([ ... raw signals ...], signal_times=[0, 5400.0, 10800.0])
    >>> print(site)
    LazySite(3 signals, times=[   0. 5400. 10800.0])
    >>> plt.plot(t_arr, elec_data[:, 0])
    """

    signal_filenames = {
        "ephys": "ephys.wav",
        "mic": "mic.wav",
        "motion": "motion.wav",
        "ephys_filtered": "ephys_filtered.wav"
    }

    @staticmethod
    def from_site(site_dir):
        experiment_dir = os.path.dirname(os.path.dirname(site_dir))
        config = load_yaml(os.path.join(site_dir, "site_config.yaml"))

        sessions = [os.path.join(experiment_dir, "sessions", session["id"]) for session in config["sessions"]]
        session_times = [session["t"] for session in config["sessions"]]

        return SiteReader(sessions, session_times)

    @staticmethod
    def from_session(session_dir):
        sessions = [session_dir]
        session_times = [0.0]
        return SiteReader(sessions, session_times)

    def __init__(self, session_dirs, session_times):
        self.session_times = session_times
        self.session_dirs = session_dirs
        self.signals = {}

        for signal, signal_key in self.signal_filenames.items():
            signal_paths = [os.path.join(session_dir, "raw", signal_key) for session_dir in self.session_dirs]
            if all([os.path.exists(path) for path in signal_paths]):
                self.signals[signal] = MultiWavReader(
                    self.session_times,
                    [AudioFile(path) for path in signal_paths]
                )
            else:
                self.signals[signal] = None

    def time_slice(self, t_start, t_stop, signal=None):
        """Access a subset of the data array by time

        Args
        ----
        t_start : float
            Time in seconds of the first sample, rounded down
        t_stop : float
            Time in seconds of the last sample, rounded down

        Kwargs
        ------
        signal : str (default=None)
            String name of the specific signal to load (e.g. "filtered",
            "mic", "raw"). If left as None, load all the signals into a
            LoadedTimeSliceDict. Specify this to avoid loading unneeded
            arrays.

        If signal is not provided, returns a LoadedTimeSliceDict whose keys
            are the signal names and values are LoadedTimeSlice objects.
            If signal is specified, directly return the LoadedTimeSlice object
            for that array.
        """
        if signal is not None and self.signals[signal]:
            return self.signals[signal].time_slice(t_start, t_stop)
        elif signal is not None:
            raise Exception("Signal {} not found in {}".format(signal, self))
        else:
            result = {}
            for signal, reader in self.signals.items():
                if reader is not None:
                    result[signal] = reader.time_slice(t_start, t_stop)
                else:
                    result[signal] = None
            return result
