"""Support for Intan "One File Per Signal Type" recordings (.dat files)

When Intan RHX is set to save in the "One File Per Signal Type" format, each
recording is a folder that looks like this::

    Bird__YYMMDD_HHMMSS/
        info.rhd          header only (no data), same header as a .rhd file
        settings.xml
        time.dat          int32 sample index, one per sample
        amplifier.dat     int16, [n_samples x n_amplifier_channels], 0.195 uV/bit
        auxiliary.dat     uint16, [n_samples x n_aux_channels], 37.4 uV/bit
                          (RHX writes aux at the full sample rate by holding
                          each aux sample for 4 amplifier samples)
        supply.dat        uint16 (optional)
        analogin.dat      uint16, [n_samples x n_adc_channels]
        digitalin.dat     uint16, one packed word per sample (bit i = DIGITAL-IN-i)
        digitalout.dat    uint16 (optional)
        lowpass.dat ...   (optional, not used by songephys)

In songephys, a recording like this is referred to by the path of its
``info.rhd`` file (e.g. ``Bird__YYMMDD_HHMMSS/info.rhd`` in a site filelist).
The recording timestamp is taken from the folder name rather than the filename.

The main entry point is :func:`read_intan_dat`, which returns a SpikeInterface
recording for one of the same stream names that neo's IntanRawIO uses, so it
can be used interchangeably with the .rhd reader in the pipeline.
"""
import os
import struct

import numpy as np

INFO_FILENAME = "info.rhd"

AMPLIFIER_STREAM = "RHD2000 amplifier channel"
AUX_STREAM = "RHD2000 auxiliary input channel"
SUPPLY_STREAM = "RHD2000 supply voltage channel"
ADC_STREAM = "USB board ADC input channel"
DIGITAL_IN_STREAM = "USB board digital input channel"
DIGITAL_OUT_STREAM = "USB board digital output channel"

# signal_type codes in the rhd header
_SIGNAL_TYPES = {
    0: AMPLIFIER_STREAM,
    1: AUX_STREAM,
    2: SUPPLY_STREAM,
    3: ADC_STREAM,
    4: DIGITAL_IN_STREAM,
    5: DIGITAL_OUT_STREAM,
}

# Data file and on-disk dtype for each stream
_STREAM_FILES = {
    AMPLIFIER_STREAM: ("amplifier.dat", np.int16),
    AUX_STREAM: ("auxiliary.dat", np.uint16),
    SUPPLY_STREAM: ("supply.dat", np.uint16),
    ADC_STREAM: ("analogin.dat", np.uint16),
    DIGITAL_IN_STREAM: ("digitalin.dat", np.uint16),
    DIGITAL_OUT_STREAM: ("digitalout.dat", np.uint16),
}

# In .rhd files the auxiliary inputs are sampled at 1/4 of the amplifier rate.
# RHX upsamples them (sample-and-hold) when writing auxiliary.dat. We take every
# 4th sample so the aux stream matches what the .rhd reader gives.
AUX_DECIMATION = 4


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

class _Reader:
    def __init__(self, fid):
        self.fid = fid

    def read(self, fmt):
        size = struct.calcsize(fmt)
        data = self.fid.read(size)
        if len(data) != size:
            raise IOError("Unexpected end of Intan header")
        return struct.unpack(fmt, data)

    def one(self, fmt):
        return self.read("<" + fmt)[0]

    def qstring(self):
        length, = self.read("<I")
        if length == 0xFFFFFFFF:
            return ""
        return self.fid.read(length).decode("utf-16-le")


def read_rhd_header(filename):
    """Parse the header of an Intan .rhd (or info.rhd) file

    Returns a dict with keys: version, sample_rate, eval_board_mode,
    reference_channel, notes, and "channels": a list of dicts with keys
    native_channel_name, custom_channel_name, native_order, custom_order,
    signal_type, stream_name, port_prefix, chip_channel, board_stream,
    impedance_magnitude, impedance_phase. Only enabled channels are listed,
    in the order they are saved in the data files.
    """
    with open(filename, "rb") as fid:
        r = _Reader(fid)
        magic = r.one("I")
        if magic != 0xC6912702:
            raise ValueError("{} is not an Intan RHD header file".format(filename))
        major, minor = r.read("<hh")
        version = (major, minor)

        header = dict(version=version)
        header["sample_rate"] = r.one("f")
        header["dsp_enabled"] = r.one("h")
        (header["actual_dsp_cutoff_frequency"],
         header["actual_lower_bandwidth"],
         header["actual_upper_bandwidth"],
         header["desired_dsp_cutoff_frequency"],
         header["desired_lower_bandwidth"],
         header["desired_upper_bandwidth"]) = r.read("<ffffff")
        header["notch_filter_mode"] = r.one("h")
        header["desired_impedance_test_frequency"], header["actual_impedance_test_frequency"] = r.read("<ff")
        header["notes"] = [r.qstring(), r.qstring(), r.qstring()]

        header["num_temp_sensor_channels"] = r.one("h") if version >= (1, 1) else 0
        header["eval_board_mode"] = r.one("h") if version >= (1, 3) else 0
        header["reference_channel"] = r.qstring() if version >= (2, 0) else ""

        channels = []
        num_groups = r.one("h")
        for _ in range(num_groups):
            group_name = r.qstring()
            group_prefix = r.qstring()
            group_enabled, group_num_channels, _group_num_amps = r.read("<hhh")
            if group_num_channels == 0 or not group_enabled:
                continue
            for _ in range(group_num_channels):
                native_name = r.qstring()
                custom_name = r.qstring()
                (native_order, custom_order, signal_type, channel_enabled,
                 chip_channel, board_stream) = r.read("<hhhhhh")
                r.read("<hhhh")  # trigger settings (unused)
                imp_mag, imp_phase = r.read("<ff")
                if not channel_enabled:
                    continue
                channels.append(dict(
                    native_channel_name=native_name,
                    custom_channel_name=custom_name,
                    native_order=native_order,
                    custom_order=custom_order,
                    signal_type=signal_type,
                    stream_name=_SIGNAL_TYPES.get(signal_type, "unknown"),
                    port_name=group_name,
                    port_prefix=group_prefix,
                    chip_channel=chip_channel,
                    board_stream=board_stream,
                    impedance_magnitude=imp_mag,
                    impedance_phase=imp_phase,
                ))
        header["channels"] = channels
    return header


# ---------------------------------------------------------------------------
# Paths and timestamps
# ---------------------------------------------------------------------------

def is_intan_dat_path(path):
    """True if path points to a one-file-per-signal-type recording

    Accepts either the ``info.rhd`` file or the recording folder itself.
    """
    path = str(path)
    if os.path.basename(path) == INFO_FILENAME:
        return True
    return os.path.isdir(path) and os.path.exists(os.path.join(path, INFO_FILENAME))


def dat_folder(path):
    """Return the recording folder given either info.rhd or the folder"""
    path = str(path)
    if os.path.basename(path) == INFO_FILENAME:
        return os.path.dirname(path)
    return path


def recording_name(path):
    """Name that carries the recording timestamp, e.g. ``Bird__YYMMDD_HHMMSS``

    For .rhd files this is the filename without extension, for dat recordings
    it is the folder name (since the header is always called info.rhd).
    """
    path = str(path).replace("\\", "/").rstrip("/")
    if is_intan_dat_path(path):
        return os.path.basename(dat_folder(path))
    return os.path.splitext(os.path.basename(path))[0]


def _stream_channels(header, stream_name):
    return [c for c in header["channels"] if c["stream_name"] == stream_name]


def get_dat_num_samples(path):
    """Number of samples in a dat recording, from the size of time.dat"""
    folder = dat_folder(path)
    return os.path.getsize(os.path.join(folder, "time.dat")) // 4


def check_dat_timestamps_continuous(path, chunk_size=30000 * 600):
    """Check that time.dat has no dropped samples

    Returns (is_continuous, num_gaps)
    """
    folder = dat_folder(path)
    t = np.memmap(os.path.join(folder, "time.dat"), dtype=np.int32, mode="r")
    num_gaps = 0
    last = None
    for start in range(0, len(t), chunk_size):
        chunk = np.asarray(t[start:start + chunk_size], dtype=np.int64)
        if last is not None:
            chunk = np.concatenate(([last], chunk))
        num_gaps += int(np.count_nonzero(np.diff(chunk) != 1))
        last = chunk[-1]
    return num_gaps == 0, num_gaps


def _stream_layout(path, stream_name, header=None):
    """Return (data_file, dtype, channels, num_columns) for a stream"""
    folder = dat_folder(path)
    if header is None:
        header = read_rhd_header(os.path.join(folder, INFO_FILENAME))
    if stream_name not in _STREAM_FILES:
        raise ValueError("Unknown stream {}. Choose from {}".format(stream_name, list(_STREAM_FILES)))
    fn, dtype = _STREAM_FILES[stream_name]
    data_file = os.path.join(folder, fn)
    if not os.path.exists(data_file):
        raise IOError("{} not found for stream '{}'".format(data_file, stream_name))
    channels = _stream_channels(header, stream_name)
    if stream_name in (DIGITAL_IN_STREAM, DIGITAL_OUT_STREAM):
        # digital channels are packed into a single uint16 word per sample
        num_columns = 1
    else:
        num_columns = len(channels)
    if num_columns == 0:
        raise ValueError("No enabled channels for stream '{}' in {}".format(stream_name, folder))

    n_samples = get_dat_num_samples(folder)
    n_values = os.path.getsize(data_file) // np.dtype(dtype).itemsize
    if n_values != n_samples * num_columns:
        raise ValueError(
            "{} has {} values, expected {} samples x {} channels from info.rhd/time.dat".format(
                data_file, n_values, n_samples, num_columns))
    return data_file, dtype, channels, num_columns


# ---------------------------------------------------------------------------
# SpikeInterface recording
# ---------------------------------------------------------------------------

def read_intan_dat(path, stream_name=AMPLIFIER_STREAM):
    """Load one stream of an Intan one-file-per-signal-type recording

    Parameters
    ----------
    path : str
        Path to the recording's info.rhd, or to the recording folder
    stream_name : str
        One of the neo IntanRawIO stream names, e.g. "RHD2000 amplifier channel",
        "RHD2000 auxiliary input channel", "USB board ADC input channel",
        "USB board digital input channel"

    Returns
    -------
    IntanDatRecordingExtractor

    Notes
    -----
    Dtypes and channel ids match what the .rhd reader returns, except that
    amplifier data is already signed int16 (in .rhd files it is uint16 offset
    by 32768). Use :func:`songephys.pipeline.file_setup.ensure_signed` rather
    than ``unsigned_to_signed`` when the data could come from either format.
    The digital input stream is a single channel holding the packed uint16
    word, matching the .rhd digital stream.
    """
    return IntanDatRecordingExtractor(path, stream_name=stream_name)


try:
    from spikeinterface.core import BaseRecording, BaseRecordingSegment
except ImportError:  # pragma: no cover - spikeinterface is a pipeline dependency
    BaseRecording = object
    BaseRecordingSegment = object


class IntanDatRecordingExtractor(BaseRecording):
    """SpikeInterface recording for one stream of an Intan .dat recording"""

    extractor_name = "IntanDat"
    mode = "folder"
    name = "intan_dat"

    def __init__(self, path, stream_name=AMPLIFIER_STREAM):
        folder = os.path.abspath(dat_folder(path))
        header = read_rhd_header(os.path.join(folder, INFO_FILENAME))
        data_file, dtype, channels, num_columns = _stream_layout(folder, stream_name, header)

        fs = float(header["sample_rate"])
        step = 1
        if stream_name == AUX_STREAM:
            step = AUX_DECIMATION
            fs = fs / AUX_DECIMATION

        if stream_name in (DIGITAL_IN_STREAM, DIGITAL_OUT_STREAM):
            channel_ids = ["DIGITAL-IN" if stream_name == DIGITAL_IN_STREAM else "DIGITAL-OUT"]
        else:
            channel_ids = [c["native_channel_name"] for c in channels]

        BaseRecording.__init__(self, sampling_frequency=fs, channel_ids=channel_ids, dtype=dtype)

        memmap = np.memmap(data_file, dtype=dtype, mode="r").reshape(-1, num_columns)
        self.add_recording_segment(IntanDatRecordingSegment(memmap, step, fs))

        gains, offsets = _gains_offsets(header, stream_name, num_columns)
        self.set_channel_gains(gains)
        self.set_channel_offsets(offsets)
        if stream_name not in (DIGITAL_IN_STREAM, DIGITAL_OUT_STREAM):
            self.set_property("channel_name", [c["custom_channel_name"] for c in channels])

        self._kwargs = dict(path=folder, stream_name=stream_name)


class IntanDatRecordingSegment(BaseRecordingSegment):
    def __init__(self, memmap, step, sampling_frequency):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._memmap = memmap
        self._step = step
        self._num_samples = memmap.shape[0] // step

    def get_num_samples(self):
        return self._num_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._num_samples
        s = self._step
        traces = self._memmap[start_frame * s:end_frame * s:s]
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        # always return a writable in-memory copy (downstream preprocessors
        # like RemoveJumps modify traces in place)
        return np.array(traces)


def _gains_offsets(header, stream_name, n):
    """Gains/offsets to microvolts (amplifier, aux, supply) or volts (ADC)"""
    if stream_name == AMPLIFIER_STREAM:
        return [0.195] * n, [0.0] * n
    if stream_name == AUX_STREAM:
        return [37.4] * n, [0.0] * n
    if stream_name == SUPPLY_STREAM:
        return [74.8] * n, [0.0] * n
    if stream_name == ADC_STREAM:
        mode = header["eval_board_mode"]
        if mode == 1:
            gain = 152.59e-6
        elif mode == 13:
            gain = 312.5e-6
        else:
            return [50.354e-6] * n, [0.0] * n
        return [gain] * n, [-32768 * gain] * n
    return [1.0] * n, [0.0] * n
