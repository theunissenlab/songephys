"""Electrical stimulation artefact handling

Stimulation pulses are found from a TTL on a digital input (e.g. a Master-8
pulse on DIGITAL-IN-01). Around each pulse the raw signal is blanked, and the
data between pulses is band-pass filtered piece by piece, so the filter never
runs across the artefact (a zero-phase filter run across it rings for about
+/-8 ms, producing fake spikes before and after each pulse).

This module holds the numpy/scipy logic so it can be tested without
SpikeInterface, luigi or pynwb. The pipeline wrappers live in
``songephys.pipeline.file_setup`` (``DetectStimEvents``, ``StimBlankBandpass``)
and ``songephys.pipeline.nwb_utils`` (``AddStimEventsNWB``).

Config
------
Defaults are in ``STIM_ARTIFACT_DEFAULTS``. They can be overridden in the
project config under ``STIM_ARTIFACT`` and per site in the site config under
``stim_artifact`` (site wins). Per-bout annotations (polarity, current, notes)
go in the site config under ``stim_bouts``::

    stim_artifact:
      enabled: true
      digital_channel: DIGITAL-IN-01
      ms_before: 0.5
      ms_after: 1.5
    stim_bouts:
      - {bout: 0, polarity: "-", current_ua: 500}
      - {bout: 1, polarity: "+", current_ua: 500, notes: "..."}
"""
import json
import re
import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt

STIM_EVENTS_VERSION = 1

STIM_ARTIFACT_DEFAULTS = dict(
    enabled=False,
    # digital line carrying one TTL edge per stimulation pulse: a native name
    # (DIGITAL-IN-01), a name from site_config['digital_channel_names'], or a bit index
    digital_channel="DIGITAL-IN-01",
    # blank window around each pulse, relative to the TTL rising edge
    ms_before=0.5,
    ms_after=1.5,
    # constant delay between the TTL edge and the artefact
    offset_ms=0.0,
    # extra data read on each side of a chunk when filtering
    margin_ms=50.0,
    # band-pass applied between pulses (same as the normal pipeline)
    freq_min=300.0,
    freq_max=8000.0,
    filter_order=5,
    # a pulse counts as delivered if its artefact score exceeds this
    delivered_threshold=20.0,
    # blank pulses whose TTL fired but no artefact was seen (e.g. stimulator off)
    blank_undelivered=False,
    # pulses separated by more than this many seconds start a new bout
    bout_gap_s=5.0,
    # RemoveJumps leaves alone the blank window plus this much after it
    remove_jumps_extra_ms=1.0,
    # --- per-channel blanking and tail subtraction (see stim_tail.py) ---
    # adaptive_blank: end each channel's blank where that channel's fast artefact
    # ends (last sample on the rail or with a big jump, searched up to
    # adaptive_search_ms after the pulse) plus adaptive_extra_samples, kept within
    # [adaptive_min_ms_after, adaptive_max_ms_after]. Replaces ms_after for
    # delivered pulses. Rail/jump thresholds are in raw counts (0.195 uV each).
    adaptive_blank=False,
    adaptive_search_ms=2.0,
    adaptive_rail_counts=32700,
    adaptive_jump_counts=1500,
    adaptive_extra_samples=10,
    adaptive_min_ms_after=0.0,
    adaptive_max_ms_after=3.0,
    # subtract_tail: fit tail_n_exp exponentials + offset to tail_fit_ms of raw data
    # after each channel's blank and subtract the fit (tapered to 0 over the last
    # tail_taper_ms) before band-passing. tail_workers: processes for the fits
    # (0 = min(8, CPU count)).
    subtract_tail=False,
    tail_n_exp=3,
    tail_fit_ms=500.0,
    tail_taper_ms=50.0,
    tail_workers=0,
    # blank_after_filter_ms: for the fitted pulses, after subtracting the tail and
    # band-passing from each channel's (adaptive) blank end, also zero the output
    # up to this many ms after the pulse. The filter still starts early (the fast
    # transient settles before this point) but the dirty window is not kept.
    # 0 = off.
    blank_after_filter_ms=0.0,
    # extra_blank_s: extra [start_s, stop_s] windows to blank on every channel, for
    # artefacts with no TTL (e.g. the transient when the stimulator is connected).
    # Like the pulse blanks, the band-pass is run separately on each side.
    extra_blank_s=(),
    # extra_artifacts_s: approximate times (s) of artefacts with no TTL that should be
    # treated like a stimulation pulse (e.g. the jump when the stimulator is connected).
    # The jump is located within extra_artifact_search_ms of each time; the blank runs
    # from ms_before before it to extra_artifact_ms_after after it (longer on channels
    # that stay on the rail, with adaptive_blank), the tail after it is fitted and
    # subtracted (subtract_tail), and each side is band-passed separately.
    # blank_after_filter_ms does not apply to these.
    extra_artifacts_s=(),
    extra_artifact_search_ms=100.0,
    extra_artifact_ms_after=1.0,
)


# ---------------------------------------------------------------------------
# Config and digital channels
# ---------------------------------------------------------------------------

def get_stim_artifact_config(site_config=None, project_config=None):
    """Merge defaults <- project STIM_ARTIFACT <- site stim_artifact

    Keys are matched case-insensitively, so ``ENABLED`` and ``enabled`` both work.
    """
    cfg = dict(STIM_ARTIFACT_DEFAULTS)
    for source in ((project_config or {}).get("STIM_ARTIFACT"),
                   (site_config or {}).get("stim_artifact")):
        for k, v in (source or {}).items():
            k = str(k).lower()
            if k not in cfg:
                warnings.warn("Unknown stim_artifact config key %r ignored" % k)
                continue
            cfg[k] = v
    for k in ("enabled", "blank_undelivered", "adaptive_blank", "subtract_tail"):
        cfg[k] = bool(cfg[k])
    cfg["extra_blank_s"] = parse_extra_blank_s(cfg["extra_blank_s"])
    cfg["extra_artifacts_s"] = parse_extra_artifacts_s(cfg["extra_artifacts_s"])
    return cfg


def parse_extra_artifacts_s(value):
    """Validate ``extra_artifacts_s``: a time or list of times in seconds (>= 0), returned sorted"""
    if value is None:
        return []
    try:
        arr = np.atleast_1d(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        raise ValueError("stim_artifact extra_artifacts_s must be a list of times in seconds, got %r" % (value,))
    if arr.ndim != 1 or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("stim_artifact extra_artifacts_s must be a list of times in seconds, got %r" % (value,))
    return np.sort(arr).tolist()


def find_jump_onset(read_traces, approx_frame, fs, n_samples, search_ms=100.0, frac=0.2):
    """Locate an artefact's jump near ``approx_frame``

    Uses the median over channels of the sample-to-sample change, so it follows
    jumps common to the probe and ignores spikes and LFP. The onset is the first
    sample whose change reaches ``frac`` of the largest change within
    ``search_ms`` either side of ``approx_frame``.

    read_traces : callable(start, stop) -> (T, C) raw amplifier data

    Returns (onset_frame, score): the first sample after the jump begins, and the
    largest change divided by the typical change in the window (like
    ``score_artifacts``; noise scores a few). (None, 0.0) if there is no data.
    """
    s = int(round(search_ms * fs / 1e3))
    a, b = max(0, int(approx_frame) - s), min(int(n_samples), int(approx_frame) + s + 1)
    if b - a < 3:
        return None, 0.0
    x = np.asarray(read_traces(a, b), dtype=np.float64)
    d = np.median(np.abs(np.diff(x, axis=0)), axis=1)
    if d.max() <= 0:
        return None, 0.0
    k = int(np.argmax(d >= frac * d.max()))
    return a + k + 1, float(d.max() / max(np.median(d), 1e-9))


def find_extra_artifacts(read_traces, cfg, fs, n_samples):
    """Jump onsets for ``extra_artifacts_s``; returns (onset_frames, scores, approx_s) of those found

    Times with no clear jump nearby (score below ``delivered_threshold``) or outside
    the recording are skipped with a warning.
    """
    onsets, scores, approx = [], [], []
    for t in parse_extra_artifacts_s(cfg.get("extra_artifacts_s")):
        f = int(round(t * fs))
        onset, score = (None, 0.0) if f >= n_samples else find_jump_onset(
            read_traces, f, fs, n_samples, cfg["extra_artifact_search_ms"])
        if onset is None or score <= cfg["delivered_threshold"]:
            warnings.warn("extra_artifacts_s: no clear jump within %.0f ms of %.3f s (score %.1f); ignored"
                          % (cfg["extra_artifact_search_ms"], t, score))
            continue
        if onsets and onset == onsets[-1]:
            continue
        onsets.append(int(onset)); scores.append(score); approx.append(float(t))
    return np.asarray(onsets, dtype=np.int64), np.asarray(scores, dtype=float), approx


def parse_extra_blank_s(value):
    """Validate ``extra_blank_s``: a list of [start_s, stop_s] pairs (stop > start >= 0)

    A single pair ([23.5, 23.8]) is accepted too. Returns a list of [float, float].
    """
    if value is None:
        return []
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("stim_artifact extra_blank_s must be a list of [start_s, stop_s] pairs, got %r" % (value,))
    if arr.size == 0:
        return []
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ValueError("stim_artifact extra_blank_s must be a list of [start_s, stop_s] pairs, got %r" % (value,))
    if (arr[:, 0] < 0).any() or (arr[:, 1] <= arr[:, 0]).any():
        raise ValueError("stim_artifact extra_blank_s: each pair needs stop_s > start_s >= 0, got %r" % (value,))
    return arr.tolist()


def extra_blank_windows(cfg, fs, n_samples):
    """``extra_blank_s`` as sorted, merged [start, stop) sample windows within the recording"""
    pairs = np.asarray(parse_extra_blank_s(cfg.get("extra_blank_s")), dtype=float).reshape(-1, 2)
    if len(pairs) == 0:
        return np.zeros((0, 2), np.int64)
    w = np.stack([np.floor(pairs[:, 0] * fs), np.ceil(pairs[:, 1] * fs)], 1).astype(np.int64)
    outside = w[:, 0] >= n_samples
    if outside.any():
        warnings.warn("extra_blank_s windows starting after the end of the recording ignored: %s"
                      % pairs[outside].tolist())
    return merge_windows(np.clip(w[~outside], 0, n_samples))


def needs_stim_tails(cfg):
    """True if per-channel blanking or tail subtraction is switched on."""
    return bool(cfg["enabled"] and (cfg["adaptive_blank"] or cfg["subtract_tail"]))


def get_digital_channel_bits(site_config, default_names=None):
    """Return [(bit, name), ...] for the enabled digital inputs

    In the packed digital word, bit i is DIGITAL-IN-(i+1) regardless of which
    channels are enabled, so the position of a name in digital_channel_names
    is not its bit when channels are skipped (e.g. DIGITAL-IN-01, 02, 09, ...).
    Uses site_config['digital_channel_bits'] if present (written by create-site),
    otherwise parses native names like DIGITAL-IN-09, and falls back to list
    position for custom names (the old behaviour).
    """
    names = site_config.get('digital_channel_names', default_names) or []
    bits = site_config.get('digital_channel_bits')
    if bits is None or len(bits) != len(names):
        bits = []
        for i, name in enumerate(names):
            m = re.fullmatch(r'DIGITAL-IN-?(\d+)', str(name).strip(), flags=re.IGNORECASE)
            bits.append(int(m.group(1)) - 1 if m else i)
    return list(zip([int(b) for b in bits], names))


def resolve_digital_bit(channel, site_config=None):
    """Bit index (in the packed digital word) of a digital input

    ``channel`` may be an int bit index, a native name such as DIGITAL-IN-01
    (bit 0), or a custom name listed in site_config['digital_channel_names'].
    """
    if isinstance(channel, (int, np.integer)):
        return int(channel)
    name = str(channel).strip()
    m = re.fullmatch(r'DIGITAL-IN-?(\d+)', name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)) - 1
    for bit, n in get_digital_channel_bits(site_config or {}):
        if str(n).strip().lower() == name.lower():
            return bit
    raise ValueError("Cannot find digital channel %r (use DIGITAL-IN-NN, a bit index, "
                     "or a name from digital_channel_names)" % channel)


def ttl_on_off(dig_data):
    """Rising and falling edge sample indices of a 0/1 digital trace

    Traces that are mostly high are inverted first. A pulse still high at the
    end of the recording ends at len(dig_data); a falling edge before the
    first rising edge is ignored. Returns arrays of equal length.
    """
    dig_data = np.asarray(dig_data, dtype=np.int8)
    if dig_data.mean() > .5:
        dig_data = 1 - dig_data
    d = np.diff(dig_data)
    rising = np.where(d == 1)[0]
    falling = np.where(d == -1)[0]
    if len(rising) == 0:
        return rising, rising.copy()
    falling = falling[falling > rising[0]]
    if len(falling) < len(rising):
        falling = np.append(falling, len(dig_data))
    return rising, falling


def find_ttl_edges(read_chunk, n_samples, bit=None, chunk_size=30_000_000):
    """Rising/falling edges of a digital line, read in chunks

    Parameters
    ----------
    read_chunk : callable(start, stop) -> 1D array
        Returns digital samples [start, stop). With ``bit`` set, these are
        packed words and the bit is extracted; otherwise nonzero = high.
    n_samples : int
    bit : int or None

    Returns
    -------
    rising, falling : int64 arrays of equal length
        ``rising`` is the first high sample of each pulse, ``falling`` the
        first low sample after it (``n_samples`` if still high at the end).
    """
    rising, falling = [], []
    prev = None  # a pulse already high at sample 0 has no rising edge
    for start in range(0, n_samples, chunk_size):
        d = np.asarray(read_chunk(start, min(start + chunk_size, n_samples)))
        b = ((d >> bit) & 1) if bit is not None else (d != 0)
        b = b.astype(np.int8)
        if prev is None:
            prev = b[0] if len(b) else 0
        db = np.diff(np.concatenate([[prev], b]))
        rising.append(np.where(db == 1)[0] + start)
        falling.append(np.where(db == -1)[0] + start)
        prev = b[-1] if len(b) else prev
    rising = np.concatenate(rising).astype(np.int64) if rising else np.zeros(0, np.int64)
    falling = np.concatenate(falling).astype(np.int64) if falling else np.zeros(0, np.int64)
    if len(rising) == 0:
        return rising, np.zeros(0, np.int64)
    falling = falling[falling > rising[0]]
    if len(falling) < len(rising):
        falling = np.concatenate([falling, [n_samples]]).astype(np.int64)
    return rising, falling[:len(rising)]


# ---------------------------------------------------------------------------
# Delivered / not delivered
# ---------------------------------------------------------------------------

def score_artifacts(read_traces, frames, fs, n_samples, gain_uv=1.0,
                    pre_ms=10.0, post_ms=1.0):
    """How strongly each TTL is followed by a stimulation artefact

    The score is the median over channels of the largest sample-to-sample
    jump within ``post_ms`` after the TTL, divided by that channel's
    sample-to-sample noise just before it. Using differences makes the score
    insensitive to LFP. Delivered pulses score in the hundreds; a TTL with
    the stimulator off scores like noise (a few).

    Parameters
    ----------
    read_traces : callable(start, stop) -> (T, C) array of raw amplifier data
    frames : int array of TTL sample indices
    gain_uv : float, microvolts per raw unit (for ``peak_uv``)

    Returns
    -------
    score : float array
    peak_uv : float array
        Median over channels of the largest deviation from the pre-pulse level
        in the same window, in microvolts.
    """
    pre, post = int(round(pre_ms * fs / 1e3)), int(round(post_ms * fs / 1e3))
    tail = int(round(2.0 * fs / 1e3))  # baseline = last 2 ms before the pulse
    score = np.full(len(frames), np.nan)
    peak = np.full(len(frames), np.nan)
    for i, f in enumerate(np.asarray(frames, dtype=np.int64)):
        a, b = max(0, f - pre), min(n_samples, f + post + 1)
        if f - a < 10 or b - f < 2:
            continue
        x = np.asarray(read_traces(a, b), dtype=np.float64)
        k = f - a
        pre_x, post_x = x[:k], x[k - 1:]
        d_noise = np.median(np.abs(np.diff(pre_x, axis=0)), axis=0) / 0.6745
        d_noise = np.maximum(d_noise, 1e-9)
        jump = np.abs(np.diff(post_x, axis=0)).max(axis=0)
        score[i] = np.median(jump / d_noise)
        base = np.median(pre_x[-tail:], axis=0)
        peak[i] = np.median(np.abs(post_x - base).max(axis=0)) * gain_uv
    return score, peak


# ---------------------------------------------------------------------------
# Bouts
# ---------------------------------------------------------------------------

def group_bouts(frames, fs, gap_s):
    """Bout index for each pulse; a new bout starts after a gap > gap_s"""
    frames = np.asarray(frames)
    if len(frames) == 0:
        return np.zeros(0, dtype=int)
    gaps = np.diff(frames) / fs > gap_s
    return np.concatenate([[0], np.cumsum(gaps)]).astype(int)


def summarize_bouts(start_frames, stop_frames, delivered, bout, fs):
    """One dict per bout: frames, pulse counts and pulse rate"""
    out = []
    for b in np.unique(bout):
        m = bout == b
        s = np.asarray(start_frames)[m]
        rate = (len(s) - 1) / ((s[-1] - s[0]) / fs) if len(s) > 1 and s[-1] > s[0] else float("nan")
        out.append(dict(
            bout=int(b),
            start_frame=int(s[0]),
            stop_frame=int(np.asarray(stop_frames)[m][-1]),
            n_pulses=int(m.sum()),
            n_delivered=int(np.asarray(delivered)[m].sum()),
            rate_hz=float(rate),
        ))
    return out


def annotate_bouts(bouts, annotations):
    """Attach polarity / current_ua / notes from site_config['stim_bouts']

    ``annotations`` is a list of dicts with a ``bout`` index. Missing values
    become "unknown" / NaN / "". Returns new dicts; warns on mismatches.
    """
    annotations = annotations or []
    by_bout = {}
    for a in annotations:
        if "bout" not in a:
            warnings.warn("stim_bouts entry without a 'bout' index ignored: %r" % (a,))
            continue
        by_bout[int(a["bout"])] = a
    known = {b["bout"] for b in bouts}
    extra = sorted(set(by_bout) - known)
    if extra:
        warnings.warn("stim_bouts annotations for bouts %s, but only %d bouts were detected"
                      % (extra, len(bouts)))
    if annotations and len(by_bout) != len(bouts):
        warnings.warn("%d stim_bouts annotations for %d detected bouts" % (len(by_bout), len(bouts)))
    out = []
    for b in bouts:
        a = by_bout.get(b["bout"], {})
        cur = a.get("current_ua")
        out.append(dict(b,
                        polarity=str(a.get("polarity", "unknown")),
                        current_ua=float(cur) if cur is not None else float("nan"),
                        notes=str(a.get("notes", ""))))
    return out


# ---------------------------------------------------------------------------
# Blank windows
# ---------------------------------------------------------------------------

def blank_windows(frames, fs, ms_before, ms_after, n_samples, offset_ms=0.0):
    """Per-pulse [start, stop) sample windows, clipped to the recording"""
    frames = np.asarray(frames, dtype=np.int64) + int(round(offset_ms * fs / 1e3))
    a = np.clip(frames - int(round(ms_before * fs / 1e3)), 0, n_samples)
    b = np.clip(frames + int(round(ms_after * fs / 1e3)), 0, n_samples)
    return np.stack([a, b], axis=1) if len(frames) else np.zeros((0, 2), np.int64)


def merge_windows(windows):
    """Merge overlapping or touching [start, stop) windows; returns sorted (N, 2)"""
    w = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
    w = w[w[:, 1] > w[:, 0]]
    if len(w) == 0:
        return np.zeros((0, 2), np.int64)
    w = w[np.argsort(w[:, 0])]
    out = [list(w[0])]
    for a, b in w[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out, dtype=np.int64)


def extend_windows(windows, before=0, after=0):
    w = np.asarray(windows, dtype=np.int64).reshape(-1, 2).copy()
    w[:, 0] -= before
    w[:, 1] += after
    return merge_windows(w)


def windows_overlapping(windows, start, stop):
    """Rows of sorted, merged ``windows`` overlapping [start, stop)"""
    w = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
    if len(w) == 0:
        return w
    i0 = np.searchsorted(w[:, 1], start, side="right")
    i1 = np.searchsorted(w[:, 0], stop, side="left")
    return w[i0:i1]


def pairs_outside_windows(starts, stops, offset, windows):
    """Keep (start, stop) index pairs (relative to ``offset``) that do not
    overlap any of the absolute ``windows``. Used by RemoveJumps."""
    starts, stops = np.asarray(starts, dtype=np.int64), np.asarray(stops, dtype=np.int64)
    if windows is None or len(windows) == 0 or len(starts) == 0:
        return starts, stops
    keep = np.array([len(windows_overlapping(windows, offset + a, offset + b + 2)) == 0
                     for a, b in zip(starts, stops)], dtype=bool)
    return starts[keep], stops[keep]


# ---------------------------------------------------------------------------
# Blank + piecewise band-pass
# ---------------------------------------------------------------------------

def make_bandpass_sos(fs, freq_min=300.0, freq_max=8000.0, order=5):
    """Butterworth band-pass, as SpikeInterface's bandpass_filter builds it"""
    return butter(order, [freq_min, freq_max], btype="bandpass", fs=fs, output="sos")


def _min_filter_length(sos):
    # sosfiltfilt's default padlen; shorter pieces cannot be filtered
    n = 2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    return 3 * n


def blank_and_filter(traces, windows, sos):
    """Zero the samples in ``windows`` and band-pass each piece in between

    Parameters
    ----------
    traces : (T, C) array
    windows : (N, 2) [start, stop) sample windows relative to ``traces``
    sos : second-order sections from :func:`make_bandpass_sos`

    Pieces too short to filter (shorter than sosfiltfilt's padding) are
    blanked too. Returns float32 (T, C).
    """
    traces = np.asarray(traces)
    T = traces.shape[0]
    out = np.zeros(traces.shape, dtype=np.float32)
    w = merge_windows(np.clip(np.asarray(windows, dtype=np.int64).reshape(-1, 2), 0, T))
    edges = np.concatenate([[0], w.ravel(), [T]]).reshape(-1, 2)  # clean pieces
    min_len = _min_filter_length(sos)
    for a, b in edges:
        if b - a > min_len:
            out[a:b] = sosfiltfilt(sos, traces[a:b].astype(np.float64), axis=0)
    return out


def blank_and_filter_chunk(read_traces, n_samples, start, stop, windows, sos, margin):
    """Output samples [start, stop) of blank + piecewise band-pass

    Reads ``margin`` extra samples on each side (clipped to the recording),
    so chunk boundaries have the same small edge effects as SpikeInterface's
    filters. ``windows`` are absolute, sorted and merged.

    read_traces : callable(start, stop) -> (T, C) array
    """
    a = max(0, start - margin)
    b = min(n_samples, stop + margin)
    raw = read_traces(a, b)
    local = windows_overlapping(windows, a, b) - a
    out = blank_and_filter(raw, local, sos)
    return out[start - a:stop - a]


# ---------------------------------------------------------------------------
# Stim events file
# ---------------------------------------------------------------------------

def build_stim_events(rising, falling, fs, n_samples, cfg, score=None, peak_uv=None,
                      digital_bit=None, extra_artifacts=None):
    """Assemble the stim-events dict saved by DetectStimEvents

    extra_artifacts : optional (onset_frames, scores, approx_s) from ``find_extra_artifacts``
    """
    rising = np.asarray(rising, dtype=np.int64)
    falling = np.asarray(falling, dtype=np.int64)
    n = len(rising)
    score = np.full(n, np.nan) if score is None else np.asarray(score, dtype=float)
    peak_uv = np.full(n, np.nan) if peak_uv is None else np.asarray(peak_uv, dtype=float)
    delivered = np.nan_to_num(score, nan=-np.inf) > cfg["delivered_threshold"]
    blanked = delivered | bool(cfg["blank_undelivered"])
    per_pulse = blank_windows(rising, fs, cfg["ms_before"], cfg["ms_after"], n_samples, cfg["offset_ms"])
    bout = group_bouts(rising, fs, cfg["bout_gap_s"])
    extra = extra_blank_windows(cfg, fs, n_samples) if cfg["enabled"] else np.zeros((0, 2), np.int64)
    ea_onset, ea_score, ea_approx = extra_artifacts if (extra_artifacts is not None and cfg["enabled"]) \
        else (np.zeros(0, np.int64), np.zeros(0), [])
    ea_onset = np.asarray(ea_onset, dtype=np.int64)
    ea_windows = blank_windows(ea_onset, fs, cfg["ms_before"], cfg["extra_artifact_ms_after"], n_samples)
    merged = merge_windows(np.concatenate([per_pulse[blanked].reshape(-1, 2), extra, ea_windows]))
    return dict(
        version=STIM_EVENTS_VERSION,
        enabled=bool(cfg["enabled"]),
        config={k: cfg[k] for k in STIM_ARTIFACT_DEFAULTS},
        sampling_rate=float(fs),
        num_samples=int(n_samples),
        digital_bit=None if digital_bit is None else int(digital_bit),
        events=dict(
            start_frame=rising.tolist(),
            stop_frame=falling.tolist(),
            delivered=delivered.tolist(),
            artifact_score=[None if np.isnan(x) else float(x) for x in score],
            artifact_peak_uv=[None if np.isnan(x) else float(x) for x in peak_uv],
            blanked=blanked.tolist(),
            blank_start_frame=[int(x) if bl else None for x, bl in zip(per_pulse[:, 0], blanked)],
            blank_stop_frame=[int(x) if bl else None for x, bl in zip(per_pulse[:, 1], blanked)],
            bout=bout.tolist(),
        ),
        bouts=summarize_bouts(rising, falling, delivered, bout, fs) if n else [],
        blank_windows=merged.tolist(),
        # the manual windows on their own (also included in blank_windows)
        extra_blank_windows=extra.tolist(),
        # artefacts with no TTL, treated like pulses (extra_artifacts_s). Their fixed
        # windows are in blank_windows; with adaptive_blank / subtract_tail the per-channel
        # windows from FitStimTails are used instead.
        extra_artifacts=dict(
            onset_frame=ea_onset.tolist(),
            artifact_score=[float(x) for x in ea_score],
            approx_s=[float(x) for x in ea_approx],
            blank_start_frame=ea_windows[:, 0].tolist(),
            blank_stop_frame=ea_windows[:, 1].tolist(),
        ),
    )


def empty_stim_events(cfg, fs=None, n_samples=None):
    return build_stim_events([], [], fs or 1.0, n_samples or 0, cfg)


def save_stim_events(path, events):
    with open(path, "w") as f:
        json.dump(events, f, indent=1)


def load_stim_events(path):
    with open(path, "r") as f:
        ev = json.load(f)
    if ev.get("version") != STIM_EVENTS_VERSION:
        raise ValueError("Unsupported stim events version %r in %s" % (ev.get("version"), path))
    return ev


def stim_events_summary(events):
    """Short human-readable summary (for logs and the memmap info json)"""
    e = events["events"]
    n, nd = len(e["start_frame"]), int(np.sum(e["delivered"]))
    lines = ["%d TTL events, %d delivered, %d without artefact, %d blanked windows" % (
        n, nd, n - nd, len(events["blank_windows"]))]
    fs = events["sampling_rate"]
    extra = events.get("extra_blank_windows") or []
    if extra:
        lines.append("  %d manual blank windows (extra_blank_s): " % len(extra)
                     + ", ".join("%.3f-%.3f s" % (a / fs, b / fs) for a, b in extra))
    und = [s / fs for s, d in zip(e["start_frame"], e["delivered"]) if not d]
    if und:
        lines.append("  without artefact at (s): " + ", ".join("%.1f" % t for t in und[:20])
                     + (" ..." if len(und) > 20 else ""))
    ea = (events.get("extra_artifacts") or {}).get("onset_frame") or []
    if ea:
        lines.append("  %d artefacts without a TTL (extra_artifacts_s), jump at (s): " % len(ea)
                     + ", ".join("%.4f" % (f / fs) for f in ea))
    for b in events["bouts"]:
        t = b["start_frame"] / fs
        lines.append("  bout %d: %dh%02dm%02ds, %d pulses (%d delivered), %.2f Hz" % (
            b["bout"], t // 3600, (t % 3600) // 60, t % 60, b["n_pulses"], b["n_delivered"], b["rate_hz"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NWB rows (pure; the NWB task only writes them)
# ---------------------------------------------------------------------------

def stim_nwb_rows(events, annotations=None, channel_name="DIGITAL-IN"):
    """Rows for the stim_pulses and stim_bouts TimeIntervals tables

    Returns (pulse_rows, bout_rows, blank_intervals), times in seconds.
    """
    fs = events["sampling_rate"]
    e = events["events"]
    bouts = annotate_bouts(events["bouts"], annotations)
    bout_info = {b["bout"]: b for b in bouts}
    nan = float("nan")
    pulse_rows = []
    for i in range(len(e["start_frame"])):
        b = bout_info.get(e["bout"][i], {})
        bs, be = e["blank_start_frame"][i], e["blank_stop_frame"][i]
        pulse_rows.append(dict(
            start_time=e["start_frame"][i] / fs,
            stop_time=e["stop_frame"][i] / fs,
            delivered=bool(e["delivered"][i]),
            artifact_score=nan if e["artifact_score"][i] is None else float(e["artifact_score"][i]),
            artifact_peak_uv=nan if e["artifact_peak_uv"][i] is None else float(e["artifact_peak_uv"][i]),
            blanked=bool(e["blanked"][i]),
            blank_start=nan if bs is None else bs / fs,
            blank_stop=nan if be is None else be / fs,
            bout=int(e["bout"][i]),
            polarity=b.get("polarity", "unknown"),
            current_ua=b.get("current_ua", nan),
            ttl_channel=channel_name,
        ))
    bout_rows = [dict(
        start_time=b["start_frame"] / fs,
        stop_time=b["stop_frame"] / fs,
        bout=b["bout"],
        n_pulses=b["n_pulses"],
        n_delivered=b["n_delivered"],
        rate_hz=b["rate_hz"],
        polarity=b["polarity"],
        current_ua=b["current_ua"],
        notes=b["notes"],
    ) for b in bouts]
    blank_intervals = [(a / fs, b / fs) for a, b in events["blank_windows"]]
    return pulse_rows, bout_rows, blank_intervals
