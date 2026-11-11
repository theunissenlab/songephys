"""One table of task events for a site, in the same time base as the spike times

Rows (one per event, sorted by time):

* ``stim``: every stimulation TTL from the stim events file, with ``delivered``,
  ``bout``, ``polarity``, ``current_ua`` (site_config stim_bouts), ``artifact_score``
  and the blank window (``blank_start_ms`` / ``blank_stop_ms`` relative to the TTL).
* one event type per named digital input (site_config ``digital_channel_names``),
  at each rising edge, with ``duration_ms`` (time high). Names are lower-cased;
  ``Match`` (FeedbackPlease target detection) is exported as ``target``.
  Unnamed inputs (``DIGITAL-IN-xx``) and the stim TTL line are skipped.
* ``target`` rows get ``with_feedback`` (a feedback pulse starts within
  ``feedback_window_s`` after the target).
* ``syllable`` rows get ``code``: the syllable code read from the
  "Syllable Code Bit1..3" inputs just after the syllable onset (0 = not classified).

Columns: time_s, sample, event, duration_ms, and the per-type columns above
(empty where they do not apply).
"""
import re

import numpy as np
import pandas as pd

from ..preprocess.ephys.stim_artifact import (
    annotate_bouts, find_ttl_edges, get_digital_channel_bits, resolve_digital_bit)

EVENT_ALIASES = {'match': 'target'}
CODE_BIT_PATTERN = re.compile(r'code\s*bit\s*(\d+)', re.IGNORECASE)
UNNAMED_PATTERN = re.compile(r'DIGITAL-IN-?\d+', re.IGNORECASE)
CODE_READ_DELAY = 3          # samples after the syllable onset to read the code bits


def _next_after(values, points, missing):
    """For each point, the first value strictly after it (``missing`` if none)."""
    values = np.asarray(values, np.int64)
    if not len(values):
        return np.full(len(points), missing, np.int64)
    j = np.searchsorted(values, points, side='right')
    out = values[np.clip(j, 0, len(values) - 1)]
    return np.where(j < len(values), out, missing)


def _read_at(read_dig, samples, block=2000):
    """Digital word at each sample index (sorted reads in blocks)."""
    samples = np.asarray(samples, np.int64)
    vals = np.empty(len(samples), np.int64)
    order = np.argsort(samples)
    for a in range(0, len(samples), block):
        idx = order[a:a + block]
        lo, hi = int(samples[idx].min()), int(samples[idx].max()) + 1
        blk = np.asarray(read_dig(lo, hi)).ravel().astype(np.int64)
        vals[idx] = blk[samples[idx] - lo]
    return vals


def digital_events(read_dig, n_samples, fs, site_config, skip_bits=(), feedback_window_s=0.1):
    """Rows for the named digital inputs (see module docstring)."""
    lines = get_digital_channel_bits(site_config)
    code_bits = {}
    events = []
    for bit, name in lines:
        name = str(name or '').strip()
        m = CODE_BIT_PATTERN.search(name)
        if m:
            code_bits[int(m.group(1))] = bit
            continue
        if not name or UNNAMED_PATTERN.fullmatch(name) or bit in skip_bits or name.lower() == 'unused':
            continue
        events.append((bit, EVENT_ALIASES.get(name.lower(), name.lower())))
    frames = []
    edges = {}
    for bit, label in events:
        r, f = find_ttl_edges(read_dig, n_samples, bit=bit)
        r = np.asarray(r, np.int64)
        fall = _next_after(f, r, n_samples)
        edges[label] = r
        frames.append(pd.DataFrame({'sample': r, 'event': label, 'duration_ms': (fall - r) / fs * 1e3}))
    if not frames:
        return pd.DataFrame(columns=['sample', 'event', 'duration_ms'])
    df = pd.concat(frames, ignore_index=True)
    if 'target' in edges and 'feedback' in edges:
        tg = edges['target']
        # feedback starting at or after the target (side='right' on tg - 1 includes equal samples)
        nxt = _next_after(edges['feedback'], tg - 1, np.iinfo(np.int64).max // 2)
        df['with_feedback'] = pd.Series(pd.NA, index=df.index, dtype='boolean')
        df.loc[df.event == 'target', 'with_feedback'] = (nxt - tg) <= feedback_window_s * fs
    if 'syllable' in edges and code_bits and len(edges['syllable']):
        words = _read_at(read_dig, np.clip(edges['syllable'] + CODE_READ_DELAY, 0, n_samples - 1))
        code = np.zeros(len(words), np.int64)
        for k, bit in sorted(code_bits.items()):
            code += ((words >> bit) & 1) << (k - 1)
        df['code'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
        df.loc[df.event == 'syllable', 'code'] = code
    return df


def stim_rows(stim_events, site_config):
    """Rows for every stimulation TTL in a stim events dict (DetectStimEvents output)."""
    ev = (stim_events or {}).get('events') or {}
    start = np.asarray(ev.get('start_frame', []), np.int64)
    if not len(start):
        return pd.DataFrame(columns=['sample', 'event'])
    fs = float(stim_events['sampling_rate'])
    n = len(start)
    get = lambda k, default: np.asarray(ev.get(k, [default] * n))
    bout = get('bout', -1).astype(np.int64)
    ann = {b['bout']: b for b in annotate_bouts(stim_events.get('bouts', []), (site_config or {}).get('stim_bouts'))}
    bs = np.asarray([np.nan if v is None else v for v in get('blank_start_frame', None)], float)
    be = np.asarray([np.nan if v is None else v for v in get('blank_stop_frame', None)], float)
    stop = get('stop_frame', None)
    dur = np.asarray([np.nan if v is None else v for v in stop], float)
    return pd.DataFrame({
        'sample': start, 'event': 'stim',
        'duration_ms': (dur - start) / fs * 1e3,
        'delivered': get('delivered', True).astype(bool),
        'bout': bout,
        'polarity': [ann.get(b, {}).get('polarity', 'unknown') for b in bout],
        'current_ua': [ann.get(b, {}).get('current_ua', np.nan) for b in bout],
        'artifact_score': get('artifact_score', np.nan).astype(float),
        'blank_start_ms': (bs - start) / fs * 1e3,
        'blank_stop_ms': (be - start) / fs * 1e3,
    })


def build_events_table(read_dig, n_samples, fs, site_config, stim_events=None, stim_bit=None,
                       feedback_window_s=0.1):
    """Events table for a site (see module docstring)."""
    skip = () if stim_bit is None else (stim_bit,)
    parts = [stim_rows(stim_events, site_config),
             digital_events(read_dig, n_samples, fs, site_config, skip_bits=skip,
                            feedback_window_s=feedback_window_s)]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) \
        else pd.DataFrame(columns=['sample', 'event'])
    df['sample'] = df['sample'].astype(np.int64)
    df.insert(0, 'time_s', df['sample'] / fs)
    return df.sort_values(['sample', 'event'], kind='stable').reset_index(drop=True)


def stim_bit_from_config(stim_events, site_config):
    """Bit of the stim TTL line (to skip it as a plain digital event), or None."""
    cfg = (stim_events or {}).get('config') or {}
    if 'digital_bit' in (stim_events or {}) and stim_events['digital_bit'] is not None:
        return int(stim_events['digital_bit'])
    try:
        return resolve_digital_bit(cfg.get('digital_channel', 'DIGITAL-IN-01'), site_config)
    except Exception:
        return None
