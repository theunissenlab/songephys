"""Truncating a site at ``tmax`` (site_config.yaml).

    tmax: 16920        # seconds from the start of the site
    tmax: "4:42:00"    # or H:MM:SS / MM:SS

Time is measured along the site's recordings played back to back (the same time
base as the memmaps and the NWB). Everything after tmax is dropped from every
stream when the single-site files are generated (ephys, LFP, audio, ADC, aux,
digital), so events, CSVs and NWB tables derived from them stop at tmax too.
"""
import glob
import json
import math
import os


def parse_tmax(value):
    """Seconds from a number or an 'H:MM:SS' / 'MM:SS' string; None stays None."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.strip().split(':')
        if len(parts) > 3 or not all(p.strip() for p in parts):
            raise ValueError("tmax %r: use seconds or H:MM:SS" % value)
        seconds = 0.0
        for p in parts:
            seconds = seconds * 60 + float(p)
    else:
        seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("tmax must be a positive number of seconds, got %r" % value)
    return seconds


def get_site_tmax(site_config):
    """tmax in seconds from a site config, or None if the site is not truncated."""
    return parse_tmax((site_config or {}).get('tmax'))


def _nwb_durations(path):
    """{acquisition name: duration s} for rate-based acquisitions in an NWB file."""
    import h5py
    out = {}
    with h5py.File(path, 'r') as f:
        for name, grp in f.get('acquisition', {}).items():
            try:
                rate = float(grp['starting_time'].attrs['rate'])
                out[name] = grp['data'].shape[0] / rate
            except Exception:
                continue
    return out


def outputs_past_tmax(site_dir, tmax, tolerance_s=1e-3):
    """Existing site outputs that are longer than tmax: [(path, duration_s), ...].

    Checks raw/*_info.json (memmaps), raw/stim_events_*.json and the site NWB.
    These have to be deleted and regenerated after tmax is set or lowered,
    because the pipeline treats existing outputs as done.
    """
    if tmax is None:
        return []
    stale = []
    raw = os.path.join(site_dir, 'raw')
    for path in sorted(glob.glob(os.path.join(raw, '*_info.json')) +
                       glob.glob(os.path.join(raw, 'stim_events_*.json'))):
        try:
            with open(path) as f:
                info = json.load(f)
            dur = float(info['num_samples']) / float(info['sampling_rate'])
        except Exception:
            continue
        if dur > tmax + tolerance_s:
            stale.append((path, dur))
    for path in sorted(glob.glob(os.path.join(raw, 'stim_tails_*.npz'))):
        try:
            import numpy as np
            with np.load(path) as z:
                n, fs = int(z['n_samples']), float(z['fs'])
            dur = n / fs
        except Exception:
            continue
        if n and dur > tmax + tolerance_s:
            stale.append((path, dur))
    for path in glob.glob(os.path.join(site_dir, '*.nwb')):
        try:
            durs = _nwb_durations(path)
        except Exception:
            continue
        if durs and max(durs.values()) > tmax + tolerance_s:
            stale.append((path, max(durs.values())))
    return stale


def check_site_outputs_respect_tmax(site_dir, site_config):
    """Raise if outputs generated before tmax was set (or lowered) still exist."""
    tmax = get_site_tmax(site_config)
    stale = outputs_past_tmax(site_dir, tmax)
    if stale:
        lines = ['  %s (%.1f s)' % (os.path.relpath(p, site_dir), d) for p, d in stale]
        raise RuntimeError(
            "site_config tmax is %.3f s but these outputs are longer; delete them (and the "
            "matching .bin/.wav files) so they are regenerated:\n%s" % (tmax, '\n'.join(lines)))


def _segments(recording):
    n = recording.get_num_segments()
    if n == 1:
        return [recording]
    return [recording.select_segments([i]) for i in range(n)]


def truncate_recordings(recordings, tmax):
    """Keep the first tmax seconds of recordings played back to back.

    recordings: SpikeInterface recordings in site order (multi-segment recordings
    are treated as their segments in order). Returns a list of single-segment
    recordings; the last one is cut with frame_slice, later ones are dropped.
    """
    if tmax is None:
        return list(recordings)
    out = []
    elapsed = 0.0
    for rec in recordings:
        for seg in _segments(rec):
            fs = seg.get_sampling_frequency()
            n = seg.get_num_samples(segment_index=0)
            keep = int(round((tmax - elapsed) * fs))
            if keep <= 0:
                return out
            if keep < n:
                out.append(seg.frame_slice(start_frame=0, end_frame=keep))
                return out
            out.append(seg)
            elapsed += n / fs
    return out
