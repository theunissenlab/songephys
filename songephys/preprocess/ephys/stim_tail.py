"""Per-channel stimulation tail removal

After each delivered stimulation pulse the amplifier sits on a large, slowly
decaying offset (a "tail", up to several mV, decaying over ~1-200 ms). The
band-pass filter removes most of it, but its curvature at the blank edge leaves
slow 50-150 uV lumps 1.5-4 ms after the pulse. This module:

1. finds, per channel and pulse, where the fast part of the artefact ends
   (last sample on the amplifier rail or with a big sample-to-sample jump, plus
   a few samples) - the *adaptive blank*;
2. fits ``n_exp`` exponentials plus an offset to the next ``tail_fit_ms`` of
   raw data and stores the fit, so it can be subtracted before filtering.

The fitter is fast: it fits bin averages (every sample for the first few ms,
log-spaced bins after), searches only the time constants (amplitudes and
offset are solved exactly for each set of time constants - variable
projection), uses an analytic Jacobian, and warm-starts from the previous
pulse of the same bout.

The data each side of the blank is still band-passed separately (see
``stim_artifact.blank_and_filter``); joining them makes the filter ring
backwards from what is left right after the blank.

Pure numpy/scipy; the pipeline wrappers are ``FitStimTails`` and
``StimBlankBandpass`` in ``songephys.pipeline.file_setup``.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import least_squares

STIM_TAILS_VERSION = 1
LOG_TAU_MIN, LOG_TAU_MAX = np.log(0.05), np.log(5000.0)   # ms
DEFAULT_TAUS = {1: (8.0,), 2: (2.0, 60.0), 3: (0.7, 8.0, 150.0)}
EXTRA_STARTS = {1: [], 2: [(0.7, 20.0)], 3: [(0.3, 3.0, 50.0), (1.5, 25.0, 400.0)]}
BASELINE_MS = (20.0, 1.0)      # baseline = mean from 20 ms to 1 ms before the blank start
MIN_TAIL_MS = 5.0              # do not fit tails shorter than this


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def make_bins(n, fs, dense_ms=5.0, n_log=350):
    """Bin edges over [0, n): one sample per bin for dense_ms, then log-spaced."""
    dense = min(int(round(dense_ms * 1e-3 * fs)), n)
    edges = np.arange(dense + 1)
    if n > dense:
        log_edges = np.unique(np.round(np.geomspace(max(dense, 1), n, n_log + 1)).astype(int))
        edges = np.unique(np.concatenate([edges, log_edges]))
    return edges


class BinnedTail:
    """Bin layout for tails of length n samples (reuse for every pulse of that length)."""

    def __init__(self, n, fs, dense_ms=5.0, n_log=350):
        self.n, self.fs = int(n), float(fs)
        self.edges = make_bins(self.n, self.fs, dense_ms, n_log)
        self.counts = np.diff(self.edges).astype(float)
        self.t_full = np.arange(self.n) / self.fs * 1e3
        self.t = np.add.reduceat(self.t_full, self.edges[:-1]) / self.counts
        self.sw = np.sqrt(self.counts)

    def bin(self, y):
        return np.add.reduceat(np.asarray(y, float), self.edges[:-1]) / self.counts


def _linear(t, sw, yb, taus):
    E = np.exp(-t[:, None] / np.asarray(taus)[None, :])
    A = np.concatenate([E, np.ones((len(t), 1))], 1) * sw[:, None]
    b = yb * sw
    Q, R = np.linalg.qr(A)
    coef = np.linalg.lstsq(R, Q.T @ b, rcond=1e-12)[0]
    return coef, A @ coef - b, Q, E


def fit_tail(tail, y, k=3, taus0=None):
    """Fit k exponentials + offset to y (length tail.n). Returns dict or None."""
    yb = tail.bin(y)
    t, sw = tail.t, tail.sw
    th0 = np.log(np.clip(np.asarray(taus0 if taus0 is not None else DEFAULT_TAUS[k], float),
                         np.exp(LOG_TAU_MIN) * 1.001, np.exp(LOG_TAU_MAX) * 0.999))

    def resid(th):
        return _linear(t, sw, yb, np.exp(th))[1]

    def jac(th):
        taus = np.exp(th)
        coef, _, Q, E = _linear(t, sw, yb, taus)
        D = E * (t[:, None] / taus[None, :]) * coef[None, :len(taus)] * sw[:, None]
        return D - Q @ (Q.T @ D)          # Kaufman's variable-projection Jacobian

    try:
        res = least_squares(resid, th0, jac=jac, method='trf',
                            bounds=(LOG_TAU_MIN, LOG_TAU_MAX), max_nfev=200)
    except Exception:
        return None
    taus = np.exp(res.x)
    coef, r, _, _ = _linear(t, sw, yb, taus)
    if not np.all(np.isfinite(coef)):
        return None
    order = np.argsort(taus)
    return dict(taus=taus[order], amps=coef[:k][order], c=float(coef[k]), cost=float((r ** 2).sum()))


def best_fit(tail, y, k=3, warm=None):
    """Lowest-cost fit over the warm start (if any), the default and extra starts."""
    starts = ([warm] if warm is not None else []) + [None] + list(EXTRA_STARTS[k])
    fits = [f for f in (fit_tail(tail, y, k, s) for s in starts) if f is not None]
    return min(fits, key=lambda f: f['cost']) if fits else None


def eval_exps(taus, amps, c, t_ms):
    """sum_k amps_k exp(-t / taus_k) + c, broadcasting taus/amps (..., k) against t."""
    t = np.asarray(t_ms, float)[..., None]
    return (amps * np.exp(-t / taus)).sum(-1) + c


# ---------------------------------------------------------------------------
# Adaptive blank end
# ---------------------------------------------------------------------------

def fast_artifact_end(x, onset, search_n, rail, jump):
    """Index of the last sample of the fast stimulation transient in x (1-D, counts).

    Looks from 2 samples before ``onset`` to ``onset + search_n`` for samples on
    the rail (|x| >= rail) or sample-to-sample jumps > jump; never earlier than
    onset + 12 samples (0.4 ms at 30 kHz).
    """
    lo, hi = max(onset - 2, 0), min(onset + search_n, len(x))
    seg = x[lo:hi]
    idx = np.arange(lo, hi)
    r = idx[np.abs(seg) >= rail]
    j = idx[1:][np.abs(np.diff(seg)) > jump]
    return max(r.max() if len(r) else 0, j.max() if len(j) else 0, onset + 12)


# ---------------------------------------------------------------------------
# Fitting all pulses x channels (parallel over channel groups)
# ---------------------------------------------------------------------------

def _fit_channels(args):
    """Worker: fit every pulse of some channels. Reads the (C, P, L) int16 block file."""
    path, shape, chans, meta = args
    blocks = np.load(path, mmap_mode='r')
    P = shape[1]
    base_n, fs, k = meta['base_n'], meta['fs'], meta['n_exp']
    valid = meta['valid_len']
    out = {key: np.full((len(chans), P) + extra, np.nan) for key, extra in
           (('b1', ()), ('baseline', ()), ('c', ()), ('r2', ()), ('taus', (k,)), ('amps', (k,)))}
    ok = np.zeros((len(chans), P), bool)
    tails = {}
    b0 = int(round(BASELINE_MS[0] * fs / 1e3)); b1_ = int(round(BASELINE_MS[1] * fs / 1e3))
    for ci, ch in enumerate(chans):
        X = np.asarray(blocks[ch], dtype=np.float64)       # (P, L)
        warm = {}
        for p in range(P):
            x = X[p, :valid[p]]
            onset = base_n
            if meta['adaptive']:
                b1 = fast_artifact_end(x, onset, meta['search_n'], meta['rail'], meta['jump']) + meta['extra'] + 1
                b1 = int(np.clip(b1, onset + meta['min_n'][p], onset + meta['max_n']))
            else:
                b1 = onset + meta['fixed_after_n'][p]
            out['b1'][ci, p] = b1 - onset
            bs = onset - meta['before_n']
            out['baseline'][ci, p] = x[max(bs - b0, 0):max(bs - b1_, 1)].mean()
            if not meta['subtract'] or valid[p] - b1 < meta['min_tail_n']:
                continue
            y = x[b1:valid[p]]
            tail = tails.get(len(y))
            if tail is None:
                tail = tails[len(y)] = BinnedTail(len(y), fs)
            f = best_fit(tail, y, k, warm.get(meta['bout'][p]))
            if f is None:
                continue
            warm[meta['bout'][p]] = list(f['taus'])
            out['taus'][ci, p], out['amps'][ci, p], out['c'][ci, p] = f['taus'], f['amps'], f['c']
            m = eval_exps(f['taus'], f['amps'], f['c'], tail.t_full)
            ss = ((y - y.mean()) ** 2).sum()
            out['r2'][ci, p] = 1 - ((y - m) ** 2).sum() / ss if ss > 0 else np.nan
            ok[ci, p] = True
    out['ok'] = ok
    return chans, out


def fit_stim_tails(read_traces, n_samples, n_channels, fs, onsets, bouts, cfg,
                   tmp_path, workers=None, progress=print, extra=None):
    """Adaptive blank ends and tail fits for every pulse x channel.

    read_traces : callable(start, stop) -> (T, C) raw signed traces (counts)
    onsets : (P,) absolute sample of each pulse (TTL edge + offset), sorted
    bouts : (P,) bout index (warm starts are shared within a bout)
    cfg : stim_artifact config (see stim_artifact.STIM_ARTIFACT_DEFAULTS)
    tmp_path : scratch .npy file for the raw blocks (deleted afterwards)
    extra : optional (P,) bool, True for artefacts without a TTL (extra_artifacts_s).
        Their blank lasts at least ``extra_artifact_ms_after`` after the jump (exactly
        that without adaptive_blank) and they get no blank_after_filter_ms.

    Returns a dict of arrays (see ``save_stim_tails``).
    """
    from concurrent.futures import ProcessPoolExecutor
    onsets = np.asarray(onsets, np.int64)
    bouts = np.asarray(bouts, np.int64)
    P = len(onsets)
    extra = np.zeros(P, bool) if extra is None else np.asarray(extra, bool)
    if np.any(np.diff(onsets) < 0):
        raise ValueError("fit_stim_tails needs onsets sorted in time")
    ms = lambda v: int(round(float(v) * fs / 1e3))
    extra_after_n = ms(cfg.get('extra_artifact_ms_after', 1.0))
    before_n = ms(cfg['ms_before'])
    base_n = ms(BASELINE_MS[0]) + before_n
    fit_n = ms(cfg['tail_fit_ms'])
    max_n = ms(cfg['adaptive_max_ms_after']) if cfg['adaptive_blank'] else ms(cfg['ms_after'])
    if extra.any():
        max_n = max(max_n, extra_after_n)
    L = base_n + max_n + fit_n
    # tails stop at the next pulse's blank (fast trains) and at the end of the recording
    next_bs = np.r_[onsets[1:] - before_n, n_samples]
    tail_end = np.minimum(onsets + max_n + fit_n, np.minimum(next_bs, n_samples))
    starts = onsets - base_n
    valid = (tail_end - starts).astype(np.int64)
    if np.any(starts < 0):
        raise ValueError("A stimulation pulse is closer than %.1f ms to the start of the recording"
                         % (base_n / fs * 1e3))

    tic = time.time()
    blocks = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=np.int16, shape=(n_channels, P, L))
    try:
        for p in range(P):
            seg = read_traces(int(starts[p]), int(tail_end[p]))
            blocks[:, p, :seg.shape[0]] = np.asarray(seg).T
        blocks.flush()
        del blocks
        progress("  read %d pulses x %d channels in %.0f s" % (P, n_channels, time.time() - tic))

        meta = dict(base_n=base_n, fs=float(fs), n_exp=int(cfg['tail_n_exp']), valid_len=valid,
                    adaptive=bool(cfg['adaptive_blank']), search_n=ms(cfg['adaptive_search_ms']),
                    rail=float(cfg['adaptive_rail_counts']), jump=float(cfg['adaptive_jump_counts']),
                    extra=int(cfg['adaptive_extra_samples']),
                    min_n=np.where(extra, extra_after_n, ms(cfg['adaptive_min_ms_after'])),
                    max_n=max_n, fixed_after_n=np.where(extra, extra_after_n, ms(cfg['ms_after'])),
                    before_n=before_n,
                    subtract=bool(cfg['subtract_tail']), min_tail_n=ms(MIN_TAIL_MS), bout=bouts)
        workers = int(workers or cfg.get('tail_workers') or min(8, os.cpu_count() or 1))
        groups = [list(g) for g in np.array_split(np.arange(n_channels), min(n_channels, max(workers * 4, 1)))]
        res = {}
        tic = time.time()
        # one BLAS/OpenMP thread per worker (set before the workers start)
        saved = {v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
        os.environ.update({v: '1' for v in saved})
        try:
            if workers <= 1:
                outs = map(_fit_channels, [(tmp_path, (n_channels, P, L), g, meta) for g in groups])
                for chans, out in outs:
                    res.update({c: (out, i) for i, c in enumerate(chans)})
            else:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for k_, (chans, out) in enumerate(ex.map(_fit_channels, [(tmp_path, (n_channels, P, L), g, meta) for g in groups])):
                        res.update({c: (out, i) for i, c in enumerate(chans)})
                        if (k_ + 1) % max(1, len(groups) // 8) == 0:
                            progress("  fitted %d/%d channels (%.0f s)" % (len(res), n_channels, time.time() - tic))
        finally:
            for v, val in saved.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    def gather(key):
        return np.stack([res[c][0][key][res[c][1]] for c in range(n_channels)], axis=1)   # (P, C, ...)

    b1 = gather('b1')
    return dict(
        version=STIM_TAILS_VERSION, fs=float(fs), n_samples=int(n_samples),
        onset=onsets, bout=bouts, extra=extra,
        blank_start=onsets - before_n,
        blank_stop=(onsets[:, None] + np.nan_to_num(b1).astype(np.int64)),
        tail_end=tail_end.astype(np.int64),
        taus=gather('taus'), amps=gather('amps'), c=gather('c'),
        baseline=gather('baseline'), r2=gather('r2'), ok=gather('ok'),
        taper_n=ms(cfg['tail_taper_ms']),
        config=json.dumps({k: cfg[k] for k in sorted(cfg)}),
    )


def stim_tails_summary(tails):
    fs = tails['fs']
    bs = (tails['blank_stop'] - tails['onset'][:, None]) / fs * 1e3
    n_extra = int(np.sum(tails['extra'])) if 'extra' in tails else 0
    lines = ["Stim tails: %d pulses x %d channels" % tails['blank_stop'].shape
             + (" (%d of them artefacts without a TTL)" % n_extra if n_extra else ""),
             "  blank ends %.2f ms after the pulse (median; 5-95%%: %.2f-%.2f, max %.2f)"
             % (np.median(bs), *np.percentile(bs, [5, 95]), bs.max())]
    ok = tails['ok']
    if ok.any():
        r2 = tails['r2'][ok]
        lines.append("  tail fits: %d/%d ok, R^2 median %.3f (5th pct %.3f)" % (ok.sum(), ok.size, np.median(r2), np.percentile(r2, 5)))
    return "\n".join(lines)


def save_stim_tails(path, tails):
    np.savez(path, **tails)


def load_stim_tails(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    for k in ('version', 'n_samples', 'taper_n'):
        out[k] = int(out[k])
    out['fs'] = float(out['fs'])
    out['config'] = json.loads(str(out['config']))
    if 'extra' not in out:                       # files written before extra_artifacts_s existed
        out['extra'] = np.zeros(len(out['onset']), bool)
    return out


def empty_stim_tails(fs=30000.0, n_samples=0, n_channels=0, k=3):
    z = np.zeros((0, n_channels))
    return dict(version=STIM_TAILS_VERSION, fs=float(fs), n_samples=int(n_samples),
                onset=np.zeros(0, np.int64), bout=np.zeros(0, np.int64), extra=np.zeros(0, bool),
                blank_start=np.zeros(0, np.int64),
                blank_stop=np.zeros((0, n_channels), np.int64), tail_end=np.zeros(0, np.int64),
                taus=np.zeros((0, n_channels, k)), amps=np.zeros((0, n_channels, k)), c=z, baseline=z, r2=z,
                ok=np.zeros((0, n_channels), bool), taper_n=0, config=json.dumps({}))


# ---------------------------------------------------------------------------
# Applying: correction to subtract and per-channel blank windows for a chunk
# ---------------------------------------------------------------------------

def tail_correction(tails, a, b, chans):
    """(b - a, len(chans)) array to subtract from raw samples [a, b) of these channels.

    For each fitted pulse x channel, over [blank_stop, tail_end): the fit minus the
    pre-pulse baseline, tapered to 0 over the last ``taper_n`` samples so the end of
    the subtraction is smooth.
    """
    chans = np.asarray(chans)
    out = np.zeros((b - a, len(chans)))
    if not len(tails['onset']):
        return out
    fs, taper = tails['fs'], tails['taper_n']
    hit = np.nonzero((tails['tail_end'] > a) & (tails['blank_stop'].min(1) < b))[0]
    for p in hit:
        bs = tails['blank_stop'][p, chans]                                  # (C,)
        lo, hi = max(a, int(bs.min())), min(b, int(tails['tail_end'][p]))
        if hi <= lo:
            continue
        s = np.arange(lo, hi)
        t = (s[:, None] - bs[None, :]) / fs * 1e3                          # (n, C) ms after blank end
        ok = tails['ok'][p, chans]
        if not ok.any():
            continue
        corr = eval_exps(tails['taus'][p, chans], tails['amps'][p, chans], tails['c'][p, chans], np.maximum(t, 0))
        corr = corr - tails['baseline'][p, chans]
        corr[(t < 0) | ~ok[None, :]] = 0
        if taper > 0:
            te = int(tails['tail_end'][p])
            w = np.clip((te - s) / taper, 0, 1)
            w = 0.5 - 0.5 * np.cos(np.pi * w)
            corr *= w[:, None]
        out[lo - a:hi - a] += corr
    return out


def post_filter_windows(tails, post_blank_n, a, b):
    """[blank_start, onset + post_blank_n) for the fitted pulses overlapping [a, b), relative to a.

    Artefacts without a TTL (``tails['extra']``) are left out: they keep their own blank only.
    """
    if post_blank_n <= 0 or not len(tails['onset']):
        return np.zeros((0, 2), np.int64)
    w = np.stack([tails['blank_start'], tails['onset'] + int(post_blank_n)], 1).astype(np.int64)
    extra = tails.get('extra')
    if extra is not None and len(extra):
        w = w[~np.asarray(extra, bool)]
    w = w[(w[:, 1] > a) & (w[:, 0] < b)]
    return np.clip(w, a, b) - a


def tail_blank_filter_chunk(read_traces, n_samples, start, stop, fixed_windows, tails, chans, sos, margin,
                            post_blank_n=0):
    """Output samples [start, stop): subtract fitted tails, blank per channel, band-pass each piece.

    Like ``stim_artifact.blank_and_filter_chunk`` but with per-channel blank windows
    and tail subtraction. ``chans`` are the parent recording's channel indices of the
    columns ``read_traces`` returns. ``post_blank_n`` > 0 also zeroes the output from
    each fitted pulse's blank start to onset + post_blank_n samples, after filtering.
    """
    from .stim_artifact import blank_and_filter
    a = max(0, start - margin)
    b = min(n_samples, stop + margin)
    raw = np.asarray(read_traces(a, b), dtype=np.float64)
    raw = raw - tail_correction(tails, a, b, chans)
    out = np.zeros(raw.shape, dtype=np.float32)
    for cols, w in channel_window_groups(tails, fixed_windows, a, b, chans):
        out[:, cols] = blank_and_filter(raw[:, cols], w, sos)
    for w0, w1 in post_filter_windows(tails, post_blank_n, a, b):
        out[w0:w1] = 0
    return out[start - a:stop - a]


def channel_window_groups(tails, fixed_windows, a, b, chans):
    """Group channels that share the same blank windows within [a, b).

    Returns [(column indices into chans, (N, 2) windows relative to a), ...].
    ``fixed_windows`` (absolute, all channels) are added to every channel.
    """
    from .stim_artifact import merge_windows, windows_overlapping
    chans = np.asarray(chans)
    fixed = windows_overlapping(merge_windows(fixed_windows), a, b)
    if len(tails['onset']):
        hit = np.nonzero((tails['blank_stop'].max(1) > a) & (tails['blank_start'] < b))[0]
    else:
        hit = np.zeros(0, int)
    if not len(hit):
        return [(np.arange(len(chans)), fixed - a)]
    groups = {}
    for ci, ch in enumerate(chans):
        w = np.stack([tails['blank_start'][hit], tails['blank_stop'][hit, ch]], 1)
        key = tuple(map(tuple, w.tolist()))
        groups.setdefault(key, []).append(ci)
    out = []
    for key, cols in groups.items():
        w = merge_windows(np.concatenate([fixed, np.asarray(key, np.int64).reshape(-1, 2)]))
        out.append((np.asarray(cols), np.clip(w, a, b) - a))
    return out
