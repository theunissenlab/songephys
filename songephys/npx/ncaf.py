"""Readers for the NCAF closed-loop outputs (NCAF_outs/{session}/)."""
import glob
import os
import re

import numpy as np
import pandas as pd

_KV = re.compile(r'(\w+)=(\S*)')


def parse_decisions_log(path):
    """decisions.log -> one row per SYLLABLE line, with the matching TRIGGER line joined on.

    Each SYLLABLE line is one detection of the targeted syllable and the online
    spike count decision; a TRIGGER line (same imec_onset_sample) is written
    when feedback was delivered.
    """
    sylls, trigs = [], []
    with open(path) as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            kv = dict(_KV.findall(parts[1]))
            if parts[0] == 'SYLLABLE':
                sylls.append(kv)
            elif parts[0] == 'TRIGGER':
                trigs.append(kv)
    syl = pd.DataFrame(sylls)
    trg = pd.DataFrame(trigs)
    for c in ('code', 'ni_onset_sample', 'imec_onset_sample', 'count', 'threshold', 'triggered'):
        syl[c] = pd.to_numeric(syl[c])
    syl['triggered'] = syl['triggered'].astype(bool)
    for c in ('sub_counts', 'pair_counts', 'spikes_ms'):
        syl[c] = [np.array([float(x) for x in v.split(',') if x != ''], dtype=float) for v in syl[c].fillna('')]
    if len(trg):
        for c in ('imec_onset_sample', 'decision_imec_sample', 'pulse_imec_sample', 'total_count', 'threshold'):
            trg[c] = pd.to_numeric(trg[c])
        trg['pulse_ms'] = pd.to_numeric(trg['pulse_ms'])
        w = trg['window_ms'].str.split('-', expand=True).astype(float)
        trg['window_start_ms'], trg['window_end_ms'] = w[0], w[1]
        trg = trg[['imec_onset_sample', 'decision_imec_sample', 'pulse_imec_sample', 'pulse_ms',
                   'window_start_ms', 'window_end_ms', 'total_count']].drop_duplicates('imec_onset_sample')
        syl = syl.merge(trg, on='imec_onset_sample', how='left')
    return syl


def read_syllable_times(path):
    """syllableTimes.txt -> DataFrame(code, ni_sample, imec_sample) (absolute samples)."""
    return pd.read_csv(path, header=None, names=['code', 'ni_sample', 'imec_sample'])


def read_spike_times(path):
    """spikeTimes.txt -> absolute imec samples of online-detected target spikes."""
    if os.path.getsize(path) == 0:
        return np.array([], dtype=np.int64)
    return np.atleast_1d(np.loadtxt(path, dtype=np.int64))


def read_filters(filter_dir):
    """filters/{kind}_{template}.bin -> one dict per target template."""
    out = []
    templates = sorted({int(re.search(r'_(\d+)\.bin$', f).group(1))
                        for f in glob.glob(os.path.join(filter_dir, '*_*.bin'))})
    for t in templates:
        def _read(kind, dtype):
            p = os.path.join(filter_dir, f'{kind}_{t}.bin')
            return np.fromfile(p, dtype=dtype) if os.path.exists(p) else None
        channels = _read('channels', '<i4')
        filt = _read('filter', '<f8')
        if filt is not None and channels is not None and len(channels) and filt.size % len(channels) == 0:
            filt = filt.reshape(len(channels), -1)
        thr = _read('threshold', '<f8')
        car = _read('car_shank', '<i4')
        out.append(dict(template=t, channels=channels, filter=filt,
                        threshold=float(thr[0]) if thr is not None else np.nan,
                        car_shank=int(car[0]) if car is not None else -1))
    return out
