"""Readers for SpikeGLX / CatGT outputs.

Adapted from SpikeGLX's readSGLX.py (as vendored in snakeSort), trimmed to what
the NWB conversion needs, plus readers for CatGT side files (ct_offsets, edge
files) and the raw per-trigger files.
"""
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def meta_path(path):
    path = Path(path)
    return path if path.suffix == '.meta' else path.with_suffix('.meta')


def read_meta(path):
    """Parse a SpikeGLX .meta file into a dict (values are strings).
    Keys starting with '~' have the '~' stripped."""
    meta = {}
    with open(meta_path(path)) as f:
        for line in f.read().splitlines():
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            meta[k.lstrip('~')] = v
    return meta


def sample_rate(meta):
    if meta['typeThis'] == 'imec':
        return float(meta['imSampRate'])
    if meta['typeThis'] == 'nidq':
        return float(meta['niSampRate'])
    return float(meta['obSampRate'])


def n_saved_chans(meta):
    return int(meta['nSavedChans'])


def n_samples(meta):
    return int(int(meta['fileSizeBytes']) / (2 * n_saved_chans(meta)))


def memmap_bin(bin_path, meta=None):
    """Memory map a SpikeGLX binary as (n_samples, n_channels) int16."""
    meta = meta if meta is not None else read_meta(bin_path)
    return np.memmap(bin_path, dtype=np.int16, mode='r',
                     shape=(n_samples(meta), n_saved_chans(meta)))


def ni_channel_counts(meta):
    """(MN, MA, XA, DW) channel counts for an nidq stream."""
    return tuple(int(x) for x in meta['snsMnMaXaDw'].split(','))


def ni_volts_per_bit(meta, chan):
    """Conversion factor (V / int16 count) for an NI analog channel index."""
    mn, ma, xa, dw = ni_channel_counts(meta)
    fI2V = float(meta['niAiRangeMax']) / float(meta['niMaxInt'])
    if chan < mn:
        gain = float(meta['niMNGain'])
    elif chan < mn + ma:
        gain = float(meta['niMAGain'])
    else:
        gain = 1.0
    return fI2V / gain


def ni_digital_word_index(meta, word=0):
    """Column index of the NI digital word in the saved binary."""
    mn, ma, xa, dw = ni_channel_counts(meta)
    if dw == 0:
        raise ValueError("No digital words saved in this nidq stream")
    return mn + ma + xa + word


def ni_digital_lines(meta):
    """List of digital line numbers saved (e.g. niXDChans1=0:7 -> [0..7])."""
    spec = meta.get('niXDChans1', '')
    lines = []
    for part in spec.split(','):
        if part == '':
            continue
        if ':' in part:
            a, b = part.split(':')
            lines.extend(range(int(a), int(b) + 1))
        else:
            lines.append(int(part))
    return lines


def imec_device_info(meta):
    keys = ['imDatPrb_pn', 'imDatPrb_sn', 'imDatPrb_type', 'imDatHs_pn', 'imDatHs_sn',
            'imDatBsc_pn', 'imDatBsc_sn', 'imDatFx_pn', 'imDatFx_sn', 'imDatApi', 'imChan0Ref']
    return {k: meta[k] for k in keys if k in meta}


# ----------------------------------------------------------------------------
# CatGT side files
# ----------------------------------------------------------------------------

def read_ct_offsets(path):
    """Read a CatGT {run}_ct_offsets.txt into a DataFrame, one row per input file.

    Columns are named after the rows of the file, e.g. smp_imap0, smp_nidq,
    sec_imap0, sec_nidq. Each value is where that input (trigger) file starts in
    the concatenated (tcat) output.
    """
    cols = {}
    with open(path) as f:
        for line in f:
            if ':' not in line:
                continue
            name, vals = line.split(':', 1)
            cols[name.strip()] = [float(x) for x in vals.split()]
    df = pd.DataFrame(cols)
    for c in df.columns:
        if c.startswith('smp_'):
            df[c] = df[c].astype(np.int64)
    return df


def read_edges(path):
    """Read a CatGT/TPrime edge-time text file (one time in seconds per line)."""
    if path is None or os.path.getsize(path) == 0:
        return np.array([], dtype=float)
    return np.atleast_1d(np.loadtxt(path, dtype=float))


def read_fyi(path):
    """Read a CatGT {run}_fyi.txt into a dict."""
    out = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                out[k] = v
    return out


def parse_catgt_cmdline(meta):
    """Pull -dir, -run, -g, -t, -prb out of the catGTCmdline0 stored in a tcat meta."""
    cmd = meta.get('catGTCmdline0', '').strip('<>')
    args = {}
    for tok in cmd.split():
        m = re.match(r'-(\w+)=(.*)', tok)
        if m:
            args.setdefault(m.group(1), m.group(2))
    return args


# ----------------------------------------------------------------------------
# Locating a CatGT run on disk
# ----------------------------------------------------------------------------

class CatGTRun(object):
    """Paths for one CatGT output run, e.g. {dest}/catgt_Choc57_d1.1_g0/"""

    def __init__(self, catgt_dir, probe=0):
        self.dir = Path(catgt_dir)
        self.probe = int(probe)
        self.run_g = self.dir.name[len('catgt_'):] if self.dir.name.startswith('catgt_') else self.dir.name
        self.ni_bin = self.dir / f'{self.run_g}_tcat.nidq.bin'
        self.imec_dir = self.dir / f'{self.run_g}_imec{self.probe}'
        self.ap_bin = self.imec_dir / f'{self.run_g}_tcat.imec{self.probe}.ap.bin'
        self.ct_offsets_path = self.dir / f'{self.run_g}_ct_offsets.txt'
        self.fyi_path = self.dir / f'{self.run_g}_fyi.txt'

    @property
    def ni_meta(self):
        return read_meta(self.ni_bin)

    @property
    def ap_meta(self):
        return read_meta(self.ap_bin)

    @property
    def ct_offsets(self):
        return read_ct_offsets(self.ct_offsets_path)

    def _one(self, pattern):
        hits = sorted(glob.glob(str(pattern)))
        if len(hits) == 0:
            raise FileNotFoundError(pattern)
        return Path(hits[0])

    @property
    def ni_sync_edges_path(self):
        # the NI sync extractor CatGT runs automatically (word, bit from syncNiChan)
        return self._one(self.dir / f'{self.run_g}_tcat.nidq.xd_*_{self.ni_meta["syncNiChan"]}_500.txt')

    @property
    def imec_sync_edges_path(self):
        return self._one(self.imec_dir / f'{self.run_g}_tcat.imec{self.probe}.ap.xd_*_6_500.txt')

    def ks_dir(self, name=None):
        if name is None:
            name = f'imec{self.probe}_ks4'
        return self.imec_dir / name

    def raw_files_csv(self):
        return self.dir / f'{self.run_g}_raw_files.csv'


def raw_file_table(raw_run_dir, run_g, probe=0):
    """Table of the raw SpikeGLX per-trigger files for a run: one row per t-index.

    Columns: t_index, ni_first_sample, ni_n_samples, imec_first_sample,
    imec_n_samples, file_create_time. `first_sample` is the stream's absolute
    sample count since acquisition started (this is what the NCAF logs use).
    """
    raw_run_dir = Path(raw_run_dir)
    rows = {}
    for p in glob.glob(str(raw_run_dir / f'{run_g}_t*.nidq.meta')):
        t = int(re.search(r'_t(\d+)\.nidq\.meta$', p).group(1))
        m = read_meta(p)
        rows.setdefault(t, {})
        rows[t].update(ni_first_sample=int(m['firstSample']), ni_n_samples=n_samples(m),
                       file_create_time=m['fileCreateTime'])
    for p in glob.glob(str(raw_run_dir / f'{run_g}_imec{probe}' / f'{run_g}_t*.imec{probe}.ap.meta')):
        t = int(re.search(r'_t(\d+)\.imec\d+\.ap\.meta$', p).group(1))
        m = read_meta(p)
        rows.setdefault(t, {})
        rows[t].update(imec_first_sample=int(m['firstSample']), imec_n_samples=n_samples(m))
    df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    df.index.name = 't_index'
    return df.reset_index()


def load_raw_file_table(catgt_run, raw_dir=None):
    """Raw file table for a CatGT run. Uses the cached CSV written by the sort step
    (so it still works once raw data is archived), else reads the raw metas."""
    csv = catgt_run.raw_files_csv()
    if csv.exists():
        return pd.read_csv(csv)
    if raw_dir is None:
        args = parse_catgt_cmdline(catgt_run.ni_meta)
        raw_dir = args.get('dir')
    raw_run_dir = Path(raw_dir) / catgt_run.run_g
    if not raw_run_dir.exists():
        return None
    df = raw_file_table(raw_run_dir, catgt_run.run_g, catgt_run.probe)
    return df


def decode_bits(word, line):
    """Boolean trace of one digital line from a digital word array."""
    return ((np.asarray(word).astype(np.uint16) >> line) & 1).astype(bool)
