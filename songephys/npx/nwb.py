"""Luigi tasks that package a sorted SpikeGLX session into an NWB file.

The layout follows the Intan songephys NWB files (acquisition/audio,
acquisition/ttl_events, units with kilosort_* columns, acquisition/unit_waveforms,
intervals/unit_intervals) and adds the Neuropixels/NCAF specific pieces. All
times are seconds on the NI clock along the CatGT-concatenated NI file; see
NWB_SCHEMA_NPX.md.
"""
import json
from datetime import datetime, date
from pathlib import Path
from uuid import uuid4

import luigi
import numpy as np
import pandas as pd
from dateutil.tz import tzlocal
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable, DynamicTableRegion, VectorData, VectorIndex
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

from ..analysis.unit_intervals import compute_unit_valid_intervals, recording_duration
from ..pipeline.targets import NWBAcquisitionTarget, NWBIntervalsTarget, NWBProcessingTarget, NWBUnitsTarget
from ..utils import pulse_edges
from .config import BirdConfig
from .sort import AlignSpikesToNI, RunCatGT
from .spikeglx import (decode_bits, load_raw_file_table, memmap_bin, ni_digital_word_index,
                       ni_volts_per_bit, read_edges, read_meta, sample_rate, imec_device_info)
from .sync import build_ni_clock


def make_table(cls, name, description, columns):
    """Build a DynamicTable/TimeIntervals from {col: (data, description[, table])}.

    A list of np.ndarrays becomes a ragged (indexed) column. Giving a third
    element (a DynamicTable) makes the column a link (DynamicTableRegion) into
    that table's rows; with ragged data each row links to 0..n rows, which is
    how optional links are stored (empty = no link)."""
    cols, n = [], 0
    for c, spec in columns.items():
        data, desc = spec[0], spec[1]
        table = spec[2] if len(spec) > 2 else None
        ragged = isinstance(data, list) and len(data) and isinstance(data[0], np.ndarray)
        flat = np.concatenate(data) if ragged else np.asarray(data)
        if table is not None:
            vd = DynamicTableRegion(name=c, description=desc, data=flat.astype(np.int64), table=table)
        else:
            vd = VectorData(name=c, description=desc, data=flat)
        if ragged:
            idx = np.cumsum([len(x) for x in data]).astype(np.int64)
            cols += [vd, VectorIndex(name=c + '_index', data=idx, target=vd)]
        else:
            cols.append(vd)
        n = len(data)
    return cls(name=name, description=description, columns=cols, id=np.arange(n))


def optional_link(idx):
    """Per-row optional link for a ragged DynamicTableRegion: -1 -> no link."""
    return [np.array([i], dtype=np.int64) if i >= 0 else np.array([], dtype=np.int64) for i in np.asarray(idx)]


def nearest(ref, x, tol):
    """Index into sorted ref of the nearest element to each x, -1 if farther than tol."""
    ref = np.asarray(ref, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), -1, dtype=np.int64)
    if len(ref) == 0 or len(x) == 0:
        return out
    j = np.clip(np.searchsorted(ref, x), 1, max(len(ref) - 1, 1)) if len(ref) > 1 else np.zeros(len(x), int)
    if len(ref) > 1:
        j = np.where(np.abs(ref[j - 1] - x) <= np.abs(ref[j] - x), j - 1, j)
    ok = np.isfinite(x) & (np.abs(ref[j] - x) <= tol)
    out[ok] = j[ok]
    return out


def compressed(data, chunk_rows=None):
    chunks = None
    if chunk_rows is not None:
        chunks = (min(chunk_rows, data.shape[0]),) + tuple(data.shape[1:])
    return H5DataIO(data, compression='gzip', compression_opts=4, chunks=chunks)


def _tz(dt, tz=None):
    if dt.tzinfo is not None:
        return dt
    if tz is None:
        return dt.replace(tzinfo=tzlocal())
    from zoneinfo import ZoneInfo
    return dt.replace(tzinfo=ZoneInfo(tz))


class NPXSessionTask(luigi.Task):
    config = luigi.Parameter()
    session = luigi.Parameter()

    @property
    def cfg(self):
        return BirdConfig(self.config)

    @property
    def nwb_path(self):
        return str(self.cfg.nwb_path(self.session))

    @property
    def run_dir(self):
        return self.cfg.catgt_run(self.session)

    def clock(self):
        run = self.run_dir
        return build_ni_clock(run, load_raw_file_table(run, self.cfg['raw_dir']))

    def base_requires(self):
        return CreateNPXSessionNWB(config=self.config, session=self.session)


# ----------------------------------------------------------------------------
class CreateNPXSessionNWB(NPXSessionTask):
    """NWB file with subject, probe, electrodes, recording segments and CatGT metadata."""

    def requires(self):
        return RunCatGT(config=self.config, session=self.session)

    def output(self):
        return luigi.LocalTarget(self.nwb_path)

    def run(self):
        cfg = self.cfg
        run = self.run_dir
        ni_meta, ap_meta = run.ni_meta, run.ap_meta
        clock = self.clock()

        # t = 0 is the start of the first trigger file in the concatenated NI stream
        start = _tz(datetime.fromisoformat(ni_meta['fileCreateTime']), cfg['timezone'])
        nwbfile = NWBFile(
            session_description=cfg['session_description'],
            identifier=str(uuid4()),
            session_start_time=start,
            session_id=self.session,
            experimenter=cfg['experimenter'],
            lab=cfg['lab'],
            institution=cfg['institution'],
            keywords=['Neuropixels', 'NCAF', 'zebra finch', 'song'],
        )
        subj = dict(cfg['subject'])
        dob = subj.pop('date_of_birth', None)
        if isinstance(dob, str):
            dob = datetime.fromisoformat(dob)
        elif isinstance(dob, date) and not isinstance(dob, datetime):
            dob = datetime(dob.year, dob.month, dob.day)
        nwbfile.subject = Subject(date_of_birth=_tz(dob, cfg['timezone']) if dob is not None else None,
                                  subject_id=subj.pop('subject_id', cfg.bird), **subj)

        self._add_probe(nwbfile, ap_meta)

        # recording segments: contiguous stretches of the triggered recording
        seg = clock.segments()
        cols = {'start_time': (seg.start_time.values, 'start in NI tcat seconds'),
                'stop_time': (seg.stop_time.values, 'stop in NI tcat seconds'),
                'first_t_index': (seg.first_t_index.values, 'first SpikeGLX trigger file (t index) in this segment'),
                'last_t_index': (seg.last_t_index.values, 'last SpikeGLX trigger file (t index) in this segment')}
        if 'acq_time_s' in seg:
            cols.update({
                'acq_time': (seg.acq_time_s.values, 'start in seconds since SpikeGLX acquisition started (NI clock); '
                             'differences between rows give the real time between segments'),
                'ni_abs_first_sample': (seg.ni_abs_first_sample.values, 'absolute NI sample of the segment start'),
                'imec_abs_first_sample': (seg.imec_abs_first_sample.values, 'absolute imec sample of the segment start'),
                'file_create_time': (seg.file_create_time.astype(str).values, 'SpikeGLX fileCreateTime of the first file'),
            })
        nwbfile.add_time_intervals(make_table(
            TimeIntervals, 'recording_segments',
            'Contiguous recorded stretches. The recording is TTL-triggered and CatGT concatenates the '
            'trigger files, so NWB time skips the gaps between rows.', cols))

        meta_mod = nwbfile.create_processing_module(
            name='metadata', description='CatGT / clock alignment metadata for this session')
        offs = run.ct_offsets
        raw = load_raw_file_table(run, cfg['raw_dir'])
        if raw is not None:
            offs = pd.concat([raw.reset_index(drop=True), offs], axis=1)
        meta_mod.add(DynamicTable.from_dataframe(offs, name='catgt_offsets',
                                                 table_description='One row per SpikeGLX trigger file: where it starts in each '
                                                 'concatenated stream (smp_*/sec_*) and its absolute first sample'))
        meta_mod.add(DynamicTable.from_dataframe(clock.piece_summary, name='ni_clock_pieces',
                                                 table_description='Per trigger file imec->NI alignment: sync edges paired and offsets'))
        ie, ne = read_edges(run.imec_sync_edges_path), read_edges(run.ni_sync_edges_path)
        meta_mod.add(make_table(DynamicTable, 'sync_edges', 'Sync pulse edge times (s) in each tcat stream',
                                {'imec_sync_sec': (np.r_[ie, np.full(max(0, len(ne) - len(ie)), np.nan)], 'imec tcat seconds'),
                                 'nidq_sync_sec': (np.r_[ne, np.full(max(0, len(ie) - len(ne)), np.nan)], 'NI tcat seconds')}))
        params = {'catgt_cmdline': ni_meta.get('catGTCmdline0', ''),
                  'ni_sample_rate': ni_meta['niSampRate'], 'imec_sample_rate': ap_meta['imSampRate'],
                  'clock': 'nidq (CatGT-concatenated NI file)', 'sync_method': cfg['sync']['method'],
                  'kilosort_settings': json.dumps(cfg['kilosort']['settings']),
                  'config_path': str(cfg.path), 'config': json.dumps(cfg.data, default=str)}
        meta_mod.add(make_table(DynamicTable, 'pipeline_params', 'Settings used to produce this file',
                                {'key': (list(params.keys()), 'name'), 'value': ([str(v) for v in params.values()], 'value')}))

        Path(self.nwb_path).parent.mkdir(parents=True, exist_ok=True)
        with NWBHDF5IO(self.nwb_path, 'w') as io:
            io.write(nwbfile)

    def _add_probe(self, nwbfile, ap_meta):
        cfg = self.cfg
        p = cfg['probe']
        info = imec_device_info(ap_meta)
        device = nwbfile.create_device(
            name=f"imec{self.run_dir.probe}", manufacturer='imec',
            description=f"Neuropixels {info.get('imDatPrb_pn', '')} " + json.dumps(info),
            serial_number=info.get('imDatPrb_sn'))
        ks_dir = cfg.ks_dir(self.session)
        pos = np.load(ks_dir / 'channel_positions.npy')
        shanks = np.load(ks_dir / 'channel_shanks.npy').astype(int)
        chan_map = np.load(ks_dir / 'channel_map.npy').ravel()

        # songephys convention: +x posterior, +y inferior, +z right (mm)
        post = int(p['posterior_shank'])
        sign = 1.0 if post == shanks.min() else -1.0  # direction of increasing probe x that is anterior
        x0 = pos[shanks == post, 0].mean()
        ap = p['ap_mm'] + sign * (pos[:, 0] - x0) / 1000.0

        groups = {}
        for s in np.unique(shanks):
            groups[s] = nwbfile.create_electrode_group(
                name=f'imec{self.run_dir.probe}_shank{s}', device=device, location=p['location'],
                description=f'shank {s} (Kilosort channel_shanks label); shank {post} is posterior')
        nwbfile.add_electrode_column(name='shank', description='Kilosort shank label (1-based)')
        nwbfile.add_electrode_column(name='channel', description='SpikeGLX/Kilosort channel number')
        for i, ch in enumerate(chan_map):
            nwbfile.add_electrode(
                id=int(ch), group=groups[shanks[i]], location=p['location'],
                x=float(-ap[i]), y=float(p['depth_mm'] - pos[i, 1] / 1000.0), z=float(p['ml_mm']),
                rel_x=float(pos[i, 0]), rel_y=float(pos[i, 1]),
                filtering=f"CatGT {read_meta(self.run_dir.ap_bin).get('catGTCmdline0', '')}",
                reference=ap_meta.get('imChan0Ref', ''), shank=int(shanks[i]), channel=int(ch))


# ----------------------------------------------------------------------------
class AddNIAcquisitionNWB(NPXSessionTask):
    """audio, feedback copy, raw digital word and ttl_events from the NI stream."""

    def requires(self):
        return self.base_requires()

    def output(self):
        return NWBAcquisitionTarget(self.nwb_path, 'ni_digital', 'ttl_events',
                                    *self.cfg['ni']['analog'].values())

    def run(self):
        cfg = self.cfg
        run = self.run_dir
        meta = run.ni_meta
        fs = sample_rate(meta)
        ni = memmap_bin(run.ni_bin, meta)
        word = np.ascontiguousarray(ni[:, ni_digital_word_index(meta)]).astype(np.uint16)

        with NWBHDF5IO(self.nwb_path, 'r+') as io:
            nwbfile = io.read()
            for chan, name in cfg['ni']['analog'].items():
                chan = int(chan)
                nwbfile.add_acquisition(TimeSeries(
                    name=name, data=compressed(np.ascontiguousarray(ni[:, chan]), chunk_rows=int(fs)),
                    unit='volts', conversion=ni_volts_per_bit(meta, chan), rate=fs, starting_time=0.0,
                    description=f'NI analog channel XA{chan} (int16 counts; multiply by conversion for volts)'))
            line_names = {int(k): v for k, v in cfg['ni']['digital_lines'].items()}
            for i, l in enumerate(cfg['ni']['syllable_code_lines']):
                line_names.setdefault(int(l), f'syllable_code_bit{i}')
            nwbfile.add_acquisition(TimeSeries(
                name='ni_digital', data=compressed(word.astype(np.uint8), chunk_rows=int(fs)),
                unit='n/a', rate=fs, starting_time=0.0,
                description='Raw NI digital word XD0; bit n = digital line n',
                comments=json.dumps({str(k): v for k, v in sorted(line_names.items())})))

            ts, typ, names, dur = [], [], [], []
            for line, name in sorted(line_names.items()):
                if name == 'sync' or name.startswith('syllable_code'):
                    continue  # sync edges are in processing/metadata, codes in intervals/syllables
                r, f = pulse_edges(decode_bits(word, line), invert=False)
                ts.append((r + 1) / fs); dur.append((f - r) / fs)
                typ.append(np.full(len(r), line)); names.append(np.full(len(r), name, dtype=object))
            order = np.argsort(np.concatenate(ts), kind='stable')
            nwbfile.add_acquisition(make_table(DynamicTable, 'ttl_events', 'Precise time and metadata about TTL pulses', {
                'timestamp': (np.concatenate(ts)[order], 'time of the TTL pulse onset (s)'),
                'ttl_type': (np.concatenate(typ)[order], 'NI digital line number'),
                'ttl_name': (np.concatenate(names)[order].astype(str), 'name of the TTL line'),
                'duration': (np.concatenate(dur)[order], 'duration of the TTL pulse (s)'),
            }))
            io.write(nwbfile)


def decode_syllable_codes(word, lines, fs, settle_ms=3.0, min_gap_ms=5.0):
    """Syllable detections from the code lines: onset sample, code, and code-pulse end sample."""
    port = np.zeros(len(word), dtype=np.uint8)
    for i, l in enumerate(lines):
        port |= (decode_bits(word, int(l)).astype(np.uint8) << i)
    on = np.flatnonzero((port[1:] != 0) & (port[:-1] == 0)) + 1
    on = on[np.r_[True, np.diff(on) > min_gap_ms * fs / 1000]]
    settle = max(1, int(settle_ms * fs / 1000))
    codes = np.array([port[i:i + settle].max() for i in on], dtype=int)
    off = np.flatnonzero((port[1:] == 0) & (port[:-1] != 0)) + 1
    ends = off[np.clip(np.searchsorted(off, on), 0, max(len(off) - 1, 0))] if len(off) else on
    return on, codes, ends


class AddSyllablesNWB(NPXSessionTask):
    """intervals/syllables: online syllable detections decoded from the NI code lines."""

    def requires(self):
        return self.base_requires()

    def output(self):
        return NWBIntervalsTarget(self.nwb_path, 'syllables')

    def run(self):
        cfg = self.cfg
        run = self.run_dir
        meta = run.ni_meta
        fs = sample_rate(meta)
        word = np.ascontiguousarray(memmap_bin(run.ni_bin, meta)[:, ni_digital_word_index(meta)]).astype(np.uint16)
        nic = cfg['ni']
        on, codes, ends = decode_syllable_codes(word, nic['syllable_code_lines'], fs,
                                                nic['syllable_settle_ms'], nic['syllable_min_gap_ms'])
        stop = ends / fs
        stop_desc = 'end of the code pulse (s)'
        extra = {}
        snd = [int(k) for k, v in nic['digital_lines'].items() if v == 'sound_threshold']
        if snd:
            # the code pulse is emitted at syllable onset, just before the sound_threshold pulse
            # for that syllable rises; pair each code with the nearest sound onset
            r, f = pulse_edges(decode_bits(word, snd[0]), invert=False)
            r, f = r + 1, f + 1
            matched = np.zeros(len(on), bool)
            j = np.zeros(len(on), int)
            if len(r):
                j = np.clip(np.searchsorted(r, on), 1, len(r) - 1) if len(r) > 1 else np.zeros(len(on), int)
                if len(r) > 1:
                    j = np.where(np.abs(r[j - 1] - on) < np.abs(r[j] - on), j - 1, j)
                matched = np.abs(r[j] - on) <= nic['syllable_sound_match_ms'] * fs / 1000
            snd_on = np.where(matched, r[j] / fs if len(r) else np.nan, np.nan)
            snd_off = np.where(matched, f[j] / fs if len(r) else np.nan, np.nan)
            stop = np.where(matched, np.maximum(snd_off, on / fs), stop)
            stop_desc = 'offset of the matched sound_threshold pulse (syllable end); end of the code pulse if unmatched'
            extra = {'sound_onset': (snd_on, 'onset of the sound_threshold pulse for this syllable (NaN if none '
                                     f"within {nic['syllable_sound_match_ms']} ms)"),
                     'sound_offset': (snd_off, 'offset of that sound_threshold pulse (NaN if none)')}
        cols = {'start_time': (on / fs, 'syllable detection time: onset of the code pulse (s)'),
                'stop_time': (stop, stop_desc),
                'code': (codes, 'syllable code = sum(line_i << i) over the syllable_code_lines')}
        cols.update(extra)
        with NWBHDF5IO(self.nwb_path, 'r+') as io:
            nwbfile = io.read()
            nwbfile.add_time_intervals(make_table(
                TimeIntervals, 'syllables',
                f"Online syllable detections from NI lines {nic['syllable_code_lines']}", cols))
            io.write(nwbfile)


# ----------------------------------------------------------------------------
class AddNPXUnits(NPXSessionTask):
    """units: Kilosort clusters (noise excluded) with NI-clock spike times."""
    kilosort_dir = luigi.OptionalParameter(default=None)

    def requires(self):
        return {'nwb': self.base_requires(),
                'aligned': AlignSpikesToNI(config=self.config, session=self.session)}

    def output(self):
        return NWBUnitsTarget(self.nwb_path)

    def ks_dir(self):
        ks = self.cfg.ks_dir(self.session)
        return ks if self.kilosort_dir is None else ks.parent / self.kilosort_dir

    def run(self):
        ks = self.ks_dir()
        st_ni = np.load(ks / 'spike_times_ni_sec.npy').ravel()
        st_imec = np.load(ks / 'spike_times_sec.npy').ravel()
        clu = np.load(ks / 'spike_clusters.npy').ravel()
        pos = np.load(ks / 'spike_positions.npy')
        chan_map = np.load(ks / 'channel_map.npy').ravel()
        templates = np.load(ks / 'templates.npy', mmap_mode='r')
        spike_templates = np.load(ks / 'spike_templates.npy').ravel()

        def tsv(name):
            p = ks / name
            return pd.read_csv(p, sep='\t').set_index('cluster_id') if p.exists() else None
        group = tsv('cluster_group.tsv')
        group = group[group.columns[-1]] if group is not None else tsv('cluster_KSLabel.tsv')['KSLabel']
        amp, contam = tsv('cluster_Amplitude.tsv'), tsv('cluster_ContamPct.tsv')

        order = np.argsort(clu, kind='stable')
        ids, starts, counts = np.unique(clu[order], return_index=True, return_counts=True)
        mean_pos = np.stack([np.bincount(clu, weights=pos[:, i], minlength=ids.max() + 1) for i in range(2)], 1)
        mean_pos = mean_pos / np.maximum(np.bincount(clu, minlength=ids.max() + 1), 1)[:, None]
        duration = recording_duration_from_meta(self.run_dir)

        with NWBHDF5IO(self.nwb_path, 'r+') as io:
            nwbfile = io.read()
            for name, desc, kw in [
                ('kilosort_dir', 'The directory where the kilosort output is stored', {}),
                ('kilosort_id', 'The id of the unit in the kilosort output', {}),
                ('group', 'The Kilosort Group (good, mua, noise)', {}),
                ('ks_x', 'The approx X position in um from Kilosort (mean spike position)', {}),
                ('ks_y', 'The approx Y position in um from Kilosort (mean spike position)', {}),
                ('peak_channel', 'channel with the largest template amplitude', {}),
                ('amplitude', 'Kilosort cluster amplitude', {}),
                ('contam_pct', 'Kilosort estimated contamination (%)', {}),
                ('n_spikes', 'number of spikes', {}),
                ('firing_rate', 'mean firing rate over the recorded time (Hz)', {}),
                ('spike_times_imec', 'spike times on the imec clock (s, tcat), before NI alignment', {'index': True}),
            ]:
                nwbfile.add_unit_column(name=name, description=desc, **kw)
            electrode_ids = list(nwbfile.electrodes.id[:])
            for cid, s0, n in zip(ids, starts, counts):
                if group.get(cid, 'unsorted') == 'noise':
                    continue
                idx = order[s0:s0 + n]
                idx = idx[np.argsort(st_ni[idx], kind='stable')]
                # after Phy merges a cluster can hold several templates; use the most common one
                t = np.asarray(templates[np.bincount(spike_templates[idx]).argmax()])
                peak = int(chan_map[np.argmax(t.max(0) - t.min(0))])
                nwbfile.add_unit(
                    spike_times=st_ni[idx], spike_times_imec=st_imec[idx],
                    electrodes=[electrode_ids.index(peak)],
                    kilosort_dir=str(ks), kilosort_id=int(cid), group=str(group.get(cid, 'unsorted')),
                    ks_x=float(mean_pos[cid, 0]), ks_y=float(mean_pos[cid, 1]), peak_channel=peak,
                    amplitude=float(amp.Amplitude.get(cid, np.nan)) if amp is not None else np.nan,
                    contam_pct=float(contam.ContamPct.get(cid, np.nan)) if contam is not None else np.nan,
                    n_spikes=int(n), firing_rate=float(n / duration))
            io.write(nwbfile)


def recording_duration_from_meta(run):
    m = run.ni_meta
    return float(m['fileTimeSecs'])


class AddNPXUnitWaveforms(NPXSessionTask):
    """acquisition/unit_waveforms: mean/sd raw (CatGT-filtered) waveforms on the top channels."""

    def requires(self):
        return AddNPXUnits(config=self.config, session=self.session)

    def output(self):
        return NWBAcquisitionTarget(self.nwb_path, 'unit_waveforms')

    def run(self):
        cfg = self.cfg['nwb']
        run = self.run_dir
        meta = run.ap_meta
        fs = sample_rate(meta)
        ap = memmap_bin(run.ap_bin, meta)
        n_neural = int(meta['snsApLfSy'].split(',')[0]) if 'snsApLfSy' in meta else ap.shape[1] - 1
        half = int(cfg['waveform_ms'] * fs / 1000)
        win = np.arange(-half, half + 1)
        rng = np.random.default_rng(0)
        ks = Path(self.requires().ks_dir())
        st = np.load(ks / 'spike_times.npy').ravel().astype(np.int64)
        clu = np.load(ks / 'spike_clusters.npy').ravel()
        templates = np.load(ks / 'templates.npy', mmap_mode='r')
        chan_map = np.load(ks / 'channel_map.npy').ravel()
        spike_templates = np.load(ks / 'spike_templates.npy').ravel()
        uv = float(meta['imAiRangeMax']) / float(meta['imMaxInt']) / float(meta.get('imChan0apGain', 80)) * 1e6

        with NWBHDF5IO(self.nwb_path, 'r+') as io:
            nwb = io.read()
            units = nwb.units.to_dataframe(index=True)
            unit_ids, elecs, means, sds, temps = [], [], [], [], []
            for uid, unit in units.iterrows():
                in_unit = clu == unit.kilosort_id
                s = st[in_unit]
                tmpl = np.bincount(spike_templates[in_unit]).argmax()
                s = s[(s > half) & (s < ap.shape[0] - half - 1)]
                s = np.sort(rng.choice(s, min(len(s), cfg['waveform_n_spikes']), replace=False))
                wf = np.asarray(ap[s[:, None] + win[None, :], :n_neural], dtype=np.float32)  # (n, t, ch)
                m, sd = wf.mean(0).T, wf.std(0).T  # (ch, t)
                top = np.argsort(np.abs(m - np.median(m, 1, keepdims=True)).sum(1))[::-1][:cfg['waveform_n_channels']]
                t = np.asarray(templates[tmpl])  # (t, ch idx into chan_map)
                tidx = np.array([np.flatnonzero(chan_map == c)[0] if np.any(chan_map == c) else -1 for c in top])
                temps.append(np.stack([t[:, i] if i >= 0 else np.zeros(t.shape[0]) for i in tidx]))
                unit_ids.append(uid); elecs.append(top); means.append(m[top]); sds.append(sd[top])
            nwb.add_acquisition(make_table(DynamicTable, 'unit_waveforms',
                f'Mean/SD waveforms on the top {cfg["waveform_n_channels"]} channels from up to '
                f'{cfg["waveform_n_spikes"]} spikes; int16 counts of the CatGT-filtered AP band '
                f'({uv:.4f} uV/count)', {
                    'unit_id': (np.array(unit_ids), 'row id of the unit in the units table'),
                    'electrodes': (np.stack(elecs), 'channels (electrode ids) of the waveforms, strongest first'),
                    'waveform_mean': (np.stack(means), 'mean waveform (channel x time)'),
                    'waveform_sd': (np.stack(sds), 'std of waveform (channel x time)'),
                    'template': (np.stack(temps), 'Kilosort template on the same channels (channel x time)'),
                }))
            io.write(nwb)


class AddNPXUnitValidIntervals(NPXSessionTask):
    def requires(self):
        return AddNPXUnits(config=self.config, session=self.session)

    def output(self):
        return NWBIntervalsTarget(self.nwb_path, 'unit_intervals')

    def run(self):
        with NWBHDF5IO(self.nwb_path, 'r+') as io:
            nwb = io.read()
            tot = recording_duration(nwb)
            starts, stops, uids = [], [], []
            for uid in range(len(nwb.units)):
                for a, b in compute_unit_valid_intervals(nwb.units['spike_times'][uid], tot):
                    starts.append(float(a)); stops.append(float(b)); uids.append(int(nwb.units.id[uid]))
            nwb.add_time_intervals(make_table(TimeIntervals, 'unit_intervals', 'Valid intervals for each unit', {
                'start_time': (np.array(starts, float), 'start (s)'), 'stop_time': (np.array(stops, float), 'stop (s)'),
                'unit_id': (np.array(uids, int), 'The id of the unit in the nwb file that this interval corresponds to')}))
            io.write(nwb)


class GenerateNPXNWB(luigi.WrapperTask):
    """Everything except units (like gen-nwb for the Intan pipeline)."""
    config = luigi.Parameter()
    session = luigi.Parameter()

    def requires(self):
        kw = dict(config=self.config, session=self.session)
        return [CreateNPXSessionNWB(**kw), AddNIAcquisitionNWB(**kw), AddSyllablesNWB(**kw)]


class AddNPXSortedUnits(luigi.WrapperTask):
    """Units, waveforms, valid intervals and the NCAF tables that link to them (like add-sorted-units)."""
    config = luigi.Parameter()
    session = luigi.Parameter()

    def requires(self):
        from .ncaf_nwb import AddNCAFNWB
        kw = dict(config=self.config, session=self.session)
        reqs = [AddNPXUnits(**kw), AddNPXUnitWaveforms(**kw), AddNPXUnitValidIntervals(**kw)]
        if BirdConfig(self.config).ncaf_dir(self.session).exists():
            reqs.append(AddNCAFNWB(**kw))
        return reqs


def clear_units(nwb_path):
    """Remove units, unit_waveforms, unit_intervals and the NCAF tables (which link to units)
    so a new sort can be added."""
    import os
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwb = io.read()
        if nwb.units is not None:
            nwb.units.reset_parent()
            nwb.fields['units'] = None
        for coll, name in ((nwb.acquisition, 'unit_waveforms'), (nwb.intervals, 'unit_intervals'),
                           (nwb.intervals, 'ncaf_trials'), (nwb.processing, 'ncaf')):
            if name in coll:
                coll[name].reset_parent()
                coll.pop(name)
        tmp = nwb_path + '.tmp'
        with NWBHDF5IO(tmp, 'w') as out:
            out.export(src_io=io, nwbfile=nwb)
    os.replace(tmp, nwb_path)
