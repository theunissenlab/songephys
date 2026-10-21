"""CatGT -> Kilosort 4 -> NI-clock alignment for SpikeGLX runs.

Replaces /home/brian/ksTest/ecephys_spike_sorting/.../sglx_multi_run_pipeline.py
for our use (same CatGT and KS4 settings by default), without the
ecephys_spike_sorting dependency. Outputs land in the same places that pipeline
used ({processed_dir}/catgt_{run}_g{gate}/{run}_g{gate}_imec{p}/imec{p}_ks4), so
runs sorted by either pipeline can be converted to NWB.

Clock: NI is the master. CatGT does not pick a clock (it only extracts sync
edges per stream); the clock is chosen when spike times are mapped. We write
  spike_times_sec.npy        spike times on the imec clock (seconds, tcat)
  spike_times_ni_sec.npy     spike times on the NI clock   (used for NWB)
  spike_times_ni_tprime.npy  TPrime with -tostream = NI sync edges (cross-check)
"""
import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

import luigi
import numpy as np

from .config import BirdConfig
from .spikeglx import read_meta, sample_rate, raw_file_table, read_edges
from .sync import build_ni_clock


def parse_trigger_range(raw_run_dir, run_g, probe, triggers):
    """'start,end' style trigger spec -> (first, last) from the files on disk."""
    first, last = [x.strip() for x in str(triggers).split(',')]
    if first == 'start' or last == 'end':
        files = glob.glob(str(Path(raw_run_dir) / f'{run_g}_imec{probe}' / f'{run_g}_t*.imec{probe}.ap.bin'))
        ts = sorted(int(re.search(r'_t(\d+)\.imec', f).group(1)) for f in files)
        if len(ts) == 0:
            raise FileNotFoundError(f"No raw ap.bin files for {run_g} imec{probe} in {raw_run_dir}")
        first = ts[0] if first == 'start' else int(first)
        last = ts[-1] if last == 'end' else int(last)
    return int(first), int(last)


def catgt_command(cfg, name):
    s = cfg.session(name)
    c = cfg['catgt']
    probes = ','.join(str(p) for p in s['probes'])
    t0, t1 = parse_trigger_range(cfg.raw_run_dir(name), s['run_g'], s['probes'][0], s['triggers'])
    args = [
        f"-dir={cfg['raw_dir']}", f"-run={name}", f"-g={s['gate']}", f"-t={t0},{t1}",
        '-ni', '-ap', f'-prb={probes}', f"-{c['car_mode']}",
    ] + c['options'].split() + c['extractors'].split() + [f"-dest={cfg['processed_dir']}"]
    return [c['path'], ' '.join(args)]


class NPXTask(luigi.Task):
    config = luigi.Parameter()
    session = luigi.Parameter()

    @property
    def cfg(self):
        return BirdConfig(self.config)


class RunCatGT(NPXTask):
    def output(self):
        run = self.cfg.catgt_run(self.session)
        return {'ni_bin': luigi.LocalTarget(str(run.ni_bin)),
                'ap_bin': luigi.LocalTarget(str(run.ap_bin)),
                'raw_files': luigi.LocalTarget(str(run.raw_files_csv()))}

    def run(self):
        cfg = self.cfg
        run = cfg.catgt_run(self.session)
        if not (run.ni_bin.exists() and run.ap_bin.exists()):
            cmd = catgt_command(cfg, self.session)
            print('Running CatGT:', ' '.join(cmd))
            # CatGT writes CatGT.log in the cwd; keep it with the output
            os.makedirs(cfg['processed_dir'], exist_ok=True)
            subprocess.run(cmd, check=True, cwd=cfg['processed_dir'])
            if not run.ap_bin.exists():
                raise RuntimeError(f"CatGT did not produce {run.ap_bin}; see CatGT.log in {cfg['processed_dir']}")
        # cache the raw per-trigger file timing so NWB conversion works after raw data is archived
        s = cfg.session(self.session)
        df = raw_file_table(cfg.raw_run_dir(self.session), s['run_g'], run.probe)
        df.to_csv(run.raw_files_csv(), index=False)


def meta_to_ks_probe(ap_meta):
    """Kilosort probe dict from a SpikeGLX imec meta (snsGeomMap). Shanks are labeled 1..N
    to match the chanMap.mat files written by ecephys_spike_sorting."""
    geom = re.findall(r'\(([^)]*)\)', ap_meta['snsGeomMap'])
    header = geom[0].split(',')
    shank_pitch = float(header[2])
    entries = [tuple(float(x) for x in g.split(':')) for g in geom[1:]]
    shank = np.array([e[0] for e in entries])
    x = np.array([e[1] for e in entries]) + shank * shank_pitch
    y = np.array([e[2] for e in entries])
    n = len(entries)
    return {'chanMap': np.arange(n), 'xc': x, 'yc': y, 'kcoords': shank + 1, 'n_chan': n}


class RunKilosort4(NPXTask):
    def requires(self):
        return RunCatGT(config=self.config, session=self.session)

    def output(self):
        ks = self.cfg.ks_dir(self.session)
        return {'spike_times': luigi.LocalTarget(str(ks / 'spike_times.npy')),
                'spike_times_sec': luigi.LocalTarget(str(ks / 'spike_times_sec.npy'))}

    def run(self):
        from kilosort import run_kilosort
        cfg = self.cfg
        run = cfg.catgt_run(self.session)
        ks_dir = cfg.ks_dir(self.session)
        meta = run.ap_meta
        fs = sample_rate(meta)
        if not (ks_dir / 'spike_times.npy').exists():
            k = cfg['kilosort']
            settings = dict(k['settings'])
            settings['n_chan_bin'] = int(meta['nSavedChans'])
            settings['fs'] = fs
            if settings.get('tmax', -1) < 0:
                settings['tmax'] = np.inf
            ks_dir.mkdir(parents=True, exist_ok=True)
            run_kilosort(settings, probe=meta_to_ks_probe(meta), filename=run.ap_bin,
                         results_dir=ks_dir, data_dtype='int16', do_CAR=k['do_CAR'],
                         save_extra_vars=k['save_extra_vars'], verbose_console=True)
        if not (ks_dir / 'spike_times_sec.npy').exists():
            st = np.load(ks_dir / 'spike_times.npy')
            np.save(ks_dir / 'spike_times_sec.npy', (st / fs).astype(np.float64))


class AlignSpikesToNI(NPXTask):
    """Spike times on the NI clock (spike_times_ni_sec.npy), plus TPrime cross-check."""

    def requires(self):
        return RunKilosort4(config=self.config, session=self.session)

    def output(self):
        ks = self.cfg.ks_dir(self.session)
        return luigi.LocalTarget(str(ks / 'spike_times_ni_sec.npy'))

    def run(self):
        cfg = self.cfg
        run = cfg.catgt_run(self.session)
        ks_dir = cfg.ks_dir(self.session)
        st = np.load(ks_dir / 'spike_times_sec.npy').ravel()

        clock = build_ni_clock(run)
        clock.piece_summary.to_csv(ks_dir / 'ni_clock_pieces.csv', index=False)
        st_ni = clock.imec_to_ni(st)

        if cfg['tprime']['run']:
            st_tp = run_tprime(cfg, run, ks_dir)
            bad = np.abs(st_tp - st_ni) > 1e-3
            print(f"TPrime (NI master) vs segmentwise: {bad.sum()} / {len(st)} spikes differ by >1 ms "
                  "(expected near trigger joins, where TPrime mis-pairs sync edges)")
            if cfg['sync']['method'] == 'tprime':
                st_ni = st_tp
        np.save(self.output().path, st_ni)


def run_tprime(cfg, run, ks_dir):
    """TPrime with the NI sync edges as -tostream (NI is the master clock)."""
    out = ks_dir / 'spike_times_ni_tprime.npy'
    cmd = [cfg['tprime']['path'],
           f"-syncperiod={cfg['sync']['period_s']}",
           f"-tostream={run.ni_sync_edges_path}",
           f"-fromstream=1,{run.imec_sync_edges_path}",
           f"-events=1,{ks_dir / 'spike_times_sec.npy'},{out}"]
    print('Running TPrime:', ' '.join(cmd))
    with open(run.dir / f'{run.run_g}_TPrime_ni_cmd.txt', 'w') as f:
        f.write(' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=str(run.dir))
    return np.load(out).ravel()


class SortSession(luigi.WrapperTask):
    config = luigi.Parameter()
    session = luigi.Parameter()

    def requires(self):
        return AlignSpikesToNI(config=self.config, session=self.session)
