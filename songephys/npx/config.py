"""Per-bird configuration for Neuropixels sessions.

One YAML per bird (see example_site_configs/npx_bird_config.yaml) lists the
paths, rig wiring (which NI line is what), probe placement, and the CatGT /
Kilosort / TPrime settings. Each entry under `sessions` is one SpikeGLX run that
becomes one NWB file.
"""
import copy
from pathlib import Path

from ..format.io import load_yaml
from .spikeglx import CatGTRun

DEFAULTS = {
    'experimenter': ['Thomas, Logan'],
    'lab': 'Gadagkar Lab',
    'institution': 'Columbia University',
    'session_description': 'NCAF Neuropixels recording',
    'timezone': None,  # IANA name of the rig's clock, e.g. America/New_York; None = this machine's
    'subject': {'species': 'Taeniopygia guttata', 'description': ''},
    'probe': {
        'location': 'unknown',
        'ml_mm': 0.0,          # +right, -left (NWB +z is right)
        'ap_mm': 0.0,          # +anterior; this is the AP of the posterior-most shank
        'depth_mm': 0.0,       # depth of the probe tip below the surface
        'posterior_shank': 1,  # KS channel_shanks label (1-4) of the posterior-most shank
        'shank_pitch_um': 250,
    },
    'ni': {
        'analog': {0: 'audio', 1: 'feedback_copy'},
        'digital_lines': {0: 'sync', 1: 'trigger', 3: 'feedback', 4: 'sound_threshold'},
        'syllable_code_lines': [5, 6],   # LSB first; code = sum(bit_i << i)
        'syllable_settle_ms': 3.0,
        'syllable_min_gap_ms': 5.0,
        'syllable_sound_match_ms': 5.0,  # max |code onset - sound_threshold onset| to pair them
    },
    'sync': {
        # Every time written to NWB is on this clock. Must stay 'nidq': the audio and
        # digital events are native NI samples, only spike times get converted.
        'master': 'nidq',
        'period_s': 1.0,
        # 'segmentwise' (songephys.npx.sync, handles triggered recordings) or 'tprime'
        'method': 'segmentwise',
    },
    'catgt': {
        'path': '/home/brian/sglx_tools/CatGT-linux/runit.sh',
        'car_mode': 'gblcar',
        'options': ('-zerofillmax=5 -prb_fld -out_prb_fld -pass1_force_ni_ob_bin '
                    '-apfilter=butter,12,300,9000 -lffilter=butter,12,1,500 -gfix=0.4,0.10,0.02'),
        # NI/imec sync edges are extracted automatically; extra extractors are optional
        # because the NWB step decodes events straight from the NI digital word.
        'extractors': '',
    },
    'kilosort': {
        'output_name': 'imec{probe}_ks4',
        'do_CAR': False,  # CAR already done by CatGT
        'save_extra_vars': True,
        'settings': {
            'Th_universal': 8, 'Th_learned': 9, 'duplicate_spike_ms': 0.25, 'nblocks': 6,
            'sig_interp': 20.0, 'whitening_range': 43, 'min_template_size': 10,
            'template_sizes': 5, 'templates_from_data': True, 'nearest_chans': 10,
            'nearest_templates': 100, 'ccg_threshold': 0.25, 'acg_threshold': 0.20,
            'cluster_init_seed': 1, 'tmin': 0, 'tmax': -1,
        },
    },
    'tprime': {
        'path': '/home/brian/sglx_tools/TPrime-linux/runit.sh',
        'run': True,   # also run TPrime (NI master) as a cross-check of the alignment
    },
    'ncaf': {
        # NCAF live outputs; per-session `ncaf_dir` overrides {raw_dir}/{run}_g{gate}/{subdir}
        'subdir': 'ncaf_outs',
        'trial_window_ms': [-50, 250],   # online spikes stored per trial, relative to syllable onset
        'link_tol_ms': 5.0,              # max gap when pairing NCAF trigger / feedback TTL / XA1 burst
        'match_window_ms': 0.5,          # online spike <-> offline (Kilosort) spike coincidence window
        'match_max_dist_um': 100.0,      # target unit must be this close to the filter channels
        'n_candidates': 5,               # offline units scored and stored as target candidates
        'xa1_threshold_v': 0.02,         # feedback-copy burst detection (as snakeSort get_feedbacks)
        'xa1_smooth_samples': 100,
        'xa1_min_ms': 5.0,
    },
    'nwb': {
        'waveform_ms': 2.0,
        'waveform_n_spikes': 1000,
        'waveform_n_channels': 4,
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class BirdConfig(object):
    def __init__(self, path):
        self.path = Path(path)
        self.data = _merge(DEFAULTS, load_yaml(self.path))
        if self.data['sync']['master'] != 'nidq':
            raise ValueError("sync.master must be 'nidq': NWB times are defined on the NI clock")

    def __getitem__(self, k):
        return self.data[k]

    @property
    def bird(self):
        return self.data['bird']

    @property
    def sessions(self):
        return list(self.data['sessions'].keys())

    def session(self, name):
        s = dict(gate='0', triggers='start,end', probes=[0])
        s.update(self.data['sessions'][name] or {})
        s['name'] = name
        first_gate = str(s['gate']).split(',')[0]
        s['run_g'] = f'{name}_g{first_gate}'
        return s

    def catgt_run(self, name, probe=None):
        s = self.session(name)
        probe = s['probes'][0] if probe is None else probe
        return CatGTRun(Path(self.data['processed_dir']) / f"catgt_{s['run_g']}", probe=probe)

    def raw_run_dir(self, name):
        return Path(self.data['raw_dir']) / self.session(name)['run_g']

    def ks_dir(self, name, probe=None):
        run = self.catgt_run(name, probe)
        return run.ks_dir(self.data['kilosort']['output_name'].format(probe=run.probe))

    def ncaf_dir(self, name):
        s = self.session(name)
        if s.get('ncaf_dir'):
            return Path(s['ncaf_dir'])
        return self.raw_run_dir(name) / self.data['ncaf']['subdir']

    def nwb_path(self, name):
        return Path(self.data['nwb_dir']) / f'{name}.nwb'
