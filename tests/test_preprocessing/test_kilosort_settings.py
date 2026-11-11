import unittest
import warnings

from songephys.preprocess.ephys.kilosort_settings import (
    build_kilosort_settings, describe_overrides, get_kilosort_overrides)

# A stand-in for kilosort.DEFAULT_SETTINGS (the real one needs torch).
KS_DEFAULTS = {
    'n_chan_bin': 385, 'fs': 30000, 'batch_size': 60000, 'nblocks': 1,
    'Th_universal': 9, 'Th_learned': 8, 'dminx': 32, 'templates_from_data': True,
    'nearest_templates': 100, 'nearest_chans': 10, 'ccg_threshold': 0.25, 'x_centers': None,
}


class TestKilosortSettings(unittest.TestCase):
    def test_defaults_match_previous_hardcoded_values(self):
        s = build_kilosort_settings(KS_DEFAULTS, 128, 128)
        self.assertEqual(s['n_chan_bin'], 128)
        self.assertEqual(s['nblocks'], 1)
        self.assertEqual(s['batch_size'], 180000)
        self.assertEqual(s['Th_universal'], 7)
        self.assertEqual(s['dminx'], 18.5)
        self.assertEqual(s['nearest_templates'], 100)
        self.assertEqual(s['nearest_chans'], 8)
        self.assertEqual(s['Th_learned'], 8)        # untouched Kilosort default
        self.assertEqual(KS_DEFAULTS['nblocks'], 1)  # input not mutated

    def test_site_overrides_project_overrides_defaults(self):
        project = {'KILOSORT': {'nblocks': 2, 'dminx': 22.5}}
        site = {'kilosort': {'nblocks': 0, 'th_universal': 8}}  # case-insensitive key
        s = build_kilosort_settings(KS_DEFAULTS, 128, 128, site_config=site, project_config=project)
        self.assertEqual(s['nblocks'], 0)
        self.assertEqual(s['dminx'], 22.5)
        self.assertEqual(s['Th_universal'], 8)

    def test_capped_by_channel_count(self):
        site = {'kilosort': {'nearest_chans': 12}}
        s = build_kilosort_settings(KS_DEFAULTS, 128, 6, site_config=site)
        self.assertEqual(s['nearest_chans'], 6)
        self.assertEqual(s['nearest_templates'], 6)

    def test_unknown_and_computed_keys_warn(self):
        site = {'kilosort': {'nblokcs': 0, 'n_chan_bin': 4}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            o = get_kilosort_overrides(KS_DEFAULTS, site_config=site)
        self.assertEqual(len(w), 2)
        self.assertEqual(o['nblocks'], 1)
        self.assertNotIn('n_chan_bin', o)

    def test_describe_overrides(self):
        s = build_kilosort_settings(KS_DEFAULTS, 128, 128, site_config={'kilosort': {'nblocks': 0}})
        lines = describe_overrides(KS_DEFAULTS, s)
        self.assertIn('nblocks: 0', lines)
        self.assertIn('Th_universal: 7', lines)
        self.assertNotIn('Th_learned: 8', lines)


if __name__ == '__main__':
    unittest.main()
