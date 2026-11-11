import importlib.util
import json
import os
import tempfile
import unittest

import numpy as np

_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'phy_plugins', 'event_aligned_view.py')
_spec = importlib.util.spec_from_file_location('event_aligned_view', _PATH)
eav = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eav)


class TestAlign(unittest.TestCase):
    def test_align_matches_brute_force(self):
        rng = np.random.default_rng(0)
        st = np.sort(rng.uniform(0, 100, 5000))
        ev = np.sort(rng.uniform(1, 99, 40))
        rel, trial = eav.align_spikes(st, ev, 0.02, 0.05)
        exp = [(s - e, i) for i, e in enumerate(ev) for s in st if -0.02 <= s - e <= 0.05]
        self.assertEqual(len(rel), len(exp))
        np.testing.assert_allclose(rel, [r for r, _ in exp])
        np.testing.assert_array_equal(trial, [i for _, i in exp])

    def test_align_empty(self):
        rel, trial = eav.align_spikes(np.array([50.0]), np.array([1.0, 2.0]), 0.01, 0.01)
        self.assertEqual(len(rel), 0)
        self.assertEqual(len(trial), 0)

    def test_psth_rate(self):
        # one spike at +1 ms in each of 10 events -> 1000 Hz in a 1 ms bin
        rel = np.full(10, 0.0015)
        edges, rate = eav.psth(rel, 10, 0.005, 0.005, 0.001)
        self.assertEqual(len(edges), 11)
        self.assertAlmostEqual(rate.max(), 1000.0)
        self.assertAlmostEqual(rate.sum() * 0.001 * 10, 10)


class TestTraceHelpers(unittest.TestCase):
    def test_marker_lines(self):
        markers = [(np.array([1.0, 2.0, 3.0]), 0.0, (1, 1, 0, 1)),
                   (np.array([1.0, 2.0, 3.0]), 1.5, (0.5, 0.5, 0.5, 1))]
        pos, col = eav.marker_lines(markers, 1.9, 2.1)
        np.testing.assert_allclose(pos, [[2.0, -1, 2.0, 1], [2.0015, -1, 2.0015, 1]])
        self.assertEqual(col.shape, (2, 4))
        self.assertEqual(eav.marker_lines(markers, 5, 6), (None, None))

    def test_neighbour_event(self):
        t = np.array([1.0, 2.0, 3.0])
        self.assertEqual(eav.neighbour_event(t, 2.0, +1), 2)   # strictly after the current one
        self.assertEqual(eav.neighbour_event(t, 2.0, -1), 0)
        self.assertEqual(eav.neighbour_event(t, 1.5, +1), 1)
        self.assertIsNone(eav.neighbour_event(t, 3.0, +1))
        self.assertIsNone(eav.neighbour_event(t, 1.0, -1))


class TestEventSources(unittest.TestCase):
    def test_stim_sets(self):
        fs = 30000.0
        d = {
            'enabled': True, 'sampling_rate': fs,
            'events': {
                'start_frame': [30000, 60000, 90000, 120000],
                'delivered': [True, True, False, True],
                'blanked': [True, True, False, True],
                'blank_start_frame': [29985, 59985, None, 119985],
                'blank_stop_frame': [30045, 60045, None, 120045],
                'bout': [0, 0, 1, 1],
            },
        }
        ann = {0: {'bout': 0, 'polarity': '-', 'current_ua': 500},
               1: {'bout': 1, 'polarity': '+', 'current_ua': 500}}
        # None values come through json as null -> nan with float dtype
        d = json.loads(json.dumps(d))
        d['events']['blank_start_frame'] = [np.nan if v is None else v for v in d['events']['blank_start_frame']]
        d['events']['blank_stop_frame'] = [np.nan if v is None else v for v in d['events']['blank_stop_frame']]
        sets = {s.name: s for s in eav.stim_event_sets(d, ann)}
        np.testing.assert_allclose(sets['stim: delivered'].times, [1, 2, 4])
        np.testing.assert_allclose(sets['stim: TTL, no artefact'].times, [3])
        np.testing.assert_allclose(sets['stim: polarity -'].times, [1, 2])
        np.testing.assert_allclose(sets['stim: polarity +'].times, [4])
        self.assertIn('stim: bout 1 (+ 500 uA)', sets)
        lo, hi = sets['stim: delivered'].shade_ms
        self.assertAlmostEqual(lo, -0.5)
        self.assertAlmostEqual(hi, 1.5)

    def test_target_feedback_sets(self):
        fb = [1.02, 3.02, 9.0]          # 9.0: manual feedback with no target
        tg = [1.0, 2.0, 3.0, 4.0]
        sets = {s.name: s.times for s in eav.target_feedback_sets(fb, tg)}
        np.testing.assert_allclose(sets['feedback'], fb)
        np.testing.assert_allclose(sets['target'], tg)
        np.testing.assert_allclose(sets['target: no feedback'], [2.0, 4.0])
        np.testing.assert_allclose(sets['target: with feedback'], [1.0, 3.0])

    def test_scan_digital_and_site(self):
        n = 100_000
        x = np.zeros(n, np.uint16)
        x[:10] |= 1 << 10                       # target line high at sample 0: not an edge
        for s in (30000, 60000):
            x[s:s + 150] |= 1 << 10              # targets
        x[30600:32100] |= 1 << 8                 # feedback 20 ms after the first target
        with tempfile.TemporaryDirectory() as site:
            os.makedirs(os.path.join(site, 'raw'))
            os.makedirs(os.path.join(site, 'ks'))
            x.tofile(os.path.join(site, 'raw', 'digital_x.bin'))
            with open(os.path.join(site, 'raw', 'digital_x_info.json'), 'w') as f:
                json.dump({'sampling_rate': 30000}, f)
            with open(os.path.join(site, 'site_config.yaml'), 'w') as f:
                f.write('digital_channel_names: [Stim pulse, Feedback, Match]\n'
                        'digital_channel_bits: [0, 8, 10]\n')
            edges = eav.scan_digital_rising(os.path.join(site, 'raw', 'digital_x.bin'), [8, 10], chunk=7777)
            np.testing.assert_array_equal(edges[10], [30000, 60000])
            np.testing.assert_array_equal(edges[8], [30600])
            for _ in range(2):  # second call reads the cache
                sets = {s.name: s.times for s in eav.load_digital_event_sets(os.path.join(site, 'ks'))}
                np.testing.assert_allclose(sets['target'], [1.0, 2.0])
                np.testing.assert_allclose(sets['target: no feedback'], [2.0])
                np.testing.assert_allclose(sets['feedback'], [1.02])
            self.assertTrue(os.path.exists(os.path.join(site, 'ks', '.phy', 'event_aligned_digital.npz')))

    def test_disabled_stim(self):
        self.assertEqual(eav.stim_event_sets({'enabled': False, 'events': {}}), [])

    def test_event_files(self):
        with tempfile.TemporaryDirectory() as d:
            np.save(os.path.join(d, 'events_song.npy'), np.array([1.5, 0.5]))
            np.save(os.path.join(d, 'events_samples.npy'), np.array([30000, 60000], dtype=np.int64))
            with open(os.path.join(d, 'events_ttl.csv'), 'w') as f:
                f.write('time,label\n2.25,a\n3.5,b\n')
            with open(os.path.join(d, 'events_plain.txt'), 'w') as f:
                f.write('# header\n10\n20\n')
            with open(os.path.join(d, 'not_events.txt'), 'w') as f:
                f.write('99\n')
            sets = {s.name: s.times for s in eav.load_event_files(d, 30000.0)}
        self.assertEqual(set(sets), {'events_song', 'events_samples', 'events_ttl', 'events_plain'})
        np.testing.assert_allclose(sets['events_song'], [0.5, 1.5])
        np.testing.assert_allclose(sets['events_samples'], [1.0, 2.0])
        np.testing.assert_allclose(sets['events_ttl'], [2.25, 3.5])
        # integers in a text file are sample indices
        np.testing.assert_allclose(sets['events_plain'], [10 / 30000.0, 20 / 30000.0])

    def test_site_layout(self):
        with tempfile.TemporaryDirectory() as site:
            os.makedirs(os.path.join(site, 'raw'))
            os.makedirs(os.path.join(site, 'ks4'))
            with open(os.path.join(site, 'raw', 'stim_events_x.json'), 'w') as f:
                json.dump({'enabled': True, 'sampling_rate': 30000.0,
                           'events': {'start_frame': [30000], 'delivered': [True], 'bout': [0]}}, f)
            with open(os.path.join(site, 'site_config.yaml'), 'w') as f:
                f.write('stim_bouts:\n- {bout: 0, polarity: "-", current_ua: 500}\n')
            names = [s.name for s in eav.load_all_event_sets(os.path.join(site, 'ks4'), 30000.0)]
        self.assertEqual(names, ['stim: delivered', 'stim: polarity -', 'stim: bout 0 (- 500 uA)'])


if __name__ == '__main__':
    unittest.main()
