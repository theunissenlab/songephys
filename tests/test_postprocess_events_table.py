import os
import tempfile
import unittest
import importlib.util

import numpy as np

from songephys.postprocess.events_table import build_events_table

_PATH = os.path.join(os.path.dirname(__file__), '..', 'phy_plugins', 'event_aligned_view.py')
_spec = importlib.util.spec_from_file_location('event_aligned_view', _PATH)
eav = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eav)

FS = 30000.0
SITE = {
    'digital_channel_names': ['Stim pulse', 'DIGITAL-IN-02', 'Feedback', 'Syllable', 'Match', 'Song',
                              'Syllable Code Bit1', 'Syllable Code Bit2', 'Syllable Code Bit3'],
    'digital_channel_bits': [0, 1, 8, 9, 10, 11, 13, 14, 15],
    'stim_bouts': [{'bout': 0, 'polarity': '-', 'current_ua': 500}],
}


def synthetic():
    n = int(10 * FS)
    d = np.zeros(n, np.uint16)
    def high(bit, a, b):
        d[a:b] |= np.uint16(1 << bit)
    for s in (30000, 60000):                         # stim TTLs (skipped as a digital line)
        high(0, s, s + 6)
    high(1, 1000, 2000)                              # unnamed line: skipped
    syl = [(90000, 1), (150000, 3), (210000, 0)]     # (onset, code)
    for s, code in syl:
        high(9, s, s + 900)
        for k in range(3):
            if code >> k & 1:
                high(13 + k, s + 1, s + 400)         # code bits set just after the onset
    for s in (90100, 150100):                         # targets
        high(10, s, s + 150)
    high(8, 90100 + 600, 90100 + 600 + 1500)         # feedback 20 ms after the first target only
    high(11, 80000, 240000)                          # one song
    stim = {'sampling_rate': FS, 'bouts': [{'bout': 0, 'start_frame': 30000, 'stop_frame': 60006,
                                           'n_pulses': 2, 'n_delivered': 1, 'rate_hz': 1.0}],
            'events': {'start_frame': [30000, 60000], 'stop_frame': [30006, 60006], 'delivered': [True, False],
                       'bout': [0, 0], 'artifact_score': [300.0, 1.0], 'blanked': [True, False],
                       'blank_start_frame': [29985, None], 'blank_stop_frame': [30045, None]},
            'digital_bit': 0, 'config': {}}
    return d, stim


class TestEventsTable(unittest.TestCase):
    def setUp(self):
        self.d, self.stim = synthetic()
        self.df = build_events_table(lambda a, b: self.d[a:b], len(self.d), FS, SITE, self.stim, stim_bit=0)

    def test_counts_and_names(self):
        self.assertEqual(self.df.event.value_counts().to_dict(),
                         {'syllable': 3, 'stim': 2, 'target': 2, 'feedback': 1, 'song': 1})
        self.assertTrue((np.diff(self.df.time_s.values) >= 0).all())

    def test_columns(self):
        df = self.df
        st = df[df.event == 'stim']
        self.assertEqual(list(st.delivered), [True, False])
        self.assertEqual(list(st.polarity), ['-', '-'])
        self.assertAlmostEqual(st.blank_start_ms.iloc[0], -0.5)
        self.assertAlmostEqual(st.blank_stop_ms.iloc[0], 1.5)
        self.assertEqual(list(df[df.event == 'target'].with_feedback), [True, False])
        self.assertEqual(list(df[df.event == 'syllable'].code), [1, 3, 0])
        self.assertAlmostEqual(df[df.event == 'feedback'].duration_ms.iloc[0], 50.0)
        self.assertAlmostEqual(df[df.event == 'song'].time_s.iloc[0], 80000 / FS)

    def test_phy_plugin_reads_table(self):
        with tempfile.TemporaryDirectory() as d:
            self.df.to_csv(os.path.join(d, 'events.csv'), index=False)
            sets = {s.name: s for s in eav.load_all_event_sets(d, FS)}
        self.assertIn('stim: delivered', sets)
        np.testing.assert_allclose(sets['stim: delivered'].times, [1.0])
        self.assertEqual(sets['stim: delivered'].shade_ms, (-0.5, 1.5))
        np.testing.assert_allclose(sets['stim: TTL, no artefact'].times, [2.0])
        self.assertIn('stim: bout 0 (- 500 uA)', sets)
        np.testing.assert_allclose(sets['target: no feedback'].times, [150100 / FS])
        np.testing.assert_allclose(sets['syllable: code 3'].times, [150000 / FS])
        self.assertNotIn('events', sets)              # the table is not also read as a plain file


if __name__ == '__main__':
    unittest.main()
