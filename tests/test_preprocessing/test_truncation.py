import json
import os
import tempfile
import unittest

import numpy as np

from songephys.pipeline.truncation import (
    check_site_outputs_respect_tmax, get_site_tmax, outputs_past_tmax, parse_tmax,
    truncate_recordings)


def _rec(n, fs=1000.0, n_seg=1, start=0):
    from spikeinterface.core import NumpyRecording
    segs = [np.arange(start + k * n, start + (k + 1) * n, dtype=np.float32)[:, None] for k in range(n_seg)]
    return NumpyRecording(segs, sampling_frequency=fs)


class TestParseTmax(unittest.TestCase):
    def test_formats(self):
        self.assertIsNone(parse_tmax(None))
        self.assertEqual(parse_tmax(16920), 16920.0)
        self.assertEqual(parse_tmax('16920.5'), 16920.5)
        self.assertEqual(parse_tmax('4:42:00'), 16920.0)
        self.assertEqual(parse_tmax('42:30'), 2550.0)
        self.assertEqual(get_site_tmax({'tmax': '0:01:00'}), 60.0)
        self.assertIsNone(get_site_tmax({}))

    def test_bad_values(self):
        for v in (0, -5, '1::2', '1:2:3:4', 'abc', float('nan')):
            with self.assertRaises(ValueError):
                parse_tmax(v)


class TestTruncateRecordings(unittest.TestCase):
    def test_no_tmax_keeps_everything(self):
        recs = [_rec(100), _rec(50)]
        self.assertEqual(len(truncate_recordings(recs, None)), 2)

    def test_cut_inside_second_file(self):
        out = truncate_recordings([_rec(1000), _rec(1000, start=1000), _rec(1000, start=2000)], 1.5)
        self.assertEqual([r.get_num_samples(0) for r in out], [1000, 500])
        np.testing.assert_array_equal(out[1].get_traces()[:, 0], np.arange(1000, 1500))

    def test_cut_at_file_boundary_and_beyond_end(self):
        out = truncate_recordings([_rec(1000), _rec(1000)], 1.0)
        self.assertEqual([r.get_num_samples(0) for r in out], [1000])
        out = truncate_recordings([_rec(1000), _rec(1000)], 10.0)
        self.assertEqual([r.get_num_samples(0) for r in out], [1000, 1000])

    def test_multi_segment_and_rate(self):
        # 2 segments of 0.5 s at 4 kHz: tmax 0.75 s -> 2000 + 1000 samples
        out = truncate_recordings([_rec(2000, fs=4000.0, n_seg=2)], 0.75)
        self.assertEqual([r.get_num_samples(0) for r in out], [2000, 1000])

    def test_concatenated_length(self):
        from spikeinterface.core import concatenate_recordings
        out = truncate_recordings([_rec(30000, fs=30000.0), _rec(30000, fs=30000.0)], 1.25)
        cat = concatenate_recordings(out)
        self.assertEqual(cat.get_total_samples(), 37500)


class TestStaleOutputs(unittest.TestCase):
    def test_detects_long_outputs(self):
        with tempfile.TemporaryDirectory() as site:
            raw = os.path.join(site, 'raw')
            os.makedirs(raw)
            for name, n, fs in (('site_info.json', 3000, 30000), ('lfp_site_info.json', 100, 1000),
                                ('digital_site_info.json', 30000, 30000)):
                with open(os.path.join(raw, name), 'w') as f:
                    json.dump({'num_samples': n, 'sampling_rate': fs}, f)
            stale = outputs_past_tmax(site, 0.1)
            self.assertEqual([os.path.basename(p) for p, _ in stale], ['digital_site_info.json'])
            self.assertEqual(outputs_past_tmax(site, None), [])
            with self.assertRaises(RuntimeError):
                check_site_outputs_respect_tmax(site, {'tmax': 0.1})
            check_site_outputs_respect_tmax(site, {'tmax': 2})  # nothing longer than 2 s
            check_site_outputs_respect_tmax(site, {})


if __name__ == '__main__':
    unittest.main()
