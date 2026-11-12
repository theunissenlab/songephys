import os
import tempfile
import unittest

import numpy as np

from songephys.preprocess.ephys.stim_artifact import (
    blank_and_filter,
    build_stim_events,
    get_stim_artifact_config,
    make_bandpass_sos,
    score_artifacts,
)
from songephys.preprocess.ephys.stim_plot import (
    pick_columns,
    pick_pulses,
    plot_stim_artifact_figures,
    read_pulse_snippets,
)

from .test_stim_artifact import FS, ms, synthetic_stim_recording


class TestHelpers(unittest.TestCase):
    def test_pick_columns_and_pulses(self):
        np.testing.assert_array_equal(pick_columns(115, 4), [0, 38, 76, 114])
        np.testing.assert_array_equal(pick_columns(2, 4), [0, 1])
        np.testing.assert_array_equal(pick_pulses(np.arange(5), 40), np.arange(5))
        p = pick_pulses(np.arange(100), 10)
        self.assertEqual(len(p), 10)
        self.assertEqual((p[0], p[-1]), (0, 99))

    def test_snippets_pad_with_nan_at_the_edges(self):
        x = np.arange(100, dtype=float)[:, None] * np.ones((1, 2))
        s = read_pulse_snippets(lambda a, b: x[a:b], [3, 50, 98], 5, 5, len(x))
        self.assertEqual(s.shape, (3, 10, 2))
        self.assertTrue(np.isnan(s[0, :2]).all())
        np.testing.assert_array_equal(s[0, 2:, 0], np.arange(0, 8))
        np.testing.assert_array_equal(s[1, :, 0], np.arange(45, 55))
        self.assertTrue(np.isnan(s[2, 7:]).all())


class TestFigures(unittest.TestCase):
    def test_writes_one_figure_per_bout_and_window(self):
        # two bouts (pulses > 5 s apart start a new bout) and one manual window
        x, frames = synthetic_stim_recording(n_ch=8, dur_s=9.0, pulse_s=(0.5, 1.5, 8.0))
        x[ms(4000):ms(4200)] += 30000
        x = np.clip(x, -32764, 32764)
        cfg = get_stim_artifact_config({"stim_artifact": {"enabled": True, "extra_blank_s": [[3.99, 4.25]]}})
        score, peak = score_artifacts(lambda a, b: x[a:b], frames, FS, len(x))
        ev = build_stim_events(frames, np.array(frames) + 6, FS, len(x), cfg, score, peak, 0)
        raw_cols = [0, 1, 2, 4, 5, 7]                       # channels 3 and 6 are "bad"
        binary = blank_and_filter(x[:, raw_cols], np.asarray(ev["blank_windows"]), make_bandpass_sos(FS))
        seen = []

        def read_raw(a, b, cols):
            seen.append(list(cols))
            return x[a:b][:, cols]

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "figs")
            files = plot_stim_artifact_figures(
                read_raw, lambda a, b: binary[a:b], raw_cols, FS, len(x), ev, out, n_channels=3,
                annotations=[{"bout": 0, "polarity": "-", "current_ua": 500}])
            self.assertEqual([os.path.basename(f) for f in files],
                             ["stim_bout_00.png", "stim_bout_01.png", "extra_blank_00.png"])
            for f in files:
                self.assertGreater(os.path.getsize(f), 5000)
        # binary columns 0, 2 (rounded 2.5 -> 2), 5 map to raw channels 0, 2, 7
        self.assertTrue(all(c == [0, 2, 7] for c in seen))

    def test_no_events_writes_nothing(self):
        ev = build_stim_events([], [], FS, 1000, get_stim_artifact_config(None))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(plot_stim_artifact_figures(None, None, [0, 1], FS, 1000, ev, d), [])


if __name__ == "__main__":
    unittest.main()
