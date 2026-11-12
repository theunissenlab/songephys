import os
import tempfile
import unittest

import numpy as np

from songephys.preprocess.ephys.stim_artifact import (
    STIM_ARTIFACT_DEFAULTS, blank_and_filter_chunk, make_bandpass_sos, needs_stim_tails)
from songephys.preprocess.ephys.stim_tail import (
    BinnedTail, best_fit, eval_exps, fast_artifact_end, fit_stim_tails, load_stim_tails,
    save_stim_tails, tail_blank_filter_chunk, tail_correction)

FS = 30000.0


def synthetic(n_ch=4, seconds=4.0, seed=0):
    """Recording with 3 stim pulses: rail for ~10 samples, then a 3-exp tail; one spike at +2.5 ms."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = rng.normal(0, 50, (n, n_ch))                     # counts (~10 uV noise)
    onsets = np.array([int(0.5 * FS), int(1.5 * FS), int(2.5 * FS)])
    t = np.arange(int(0.6 * FS)) / FS * 1e3
    taus = np.array([0.8, 9.0, 130.0])
    for p, on in enumerate(onsets):
        for c in range(n_ch):
            sign = 1 if (p + c) % 2 else -1
            x[on - 1:on + 10, c] = sign * 32764              # saturated
            amps = sign * np.array([8000, 6000, 3000]) * (1 + 0.2 * c)
            tail = eval_exps(taus, amps, 0, t)
            x[on + 10:on + 10 + len(t), c] += tail[:len(x) - on - 10]
    spike = -np.exp(-0.5 * ((np.arange(30) - 10) / 3.0) ** 2) * 800
    for on in onsets:
        x[on + 75 - 10:on + 75 + 20, 0] += spike                 # trough at +2.5 ms on channel 0
    return x, onsets


def cfg(**kw):
    c = dict(STIM_ARTIFACT_DEFAULTS, enabled=True, adaptive_blank=True, subtract_tail=True)
    c.update(kw)
    return c


class TestFit(unittest.TestCase):
    def test_recovers_three_exponentials(self):
        tail = BinnedTail(15000, FS)
        y = eval_exps(np.array([0.9, 10.0, 140.0]), np.array([3000.0, 2000.0, 800.0]), -50.0, tail.t_full)
        y = y + np.random.default_rng(1).normal(0, 5, len(y))
        f = best_fit(tail, y, 3)
        np.testing.assert_allclose(f['taus'], [0.9, 10.0, 140.0], rtol=0.1)
        resid = y - eval_exps(f['taus'], f['amps'], f['c'], tail.t_full)
        self.assertLess(resid.std(), 6)

    def test_fast_artifact_end(self):
        x = np.zeros(200)
        x[100:108] = 32764
        x[108:111] = [20000, 5000, 4000]           # big jumps until index 109
        self.assertEqual(fast_artifact_end(x, 100, 60, 32700, 1500), 112)   # never before onset + 12
        x[118] = 9000                              # a late jump inside the search window
        self.assertEqual(fast_artifact_end(x, 100, 60, 32700, 1500), 119)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.x, self.onsets = synthetic()
        self.read = lambda a, b: self.x[a:b]
        self.tmp = tempfile.mkdtemp()

    def fit(self, **kw):
        return fit_stim_tails(self.read, len(self.x), self.x.shape[1], FS, self.onsets, np.zeros(3, int),
                              cfg(**kw), os.path.join(self.tmp, 'blocks.npy'), workers=1, progress=lambda s: None)

    def test_defaults_are_off(self):
        self.assertFalse(needs_stim_tails(dict(STIM_ARTIFACT_DEFAULTS, enabled=True)))
        self.assertTrue(needs_stim_tails(cfg()))

    def test_blank_end_and_fit(self):
        tails = self.fit()
        self.assertFalse(os.path.exists(os.path.join(self.tmp, 'blocks.npy')))   # scratch file removed
        rel = (tails['blank_stop'] - tails['onset'][:, None])
        # railed through onset+9, so the fast part ends at onset+12 at the earliest -> +23 samples
        self.assertTrue(np.all(rel >= 23) and np.all(rel <= 30))
        self.assertTrue(tails['ok'].all())
        self.assertGreater(np.median(tails['r2']), 0.99)
        # the correction tapers to exactly 0 at the end of each tail
        te = int(tails['tail_end'][0])
        corr = tail_correction(tails, te - 5, te + 5, np.arange(4))
        self.assertTrue(np.all(np.abs(corr[5:]) == 0))
        self.assertLess(np.abs(corr[:5]).max(), 1.0)

    def test_subtraction_removes_tail_keeps_spike(self):
        tails = self.fit()
        sos = make_bandpass_sos(FS, 300, 8000, 5)
        on = int(self.onsets[1])
        old_w = np.array([[on - 15, on + 45]])
        a, b = on - 3000, on + 3000
        new = tail_blank_filter_chunk(self.read, len(self.x), a, b, np.zeros((0, 2), np.int64), tails, np.arange(4), sos, 1500)
        old = blank_and_filter_chunk(self.read, len(self.x), a, b, old_w, sos, 1500)
        w = slice(on + 50 - a, on + 150 - a)                            # 1.7 - 5 ms after the pulse
        clean = slice(on + 300 - a, on + 2500 - a)                      # 10 - 83 ms (noise only)
        for c in (1, 2, 3):                                             # channels without the spike
            self.assertLess(np.abs(new[w, c]).max(), np.abs(old[w, c]).max())
            self.assertLess(new[w, c].std(), 3 * new[clean, c].std())
        # the spike on channel 0 survives: same trough as filtering the spike alone
        self.assertLess(new[on + 75 - a - 3:on + 75 - a + 3, 0].min(), -300)

    def test_chunking_does_not_change_output(self):
        tails = self.fit()
        sos = make_bandpass_sos(FS, 300, 8000, 5)
        n = len(self.x)
        whole = tail_blank_filter_chunk(self.read, n, 0, n, np.zeros((0, 2), np.int64), tails, np.arange(4), sos, 1500)
        parts = np.concatenate([tail_blank_filter_chunk(self.read, n, s, min(n, s + 7001), np.zeros((0, 2), np.int64),
                                                        tails, np.arange(4), sos, 1500) for s in range(0, n, 7001)])
        far = np.ones(n, bool)
        for s in range(0, n, 7001):            # sosfiltfilt chunk-edge effects, as for bandpass_filter
            far[max(0, s - 1500):s + 1500] = False
        np.testing.assert_allclose(parts[far], whole[far], atol=1e-3)
        # channel subset gives the same columns
        sub = tail_blank_filter_chunk(lambda a, b: self.x[a:b, [3, 1]], n, 40000, 52000, np.zeros((0, 2), np.int64),
                                      tails, np.array([3, 1]), sos, 1500)
        full = tail_blank_filter_chunk(self.read, n, 40000, 52000, np.zeros((0, 2), np.int64), tails, np.arange(4), sos, 1500)
        np.testing.assert_allclose(sub, full[:, [3, 1]], atol=1e-4)

    def test_blank_after_filter(self):
        tails = self.fit()
        sos = make_bandpass_sos(FS, 300, 8000, 5)
        n = len(self.x)
        none = np.zeros((0, 2), np.int64)
        plain = tail_blank_filter_chunk(self.read, n, 0, n, none, tails, np.arange(4), sos, 1500)
        post = tail_blank_filter_chunk(self.read, n, 0, n, none, tails, np.arange(4), sos, 1500, post_blank_n=45)
        on = int(self.onsets[1])
        self.assertTrue(np.all(post[on - 15:on + 45] == 0))          # zeroed to 1.5 ms after the pulse
        self.assertTrue(np.any(plain[on + 30:on + 45] != 0))         # (the plain output had data there)
        np.testing.assert_array_equal(post[on + 45:on + 3000], plain[on + 45:on + 3000])   # unchanged after
        # chunked output agrees away from chunk edges
        parts = np.concatenate([tail_blank_filter_chunk(self.read, n, s, min(n, s + 7001), none, tails, np.arange(4),
                                                        sos, 1500, post_blank_n=45) for s in range(0, n, 7001)])
        np.testing.assert_array_equal(parts[on - 15:on + 45], 0)

    def test_save_load_and_parallel(self):
        t1 = self.fit()
        t2 = fit_stim_tails(self.read, len(self.x), 4, FS, self.onsets, np.zeros(3, int), cfg(),
                            os.path.join(self.tmp, 'blocks2.npy'), workers=2, progress=lambda s: None)
        np.testing.assert_array_equal(t1['blank_stop'], t2['blank_stop'])
        np.testing.assert_allclose(t1['taus'], t2['taus'])
        path = os.path.join(self.tmp, 'tails.npz')
        save_stim_tails(path, t1)
        t3 = load_stim_tails(path)
        np.testing.assert_array_equal(t3['blank_stop'], t1['blank_stop'])
        self.assertTrue(t3['config']['subtract_tail'])

    def test_extra_artifact_blank_and_no_post_blank(self):
        # the middle pulse stands in for an artefact without a TTL
        extra = np.array([False, True, False])
        fit = lambda **kw: fit_stim_tails(self.read, len(self.x), 4, FS, self.onsets, np.arange(3), cfg(**kw),
                                          os.path.join(self.tmp, 'blocks.npy'), workers=1,
                                          progress=lambda s: None, extra=extra)
        tails = fit()
        rel = tails['blank_stop'] - tails['onset'][:, None]
        np.testing.assert_array_equal(tails['extra'], extra)
        self.assertTrue((rel[[0, 2]] < 30).all())              # pulses: where the fast part ends (~0.7 ms)
        np.testing.assert_array_equal(rel[1], 30)              # extra: at least extra_artifact_ms_after = 1 ms
        self.assertTrue(tails['ok'].all())                     # tail fitted for it too
        # without adaptive_blank the extra artefact is blanked for exactly 1 ms, pulses for ms_after
        rel = fit(adaptive_blank=False)
        np.testing.assert_array_equal((rel['blank_stop'] - rel['onset'][:, None])[:, 0], [45, 30, 45])
        # on the rail for longer than 1 ms: the blank follows the saturation
        x = self.x.copy()
        on = int(self.onsets[1])
        x[on:on + 50] = 32764
        t2 = fit_stim_tails(lambda a, b: x[a:b], len(x), 4, FS, self.onsets, np.arange(3), cfg(),
                            os.path.join(self.tmp, 'blocks.npy'), workers=1, progress=lambda s: None, extra=extra)
        # (the drop off the rail at sample 50 is the last big jump, plus adaptive_extra_samples + 1)
        np.testing.assert_array_equal((t2['blank_stop'] - t2['onset'][:, None])[1], 61)
        # blank_after_filter_ms applies to the pulses only
        sos = make_bandpass_sos(FS, 300, 8000, 5)
        n = len(self.x)
        out = tail_blank_filter_chunk(self.read, n, 0, n, np.zeros((0, 2), np.int64), tails, np.arange(4), sos,
                                      1500, post_blank_n=60)
        on0 = int(self.onsets[0])
        self.assertTrue(np.all(out[on0 - 15:on0 + 60] == 0))
        self.assertTrue(np.all(out[on - 15:on + 30] == 0))
        self.assertTrue(np.all(np.any(out[on + 30:on + 60] != 0, axis=0)))   # data kept from 1 ms on
        # the tail is removed there too: no slow lump after the blank
        self.assertLess(np.abs(out[on + 30:on + 300, 1:]).max(), 400)
        # files written before this option load with extra = False
        path = os.path.join(self.tmp, 'old.npz')
        save_stim_tails(path, {k: v for k, v in tails.items() if k != 'extra'})
        self.assertFalse(load_stim_tails(path)['extra'].any())

    def test_adaptive_only_and_fixed_tail(self):
        t = self.fit(subtract_tail=False)
        self.assertFalse(t['ok'].any())                     # no fits, just blank ends
        t = self.fit(adaptive_blank=False)                  # fixed blank end at ms_after = 1.5 ms
        np.testing.assert_array_equal(t['blank_stop'] - t['onset'][:, None], 45)
        self.assertTrue(t['ok'].all())


if __name__ == '__main__':
    unittest.main()
