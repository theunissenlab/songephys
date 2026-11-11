import unittest

import numpy as np

from songephys.postprocess.merge_check import cluster_band_metrics, fit_two_gaussians


class TestMergeCheck(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.T = 3600.0

    def unit(self, n, amp, cv=0.1, drift=0.0):
        t = np.sort(self.rng.uniform(0, self.T, n))
        a = amp * (1 + drift * t / self.T) * np.exp(self.rng.normal(0, cv, n))
        return t, a

    def test_em_recovers_two_bands(self):
        x = np.r_[self.rng.normal(0, 0.1, 3000), self.rng.normal(1, 0.1, 1000)]
        mu, sd, w = fit_two_gaussians(x)
        np.testing.assert_allclose(mu, [0, 1], atol=0.03)
        np.testing.assert_allclose(w, [0.75, 0.25], atol=0.03)

    def test_merged_pair_is_flagged_even_with_drift(self):
        t1, a1 = self.unit(30000, 14, drift=0.3)
        t2, a2 = self.unit(6000, 31, drift=0.3)
        t, a = np.r_[t1, t2], np.r_[a1, a2]
        o = np.argsort(t)
        m = cluster_band_metrics(t[o], a[o], self.T)
        self.assertGreater(m['persistence'], 0.9)
        self.assertLess(m['valley'], 0.3)
        self.assertAlmostEqual(m['amp_ratio'], 31 / 14, delta=0.2)

    def test_single_unit_with_drift_is_not_flagged(self):
        t, a = self.unit(30000, 20, cv=0.15, drift=1.0)
        m = cluster_band_metrics(t, a, self.T)
        self.assertLess(m['persistence'], 0.3)

    def test_threshold_truncated_unit_is_not_flagged(self):
        t, a = self.unit(60000, 10, cv=0.35)
        keep = a > 9                                    # cut by the detection threshold
        m = cluster_band_metrics(t[keep], a[keep], self.T)
        self.assertFalse(m['persistence'] >= 0.5 and m['valley'] <= 0.5)


if __name__ == '__main__':
    unittest.main()
