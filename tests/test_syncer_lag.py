import unittest

import numpy as np

from motion_sync.syncer import _corr_demeaned_cosine, _corr_motion_weighted, estimate_lag
from tests.support import shifted_foot_speed_signals


class TestCorrelationHelpers(unittest.TestCase):
    def test_demeaned_cosine_perfect_alignment(self):
        y = np.random.randn(50, 2)
        score = _corr_demeaned_cosine(y, y.copy())
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_demeaned_cosine_opposite(self):
        y = np.random.randn(50, 2)
        score = _corr_demeaned_cosine(y, -y)
        self.assertLess(score, -0.99)

    def test_motion_weighted_static_overlap_is_weak(self):
        n = 80
        static = np.zeros((n, 2))
        moving = np.column_stack([np.sin(np.linspace(0, 4 * np.pi, n)), np.zeros(n)])
        score = _corr_motion_weighted(static, moving, floor_quantile=0.12)
        self.assertEqual(score, float("-inf"))

    def test_motion_weighted_aligned_motion(self):
        t1, x1, t2, x2 = shifted_foot_speed_signals(true_lag=0.0, n=120)
        score = _corr_motion_weighted(x1, x2, floor_quantile=0.12)
        self.assertGreater(score, 0.95)


class TestEstimateLag(unittest.TestCase):
    def test_recovers_known_lag(self):
        true_lag = 0.35
        t1, x1, t2, x2 = shifted_foot_speed_signals(true_lag=true_lag, n=250)
        lag, corr, _info = estimate_lag(
            t1,
            x1,
            t2,
            x2,
            motion_weighted=False,
            max_abs_lag_seconds=2.0,
        )
        self.assertGreater(corr, 0.9)
        self.assertAlmostEqual(lag, true_lag, delta=0.05)

    def test_motion_weighted_lag(self):
        true_lag = 0.35
        t1, x1, t2, x2 = shifted_foot_speed_signals(true_lag=true_lag, n=250)
        lag, corr, _info = estimate_lag(
            t1,
            x1,
            t2,
            x2,
            motion_weighted=True,
            motion_weight_floor_quantile=0.12,
            max_abs_lag_seconds=2.0,
        )
        self.assertGreater(corr, 0.5)
        self.assertAlmostEqual(lag, true_lag, delta=0.08)

    def test_max_abs_lag_clamp_rejects_wide_search(self):
        n = 120
        dt = 0.01
        t1 = np.arange(n) * dt + 10.0
        t2 = np.arange(n) * dt
        x1 = np.column_stack([np.sin(t1), np.cos(t1)])
        x2 = np.column_stack([np.sin(t2 + 7.0), np.cos(t2 + 7.0)])
        with self.assertRaises(ValueError):
            estimate_lag(
                t1,
                x1,
                t2,
                x2,
                max_abs_lag_seconds=1.0,
                min_overlap=0.05,
            )


if __name__ == "__main__":
    unittest.main()
