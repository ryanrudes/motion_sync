import unittest

import numpy as np

from retargeting.rigid_body_model_estimator import (
    _robust_distance_estimate,
    estimate_pairwise_distances,
    fit_plane,
)
from tests.support import square_marker_tracks


class TestRobustDistanceEstimate(unittest.TestCase):
    def test_median_on_clean_data(self):
        values = np.array([1.0, 1.01, 0.99, 1.0])
        med, sigma = _robust_distance_estimate(values)
        self.assertAlmostEqual(med, 1.0, places=2)
        self.assertLess(sigma, 0.05)

    def test_empty_returns_nan(self):
        med, sigma = _robust_distance_estimate(np.array([]))
        self.assertTrue(np.isnan(med))
        self.assertEqual(sigma, float("inf"))


class TestEstimatePairwiseDistances(unittest.TestCase):
    def test_square_geometry(self):
        tracks = square_marker_tracks()
        names, d_mat, w_mat = estimate_pairwise_distances(tracks, min_common_frames=5)
        self.assertEqual(len(names), 4)
        i_a = names.index("a")
        i_b = names.index("b")
        self.assertAlmostEqual(d_mat[i_a, i_b], 1.0, places=2)
        self.assertGreater(w_mat[i_a, i_b], 0.0)


class TestFitPlane(unittest.TestCase):
    def test_horizontal_plane(self):
        names = ["p0", "p1", "p2"]
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        plane = fit_plane(positions, names, names)
        self.assertLess(plane.stress, 1e-10)
        self.assertAlmostEqual(plane.intercept, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
