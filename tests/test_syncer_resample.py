import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from motion_sync.syncer import resample_feature


class TestResampleFeature(unittest.TestCase):
    def test_linear_interp(self):
        t_src = np.array([0.0, 1.0, 2.0])
        x_src = np.array([[0.0], [2.0], [4.0]])
        t_tgt = np.array([0.5, 1.5])
        y = resample_feature(t_src, x_src, t_tgt, mode="linear")
        np.testing.assert_allclose(y[:, 0], [1.0, 3.0])

    def test_nearest_no_extrapolation(self):
        t_src = np.array([0.0, 1.0, 2.0])
        x_src = np.array([10.0, 20.0, 30.0])
        t_tgt = np.array([-0.5, 0.4, 2.5])
        y = resample_feature(t_src, x_src, t_tgt, mode="nearest")
        self.assertTrue(np.isnan(y[0]))
        self.assertEqual(y[1], 10.0)
        self.assertTrue(np.isnan(y[2]))

    def test_quat_slerp_matches_scipy(self):
        t_src = np.linspace(0.0, 1.0, 5)
        angles = np.column_stack(
            [np.linspace(0.0, 1.0, 5), np.zeros(5), np.zeros(5)]
        )
        rots = R.from_euler("xyz", angles)
        q_src = rots.as_quat()
        t_tgt = np.array([0.25, 0.75])
        y = resample_feature(t_src, q_src, t_tgt, mode="quat")
        ref = Slerp(t_src, rots)(t_tgt).as_quat()
        np.testing.assert_allclose(y, ref, atol=1e-10)

    def test_rotvec_interp(self):
        t_src = np.linspace(0.0, 1.0, 4)
        rv_src = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
        t_tgt = np.array([0.5])
        y = resample_feature(t_src, rv_src, t_tgt, mode="rotvec")
        self.assertEqual(y.shape, (1, 3))
        self.assertTrue(np.all(np.isfinite(y)))


if __name__ == "__main__":
    unittest.main()
