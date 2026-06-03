import unittest

import numpy as np

from motion_sync.syncer import (
    finite_time_mask,
    sanitize_quaternions,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class TestQuaternionHelpers(unittest.TestCase):
    def test_wxyz_xyzw_round_trip(self):
        q_wxyz = np.array([1.0, 0.1, 0.2, 0.3])
        q_xyzw = wxyz_to_xyzw(q_wxyz)
        np.testing.assert_allclose(q_xyzw, [0.1, 0.2, 0.3, 1.0])
        back = xyzw_to_wxyz(q_xyzw)
        np.testing.assert_allclose(back, q_wxyz)

    def test_batch_shape_preserved(self):
        q = np.random.randn(5, 3, 4)
        q[..., 0] = 1.0
        out = wxyz_to_xyzw(q)
        self.assertEqual(out.shape, (5, 3, 4))
        back = xyzw_to_wxyz(out)
        np.testing.assert_allclose(back, q)

    def test_sanitize_zero_norm_to_identity(self):
        q = np.zeros((2, 4))
        out = sanitize_quaternions(q, replacement="identity")
        np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(out[1], [0.0, 0.0, 0.0, 1.0])

    def test_sanitize_invalid_to_nan(self):
        q = np.array([[np.nan, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        out = sanitize_quaternions(q, replacement="nan")
        self.assertTrue(np.all(np.isnan(out[0])))
        np.testing.assert_allclose(out[1], [0.0, 0.0, 0.0, 1.0])

    def test_finite_time_mask(self):
        x1d = np.array([1.0, np.nan, 3.0])
        np.testing.assert_array_equal(finite_time_mask(x1d), [True, False, True])

        x2d = np.array([[1.0, 2.0], [np.nan, 1.0], [1.0, 1.0]])
        np.testing.assert_array_equal(finite_time_mask(x2d), [True, False, True])


if __name__ == "__main__":
    unittest.main()
