import unittest

import numpy as np

from motion_sync.app.convert import merge_tf_and_marker_data, order_preserving_unique
from tests.support import synthetic_convert_inputs


class TestOrderPreservingUnique(unittest.TestCase):
    def test_preserves_first_occurrence_order(self):
        arr = np.array(["b", "a", "b", "c", "a"])
        out = order_preserving_unique(arr)
        np.testing.assert_array_equal(out, ["b", "a", "c"])


class TestMergeTfAndMarkerData(unittest.TestCase):
    def test_vectorized_merge_shapes(self):
        tf_data, marker_data = synthetic_convert_inputs()
        lookup = merge_tf_and_marker_data(tf_data, marker_data)

        self.assertEqual(len(lookup["stamp"]), 4)
        self.assertEqual(lookup["body_pos"].shape, (4, 3, 3))
        self.assertEqual(lookup["body_quat"].shape, (4, 3, 4))
        self.assertEqual(lookup["marker_pos"].shape, (4, 28, 3))
        self.assertEqual(len(lookup["body_names"]), 3)
        self.assertEqual(len(lookup["marker_names"]), 28)

    def test_tf_values_land_on_expected_frames(self):
        tf_data, marker_data = synthetic_convert_inputs()
        lookup = merge_tf_and_marker_data(tf_data, marker_data)
        left_idx = list(lookup["body_names"]).index("Left_Shoe")
        frame0 = 0
        np.testing.assert_allclose(lookup["body_pos"][frame0, left_idx], [0.0, 0.0, 0.0])
        self.assertFalse(lookup["body_occluded"][frame0, left_idx])


if __name__ == "__main__":
    unittest.main()
