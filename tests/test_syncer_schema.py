import unittest

import numpy as np

from motion_sync.config import load_config
from motion_sync.syncer import (
    TimelineStitcher,
    parse_extra_feature,
    register_schema_features,
    resolve_vicon_schema_for_sync,
)
from tests.support import MINIMAL_CONFIG


class TestSchemaHelpers(unittest.TestCase):
    def setUp(self):
        self.config = load_config(MINIMAL_CONFIG)

    def test_parse_extra_feature_valid(self):
        arr = np.ones((3, 2))
        x, mode, quat_order, allow_extrap, required = parse_extra_feature(
            {"array": arr, "mode": "linear", "required": False}
        )
        np.testing.assert_array_equal(x, arr)
        self.assertEqual(mode, "linear")
        self.assertFalse(required)
        self.assertFalse(allow_extrap)
        self.assertIsNone(quat_order)

    def test_parse_extra_feature_rejects_implicit(self):
        with self.assertRaises(ValueError):
            parse_extra_feature(np.ones(3))

    def test_resolve_vicon_schema_adds_shoe_indices(self):
        vicon = {"body_names": ["Left_Shoe", "Right_Shoe", "Skateboard"]}
        schema = resolve_vicon_schema_for_sync(vicon, self.config, user_schema=None)
        idx = schema["vicon/body_pos"]["validity_body_indices"]
        self.assertEqual(idx, (0, 1))

    def test_register_schema_features(self):
        t = np.linspace(0.0, 1.0, 4)
        source = {
            "body_pos": np.ones((4, 2, 3)),
            "body_quat": np.tile([1.0, 0.0, 0.0, 0.0], (4, 2, 1)),
        }
        schema = {
            "vicon/body_pos": {"array_key": "body_pos", "mode": "linear"},
            "vicon/body_quat": {
                "array_key": "body_quat",
                "mode": "quat",
                "quat_order": "wxyz",
                "required": False,
            },
        }
        stitcher = TimelineStitcher()
        register_schema_features(stitcher, source, t, schema)
        self.assertIn("vicon/body_pos", stitcher.features)
        self.assertIn("vicon/body_quat", stitcher.features)


if __name__ == "__main__":
    unittest.main()
