import unittest
from enum import StrEnum

import numpy as np

from motion_sync.mocap_schema import MocapSchema
from motion_sync.synced_dataset import SyncClip


class _Bodies(StrEnum):
    A = "A"


class _AMarkers(StrEnum):
    M1 = "m1"
    M2 = "m2"


_SCHEMA = MocapSchema(
    bodies=_Bodies,
    body_markers={_Bodies.A: _AMarkers},
)


def _clip_with_marker_gaps() -> SyncClip[_Bodies]:
    frames = 6
    m1 = np.full((frames, 3), np.nan)
    m2 = np.full((frames, 3), np.nan)
    m1[3:6] = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m2[2:5] = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    return SyncClip(
        time_s=np.arange(frames, dtype=np.float64),
        vicon={
            "body_names": ("A",),
            "body_positions": np.zeros((frames, 1, 3)),
            "markers": {
                "names": ("m1", "m2"),
                "positions": np.stack([m1, m2], axis=1),
            },
        },
        video={
            "joints": np.zeros((frames, 2, 3)),
            "transl": np.zeros((frames, 3)),
            "global_orient": np.zeros((frames, 3)),
            "body_pose": np.zeros((frames, 63)),
            "betas": np.zeros((frames, 10)),
        },
        metadata={"lag_s": 0.0},
    ).register_mocap(_SCHEMA)


class TestMarkerVisibility(unittest.TestCase):
    def test_mask_and_find_middle(self):
        clip = _clip_with_marker_gaps()
        mask = clip.all_markers_visible_mask(_Bodies.A)
        np.testing.assert_array_equal(mask, [False, False, False, True, True, False])
        self.assertEqual(clip.find_frame_all_markers_visible(_Bodies.A), 4)

    def test_raises_when_never_all_visible(self):
        clip = _clip_with_marker_gaps()
        m1 = clip.markers_for_body(_Bodies.A)[_AMarkers.M1]
        m1[:] = np.nan
        with self.assertRaises(ValueError):
            clip.find_frame_all_markers_visible(_Bodies.A)


if __name__ == "__main__":
    unittest.main()
