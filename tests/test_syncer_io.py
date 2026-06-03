import tempfile
import unittest
from pathlib import Path

import numpy as np

from motion_sync.synced_dataset import SyncClip
from motion_sync.syncer import save_aligned_npz
from motion_sync.sync_trim_video import synced_time_range
from tests.support import write_synced_clip_timeline


class TestSaveAlignedClip(unittest.TestCase):
    def test_round_trip_through_sync_clip(self):
        aligned = {
            "t": np.array([0.0, 1.0]),
            "vicon/body_pos": np.zeros((2, 1, 3)),
            "video/joints": np.ones((2, 4, 3)),
            "video/transl": np.zeros((2, 3)),
            "video/global_orient": np.zeros((2, 3)),
            "video/body_pose": np.zeros((2, 63)),
            "video/betas": np.zeros((2, 10)),
            "__valid_masks__": {"video/joints": np.array([True, True])},
        }
        meta = {"lag": 0.5, "corr": 0.9, "body_names": ["Left_Shoe"]}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.npz"
            save_aligned_npz(path, aligned, meta)
            clip = SyncClip.load(path)
            self.assertAlmostEqual(clip.metadata.lag_s, 0.5)
            self.assertAlmostEqual(clip.metadata.correlation or 0.0, 0.9)
            self.assertEqual(clip.vicon.body_names, ("Left_Shoe",))
            self.assertEqual(clip.video.joints.shape, (2, 4, 3))


class TestSyncedTimeRange(unittest.TestCase):
    def test_reads_min_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo"
            write_synced_clip_timeline(path, np.array([1.0, 2.5, 4.0]))
            lo, hi = synced_time_range(path)
            self.assertEqual(lo, 1.0)
            self.assertEqual(hi, 4.0)

    def test_empty_timeline_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo"
            write_synced_clip_timeline(path, np.array([]))
            with self.assertRaises(ValueError):
                synced_time_range(path)


if __name__ == "__main__":
    unittest.main()
