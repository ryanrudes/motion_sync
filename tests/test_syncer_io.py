import tempfile
import unittest
from pathlib import Path

import numpy as np

from retargeting.syncer import save_aligned_npz
from retargeting.sync_trim_video import synced_time_range_from_unified
from tests.support import write_unified_npz


class TestSaveAlignedNpz(unittest.TestCase):
    def test_key_mangling_and_metadata(self):
        aligned = {
            "t": np.array([0.0, 1.0]),
            "video/joints": np.ones((2, 3)),
            "__valid_masks__": {"video/joints": np.array([True, True])},
        }
        meta = {"lag": 0.5, "corr": 0.9, "body_names": ["Left_Shoe"]}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.npz"
            save_aligned_npz(path, aligned, meta)
            data = np.load(path, allow_pickle=True)
            self.assertIn("video__joints", data.files)
            self.assertNotIn("video/joints", data.files)
            self.assertNotIn("__valid_masks__", data.files)
            self.assertAlmostEqual(float(data["lag"]), 0.5)
            self.assertAlmostEqual(float(data["corr"]), 0.9)
            names = list(data["vicon__body_names"])
            self.assertEqual(names, ["Left_Shoe"])


class TestSyncedTimeRangeFromUnified(unittest.TestCase):
    def test_reads_min_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unified.npz"
            write_unified_npz(path, np.array([1.0, 2.5, 4.0]))
            lo, hi = synced_time_range_from_unified(path)
            self.assertEqual(lo, 1.0)
            self.assertEqual(hi, 4.0)

    def test_empty_timeline_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unified.npz"
            np.savez_compressed(path, t=np.array([]))
            with self.assertRaises(ValueError):
                synced_time_range_from_unified(path)


if __name__ == "__main__":
    unittest.main()
