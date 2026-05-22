import tempfile
import unittest
from pathlib import Path

import numpy as np

from retargeting.sync_trim_video import synced_time_range_from_unified, time_range_to_frame_range
from tests.support import write_unified_npz


class TestTimeRangeToFrameRange(unittest.TestCase):
    def test_maps_closed_interval_to_half_open_frames(self):
        start, end = time_range_to_frame_range(0.0, 1.0, fps=10.0, n_frames=20)
        self.assertEqual(start, 0)
        self.assertEqual(end, 11)

    def test_collapsed_window_raises(self):
        with self.assertRaises(ValueError):
            time_range_to_frame_range(5.0, 5.0, fps=30.0, n_frames=100)

    def test_invalid_fps_raises(self):
        with self.assertRaises(ValueError):
            time_range_to_frame_range(0.0, 1.0, fps=0.0, n_frames=10)


class TestSyncedTimeRangeEdgeCases(unittest.TestCase):
    def test_single_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.npz"
            write_unified_npz(path, np.array([2.5]))
            lo, hi = synced_time_range_from_unified(path)
            self.assertEqual(lo, 2.5)
            self.assertEqual(hi, 2.5)

    def test_non_monotonic_uses_min_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u.npz"
            write_unified_npz(path, np.array([3.0, 1.0, 2.0]))
            lo, hi = synced_time_range_from_unified(path)
            self.assertEqual(lo, 1.0)
            self.assertEqual(hi, 3.0)


if __name__ == "__main__":
    unittest.main()
