import unittest
from unittest.mock import patch

import numpy as np

from retargeting.syncer import TimelineStitcher, support_overlap_video_clock


class TestTimelineStitcher(unittest.TestCase):
    def _build_stitcher(self):
        t_video = np.linspace(0.0, 1.0, 11)
        t_vicon = np.linspace(0.2, 0.8, 7)
        stitcher = TimelineStitcher()
        stitcher.add_source_time("video", t_video)
        stitcher.add_source_time("vicon", t_vicon)
        stitcher.add("video/signal", t_video, np.sin(t_video)[:, None], mode="linear")
        stitcher.add(
            "vicon/signal",
            t_vicon,
            np.cos(t_vicon)[:, None],
            mode="linear",
        )
        return stitcher, t_video, t_vicon

    @patch("builtins.print")
    def test_crop_none_keeps_full_video_timeline(self, _print):
        stitcher, t_video, _t_vicon = self._build_stitcher()
        out = stitcher.build(timeline="video", crop="none")
        np.testing.assert_allclose(out["t"], t_video)
        self.assertEqual(len(out["video/signal"]), len(t_video))

    @patch("builtins.print")
    def test_crop_support_trims_to_overlap(self, _print):
        stitcher, t_video, t_vicon = self._build_stitcher()
        out = stitcher.build(timeline="video", crop="support")
        expected_lo = max(t_video[0], t_vicon[0])
        expected_hi = min(t_video[-1], t_vicon[-1])
        self.assertAlmostEqual(out["t"][0], expected_lo, places=5)
        self.assertAlmostEqual(out["t"][-1], expected_hi, places=5)
        self.assertLess(len(out["t"]), len(t_video))

    @patch("builtins.print")
    def test_crop_valid_respects_finiteness(self, _print):
        t_video = np.linspace(0.0, 1.0, 5)
        t_vicon = np.linspace(0.0, 1.0, 5)
        x_video = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0]])
        x_vicon = np.ones((5, 1))

        stitcher = TimelineStitcher()
        stitcher.add_source_time("video", t_video)
        stitcher.add_source_time("vicon", t_vicon)
        stitcher.add("video/signal", t_video, x_video, mode="linear")
        stitcher.add("vicon/signal", t_vicon, x_vicon, mode="linear")

        out = stitcher.build(timeline="video", crop="valid")
        self.assertLess(len(out["t"]), len(t_video))
        self.assertTrue(np.all(np.isfinite(out["video/signal"])))


class TestSupportOverlapVideoClock(unittest.TestCase):
    def test_matches_stitcher_support_window(self):
        t_video = np.linspace(0.0, 2.0, 21)
        t_vicon = np.linspace(0.0, 2.0, 41)
        lag = 0.25
        lo, hi = support_overlap_video_clock(t_video, t_vicon, lag)
        t_vicon_shifted = t_vicon - lag
        expected = (max(t_video[0], t_vicon_shifted[0]), min(t_video[-1], t_vicon_shifted[-1]))
        self.assertAlmostEqual(lo, expected[0])
        self.assertAlmostEqual(hi, expected[1])


if __name__ == "__main__":
    unittest.main()
