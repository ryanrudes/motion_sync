import unittest
from enum import StrEnum

import matplotlib

matplotlib.use("Agg")

import numpy as np

from motion_sync.body_marker_plot import plot_body_markers
from motion_sync.mocap_schema import MocapSchema
from motion_sync.synced_dataset import SyncClip


class _Bodies(StrEnum):
    A = "A"


class _Markers(StrEnum):
    M1 = "m1"
    M2 = "m2"


_SCHEMA = MocapSchema(
    bodies=_Bodies,
    body_markers={_Bodies.A: _Markers},
)


class TestPlotBodyMarkers(unittest.TestCase):
    def test_static_plot(self):
        frames = 4
        clip = SyncClip(
            time_s=np.arange(frames, dtype=np.float64),
            vicon={
                "body_names": ("A",),
                "body_positions": np.tile([0.0, 0.0, 0.0], (frames, 1, 1)),
                "markers": {
                    "names": ("m1", "m2"),
                    "positions": np.stack(
                        [
                            np.tile([1.0, 0.0, 0.0], (frames, 1)),
                            np.tile([0.0, 1.0, 0.0], (frames, 1)),
                        ],
                        axis=1,
                    ),
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

        fig, ax = plot_body_markers(clip, _Bodies.A, frame=0, show=False)
        self.assertEqual(len(ax.texts), 2)
        x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
        z_span = ax.get_zlim()[1] - ax.get_zlim()[0]
        self.assertAlmostEqual(x_span, y_span)
        self.assertAlmostEqual(x_span, z_span)
        self.assertLess(x_span, 2.0)
        fig.clf()

    def test_requires_mocap_registration(self):
        clip = SyncClip(
            time_s=np.array([0.0]),
            vicon={
                "body_names": ("A",),
                "body_positions": np.zeros((1, 1, 3)),
                "markers": {"names": ("m1",), "positions": np.zeros((1, 1, 3))},
            },
            video={
                "joints": np.zeros((1, 2, 3)),
                "transl": np.zeros((1, 3)),
                "global_orient": np.zeros((1, 3)),
                "body_pose": np.zeros((1, 63)),
                "betas": np.zeros((1, 10)),
            },
            metadata={"lag_s": 0.0},
        )
        with self.assertRaises(RuntimeError):
            plot_body_markers(clip, _Bodies.A, show=False)


if __name__ == "__main__":
    unittest.main()
