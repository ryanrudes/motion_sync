import tempfile
import unittest
from enum import StrEnum
from pathlib import Path

import numpy as np

from motion_sync.mocap_schema import MocapSchema
from motion_sync.synced_dataset import RigidBodyPose, SyncClip, validate_body_enum


class _SingleBody(StrEnum):
    A = "A"


class _SkateBodies(StrEnum):
    LEFT_SHOE = "Left_Shoe"
    RIGHT_SHOE = "Right_Shoe"
    SKATEBOARD = "Skateboard"


class TestValidateBodyEnum(unittest.TestCase):
    def test_rejects_missing_and_extra(self):
        validate_body_enum(_SkateBodies, ("Left_Shoe", "Right_Shoe", "Skateboard"))
        with self.assertRaises(ValueError):
            validate_body_enum(_SingleBody, ("Left_Shoe", "Right_Shoe", "Skateboard"))


class TestSyncClipRoundTrip(unittest.TestCase):
    def test_save_load_preserves_core_fields(self):
        t = np.linspace(0.0, 1.0, 5)
        bodies = ("Left_Shoe", "Right_Shoe", "Skateboard")
        clip = SyncClip(
            name="demo",
            time_s=t,
            vicon={
                "body_names": bodies,
                "body_positions": np.random.randn(5, 3, 3),
                "body_orientations": np.tile([0, 0, 0, 1], (5, 3, 1)).reshape(5, 3, 4),
            },
            video={
                "joints": np.random.randn(5, 10, 3),
                "transl": np.random.randn(5, 3),
                "global_orient": np.random.randn(5, 3),
                "body_pose": np.random.randn(5, 63),
                "betas": np.random.randn(5, 10),
            },
            metadata={"lag_s": 0.5, "correlation": 0.9},
            valid=np.array([True, True, True, False, True]),
        )

        with tempfile.TemporaryDirectory() as tmp:
            demo_dir = Path(tmp) / "demo"
            clip.save(demo_dir)
            loaded = SyncClip.load(demo_dir, name="from_disk")
            self.assertEqual(loaded.name, "from_disk")
            self.assertEqual(loaded.frame_count, 5)
            self.assertEqual(loaded.vicon.body_names, bodies)
            np.testing.assert_allclose(loaded.time_s, t)
            self.assertAlmostEqual(loaded.metadata.lag_s, 0.5)
            self.assertAlmostEqual(loaded.metadata.correlation or 0.0, 0.9)

    def test_body_accessor(self):
        pos = np.arange(12, dtype=np.float64).reshape(4, 1, 3)
        clip: SyncClip[_SingleBody] = SyncClip(
            time_s=np.arange(4, dtype=np.float64),
            vicon={
                "body_names": ("A",),
                "body_positions": pos,
            },
            video=_minimal_video(4),
            metadata={"lag_s": 0.0},
        ).register_bodies(_SingleBody)
        track = clip.body(_SingleBody.A)
        np.testing.assert_allclose(track.positions[:, 0], [0, 3, 6, 9])

    def test_register_bodies_and_pose_at(self):
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1, 1)).reshape(2, 1, 4)

        class OneShoe(StrEnum):
            LEFT_SHOE = "Left_Shoe"

        clip: SyncClip[OneShoe] = SyncClip(
            time_s=np.array([0.0, 1.0]),
            vicon={
                "body_names": ("Left_Shoe",),
                "body_positions": pos.reshape(2, 1, 3),
                "body_orientations": quat,
            },
            video=_minimal_video(2),
            metadata={"lag_s": 0.0},
        ).register_bodies(OneShoe)
        track = clip.body(OneShoe.LEFT_SHOE)
        pose = track.pose_at(1)
        self.assertIsInstance(pose, RigidBodyPose)
        np.testing.assert_allclose(pose.position, [4.0, 5.0, 6.0])

    def test_body_requires_registration(self):
        clip = SyncClip(
            time_s=np.array([0.0]),
            vicon={"body_names": ("A",), "body_positions": np.zeros((1, 1, 3))},
            video=_minimal_video(1),
            metadata={"lag_s": 0.0},
        )
        foreign = StrEnum("Foreign", {"A": "A"})
        with self.assertRaises(RuntimeError):
            clip.body(foreign.A)

    def test_wrong_enum_after_register(self):
        clip = SyncClip(
            time_s=np.array([0.0]),
            vicon={"body_names": ("A",), "body_positions": np.zeros((1, 1, 3))},
            video=_minimal_video(1),
            metadata={"lag_s": 0.0},
        ).register_bodies(_SingleBody)
        foreign = StrEnum("Foreign", {"A": "A"})
        with self.assertRaises(TypeError):
            clip.body(foreign.A)


class _TinyBodies(StrEnum):
    A = "A"


class _TinyMarkers(StrEnum):
    M1 = "m1"


class TestSyncClipMocapRegistration(unittest.TestCase):
    def test_register_mocap_and_markers_for_body(self):
        pos = np.zeros((3, 1, 3))
        marker_pos = np.arange(9, dtype=np.float64).reshape(3, 1, 3)
        schema = MocapSchema(
            bodies=_TinyBodies,
            body_markers={_TinyBodies.A: _TinyMarkers},
        )
        clip = SyncClip(
            time_s=np.arange(3, dtype=np.float64),
            vicon={
                "body_names": ("A",),
                "body_positions": pos,
                "markers": {"names": ("m1",), "positions": marker_pos},
            },
            video=_minimal_video(3),
            metadata={"lag_s": 0.0},
        ).register_mocap(schema)
        track = clip.marker(_TinyMarkers.M1)
        self.assertEqual(track.shape, (3, 3))
        by_body = clip.markers_for_body(_TinyBodies.A)
        self.assertIn(_TinyMarkers.M1, by_body)
        self.assertIs(clip._marker_enum_for_body(_TinyBodies.A), _TinyMarkers)


class TestSyncClipFixture(unittest.TestCase):
    def test_load_pushoff5_if_present(self):
        path = Path("output/synced/pushoff5_twoshoes/synced.npz")
        if not path.is_file():
            self.skipTest("no local synced fixture")
        clip: SyncClip[_SkateBodies] = SyncClip.load(path).register_bodies(_SkateBodies)
        self.assertGreater(clip.frame_count, 100)
        left = clip.body(_SkateBodies.LEFT_SHOE)
        self.assertEqual(left.positions.shape, (clip.frame_count, 3))
        fps = clip.mean_fps()
        self.assertTrue(np.isfinite(fps) and fps > 0)


def _minimal_video(frames: int) -> dict:
    return {
        "joints": np.zeros((frames, 4, 3)),
        "transl": np.zeros((frames, 3)),
        "global_orient": np.zeros((frames, 3)),
        "body_pose": np.zeros((frames, 63)),
        "betas": np.zeros((frames, 10)),
    }


if __name__ == "__main__":
    unittest.main()
