"""ClipSession and SyncClip.load registration."""

import unittest
from enum import StrEnum

import numpy as np

from motion_sync.contact_registration import ContactSchema
from motion_sync.contacts.foot_support import FootSupport
from motion_sync.mocap_schema import MocapSchema
from motion_sync.schemas.skateboarding import Bodies, SKATE_SESSION
from motion_sync.session import ClipSession, apply_clip_registration
from motion_sync.synced_dataset import SyncClip


class _Bodies(StrEnum):
    A = "A"


class _Markers(StrEnum):
    M1 = "m1"


def _clip_with_markers(frames: int = 4) -> SyncClip:
    return SyncClip(
        time_s=np.linspace(0.0, 0.3, frames),
        vicon={
            "body_names": ("A",),
            "body_positions": np.zeros((frames, 1, 3)),
            "markers": {
                "names": ("m1",),
                "positions": np.zeros((frames, 1, 3)),
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
    )


_MOCAP = MocapSchema(bodies=_Bodies, body_markers={_Bodies.A: _Markers})


class TestClipSession(unittest.TestCase):
    def test_register_mocap_with_markers(self):
        clip = ClipSession(mocap=_MOCAP).register(_clip_with_markers())
        self.assertIs(clip.registered_bodies, _Bodies)
        self.assertIsNotNone(clip.registered_body_marker_enums)

    def test_register_mocap_bodies_only_without_markers(self):
        clip = SyncClip(
            time_s=np.array([0.0, 0.1]),
            vicon={"body_names": ("A",), "body_positions": np.zeros((2, 1, 3))},
            video={
                "joints": np.zeros((2, 2, 3)),
                "transl": np.zeros((2, 3)),
                "global_orient": np.zeros((2, 3)),
                "body_pose": np.zeros((2, 63)),
                "betas": np.zeros((2, 10)),
            },
            metadata={"lag_s": 0.0},
        )
        session = ClipSession(mocap=_MOCAP)
        clip = session.register(clip)
        self.assertIs(clip.registered_bodies, _Bodies)
        self.assertIsNone(clip.registered_body_marker_enums)

    def test_session_and_mocap_kwargs_conflict(self):
        clip = _clip_with_markers()
        with self.assertRaises(ValueError):
            apply_clip_registration(clip, session=SKATE_SESSION, mocap=_MOCAP)

    def test_load_with_session_registers_skate(self):
        import tempfile
        from pathlib import Path

        clip = SyncClip(
            time_s=np.linspace(0.0, 0.2, 3),
            vicon={
                "body_names": ("Left_Shoe", "Right_Shoe", "Skateboard"),
                "body_positions": np.zeros((3, 3, 3)),
            },
            video={
                "joints": np.zeros((3, 24, 3)),
                "transl": np.zeros((3, 3)),
                "global_orient": np.zeros((3, 3)),
                "body_pose": np.zeros((3, 63)),
                "betas": np.zeros((3, 10)),
            },
            metadata={"lag_s": 0.0},
        )
        with tempfile.TemporaryDirectory() as tmp:
            demo = Path(tmp) / "demo"
            clip.save(demo)
            loaded = SyncClip.load(demo, session=SKATE_SESSION)
            self.assertIs(loaded.registered_bodies, Bodies)
            self.assertIn(SKATE_SESSION.contacts.types[0].layer_id, loaded.registered_contacts)
            self.assertIsNotNone(loaded.registered_video)


if __name__ == "__main__":
    unittest.main()
