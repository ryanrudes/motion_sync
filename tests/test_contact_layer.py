"""Contact layers and registered contact types on SyncClip."""

import tempfile
import unittest
from enum import StrEnum
from pathlib import Path

import numpy as np

from motion_sync.contact_layer import ContactLayer, decode_contact_layers, encode_contact_layers
from motion_sync.contacts.foot_support import (
    FootSupport,
    FootSupportState,
    layer_from_foot_classification,
)
from motion_sync.contacts.shoe_board_grip import ShoeBoardGrip
from motion_sync.schemas.skateboarding import SKATE_SHOE_BOARD_GRIP
from motion_sync.schemas.skateboarding import SKATE_CONTACTS, SKATE_FOOT_SUPPORT, SKATE_MOCAP
from motion_sync.synced_dataset import SyncClip


class _Bodies(StrEnum):
    LEFT_SHOE = "Left_Shoe"
    RIGHT_SHOE = "Right_Shoe"
    SKATEBOARD = "Skateboard"


_SKATE_FOOT = FootSupport(
    left=_Bodies.LEFT_SHOE,
    right=_Bodies.RIGHT_SHOE,
    board=_Bodies.SKATEBOARD,
)


def _minimal_clip(frames: int = 5) -> SyncClip:
    return SyncClip(
        time_s=np.linspace(0.0, 0.4, frames),
        vicon={
            "body_names": ("Left_Shoe", "Right_Shoe", "Skateboard"),
            "body_positions": np.zeros((frames, 3, 3)),
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


class _FakeClassification:
    def __init__(self, frames: int) -> None:
        self.t = np.linspace(0.0, 1.0, frames)
        self.states = {
            "Left_Shoe": np.array([0, 1, 1, 2, 0][:frames], dtype=np.int8),
            "Right_Shoe": np.array([0, 0, 2, 2, 1][:frames], dtype=np.int8),
        }
        self.floor_model = "height"
        self.floor_height = 0.0
        self.floor_normal = None
        self.floor_origin = None
        self.board_contact_offsets = {"Left_Shoe": 0.05, "Right_Shoe": 0.06}


class TestContactLayerStorage(unittest.TestCase):
    def test_labels_from_state_enum(self):
        self.assertEqual(
            FootSupport.labels_on_disk(),
            ("air", "ground", "skateboard"),
        )

    def test_encode_decode_round_trip(self):
        layer = layer_from_foot_classification(_FakeClassification(5), FootSupport.layer_id)
        blob = encode_contact_layers({layer.layer_id: layer})
        loaded = decode_contact_layers(list(blob.keys()), lambda key: blob[key])
        np.testing.assert_array_equal(
            loaded[FootSupport.layer_id].states,
            layer.states,
        )

    def test_sync_clip_save_load_contact(self):
        clip = _minimal_clip(6)
        layer = ContactLayer(
            layer_id="grip_test",
            kind="binary",
            subjects=("Left_Shoe",),
            labels=(),
            mask=np.array([[True], [False], [True], [False], [True], [False]]),
            metadata={},
        )
        clip = clip.attach_contact(layer)

        with tempfile.TemporaryDirectory() as tmp:
            demo = Path(tmp) / "demo"
            clip.save(demo)
            loaded = SyncClip.load(demo)
            np.testing.assert_array_equal(
                loaded.contact_layer("grip_test").mask,
                layer.mask,
            )


class TestRegisteredFootSupport(unittest.TestCase):
    def test_contact_requires_registration(self):
        clip = _minimal_clip(5).register_bodies(_Bodies)
        clip = clip.attach_contact(
            layer_from_foot_classification(_FakeClassification(5), FootSupport.layer_id)
        )
        with self.assertRaises(RuntimeError):
            clip.contact(SKATE_FOOT_SUPPORT)

    def test_register_and_read(self):
        clip = _minimal_clip(5).register_bodies(_Bodies).register_contacts(_SKATE_FOOT)
        clip = clip.attach_contact(
            layer_from_foot_classification(_FakeClassification(5), FootSupport.layer_id)
        )
        data = clip.contact(_SKATE_FOOT)
        self.assertEqual(data.state(_Bodies.LEFT_SHOE, 1), FootSupportState.GROUND)
        self.assertEqual(data.stance_matrix().shape, (5, 2))

    def test_track_states_and_intervals(self):
        clip = _minimal_clip(5).register_bodies(_Bodies).register_contacts(_SKATE_FOOT)
        clip = clip.attach_contact(
            layer_from_foot_classification(_FakeClassification(5), FootSupport.layer_id)
        )
        data = clip.contact(_SKATE_FOOT)
        left = data.track(_Bodies.LEFT_SHOE)
        np.testing.assert_array_equal(left.states, [0, 1, 1, 2, 0])
        np.testing.assert_array_equal(left.stance, [False, True, True, True, False])
        ground = left.intervals(FootSupportState.GROUND)
        self.assertEqual(len(ground), 1)
        self.assertAlmostEqual(ground[0][0], clip.time_s[1])
        self.assertAlmostEqual(ground[0][1], clip.time_s[2])

        tracks = data.tracks()
        self.assertEqual(set(tracks.keys()), {_Bodies.LEFT_SHOE, _Bodies.RIGHT_SHOE})

    def test_has_contact(self):
        clip = _minimal_clip(3)
        self.assertFalse(clip.has_contact(SKATE_FOOT_SUPPORT))
        clip = clip.attach_contact(
            layer_from_foot_classification(_FakeClassification(3), FootSupport.layer_id)
        )
        self.assertTrue(clip.has_contact(SKATE_FOOT_SUPPORT))

    def test_detect_integration(self):
        try:
            import contact_detection  # noqa: F401
        except ImportError:
            self.skipTest("contact-detection not installed")

        frames = 40
        t = np.linspace(0.0, 2.0, frames)
        pos = np.zeros((frames, 3, 3))
        pos[:, 0, 2] = 0.05
        pos[:, 1, 2] = 0.05
        pos[:, 2, 2] = 0.08
        clip = SyncClip(
            time_s=t,
            vicon={
                "body_names": ("Left_Shoe", "Right_Shoe", "Skateboard"),
                "body_positions": pos,
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
        clip = clip.register_bodies(SKATE_MOCAP.bodies).register_contacts(SKATE_CONTACTS)
        clip = clip.detect(SKATE_FOOT_SUPPORT)
        self.assertEqual(clip.contact(SKATE_FOOT_SUPPORT).stance_matrix().shape, (frames, 2))

    def test_shoe_board_grip_from_foot_support(self):
        clip = _minimal_clip(5).register_bodies(_Bodies).register_contacts(SKATE_CONTACTS)
        clip = clip.attach_contact(
            layer_from_foot_classification(_FakeClassification(5), FootSupport.layer_id)
        )
        clip = clip.detect(SKATE_SHOE_BOARD_GRIP)
        grip = clip.contact(SKATE_SHOE_BOARD_GRIP)
        matrix = grip.mask_matrix()
        self.assertEqual(matrix.shape, (5, 2))
        foot = clip.contact(SKATE_FOOT_SUPPORT)
        left_board = foot.track(_Bodies.LEFT_SHOE).states == FootSupportState.SKATEBOARD
        np.testing.assert_array_equal(matrix[:, 0], left_board)


if __name__ == "__main__":
    unittest.main()
