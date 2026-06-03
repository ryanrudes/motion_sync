"""Contact detection metadata, freshness, and idempotent detect."""

import unittest
import warnings
from enum import StrEnum

import numpy as np

from motion_sync.contact_metadata import (
    META_SOURCE_FRAME_COUNT,
    META_TIME_FINGERPRINT,
    contact_layer_is_fresh,
    fingerprint_timeline,
    stamp_detection_metadata,
    warn_if_stale_contact_layer,
)
from motion_sync.contacts.foot_support import FootSupport, layer_from_foot_classification
from motion_sync.schemas.skateboarding import SKATE_CONTACTS, SKATE_FOOT_SUPPORT
from motion_sync.synced_dataset import SyncClip


class _Bodies(StrEnum):
    LEFT_SHOE = "Left_Shoe"
    RIGHT_SHOE = "Right_Shoe"
    SKATEBOARD = "Skateboard"


class _FakeClassification:
    def __init__(self, frames: int) -> None:
        self.states = {
            "Left_Shoe": np.zeros(frames, dtype=np.int8),
            "Right_Shoe": np.zeros(frames, dtype=np.int8),
        }
        self.floor_model = "height"
        self.floor_height = 0.0
        self.floor_normal = None
        self.floor_origin = None
        self.board_contact_offsets = {}


def _skate_clip(frames: int) -> SyncClip:
    return SyncClip(
        time_s=np.linspace(0.0, 1.0, frames),
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
    ).register_bodies(_Bodies).register_contacts(SKATE_CONTACTS)


class TestContactMetadata(unittest.TestCase):
    def test_stamp_and_fresh_match(self):
        clip = _skate_clip(10)
        layer = stamp_detection_metadata(
            layer_from_foot_classification(_FakeClassification(10), FootSupport.layer_id),
            clip,
            config=None,
        )
        self.assertEqual(layer.metadata[META_SOURCE_FRAME_COUNT], 10)
        self.assertTrue(contact_layer_is_fresh(layer, clip))

    def test_stale_when_frame_count_changes(self):
        clip = _skate_clip(10)
        layer = stamp_detection_metadata(
            layer_from_foot_classification(_FakeClassification(10), FootSupport.layer_id),
            clip,
        )
        clip2 = clip.model_copy(update={"time_s": np.linspace(0.0, 2.0, 12)})
        self.assertFalse(contact_layer_is_fresh(layer, clip2))

    def test_stale_when_timeline_endpoints_change(self):
        clip = _skate_clip(10)
        layer = stamp_detection_metadata(
            layer_from_foot_classification(_FakeClassification(10), FootSupport.layer_id),
            clip,
        )
        clip2 = clip.model_copy(update={"time_s": np.linspace(0.0, 3.0, 10)})
        self.assertNotEqual(
            fingerprint_timeline(clip.time_s),
            fingerprint_timeline(clip2.time_s),
        )
        self.assertFalse(contact_layer_is_fresh(layer, clip2))

    def test_warn_if_stale(self):
        clip = _skate_clip(8)
        layer = stamp_detection_metadata(
            layer_from_foot_classification(_FakeClassification(8), FootSupport.layer_id),
            clip,
        )
        clip_long = clip.model_copy(update={"time_s": np.linspace(0.0, 1.0, 12)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(warn_if_stale_contact_layer(layer, clip_long))
        self.assertEqual(len(caught), 1)
        self.assertIn("stale", str(caught[0].message).lower())


class TestIdempotentDetect(unittest.TestCase):
    def test_detect_skips_when_fresh(self):
        clip = _skate_clip(6)
        layer = stamp_detection_metadata(
            layer_from_foot_classification(_FakeClassification(6), FootSupport.layer_id),
            clip,
        )
        clip = clip.attach_contact(layer)
        states_before = clip.contact_layer("foot_support").states.copy()
        clip = clip.detect(SKATE_FOOT_SUPPORT, force=False)
        np.testing.assert_array_equal(
            clip.contact_layer("foot_support").states,
            states_before,
        )

    def test_contact_is_fresh_false_when_metadata_mismatches(self):
        clip = _skate_clip(7)
        layer = layer_from_foot_classification(_FakeClassification(7), FootSupport.layer_id)
        stale_meta = {
            **layer.metadata,
            META_SOURCE_FRAME_COUNT: 5,
            META_TIME_FINGERPRINT: fingerprint_timeline(np.linspace(0.0, 1.0, 5)),
        }
        clip = clip.attach_contact(layer.model_copy(update={"metadata": stale_meta}))
        self.assertFalse(clip.contact_is_fresh(SKATE_FOOT_SUPPORT))

    def test_contact_is_fresh(self):
        clip = _skate_clip(4)
        self.assertFalse(clip.contact_is_fresh(SKATE_FOOT_SUPPORT))
        clip = clip.attach_contact(
            stamp_detection_metadata(
                layer_from_foot_classification(_FakeClassification(4), FootSupport.layer_id),
                clip,
            )
        )
        self.assertTrue(clip.contact_is_fresh(SKATE_FOOT_SUPPORT))


if __name__ == "__main__":
    unittest.main()
