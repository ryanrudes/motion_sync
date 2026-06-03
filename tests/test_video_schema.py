"""VideoSchema and register_video on SyncClip."""

import unittest

import numpy as np

from motion_sync.schemas.skateboarding import (
    SKATE_SESSION,
    SKATE_VIDEO,
    SmplxCoreJoints,
)
from motion_sync.synced_dataset import SyncClip
from motion_sync.video_schema import SMPLX_CORE_SOURCE_INDICES, VideoSchema


def _skate_clip(frames: int = 6, joint_count: int = 24) -> SyncClip:
    return SyncClip(
        time_s=np.linspace(0.0, 0.5, frames),
        vicon={
            "body_names": ("Left_Shoe", "Right_Shoe", "Skateboard"),
            "body_positions": np.zeros((frames, 3, 3)),
        },
        video={
            "joints": np.arange(frames * joint_count * 3, dtype=np.float64).reshape(
                frames, joint_count, 3
            ),
            "transl": np.zeros((frames, 3)),
            "global_orient": np.zeros((frames, 3)),
            "body_pose": np.zeros((frames, 63)),
            "betas": np.zeros((frames, 10)),
        },
        metadata={"lag_s": 0.0},
    )


class TestVideoSchema(unittest.TestCase):
    def test_smplx_core_indices_length(self):
        self.assertEqual(len(SKATE_VIDEO.source_indices), len(SmplxCoreJoints))
        self.assertEqual(len(SKATE_VIDEO.source_indices), len(SMPLX_CORE_SOURCE_INDICES))

    def test_register_joint_and_core_stack(self):
        clip = _skate_clip().register_video(SKATE_VIDEO)
        left = clip.joint(SmplxCoreJoints.L_FOOT)
        self.assertEqual(left.name, "L_Foot")
        self.assertEqual(left.positions.shape, (6, 3))

        core = clip.core_joint_positions()
        self.assertEqual(core.shape, (6, len(SmplxCoreJoints), 3))
        idx = SKATE_VIDEO.core_index(SmplxCoreJoints.L_FOOT)
        np.testing.assert_array_equal(core[:, idx, :], left.positions)

    def test_rejects_out_of_range_index(self):
        from enum import StrEnum

        class _TinyJoints(StrEnum):
            A = "Pelvis"
            B = "L_Hip"

        schema = VideoSchema(joints=_TinyJoints, source_indices=(0, 99))
        clip = _skate_clip(joint_count=3)
        with self.assertRaises(ValueError):
            clip.register_video(schema)

    def test_session_load_registers_video(self):
        import tempfile
        from pathlib import Path

        clip = _skate_clip(joint_count=24)
        with tempfile.TemporaryDirectory() as tmp:
            demo = Path(tmp) / "demo"
            clip.save(demo)
            loaded = SyncClip.load(demo, session=SKATE_SESSION)
            self.assertIsNotNone(loaded.registered_video)
            self.assertIn(SmplxCoreJoints.L_FOOT.value, loaded.video_joint_map)


if __name__ == "__main__":
    unittest.main()
