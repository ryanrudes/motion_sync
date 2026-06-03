"""User-defined SMPL-X / video joint naming (parallel to :class:`~motion_sync.mocap_schema.MocapSchema`)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

JointT = TypeVar("JointT", bound=StrEnum)

# Retarget ``MotionFormat.SMPLX`` core joint order (20 joints).
SMPLX_CORE_JOINT_NAMES: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
)

# Full SMPL-X FK indices for each core joint (GVHMR ``joints.npy`` layout).
SMPLX_CORE_SOURCE_INDICES: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
)


def validate_joint_enum(joint_enum: type[JointT], expected_names: tuple[str, ...]) -> None:
    """Ensure enum members match ``expected_names`` in order (e.g. retarget smplx core)."""
    if not issubclass(joint_enum, StrEnum):
        raise TypeError(f"joint_enum must be a StrEnum subclass, got {joint_enum!r}")
    actual = tuple(member.value for member in joint_enum)
    if actual != expected_names:
        raise ValueError(
            f"{joint_enum.__name__} must list joints in retarget smplx core order; "
            f"expected {expected_names!r}, got {actual!r}"
        )


@dataclass(frozen=True)
class VideoSchema(Generic[JointT]):
    """Project video vocabulary: logical joint enum plus FK indices into ``video__joints``.

    Pass to :meth:`SyncClip.register_video` for typed :meth:`~SyncClip.joint` and
    :meth:`~SyncClip.core_joint_positions`.
    """

    joints: type[JointT]
    source_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        members = self.joint_members()
        if len(self.source_indices) != len(members):
            raise ValueError(
                f"source_indices length {len(self.source_indices)} != "
                f"{len(members)} joint enum members"
            )

    def joint_members(self) -> tuple[JointT, ...]:
        return tuple(self.joints)

    def core_index(self, joint: JointT) -> int:
        """Column index in :meth:`~SyncClip.core_joint_positions` for ``joint``."""
        return self.joint_members().index(joint)

    def validate_against_clip(self, joint_count: int) -> dict[str, int]:
        """Validate FK indices against ``clip.video.joint_count``; return value→index map."""
        if joint_count <= 0:
            raise ValueError("clip has no video joints to register against")
        mapping: dict[str, int] = {}
        for member, idx in zip(self.joint_members(), self.source_indices, strict=True):
            if idx < 0 or idx >= joint_count:
                raise ValueError(
                    f"joint {member.value!r} FK index {idx} out of range for "
                    f"video joint_count={joint_count}"
                )
            mapping[member.value] = int(idx)
        return mapping

    @classmethod
    def smplx_core(cls, joints: type[JointT]) -> VideoSchema[JointT]:
        """Schema for retarget's 20-joint SMPL-X core using standard GVHMR FK indices."""
        validate_joint_enum(joints, SMPLX_CORE_JOINT_NAMES)
        return cls(joints=joints, source_indices=SMPLX_CORE_SOURCE_INDICES)
