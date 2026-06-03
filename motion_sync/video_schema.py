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
    """Ensure enum members match ``expected_names`` in order.

    Args:
        joint_enum: Project joint :class:`StrEnum`.
        expected_names: Required member values in declaration order (e.g.
            :const:`SMPLX_CORE_JOINT_NAMES`).

    Raises:
        TypeError: If ``joint_enum`` is not a :class:`StrEnum` subclass.
        ValueError: If member values or order differ from ``expected_names``.
    """
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

    Pass to :meth:`~motion_sync.synced_dataset.SyncClip.register_video` for typed
    :meth:`~motion_sync.synced_dataset.SyncClip.joint` and
    :meth:`~motion_sync.synced_dataset.SyncClip.core_joint_positions`.

    Attributes:
        joints (type[JointT]): Logical joint :class:`StrEnum` for the project.
        source_indices (tuple[int, ...]): FK column index per enum member (same order as
            :meth:`joint_members`).
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
        """Return enum members in declaration order.

        Returns:
            Tuple of all members of :attr:`joints`.
        """
        return tuple(self.joints)

    def core_index(self, joint: JointT) -> int:
        """Return column index in :meth:`~motion_sync.synced_dataset.SyncClip.core_joint_positions`.

        Args:
            joint: Member of :attr:`joints`.

        Returns:
            Zero-based index along the stacked core-joint axis.

        Raises:
            ValueError: If ``joint`` is not in :meth:`joint_members`.
        """
        return self.joint_members().index(joint)

    def validate_against_clip(self, joint_count: int) -> dict[str, int]:
        """Validate FK indices against a clip's joint count.

        Args:
            joint_count: Number of joints in ``video__joints`` (second axis).

        Returns:
            Map ``joint_enum_value → column_index`` for :attr:`SyncClip.video_joint_map`.

        Raises:
            ValueError: If ``joint_count`` is zero or any index is out of range.
        """
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
        """Build a schema for retarget's 20-joint SMPL-X core (standard GVHMR FK indices).

        Args:
            joints: Project enum whose values match :const:`SMPLX_CORE_JOINT_NAMES` in order.

        Returns:
            Schema using :const:`SMPLX_CORE_SOURCE_INDICES`.

        Raises:
            ValueError: If ``joints`` does not match the canonical name list.
            TypeError: If ``joints`` is not a :class:`StrEnum` subclass.
        """
        validate_joint_enum(joints, SMPLX_CORE_JOINT_NAMES)
        return cls(joints=joints, source_indices=SMPLX_CORE_SOURCE_INDICES)
