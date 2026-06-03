"""Per-foot air / ground / skateboard support (skate trials)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from motion_sync.contact_layer import ContactLayer
from motion_sync.contacts.categorical import (
    CategoricalContact,
    CategoricalContactData,
    CategoricalSubjectTrack,
)
from motion_sync.types import BoolArray, FloatArray

BodyT = TypeVar("BodyT", bound=StrEnum)


class FootSupportState(IntEnum):
    """Per-frame support state (values match ``contact_detection.FootSupportState``).

    Attributes:
        AIR (int): Foot is airborne (``0``).
        GROUND (int): Foot contacts the floor (``1``).
        SKATEBOARD (int): Foot contacts the skateboard deck (``2``).
    """

    AIR = 0
    GROUND = 1
    SKATEBOARD = 2

    @classmethod
    def stance_states(cls) -> tuple[FootSupportState, FootSupportState]:
        """States counted as weight-bearing (ground or board).

        Returns:
            ``(GROUND, SKATEBOARD)``.
        """
        return (cls.GROUND, cls.SKATEBOARD)


@dataclass(frozen=True)
class FootSupportTrack(CategoricalSubjectTrack[BodyT, FootSupportState]):
    """One foot's support state over the clip timeline.

    Attributes:
        subject (BodyT): Shoe body enum for this foot.
        time_s (FloatArray): Clip timeline in seconds.
        values (np.ndarray): Per-frame :class:`FootSupportState` codes.
        state_type (type[FootSupportState]): Always :class:`FootSupportState`.
    """

    @property
    def foot(self) -> BodyT:
        """Shoe body enum (alias for :attr:`subject`)."""
        return self.subject

    @property
    def stance(self) -> BoolArray:
        """True on ground or skateboard support, shape ``(T,)``."""
        return np.isin(self.values, FootSupportState.stance_states())


@dataclass(frozen=True)
class FootSupport(CategoricalContact[FootSupportState, "FootSupportData"], Generic[BodyT]):
    """Contact type: classify each shoe as air, ground, or skateboard.

    Detection delegates to :func:`contact_detection.classify_foot_support_states` using
    Vicon rigid-body poses exported from the clip.

    Attributes:
        layer_id (str): ``"foot_support"``.
        State (type[FootSupportState]): :class:`FootSupportState`.
        left (BodyT): Left shoe Vicon body.
        right (BodyT): Right shoe Vicon body.
        board (BodyT): Skateboard rigid body (used for board-contact geometry).
    """

    layer_id: ClassVar[str] = "foot_support"
    State: ClassVar[type[IntEnum]] = FootSupportState

    left: BodyT
    right: BodyT
    board: BodyT

    @property
    def _foot_subjects(self) -> tuple[BodyT, BodyT]:
        return (self.left, self.right)

    def detect(self, clip: Any, config: Any | None = None) -> ContactLayer:
        """Run foot-support classification on Vicon bodies in ``clip``.

        Args:
            clip: :class:`~motion_sync.synced_dataset.SyncClip` (or compatible) with Vicon export.
            config: Optional :class:`~contact_detection.FootSupportConfig`; defaults from
                :func:`foot_support_config` when omitted.

        Returns:
            Categorical layer with subjects ``(left.value, right.value)`` aligned to the clip.

        Raises:
            ImportError: If ``contact_detection`` is not installed.
            ValueError: If detector frame count or subjects do not match the clip.
        """
        if config is None:
            config = foot_support_config(self.left, self.right, self.board)
        try:
            from contact_detection import classify_foot_support_states
        except ImportError as exc:
            raise ImportError(
                "Install contact-detection "
                "(e.g. uv pip install -e ../event_detection)."
            ) from exc

        t, body_names, body_pos = clip.export_vicon_bodies(
            zero_time=True,
            apply_valid_mask=False,
        )
        result = classify_foot_support_states(t, body_names, body_pos, config=config)
        layer = layer_from_foot_classification(result, self.layer_id)
        expected = (self.left.value, self.right.value)
        if layer.subjects != expected:
            raise ValueError(
                f"foot support subjects {layer.subjects!r} != expected {expected!r}"
            )
        if layer.frame_count != clip.frame_count:
            raise ValueError(
                f"detector returned {layer.frame_count} frames but clip has {clip.frame_count}"
            )
        return layer

    def read(self, clip: Any, layer: ContactLayer) -> FootSupportData[BodyT]:
        """Typed reader for an attached foot-support layer.

        Args:
            clip: Clip providing :attr:`~motion_sync.synced_dataset.SyncClip.time_s`.
            layer: Categorical foot-support layer to validate and wrap.

        Returns:
            :class:`FootSupportData` view over ``layer``.
        """
        self._validate_layer(layer)
        return FootSupportData(layer, contact=self, time_s=clip.time_s)


def layer_from_foot_classification(classification: Any, layer_id: str) -> ContactLayer:
    """Build a foot-support layer from :class:`~contact_detection.FootSupportClassification`.

    Args:
        classification: Detector output with ``states``, floor model fields, and offsets.
        layer_id: Layer id to store (usually ``"foot_support"``).

    Returns:
        Categorical :class:`~motion_sync.contact_layer.ContactLayer` with detector metadata.
    """
    foot_names = tuple(classification.states.keys())
    if not foot_names:
        raise ValueError("classification has no foot states")

    metadata: dict[str, Any] = {
        "floor_model": str(classification.floor_model),
        "floor_height": float(classification.floor_height),
        "board_contact_offsets": dict(classification.board_contact_offsets),
    }
    if classification.floor_normal is not None:
        metadata["floor_normal"] = np.asarray(classification.floor_normal, dtype=np.float64)
    if classification.floor_origin is not None:
        metadata["floor_origin"] = np.asarray(classification.floor_origin, dtype=np.float64)

    layer = FootSupport.build_layer(
        subjects=foot_names,
        states=np.stack(
            [
                np.asarray(classification.states[name], dtype=np.int8)
                for name in foot_names
            ],
            axis=1,
        ),
        metadata=metadata,
    )
    if layer.layer_id != layer_id:
        return layer.model_copy(update={"layer_id": layer_id})
    return layer


class FootSupportData(CategoricalContactData[BodyT, FootSupportState], Generic[BodyT]):
    """Read API for an attached foot-support layer."""

    def __init__(
        self,
        layer: ContactLayer,
        *,
        contact: FootSupport[BodyT],
        time_s: FloatArray,
    ) -> None:
        """Wrap a foot-support layer.

        Args:
            layer: Attached categorical layer.
            contact: Registration object defining left/right shoe bodies.
            time_s: Clip timeline in seconds.
        """
        super().__init__(
            layer,
            subjects=contact._foot_subjects,
            time_s=time_s,
            state_enum=FootSupportState,
        )
        self._contact = contact

    def track(self, foot: BodyT | str) -> FootSupportTrack[BodyT]:
        """Per-foot support track with stance helper.

        Args:
            foot: Left or right shoe body enum or Vicon name.

        Returns:
            :class:`FootSupportTrack` for that foot.
        """
        base = super().track(foot)
        return FootSupportTrack(
            subject=base.subject,
            time_s=base.time_s,
            values=base.values,
            state_type=FootSupportState,
        )

    def tracks(self) -> dict[BodyT, FootSupportTrack[BodyT]]:
        """Left and right foot tracks keyed by shoe body enum."""
        return {
            self._contact.left: self.track(self._contact.left),
            self._contact.right: self.track(self._contact.right),
        }

    def state(self, foot: BodyT | str, frame: int) -> FootSupportState:
        """Support state at one frame.

        Args:
            foot: Shoe body enum or Vicon name.
            frame: Zero-based frame index.

        Returns:
            :class:`FootSupportState` for that frame.
        """
        return self.track(foot).state_at(frame)

    def stance_mask(self, foot: BodyT | str) -> BoolArray:
        """Weight-bearing mask for one foot (ground or skateboard).

        Args:
            foot: Shoe body enum or Vicon name.

        Returns:
            Boolean array with shape ``(T,)``.
        """
        return self.track(foot).stance

    def stance_matrix(self) -> BoolArray:
        """Weight-bearing flags for left then right foot.

        Returns:
            Boolean array with shape ``(T, 2)``.
        """
        left = self.track(self._contact.left).stance
        right = self.track(self._contact.right).stance
        return np.column_stack([left, right])

    @property
    def floor_height(self) -> float:
        """Scalar floor height from detector metadata (meters)."""
        return float(self._layer.metadata["floor_height"])

    @property
    def board_contact_offsets(self) -> dict[str, float]:
        """Per-foot vertical offsets used for board proximity (from metadata)."""
        return dict(self._layer.metadata.get("board_contact_offsets", {}))

    def classification(self) -> Any:
        """Rebuild :class:`~contact_detection.FootSupportClassification` for plotting.

        Returns:
            Classification object with ``t``, ``states``, and floor fields; ``features`` empty.
        """
        from contact_detection import FootSupportClassification

        states = {
            name: self._layer.states[:, idx]  # type: ignore[index]
            for idx, name in enumerate(self._layer.subjects)
        }
        return FootSupportClassification(
            t=self._time_s,
            states=states,
            floor_model=self._layer.metadata["floor_model"],
            floor_height=self._layer.metadata["floor_height"],
            floor_normal=self._layer.metadata.get("floor_normal"),
            floor_origin=self._layer.metadata.get("floor_origin"),
            board_contact_offsets=self._layer.metadata.get("board_contact_offsets", {}),
            features={},
        )


def foot_support_config(
    left: BodyT | str,
    right: BodyT | str,
    board: BodyT | str,
) -> Any:
    """Build :class:`~contact_detection.FootSupportConfig` from body names.

    Args:
        left: Left shoe body enum or Vicon name.
        right: Right shoe body enum or Vicon name.
        board: Skateboard body enum or Vicon name.

    Returns:
        Config wired for :func:`contact_detection.classify_foot_support_states`.
    """
    from contact_detection import FootSupportConfig

    def _name(ref: BodyT | str) -> str:
        return ref.value if isinstance(ref, StrEnum) else ref

    return FootSupportConfig(
        foot_names=(_name(left), _name(right)),
        board_name=_name(board),
    )
