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
    """Per-frame support state (values match ``contact_detection.FootSupportState``)."""

    AIR = 0
    GROUND = 1
    SKATEBOARD = 2

    @classmethod
    def stance_states(cls) -> tuple[FootSupportState, FootSupportState]:
        return (cls.GROUND, cls.SKATEBOARD)


@dataclass(frozen=True)
class FootSupportTrack(CategoricalSubjectTrack[BodyT, FootSupportState]):
    """One foot's support state over the clip timeline."""

    @property
    def foot(self) -> BodyT:
        return self.subject

    @property
    def stance(self) -> BoolArray:
        """True on ground or skateboard support."""
        return np.isin(self.values, FootSupportState.stance_states())


@dataclass(frozen=True)
class FootSupport(CategoricalContact[FootSupportState, "FootSupportData"], Generic[BodyT]):
    """Contact type: classify each shoe as air, ground, or skateboard."""

    layer_id: ClassVar[str] = "foot_support"
    State: ClassVar[type[IntEnum]] = FootSupportState

    left: BodyT
    right: BodyT
    board: BodyT

    @property
    def _foot_subjects(self) -> tuple[BodyT, BodyT]:
        return (self.left, self.right)

    def detect(self, clip: Any, config: Any | None = None) -> ContactLayer:
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
        self._validate_layer(layer)
        return FootSupportData(layer, contact=self, time_s=clip.time_s)


def layer_from_foot_classification(classification: Any, layer_id: str) -> ContactLayer:
    """Build a foot-support layer from :class:`~contact_detection.FootSupportClassification`."""
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
        super().__init__(
            layer,
            subjects=contact._foot_subjects,
            time_s=time_s,
            state_enum=FootSupportState,
        )
        self._contact = contact

    def track(self, foot: BodyT | str) -> FootSupportTrack[BodyT]:
        base = super().track(foot)
        return FootSupportTrack(
            subject=base.subject,
            time_s=base.time_s,
            values=base.values,
            state_type=FootSupportState,
        )

    def tracks(self) -> dict[BodyT, FootSupportTrack[BodyT]]:
        return {
            self._contact.left: self.track(self._contact.left),
            self._contact.right: self.track(self._contact.right),
        }

    def state(self, foot: BodyT | str, frame: int) -> FootSupportState:
        return self.track(foot).state_at(frame)

    def stance_mask(self, foot: BodyT | str) -> BoolArray:
        return self.track(foot).stance

    def stance_matrix(self) -> BoolArray:
        left = self.track(self._contact.left).stance
        right = self.track(self._contact.right).stance
        return np.column_stack([left, right])

    @property
    def floor_height(self) -> float:
        return float(self._layer.metadata["floor_height"])

    @property
    def board_contact_offsets(self) -> dict[str, float]:
        return dict(self._layer.metadata.get("board_contact_offsets", {}))

    def classification(self) -> Any:
        """Rebuild :class:`~contact_detection.FootSupportClassification` for plotting."""
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
    from contact_detection import FootSupportConfig

    def _name(ref: BodyT | str) -> str:
        return ref.value if isinstance(ref, StrEnum) else ref

    return FootSupportConfig(
        foot_names=(_name(left), _name(right)),
        board_name=_name(board),
    )
