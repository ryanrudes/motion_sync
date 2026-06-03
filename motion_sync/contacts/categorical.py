"""Shared helpers for multi-state (categorical) contact layers."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from motion_sync.contact_layer import ContactLayer
from motion_sync.contact_registration import ContactType
from motion_sync.types import FloatArray

BodyT = TypeVar("BodyT", bound=StrEnum)
StateT = TypeVar("StateT", bound=IntEnum)
ViewT = TypeVar("ViewT")
IntervalList = list[tuple[float, float]]


class CategoricalContact(ContactType[StateT, ViewT], ABC, Generic[StateT, ViewT]):
    """Contact type with per-frame categorical states per subject."""

    layer_id: ClassVar[str]
    State: ClassVar[type[IntEnum]]

    @classmethod
    def build_layer(
        cls,
        *,
        subjects: tuple[str, ...],
        states: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> ContactLayer:
        """Build a categorical layer aligned to ``states`` shape ``(frames, subjects)``."""
        arr = np.asarray(states, dtype=np.int8)
        if arr.ndim != 2 or arr.shape[1] != len(subjects):
            raise ValueError(
                f"states must have shape (frames, {len(subjects)}), got {arr.shape}"
            )
        return ContactLayer(
            layer_id=cls.layer_id,
            kind="categorical",
            subjects=subjects,
            labels=cls.labels_on_disk(),
            states=arr,
            metadata=dict(metadata or {}),
        )

    def _validate_layer(self, layer: ContactLayer) -> None:
        if layer.layer_id != self.layer_id:
            raise ValueError(f"expected layer {self.layer_id!r}, got {layer.layer_id!r}")
        if layer.kind != "categorical":
            raise ValueError(f"{self.layer_id!r} requires a categorical layer")
        if tuple(layer.labels) != self.labels_on_disk():
            raise ValueError(
                f"{self.layer_id!r} labels must be {self.labels_on_disk()}, got {layer.labels}"
            )
        if layer.states is None:
            raise ValueError(f"{self.layer_id!r} layer has no states")


@dataclass(frozen=True)
class CategoricalSubjectTrack(Generic[BodyT, StateT]):
    """One subject's categorical state over the clip timeline."""

    subject: BodyT
    time_s: FloatArray
    values: np.ndarray
    state_type: type[StateT]

    @property
    def frame_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def states(self) -> np.ndarray:
        """Per-frame state codes (``State`` ints), shape ``(T,)``."""
        return self.values

    def state_at(self, frame: int) -> StateT:
        return self.state_type(int(self.values[frame]))

    def intervals(
        self,
        state: StateT,
        *,
        min_duration: float = 0.0,
    ) -> IntervalList:
        from motion_sync.intervals import intervals_from_mask

        mask = self.values == int(state)
        return intervals_from_mask(self.time_s, mask, min_duration=min_duration)


class CategoricalContactData(Generic[BodyT, StateT]):
    """Read API for an attached categorical contact layer."""

    def __init__(
        self,
        layer: ContactLayer,
        *,
        subjects: tuple[BodyT, ...],
        time_s: FloatArray,
        state_enum: type[StateT],
    ) -> None:
        self._layer = layer
        self._subjects = subjects
        self._time_s = np.asarray(time_s, dtype=np.float64)
        self._state_enum = state_enum
        if layer.states is not None and layer.states.shape[0] != self._time_s.shape[0]:
            raise ValueError("time_s length must match contact layer frame count")

    @property
    def layer(self) -> ContactLayer:
        return self._layer

    @property
    def time_s(self) -> FloatArray:
        return self._time_s

    def _resolve_subject_name(self, ref: BodyT | str) -> str:
        if isinstance(ref, StrEnum):
            return ref.value
        for candidate in self._subjects:
            if candidate.value == ref:
                return ref
        allowed = tuple(s.value for s in self._subjects)
        raise KeyError(f"subject {ref!r} not in {allowed!r}")

    def _column(self, ref: BodyT | str) -> np.ndarray:
        name = self._resolve_subject_name(ref)
        if name not in self._layer.subjects:
            raise KeyError(f"subject {name!r} not in layer subjects {self._layer.subjects!r}")
        idx = self._layer.subjects.index(name)
        return self._layer.states[:, idx]  # type: ignore[index]

    def track(self, ref: BodyT | str) -> CategoricalSubjectTrack[BodyT, StateT]:
        subject: BodyT | str = ref
        if isinstance(ref, str):
            for candidate in self._subjects:
                if candidate.value == ref:
                    subject = candidate
                    break
        return CategoricalSubjectTrack(
            subject=subject,  # type: ignore[arg-type]
            time_s=self._time_s,
            values=np.asarray(self._column(ref), dtype=np.int8),
            state_type=self._state_enum,
        )

    def tracks(self) -> dict[BodyT, CategoricalSubjectTrack[BodyT, StateT]]:
        return {subject: self.track(subject) for subject in self._subjects}
