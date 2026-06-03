"""Shared helpers for on/off (binary) contact layers."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from motion_sync.contact_layer import ContactLayer
from motion_sync.contact_registration import ContactType
from motion_sync.types import BoolArray, FloatArray

BodyT = TypeVar("BodyT", bound=StrEnum)
ViewT = TypeVar("ViewT")
IntervalList = list[tuple[float, float]]


class BinaryContact(ContactType[Any, ViewT], ABC, Generic[ViewT]):
    """Contact type with a boolean mask per subject.

    Subclasses set :attr:`layer_id` and implement :meth:`detect` and :meth:`read`.
    Binary layers store no categorical labels on disk.

    Attributes:
        layer_id (str): Stable identifier stored in :class:`~motion_sync.contact_layer.ContactLayer`.
        State: Always ``None`` (binary layers have no :class:`IntEnum` states).
    """

    layer_id: ClassVar[str]
    State = None  # type: ignore[assignment]

    @classmethod
    def labels_on_disk(cls) -> tuple[str, ...]:
        """Binary contacts do not persist categorical labels on disk."""

        return ()

    @classmethod
    def build_layer(
        cls,
        *,
        subjects: tuple[str, ...],
        mask: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> ContactLayer:
        """Build a binary layer aligned to ``mask``.

        Args:
            subjects: Subject names, one column per entry.
            mask: Boolean contact flags with shape ``(frames, len(subjects))``.
            metadata: Optional sidecar fields (e.g. ``derived_from`` layer id).

        Returns:
            A ``kind="binary"`` :class:`~motion_sync.contact_layer.ContactLayer`.
        """
        arr = np.asarray(mask, dtype=bool)
        if arr.ndim != 2 or arr.shape[1] != len(subjects):
            raise ValueError(
                f"mask must have shape (frames, {len(subjects)}), got {arr.shape}"
            )
        return ContactLayer(
            layer_id=cls.layer_id,
            kind="binary",
            subjects=subjects,
            labels=(),
            mask=arr,
            metadata=dict(metadata or {}),
        )

    def _validate_layer(self, layer: ContactLayer) -> None:
        if layer.layer_id != self.layer_id:
            raise ValueError(f"expected layer {self.layer_id!r}, got {layer.layer_id!r}")
        if layer.kind != "binary":
            raise ValueError(f"{self.layer_id!r} requires a binary layer")
        if layer.mask is None:
            raise ValueError(f"{self.layer_id!r} layer has no mask")


@dataclass(frozen=True)
class BinarySubjectTrack(Generic[BodyT]):
    """One subject's binary contact flag over the clip timeline.

    Attributes:
        subject (BodyT): Body enum or subject key for this column.
        time_s (FloatArray): Clip timeline in seconds, shape ``(T,)``.
        values (BoolArray): Per-frame contact flag, shape ``(T,)``.
    """

    subject: BodyT
    time_s: FloatArray
    values: BoolArray

    @property
    def frame_count(self) -> int:
        """Number of frames in this track."""
        return int(self.values.shape[0])

    @property
    def active(self) -> BoolArray:
        """Per-frame contact flag, shape ``(T,)`` (alias for :attr:`values`)."""
        return self.values

    def intervals(self, *, min_duration: float = 0.0) -> IntervalList:
        """Return contiguous time intervals where contact is active.

        Args:
            min_duration: Drop intervals shorter than this many seconds.

        Returns:
            List of ``(start_s, end_s)`` pairs on :attr:`time_s`.
        """
        from motion_sync.intervals import intervals_from_mask

        return intervals_from_mask(self.time_s, self.values, min_duration=min_duration)


class BinaryContactData(Generic[BodyT]):
    """Read API for an attached binary contact layer."""

    def __init__(
        self,
        layer: ContactLayer,
        *,
        subjects: tuple[BodyT, ...],
        time_s: FloatArray,
    ) -> None:
        """Attach a binary layer to a clip timeline.

        Args:
            layer: On-disk or in-memory binary contact layer.
            subjects: Registered body enums matching layer column order.
            time_s: Clip timeline in seconds; length must match layer frame count.
        """
        self._layer = layer
        self._subjects = subjects
        self._time_s = np.asarray(time_s, dtype=np.float64)
        if layer.mask is not None and layer.mask.shape[0] != self._time_s.shape[0]:
            raise ValueError("time_s length must match contact layer frame count")

    @property
    def layer(self) -> ContactLayer:
        """Underlying contact layer."""
        return self._layer

    @property
    def time_s(self) -> FloatArray:
        """Clip timeline in seconds, shape ``(T,)``."""
        return self._time_s

    def track(self, ref: BodyT | str) -> BinarySubjectTrack[BodyT]:
        """Per-subject binary track.

        Args:
            ref: Body enum member or subject string.

        Returns:
            Timeline track for that subject's layer column.
        """
        subject: BodyT | str = ref
        if isinstance(ref, str):
            for candidate in self._subjects:
                if candidate.value == ref:
                    subject = candidate
                    break
        return BinarySubjectTrack(
            subject=subject,  # type: ignore[arg-type]
            time_s=self._time_s,
            values=np.asarray(self._column(ref), dtype=bool),
        )

    def tracks(self) -> dict[BodyT, BinarySubjectTrack[BodyT]]:
        """All registered subjects keyed by body enum.

        Returns:
            Mapping from each registered :class:`StrEnum` body to its track.
        """
        return {subject: self.track(subject) for subject in self._subjects}

    def mask_matrix(self) -> BoolArray:
        """Per-subject flags, columns in registration order.

        Returns:
            Boolean array with shape ``(T, len(subjects))``.
        """
        return np.column_stack([self.track(subject).active for subject in self._subjects])

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
        return self._layer.mask[:, idx]  # type: ignore[index]
