"""Typed contact-surface vocabulary: body/marker enums → detection geometry.

Mirrors :mod:`motion_sync.mocap_schema`: project enums define the vocabulary;
:class:`BodyMarkerPatch` is the authoring unit; :meth:`BodyMarkerPatch.to_marker_patch`
bridges to :mod:`contact_detection.geometry` for fitting and classification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar, overload

import numpy as np
from contact_detection.geometry import (
    BodyContactSurface,
    ContactFrameSpec,
    ContactSurfaceSet,
    MarkerAnchoredPatch,
    RigidTransform,
)
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

BodyT = TypeVar("BodyT", bound=StrEnum)
MarkerT = TypeVar("MarkerT", bound=StrEnum)
LeftMarkerT = TypeVar("LeftMarkerT", bound=StrEnum)
RightMarkerT = TypeVar("RightMarkerT", bound=StrEnum)
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BodyLocalVector:
    """Displacement in a rigid body's local frame (meters).

    Attributes:
        x: Body +X component.
        y: Body +Y component.
        z: Body +Z component (negative often moves toward the sole from a raised marker).
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        """Return ``(x, y, z)`` for :class:`~contact_detection.MarkerAnchoredPatch`."""
        return (self.x, self.y, self.z)


def _coerce_body_local_vector(
    value: BodyLocalVector | tuple[float, float, float] | Mapping[str, float],
) -> BodyLocalVector:
    if isinstance(value, BodyLocalVector):
        return value
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError("tuple offsets must have length 3.")
        return BodyLocalVector(x=float(value[0]), y=float(value[1]), z=float(value[2]))
    if isinstance(value, Mapping):
        return BodyLocalVector(
            x=float(value.get("x", 0.0)),
            y=float(value.get("y", 0.0)),
            z=float(value.get("z", 0.0)),
        )
    raise TypeError(f"offset must be BodyLocalVector or length-3 tuple, got {type(value)!r}")


def offsets_for_markers(
    markers: tuple[MarkerT, ...],
    offsets: Mapping[MarkerT, BodyLocalVector | tuple[float, float, float]] | None = None,
) -> tuple[BodyLocalVector, ...]:
    """Build a marker-ordered offset tuple; missing entries default to zero."""
    if not markers:
        raise ValueError("markers must not be empty.")
    if len(frozenset(markers)) != len(markers):
        raise ValueError("markers must not contain duplicates.")
    if offsets is None:
        return tuple(BodyLocalVector() for _ in markers)
    extra = frozenset(offsets) - frozenset(markers)
    if extra:
        raise ValueError(
            f"offsets include markers not in patch: {[m.value for m in extra]!r}"
        )
    return tuple(_coerce_body_local_vector(offsets.get(marker, BodyLocalVector())) for marker in markers)


@dataclass(frozen=True)
class BodyMarkerPatch(Generic[BodyT, MarkerT]):
    """Marker-anchored contact patch for one rigid body (authoring layer).

    Attributes:
        body: Rigid-body enum member (``.value`` is the Vicon subject name).
        markers: Patch-defining markers in floor-fit / visualization column order.
        offsets_body: Body-local sample offsets, one per :attr:`markers` entry (same order).
        frame_spec: How to derive ``T_body_contact`` when compiling.
    """

    body: BodyT
    markers: tuple[MarkerT, ...]
    offsets_body: tuple[BodyLocalVector, ...]
    frame_spec: ContactFrameSpec = field(
        default_factory=ContactFrameSpec.fit_plane_from_samples
    )

    def __post_init__(self) -> None:
        if not self.markers:
            raise ValueError("markers must not be empty.")
        if len(self.markers) != len(self.offsets_body):
            raise ValueError("offsets_body length must match markers.")
        if len(frozenset(self.markers)) != len(self.markers):
            raise ValueError("markers must not contain duplicates.")

    @overload
    @classmethod
    def define(
        cls,
        body: BodyT,
        markers: Sequence[MarkerT],
        *,
        frame_spec: ContactFrameSpec | None = None,
    ) -> BodyMarkerPatch[BodyT, MarkerT]: ...

    @overload
    @classmethod
    def define(
        cls,
        body: BodyT,
        marker_offsets: Mapping[MarkerT, BodyLocalVector | tuple[float, float, float]],
        *,
        frame_spec: ContactFrameSpec | None = None,
    ) -> BodyMarkerPatch[BodyT, MarkerT]: ...

    @classmethod
    def define(
        cls,
        body: BodyT,
        markers_or_offsets: Sequence[MarkerT]
        | Mapping[MarkerT, BodyLocalVector | tuple[float, float, float]],
        *,
        frame_spec: ContactFrameSpec | None = None,
    ) -> BodyMarkerPatch[BodyT, MarkerT]:
        """Define a patch from marker order or a marker→offset map.

        Args:
            body: Rigid-body enum member.
            markers_or_offsets: Either an ordered marker sequence (zero offsets) or a
                mapping whose **insertion order** defines marker order and per-marker
                body-local offsets.
            frame_spec: Optional contact-frame derivation; defaults to plane fit.
        """
        if isinstance(markers_or_offsets, Mapping):
            markers = tuple(markers_or_offsets.keys())
            offsets: Mapping[MarkerT, BodyLocalVector | tuple[float, float, float]] | None = (
                markers_or_offsets
            )
        else:
            markers = tuple(markers_or_offsets)
            offsets = None
        spec = (
            frame_spec
            if frame_spec is not None
            else ContactFrameSpec.fit_plane_from_samples()
        )
        return cls(
            body=body,
            markers=markers,
            offsets_body=offsets_for_markers(markers, offsets),
            frame_spec=spec,
        )

    @property
    def attach_body(self) -> str:
        """Vicon rigid-body name."""
        return self.body.value

    @property
    def members(self) -> frozenset[MarkerT]:
        """Marker enum members in this patch."""
        return frozenset(self.markers)

    @property
    def marker_names(self) -> tuple[str, ...]:
        """Vicon marker strings in :attr:`markers` order."""
        return tuple(marker.value for marker in self.markers)

    @property
    def up_axis(self) -> int:
        """World vertical axis from :attr:`frame_spec`."""
        return self.frame_spec.up_axis

    def _sample_offsets_mapping(self) -> dict[MarkerT, tuple[float, float, float]]:
        return {
            marker: offset.as_tuple()
            for marker, offset in zip(self.markers, self.offsets_body, strict=True)
        }

    def to_marker_patch(self) -> MarkerAnchoredPatch[MarkerT]:
        """Bridge to :mod:`contact_detection` for floor-fit and support detection."""
        return MarkerAnchoredPatch(
            patch_markers=self.markers,
            sample_offsets_body=self._sample_offsets_mapping(),
            attach_body=self.attach_body,
            frame_spec=self.frame_spec,
        )

    def patch_world_positions_at_frame(
        self,
        marker_trajs: Mapping[MarkerT, FloatArray],
        frame: int,
        body_track: Any,
        *,
        quaternion_scalar_last: bool = True,
    ) -> FloatArray:
        """World positions with body-local offsets applied for one frame."""
        points = np.stack(
            [np.asarray(marker_trajs[marker][frame], dtype=np.float64) for marker in self.markers],
            axis=0,
        )
        orientation = body_track.orientation_at(frame)
        if orientation is None:
            raise ValueError(f"{self.attach_body!r} has no orientation at frame {frame}.")
        return self.to_marker_patch().world_sample_positions(
            points,
            body_quaternion_xyzw=np.asarray(orientation, dtype=np.float64),
            quaternion_scalar_last=quaternion_scalar_last,
        )

    def compile_at_frame(
        self,
        marker_trajs: Mapping[MarkerT, FloatArray],
        frame: int,
        body_track: Any,
    ) -> BodyContactSurface:
        """Derive :class:`~contact_detection.BodyContactSurface` for one clip frame."""
        points = np.stack(
            [np.asarray(marker_trajs[marker][frame], dtype=np.float64) for marker in self.markers],
            axis=0,
        )
        orientation = body_track.orientation_at(frame)
        if orientation is None:
            raise ValueError(f"{self.attach_body!r} has no orientation at frame {frame}.")
        return self.to_marker_patch().compile(
            marker_positions_world=points,
            body_translation=np.asarray(body_track.positions[frame], dtype=np.float64),
            body_quaternion_xyzw=np.asarray(orientation, dtype=np.float64),
        )

    def world_contact_frame_at_frame(
        self,
        surface: BodyContactSurface,
        body_track: Any,
        frame: int,
    ) -> RigidTransform | None:
        """Return ``T_world_contact`` from compiled ``T_body_contact`` and body pose."""
        orientation = body_track.orientation_at(frame)
        if orientation is None:
            return None
        rotation = Rotation.from_quat(np.asarray(orientation, dtype=np.float64))
        return surface.frame.compose_world(
            np.asarray(body_track.positions[frame], dtype=np.float64),
            rotation,
        )

    def plane_corners_world_at_frame(
        self,
        surface: BodyContactSurface,
        body_track: Any,
        frame: int,
        *,
        half_width: float,
        half_length: float,
    ) -> FloatArray | None:
        """Contact-plane rectangle in world coordinates."""
        world_frame = self.world_contact_frame_at_frame(surface, body_track, frame)
        if world_frame is None:
            return None
        return world_frame.rectangle_corners_contact(half_width, half_length)


@dataclass(frozen=True)
class BipedSolePatches(Generic[BodyT, LeftMarkerT, RightMarkerT]):
    """Left and right sole patches (separate marker enums per foot).

    Attributes:
        left: Sole patch on the left-shoe body.
        right: Sole patch on the right-shoe body.
    """

    left: BodyMarkerPatch[BodyT, LeftMarkerT]
    right: BodyMarkerPatch[BodyT, RightMarkerT]

    def __post_init__(self) -> None:
        if self.left.up_axis != self.right.up_axis:
            raise ValueError("left and right sole patches must use the same frame_spec.up_axis.")

    @property
    def up_axis(self) -> int:
        """Shared world vertical axis."""
        return self.left.up_axis

    def patch_for_body(self, body: BodyT) -> BodyMarkerPatch[BodyT, Any]:
        """Return the sole patch registered for ``body``."""
        if body == self.left.body:
            return self.left
        if body == self.right.body:
            return self.right
        raise KeyError(f"{body!r} is not a configured sole body.")

    def contact_surface_set(self) -> ContactSurfaceSet:
        """Both feet as a :class:`~contact_detection.ContactSurfaceSet`."""
        return ContactSurfaceSet.from_marker_patches(
            self.left.to_marker_patch(),
            self.right.to_marker_patch(),
        )

    def floor_fit_marker_names(self) -> tuple[str, ...]:
        """Vicon names for stacked floor-fit columns (left, then right)."""
        return self.left.marker_names + self.right.marker_names
