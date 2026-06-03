"""User-defined mocap naming: rigid bodies, per-body marker enums, and grouping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

BodyT = TypeVar("BodyT", bound=StrEnum)


def validate_body_enum(body_enum: type[BodyT], body_names: tuple[str, ...]) -> None:
    """Ensure a user-defined :class:`StrEnum` matches this clip's Vicon rigid-body names."""
    if not issubclass(body_enum, StrEnum):
        raise TypeError(f"body_enum must be a StrEnum subclass, got {body_enum!r}")
    if not body_names:
        raise ValueError("clip has no body_names to register against")
    enum_values = {member.value for member in body_enum}
    expected = set(body_names)
    missing = expected - enum_values
    extra = enum_values - expected
    if missing:
        raise ValueError(
            f"{body_enum.__name__} is missing Vicon bodies present in the clip: "
            f"{sorted(missing)}"
        )
    if extra:
        raise ValueError(
            f"{body_enum.__name__} defines bodies not in this clip: {sorted(extra)}"
        )


def validate_marker_enum_for_clip(
    marker_enum: type[StrEnum],
    marker_names: tuple[str, ...],
) -> None:
    """Ensure every member of a body marker enum exists on the clip (extras allowed on clip)."""
    if not issubclass(marker_enum, StrEnum):
        raise TypeError(f"marker_enum must be a StrEnum subclass, got {marker_enum!r}")
    clip_names = set(marker_names)
    missing = [m.value for m in marker_enum if m.value not in clip_names]
    if missing:
        raise ValueError(
            f"{marker_enum.__name__} references markers not in this clip: {sorted(missing)}"
        )
    values = [m.value for m in marker_enum]
    if len(values) != len(set(values)):
        raise ValueError(f"{marker_enum.__name__} has duplicate Vicon marker values")


def validate_body_marker_enums(
    body_enum: type[BodyT],
    body_markers: dict[BodyT, type[StrEnum]],
    marker_names: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Validate per-body marker enums partition the clip; return ``{body_value: (marker_value, ...)}``."""
    if not body_markers:
        raise ValueError("body_markers must not be empty")

    bodies_in_map = set(body_markers.keys())
    all_bodies = set(body_enum)
    if bodies_in_map != all_bodies:
        missing = all_bodies - bodies_in_map
        extra = bodies_in_map - all_bodies
        parts: list[str] = []
        if missing:
            parts.append(f"missing keys for {sorted(m.value for m in missing)}")
        if extra:
            parts.append(f"unknown keys {sorted(m.value for m in extra)}")
        raise ValueError(f"body_markers must have exactly one entry per body ({'; '.join(parts)})")

    clip_names = set(marker_names)
    seen_values: dict[str, BodyT] = {}
    stored: dict[str, tuple[str, ...]] = {}

    for body, marker_enum in body_markers.items():
        if not issubclass(marker_enum, StrEnum):
            raise TypeError(
                f"marker enum for {body.value!r} must be a StrEnum subclass, got {marker_enum!r}"
            )
        members = tuple(marker_enum)
        if not members:
            raise ValueError(f"{marker_enum.__name__} has no members")
        validate_marker_enum_for_clip(marker_enum, marker_names)
        body_values: list[str] = []
        for marker in members:
            if marker.value in seen_values:
                prev = seen_values[marker.value]
                raise ValueError(
                    f"Vicon marker {marker.value!r} is listed on both "
                    f"{prev.value!r} and {body.value!r}"
                )
            seen_values[marker.value] = body
            body_values.append(marker.value)
        stored[body.value] = tuple(body_values)

    if set(seen_values.keys()) != clip_names:
        missing = sorted(clip_names - set(seen_values.keys()))
        extra = sorted(set(seen_values.keys()) - clip_names)
        parts: list[str] = []
        if missing:
            parts.append(f"unassigned on a body: {missing}")
        if extra:
            parts.append(f"not on clip: {extra}")
        raise ValueError(
            f"body marker enums must partition clip markers exactly ({'; '.join(parts)})"
        )
    return stored


@dataclass(frozen=True)
class MocapSchema(Generic[BodyT]):
    """Project mocap vocabulary: body enum plus one marker :class:`StrEnum` per body.

    Each body gets its own marker enum so logical names (e.g. ``HEEL``) can repeat across
    feet without conflicting. Enum **values** are the Vicon marker strings.

    Pass to :meth:`SyncClip.register_mocap` for typed :meth:`~SyncClip.body` /
    :meth:`~SyncClip.markers_for_body` access.
    """

    bodies: type[BodyT]
    body_markers: dict[BodyT, type[StrEnum]]

    def marker_enum_for(self, body: BodyT) -> type[StrEnum]:
        """Marker enum class registered for ``body``."""
        return self.body_markers[body]

    def validate_against_clip(
        self,
        body_names: tuple[str, ...],
        marker_names: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
        """Validate against a loaded clip; return stored body→marker value map."""
        validate_body_enum(self.bodies, body_names)
        return validate_body_marker_enums(self.bodies, self.body_markers, marker_names)
