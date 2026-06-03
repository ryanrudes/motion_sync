"""User-defined mocap naming: rigid bodies, per-body marker enums, and grouping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

BodyT = TypeVar("BodyT", bound=StrEnum)


def validate_body_enum(body_enum: type[BodyT], body_names: tuple[str, ...]) -> None:
    """Ensure a user-defined :class:`StrEnum` matches this clip's Vicon rigid-body names.

    Args:
        body_enum: Project body enum; member **values** must equal Vicon subject strings.
        body_names: Rigid-body names from the loaded clip (e.g. :attr:`SyncClip.body_names`).

    Raises:
        TypeError: If ``body_enum`` is not a :class:`StrEnum` subclass.
        ValueError: If the enum and clip disagree on which bodies exist.
    """
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
    """Ensure every member of a body marker enum exists on the clip.

    Extra markers on the clip are allowed; every enum value must appear in ``marker_names``.

    Args:
        marker_enum: Per-body marker :class:`StrEnum` (values are Vicon marker strings).
        marker_names: Marker names from the loaded clip.

    Raises:
        TypeError: If ``marker_enum`` is not a :class:`StrEnum` subclass.
        ValueError: If a member references a missing marker or values are duplicated.
    """
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
    """Validate per-body marker enums partition the clip's marker set.

    Args:
        body_enum: Project body enum.
        body_markers: One marker enum per body; keys must cover every body member.
        marker_names: All marker names on the clip.

    Returns:
        Map ``body_value → (marker_value, …)`` in enum iteration order.

    Raises:
        ValueError: If bodies or markers are missing, duplicated across bodies, or
            do not partition ``marker_names`` exactly.
        TypeError: If a marker enum is not a :class:`StrEnum` subclass.
    """
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

    Pass to :meth:`~motion_sync.synced_dataset.SyncClip.register_mocap` for typed
    :meth:`~motion_sync.synced_dataset.SyncClip.body` /
    :meth:`~motion_sync.synced_dataset.SyncClip.markers_for_body` access.

    Attributes:
        bodies (type[BodyT]): Rigid-body :class:`StrEnum`; values match Vicon subject names.
        body_markers (dict[BodyT, type[StrEnum]]): Marker enum per body member.
    """

    bodies: type[BodyT]
    body_markers: dict[BodyT, type[StrEnum]]

    def marker_enum_for(self, body: BodyT) -> type[StrEnum]:
        """Return the marker enum class registered for ``body``.

        Args:
            body: Member of :attr:`bodies`.

        Returns:
            That body's marker :class:`StrEnum` subclass.

        Raises:
            KeyError: If ``body`` is not a key of :attr:`body_markers`.
        """
        return self.body_markers[body]

    def validate_against_clip(
        self,
        body_names: tuple[str, ...],
        marker_names: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
        """Validate this schema against a loaded clip.

        Args:
            body_names: Clip rigid-body names.
            marker_names: Clip marker names.

        Returns:
            Stored body→marker value map for :attr:`SyncClip.body_marker_map`.

        Raises:
            ValueError: If bodies or markers do not match the schema rules.
            TypeError: If enums are not :class:`StrEnum` subclasses.
        """
        validate_body_enum(self.bodies, body_names)
        return validate_body_marker_enums(self.bodies, self.body_markers, marker_names)
