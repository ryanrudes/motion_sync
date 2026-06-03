"""Skate-specific foot support: sole patches, surface offsets, and plane floor fit."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Generic, TypeVar

import numpy as np

from motion_sync.contacts.foot_support import FootSupport, foot_support_config
from motion_sync.surface_schema import BipedSolePatches, BodyMarkerPatch

BodyT = TypeVar("BodyT", bound=StrEnum)


@dataclass(frozen=True)
class SkateFootSupport(FootSupport[BodyT], Generic[BodyT]):
    """Foot support for skate trials using sole markers for plane floor calibration.

    Extends :class:`FootSupport` with :class:`~motion_sync.surface_schema.BipedSolePatches`.
    :meth:`detect` defaults to :meth:`default_config`; use :meth:`run_classification`
    when full detector feature traces are needed.

    Attributes:
        sole_patches: Left and right :class:`~motion_sync.surface_schema.BodyMarkerPatch` specs.
    """

    sole_patches: BipedSolePatches[BodyT, Any, Any] = field(kw_only=True)

    def sole_for_shoe(self, shoe: BodyT) -> BodyMarkerPatch[BodyT, Any]:
        """Sole patch for one shoe body."""
        return self.sole_patches.patch_for_body(shoe)

    def default_config(self, **overrides: Any) -> Any:
        """Skate :class:`~contact_detection.FootSupportConfig` (plane floor, sole markers)."""
        config = replace(
            foot_support_config(self.left, self.right, self.board),
            up_axis=self.sole_patches.up_axis,
            contact_surface_set=self.sole_patches.contact_surface_set(),
        )
        if overrides:
            config = replace(config, **overrides)
        return config

    def detect(self, clip: Any, config: Any | None = None) -> Any:
        """Classify foot support; uses :meth:`default_config` when ``config`` is omitted."""
        config = config if config is not None else self.default_config()
        return super().detect(
            clip,
            config,
            body_rotations=_body_rotations_for_feet(clip, self.left, self.right),
        )

    def run_classification(self, clip: Any, config: Any | None = None) -> Any:
        """Run the detector and return the full classification (including ``features``)."""
        from contact_detection import classify_foot_support_states

        from motion_sync.contacts.foot_support import stack_floor_fit_marker_pos

        config = config if config is not None else self.default_config()
        t, body_names, body_pos = clip.export_vicon_bodies(
            zero_time=True,
            apply_valid_mask=False,
        )
        floor_fit_pos, floor_fit_names = stack_floor_fit_marker_pos(
            clip,
            config.floor_fit_marker_names,
        )
        body_rotations = _body_rotations_for_feet(clip, self.left, self.right)
        return classify_foot_support_states(
            t,
            body_names,
            body_pos,
            config=config,
            floor_fit_marker_pos=floor_fit_pos,
            floor_fit_marker_names=floor_fit_names,
            body_rotations=body_rotations,
        )


def _body_rotations_for_feet(clip: Any, left: Any, right: Any) -> dict[str, Any] | None:
    """Collect per-foot quaternions when the clip exposes body orientations."""
    rotations: dict[str, Any] = {}
    for shoe in (left, right):
        body = clip.body(shoe)
        if body.orientations is None:
            continue
        rotations[shoe.value] = np.asarray(body.orientations, dtype=np.float64)
    return rotations or None
