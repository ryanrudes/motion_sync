"""Project-friendly constructors for contact_detection body-local models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TypeVar

from contact_detection import (
    Z_UP_AXES,
    AxisConvention,
    BodyFrameTranslation,
    ContactRegion,
    MarkerTranslation,
    PatchCalibration,
    RigidBodyContactModel,
)

BodyT = TypeVar("BodyT", bound=StrEnum)
MarkerT = TypeVar("MarkerT", bound=StrEnum)


def marker_name(marker: StrEnum | str) -> str:
    """Return the Vicon marker/body string for an enum member or raw name."""

    return marker.value if isinstance(marker, StrEnum) else str(marker)


def body_name(body: StrEnum | str) -> str:
    """Return the Vicon rigid-body string for an enum member or raw name."""

    return body.value if isinstance(body, StrEnum) else str(body)


def patch_calibration(
    marker_offsets: Sequence[MarkerT] | Mapping[MarkerT, MarkerTranslation | Sequence[float]],
    *,
    region: ContactRegion | None = None,
) -> PatchCalibration[MarkerT]:
    """Build a contact patch calibration from markers or marker offset mappings."""

    if isinstance(marker_offsets, Mapping):
        translations = {
            marker: _coerce_translation(offset)
            for marker, offset in marker_offsets.items()
        }
        return PatchCalibration(marker_translations=translations, region=region)
    return PatchCalibration.from_markers(tuple(marker_offsets), region=region)


def rigid_body_contact_model(
    body: BodyT | str,
    *,
    marker_type: type[MarkerT] | None = None,
    patches: Mapping[str, PatchCalibration[MarkerT]],
    axis_convention: AxisConvention = Z_UP_AXES,
) -> RigidBodyContactModel[MarkerT]:
    """Build a body-local contact model for one tracked rigid body."""

    return RigidBodyContactModel(
        body_name=body_name(body),
        marker_type=marker_type,
        axis_convention=axis_convention,
        patch_calibrations=dict(patches),
    )


def _coerce_translation(value: MarkerTranslation | Sequence[float]) -> MarkerTranslation:
    if isinstance(value, BodyFrameTranslation):
        return value
    if hasattr(value, "resolve"):
        return value  # SemanticAxisTranslation
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError("marker offset sequences must have length 3.")
    return BodyFrameTranslation(values)
