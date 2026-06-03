"""YAML-backed motion-sync configuration models."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PlaneSolverConfig(BaseModel):
    """Acceptance threshold for rigid-body plane fits."""

    max_stress: float


class RigidBodySolverConfig(BaseModel):
    """Rigid-body model estimation quality gates."""

    max_stress: float
    plane_solver: PlaneSolverConfig


class RigidBodyConfig(BaseModel):
    """Marker layout and optional coplanar groups for one Vicon rigid body."""

    markers: list[str]
    planes: dict[str, list[str]] = Field(default_factory=dict)


class PathsConfig(BaseModel):
    """Filesystem paths referenced by the pipeline."""

    smplx_models: Path


class TimeSyncSolverConfig(BaseModel):
    """Foot-speed cross-correlation lag search and acceptance rules."""

    smplx_joints: dict[str, list[str]]
    min_correlation: float = Field(
        default=0.28,
        ge=0.0,
        le=1.0,
        description="Reject time sync when the best score is below this. Set to 0 to disable.",
    )
    max_abs_lag_seconds: float | None = Field(
        default=90.0,
        gt=0.0,
        description="Clamp lag search to [-max,+max] (intersected with valid overlap) and "
        "reject if |lag| exceeds this after refinement. Null disables both.",
    )
    motion_weighted_sync: bool = Field(
        default=True,
        description="Weight lag correlation by foot-speed activity so quiet overlap does not dominate.",
    )
    motion_weight_floor_quantile: float = Field(
        default=0.12,
        ge=0.0,
        le=0.45,
        description="Per-overlap-window quantile subtracted from min(L,R) foot speed for weights.",
    )


class RateConfig(BaseModel):
    """Nominal sample rates for video and mocap streams."""

    video: float | None
    mocap: float | None


class MotionSyncConfig(BaseModel):
    """Root config loaded from ``configs/motion_sync.yaml``."""

    paths: PathsConfig
    rate: RateConfig
    rigid_body_solver: RigidBodySolverConfig
    time_sync_solver: TimeSyncSolverConfig
    bodies: dict[str, RigidBodyConfig]


def load_config(path: Path) -> MotionSyncConfig:
    """Load and validate a motion-sync YAML config file.

    Args:
        path: Path to a ``motion_sync.yaml`` (or compatible) file.

    Returns:
        Validated :class:`MotionSyncConfig` instance.
    """
    with open(path) as file:
        data = yaml.safe_load(file)

    config = MotionSyncConfig.model_validate(data)
    return config
