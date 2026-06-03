"""Human-friendly access to multi-source ``synced.npz`` clips.

The on-disk layout is column-oriented (SoA) for compression. :class:`SyncClip` presents
the same data as named rigid bodies, marker trajectories, and a video/SMPL stream with
utility methods for indexing, masking, and resampling.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, Literal, Self, TypeAlias, TypeVar, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from motion_sync import _storage
from motion_sync.contact_layer import ContactLayer, decode_contact_layers, encode_contact_layers
from motion_sync.contact_metadata import (
    contact_layer_is_fresh,
    stamp_detection_metadata,
    warn_if_stale_contact_layer,
)
from motion_sync.contact_registration import (
    ContactSchema,
    ContactType,
    merge_registered_contacts,
)
from motion_sync.mocap_schema import MocapSchema, validate_body_enum
from motion_sync.session import ClipSession, apply_clip_registration
from motion_sync.types import BoolArray, FloatArray, as_float_array
from motion_sync.video_schema import VideoSchema

MarkerRef: TypeAlias = str
BodyT = TypeVar("BodyT", bound=StrEnum)
JointT = TypeVar("JointT", bound=StrEnum)


class AxisConvention(StrEnum):
    """World-frame axis convention for a motion stream.

    Attributes:
        Z_UP_RIGHT_HANDED (str): Z-up, right-handed (typical mocap).
        Y_UP_RIGHT_HANDED (str): Y-up, right-handed (typical video/SMPL).
    """

    Z_UP_RIGHT_HANDED = "z_up_right_handed"
    Y_UP_RIGHT_HANDED = "y_up_right_handed"


class QuaternionOrder(StrEnum):
    """Component order for length-4 orientation arrays.

    Attributes:
        WXYZ (str): Scalar-first ``(w, x, y, z)``.
        XYZW (str): Scalar-last ``(x, y, z, w)``.
    """

    WXYZ = "wxyz"
    XYZW = "xyzw"


class SyncMetadata(BaseModel):
    """How Vicon was aligned to the video clock when the clip was built.

    Attributes:
        lag_s (float): Shift applied so video clock equals Vicon time minus lag (seconds).
        correlation (float | None): Foot-speed cross-correlation at the chosen lag, if stored.
        source_path (Path | None): Synced export path when the clip was loaded from disk.
    """

    lag_s: float = Field(description="Applied as t_video = t_vicon - lag_s.")
    correlation: float | None = None
    source_path: Path | None = None

    @property
    def lag(self) -> float:
        """Alias for :attr:`lag_s` (seconds).

        Returns:
            Lag in seconds (same as :attr:`lag_s`).
        """
        return self.lag_s


class RigidBodyPose(BaseModel):
    """Position and orientation for one rigid body at a single frame.

    Attributes:
        position (FloatArray): Translation ``(3,)`` in the stream's world frame.
        orientation (FloatArray | None): Unit quaternion ``(4,)``, or ``None`` if absent.
        quaternion_order (QuaternionOrder): Component order of :attr:`orientation`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: FloatArray
    orientation: FloatArray | None = None
    quaternion_order: QuaternionOrder = QuaternionOrder.XYZW

    @field_validator("position", mode="before")
    @classmethod
    def _position(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="position")
        if arr.shape != (3,):
            raise ValueError("position must have shape (3,)")
        return arr

    @field_validator("orientation", mode="before")
    @classmethod
    def _orientation(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, name="orientation")
        if arr.shape != (4,):
            raise ValueError("orientation must have shape (4,)")
        return arr

    @property
    def is_finite(self) -> bool:
        """True if position and orientation (if present) contain no NaN/Inf.

        Returns:
            Whether the pose is fully finite.
        """
        if not np.all(np.isfinite(self.position)):
            return False
        if self.orientation is None:
            return True
        return bool(np.all(np.isfinite(self.orientation)))


class JointTrack(BaseModel):
    """World positions for one SMPL-X / video joint over time (Y-up by default).

    Attributes:
        name (str): Logical joint name (enum value or index label).
        positions (FloatArray): ``(frames, 3)`` trajectory in the video stream frame.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    positions: FloatArray

    @field_validator("positions", mode="before")
    @classmethod
    def _positions(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="positions")
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("positions must have shape (frames, 3)")
        return arr

    @property
    def frame_count(self) -> int:
        """Number of frames in :attr:`positions`.

        Returns:
            Length of the time axis.
        """
        return int(self.positions.shape[0])

    def speeds_at_times(self, time_s: FloatArray) -> FloatArray:
        """Compute per-frame scalar speed from positions and video-clock times.

        Args:
            time_s: Monotonic times with shape ``(frames,)`` (seconds).

        Returns:
            Speed in m/s with shape ``(frames,)``; NaN where differentiation is invalid.
        """
        return scalar_speed_from_positions(self.positions, time_s)


class RigidBodyTrack(BaseModel):
    """World positions (and optional orientations) for one rigid body over time.

    Attributes:
        name (str): Vicon subject / rigid-body name.
        positions (FloatArray): ``(frames, 3)`` translations.
        orientations (FloatArray | None): ``(frames, 4)`` quaternions, or ``None``.
        quaternion_order (QuaternionOrder): Layout of rows in :attr:`orientations`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    positions: FloatArray
    orientations: FloatArray | None = None
    quaternion_order: QuaternionOrder = QuaternionOrder.XYZW

    @field_validator("positions", mode="before")
    @classmethod
    def _positions(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="positions")
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("positions must have shape (frames, 3)")
        return arr

    @field_validator("orientations", mode="before")
    @classmethod
    def _orientations(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, name="orientations")
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("orientations must have shape (frames, 4)")
        return arr

    @model_validator(mode="after")
    def _matching_lengths(self) -> Self:
        if self.orientations is not None and self.orientations.shape[0] != self.positions.shape[0]:
            raise ValueError("orientations frame count must match positions")
        return self

    @property
    def frame_count(self) -> int:
        """Number of frames in :attr:`positions`.

        Returns:
            Length of the time axis.
        """
        return int(self.positions.shape[0])

    def finite_mask(self) -> BoolArray:
        """Build a per-frame finiteness mask.

        Returns:
            Boolean array of shape ``(frames,)``; True where position and orientation
            (if present) are finite.
        """
        ok = np.isfinite(self.positions).all(axis=1)
        if self.orientations is not None:
            ok &= np.isfinite(self.orientations).all(axis=1)
        return ok

    def speeds(self) -> FloatArray:
        """Estimate per-frame scalar speed using unit time steps.

        Uses ``dt = 1`` between frames; scale externally if you need true m/s from
        mocap times. NaN where consecutive frames are not both finite.

        Returns:
            Speed array of shape ``(frames,)``.
        """
        n = self.frame_count
        out = np.full(n, np.nan, dtype=np.float64)
        if n < 2:
            return out
        dt = 1.0  # caller should scale by actual Δt if needed
        dpos = np.diff(self.positions, axis=0)
        step = np.linalg.norm(dpos, axis=1)
        finite = self.finite_mask()
        valid_step = finite[:-1] & finite[1:]
        out[1:][valid_step] = step[valid_step] / dt
        if n > 1 and finite[0] and finite[1]:
            out[0] = out[1]
        return out

    def speeds_at_times(self, time_s: FloatArray) -> FloatArray:
        """Compute per-frame speed (m/s) using ``time_s`` for differentiation.

        Args:
            time_s: Times with shape ``(frames,)`` (seconds).

        Returns:
            Speed array of shape ``(frames,)``.
        """
        return scalar_speed_from_positions(self.positions, time_s, finite_mask=self.finite_mask())

    def position_at(self, frame: int) -> FloatArray:
        """Return translation at one frame.

        Args:
            frame: Frame index.

        Returns:
            Position with shape ``(3,)``.
        """
        return np.asarray(self.positions[frame], dtype=np.float64)

    def orientation_at(self, frame: int) -> FloatArray | None:
        """Return orientation at one frame.

        Args:
            frame: Frame index.

        Returns:
            Quaternion with shape ``(4,)``, or ``None`` if :attr:`orientations` is absent.
        """
        if self.orientations is None:
            return None
        return np.asarray(self.orientations[frame], dtype=np.float64)

    def pose_at(self, frame: int) -> RigidBodyPose:
        """Bundle position and orientation at one frame.

        Args:
            frame: Frame index.

        Returns:
            :class:`RigidBodyPose` for this frame.
        """
        return RigidBodyPose(
            position=self.position_at(frame),
            orientation=self.orientation_at(frame),
            quaternion_order=self.quaternion_order,
        )


class MarkerTracks(BaseModel):
    """Named OptiTrack markers sampled on the clip timeline.

    Attributes:
        names (tuple[str, ...]): Unique Vicon marker strings.
        positions (FloatArray): ``(frames, markers, 3)`` world positions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    names: tuple[str, ...]
    positions: FloatArray

    @field_validator("positions", mode="before")
    @classmethod
    def _positions(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="positions")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("positions must have shape (frames, markers, 3)")
        return arr

    @model_validator(mode="after")
    def _names_match_markers(self) -> Self:
        if len(self.names) != self.positions.shape[1]:
            raise ValueError("len(names) must match positions.shape[1]")
        if len(set(self.names)) != len(self.names):
            raise ValueError("marker names must be unique")
        return self

    @property
    def frame_count(self) -> int:
        """Number of frames along the time axis.

        Returns:
            Length of axis 0 of :attr:`positions`.
        """
        return int(self.positions.shape[0])

    @property
    def marker_count(self) -> int:
        """Number of named markers.

        Returns:
            ``len(names)``.
        """
        return len(self.names)

    def index(self, name: str) -> int:
        """Look up a marker's column index.

        Args:
            name: Vicon marker string.

        Returns:
            Index into axis 1 of :attr:`positions`.

        Raises:
            KeyError: If ``name`` is not in :attr:`names`.
        """
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown marker {name!r}") from exc

    def marker(self, name: MarkerRef) -> FloatArray:
        """Return one marker trajectory.

        Args:
            name: Marker name.

        Returns:
            Positions with shape ``(frames, 3)``.

        Raises:
            KeyError: If ``name`` is unknown.
        """
        return self.positions[:, self.index(name), :]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.names

    def __iter__(self) -> Iterator[tuple[str, FloatArray]]:
        for name in self.names:
            yield name, self.marker(name)

    def markers(self) -> dict[str, FloatArray]:
        """Map each marker name to its ``(frames, 3)`` trajectory.

        Returns:
            Dict keyed by :attr:`names`.
        """
        return {name: self.marker(name) for name in self.names}

    def position_at(self, name: MarkerRef, frame: int) -> FloatArray:
        """Return one marker's XYZ at a single frame.

        Args:
            name: Marker name.
            frame: Frame index.

        Returns:
            Position with shape ``(3,)``.
        """
        return np.asarray(self.marker(name)[frame], dtype=np.float64)


class ViconMocap(BaseModel):
    """Vicon rigid bodies (and optional markers) on the synced video-clock timeline.

    Attributes:
        body_names (tuple[str, ...]): Vicon subject names (one per rigid body).
        body_positions (FloatArray): ``(frames, bodies, 3)`` translations (Z-up default).
        body_orientations (FloatArray | None): ``(frames, bodies, 4)`` quaternions (xyzw on disk).
        markers (MarkerTracks | None): OptiTrack markers resampled to the clip, if present.
        frame (AxisConvention): World axis convention for positions.
        quaternion_order (QuaternionOrder): Layout of :attr:`body_orientations` rows.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    body_names: tuple[str, ...]
    body_positions: FloatArray
    body_orientations: FloatArray | None = None
    markers: MarkerTracks | None = None
    frame: AxisConvention = AxisConvention.Z_UP_RIGHT_HANDED
    quaternion_order: QuaternionOrder = QuaternionOrder.XYZW

    @field_validator("body_positions", mode="before")
    @classmethod
    def _body_positions(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="body_positions")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("body_positions must have shape (frames, bodies, 3)")
        return arr

    @field_validator("body_orientations", mode="before")
    @classmethod
    def _body_orientations(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, name="body_orientations")
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError("body_orientations must have shape (frames, bodies, 4)")
        return arr

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        n_bodies = len(self.body_names)
        if self.body_positions.shape[1] != n_bodies:
            raise ValueError("body_names length must match body_positions.shape[1]")
        if len(set(self.body_names)) != n_bodies:
            raise ValueError("body_names must be unique")
        if self.body_orientations is not None and self.body_orientations.shape[:2] != self.body_positions.shape[:2]:
            raise ValueError("body_orientations shape must match body_positions on (frames, bodies)")
        if self.markers is not None and self.markers.frame_count != self.body_positions.shape[0]:
            raise ValueError("markers frame count must match body_positions")
        return self

    @property
    def frame_count(self) -> int:
        """Number of synced frames.

        Returns:
            Length of the time axis in :attr:`body_positions`.
        """
        return int(self.body_positions.shape[0])

    def body_index(self, name: str) -> int:
        """Look up a rigid body's column index.

        Args:
            name: Vicon subject string.

        Returns:
            Index into axis 1 of :attr:`body_positions`.

        Raises:
            KeyError: If ``name`` is not in :attr:`body_names`.
        """
        try:
            return self.body_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown Vicon body {name!r}; have {self.body_names}") from exc

    def has_body(self, name: str) -> bool:
        """Return whether a rigid body name is present.

        Args:
            name: Vicon subject string.

        Returns:
            True if ``name`` is in :attr:`body_names`.
        """
        return name in self.body_names

    def body(self, name: str) -> RigidBodyTrack:
        """Return one rigid body's pose track.

        Args:
            name: Vicon subject string.

        Returns:
            :class:`RigidBodyTrack` view sharing underlying arrays.

        Raises:
            KeyError: If ``name`` is unknown.
        """
        i = self.body_index(name)
        orient = None if self.body_orientations is None else self.body_orientations[:, i, :]
        return RigidBodyTrack(
            name=name,
            positions=self.body_positions[:, i, :],
            orientations=orient,
            quaternion_order=self.quaternion_order,
        )

    def bodies(self) -> dict[str, RigidBodyTrack]:
        """Map each body name to its :class:`RigidBodyTrack`.

        Returns:
            Dict keyed by :attr:`body_names`.
        """
        return {name: self.body(name) for name in self.body_names}

    def foot_speeds(
        self,
        left_name: str,
        right_name: str,
        *,
        time_s: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute scalar foot speeds for time-sync diagnostics.

        Args:
            left_name: Vicon name of the left foot / shoe body.
            right_name: Vicon name of the right foot / shoe body.
            time_s: Optional video-clock times; uses unit steps if ``None``.

        Returns:
            ``(left_speeds, right_speeds)`` each with shape ``(frames,)`` in m/s when
            ``time_s`` is provided.

        Raises:
            KeyError: If a body name is unknown.
        """
        if time_s is None:
            return self.body(left_name).speeds(), self.body(right_name).speeds()
        return (
            self.body(left_name).speeds_at_times(time_s),
            self.body(right_name).speeds_at_times(time_s),
        )


class VideoSmplx(BaseModel):
    """GVHMR / SMPL-X streams resampled onto the synced timeline (Y-up).

    Attributes:
        joints (FloatArray): ``(frames, J, 3)`` FK joint positions.
        transl (FloatArray): ``(frames, 3)`` root translation.
        global_orient (FloatArray): ``(frames, 3)`` root orientation (axis-angle).
        body_pose (FloatArray): ``(frames, 63)`` body pose parameters.
        betas (FloatArray): Shape parameters (per frame or broadcast).
        vertices (FloatArray | None): ``(frames, V, 3)`` mesh vertices when FK was run.
        frame (AxisConvention): World axis convention (Y-up for GVHMR).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    joints: FloatArray
    transl: FloatArray
    global_orient: FloatArray
    body_pose: FloatArray
    betas: FloatArray
    vertices: FloatArray | None = None
    frame: AxisConvention = AxisConvention.Y_UP_RIGHT_HANDED

    @field_validator("joints", mode="before")
    @classmethod
    def _joints(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="joints")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("joints must have shape (frames, J, 3)")
        return arr

    @field_validator("transl", "global_orient", mode="before")
    @classmethod
    def _vec3(cls, value: Any, info: ValidationInfo) -> FloatArray:
        arr = as_float_array(value, name=str(info.field_name))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{info.field_name} must have shape (frames, 3)")
        return arr

    @field_validator("body_pose", mode="before")
    @classmethod
    def _body_pose(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="body_pose")
        if arr.ndim != 2:
            raise ValueError("body_pose must have shape (frames, 63)")
        return arr

    @field_validator("betas", mode="before")
    @classmethod
    def _betas(cls, value: Any) -> FloatArray:
        return as_float_array(value, name="betas")

    @field_validator("vertices", mode="before")
    @classmethod
    def _vertices(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, name="vertices")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("vertices must have shape (frames, V, 3)")
        return arr

    @model_validator(mode="after")
    def _frame_counts(self) -> Self:
        n = self.joints.shape[0]
        for field in ("transl", "global_orient", "body_pose", "betas"):
            arr = getattr(self, field)
            if arr.shape[0] != n:
                raise ValueError(f"{field} frame count must match joints")
        if self.vertices is not None and self.vertices.shape[0] != n:
            raise ValueError("vertices frame count must match joints")
        return self

    @property
    def frame_count(self) -> int:
        """Number of synced frames.

        Returns:
            Length of axis 0 of :attr:`joints`.
        """
        return int(self.joints.shape[0])

    @property
    def joint_count(self) -> int:
        """Number of FK joints in :attr:`joints`.

        Returns:
            Size of axis 1 of :attr:`joints`.
        """
        return int(self.joints.shape[1])

    def joint(self, index: int) -> FloatArray:
        """Return positions for one SMPL-X FK joint column.

        Args:
            index: Column index in :attr:`joints`.

        Returns:
            Trajectory with shape ``(frames, 3)``.
        """
        return self.joints[:, index, :]


class SyncClip(BaseModel, Generic[BodyT]):
    """One demo: Vicon + video/SMPL-X on a shared video-clock timeline.

    Register a :class:`~motion_sync.mocap_schema.MocapSchema` with one marker :class:`StrEnum`
    per body (e.g. ``LeftShoeMarkers.HEEL`` and ``RightShoeMarkers.HEEL``) via
    :meth:`register_mocap`, or pass a :class:`~motion_sync.session.ClipSession` to
    :meth:`load`.

    Attributes:
        name (str): Demo identifier (often the parent directory name).
        time_s (FloatArray): Video-clock times ``(frames,)`` in seconds.
        vicon (ViconMocap): Vicon bodies and optional markers on the synced timeline.
        video (VideoSmplx): GVHMR / SMPL-X streams resampled to ``time_s``.
        metadata (SyncMetadata): Lag and correlation from time sync.
        valid (BoolArray | None): Optional per-frame validity mask from the sync crop.
        registered_bodies (type[BodyT] | None): User body enum after registration (runtime).
        registered_body_marker_enums (dict[str, type[StrEnum]] | None): Body value → marker enum.
        body_marker_map (dict[str, tuple[str, ...]] | None): Body value → marker value names.
        contact_layers (dict[str, ContactLayer]): Persisted contact layers keyed by id.
        registered_contacts (dict[str, ContactType[Any, Any]] | None): Types for :meth:`contact`.
        registered_video (VideoSchema[Any] | None): Schema for :meth:`joint` (runtime).
        video_joint_map (dict[str, int] | None): Joint enum value → FK column index.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    name: str = ""
    time_s: FloatArray
    vicon: ViconMocap
    video: VideoSmplx
    metadata: SyncMetadata
    valid: BoolArray | None = None
    registered_bodies: type[BodyT] | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="User StrEnum set via register_bodies/register_mocap; required before body().",
    )
    registered_body_marker_enums: dict[str, type[StrEnum]] | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="body value → marker StrEnum class; set via register_mocap().",
    )
    body_marker_map: dict[str, tuple[str, ...]] | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="body value → marker values; set via register_mocap().",
    )
    contact_layers: dict[str, ContactLayer] = Field(
        default_factory=dict,
        description="Named contact/support annotations aligned to time_s.",
    )
    registered_contacts: dict[str, ContactType[Any, Any]] | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Contact types from register_contacts(); keyed by layer_id; not persisted.",
    )
    registered_video: VideoSchema[Any] | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Video joint schema from register_video(); required before joint().",
    )
    video_joint_map: dict[str, int] | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Logical joint value → index in video.joints; set via register_video().",
    )

    @field_validator("time_s", mode="before")
    @classmethod
    def _time(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="time_s")
        if arr.ndim != 1:
            raise ValueError("time_s must be 1-D")
        return arr

    @model_validator(mode="after")
    def _aligned_lengths(self) -> Self:
        n = self.frame_count
        if self.vicon.frame_count != n or self.video.frame_count != n:
            raise ValueError("vicon, video, and time_s must share the same frame count")
        if self.valid is not None and self.valid.shape != (n,):
            raise ValueError("valid mask must have shape (frames,)")
        for layer in self.contact_layers.values():
            layer.validate_frame_count(n)
        return self

    @property
    def frame_count(self) -> int:
        """Number of frames on the synced timeline.

        Returns:
            Length of :attr:`time_s`.
        """
        return int(self.time_s.shape[0])

    @property
    def duration_s(self) -> float:
        """Span of finite :attr:`time_s` values.

        Returns:
            ``time_s[-1] - time_s[0]`` over finite samples, or ``0.0`` if empty.
        """
        if self.frame_count == 0:
            return 0.0
        t = self.time_s[np.isfinite(self.time_s)]
        if t.size == 0:
            return 0.0
        return float(t[-1] - t[0])

    def mean_fps(self) -> float:
        """Estimate median sample rate from :attr:`time_s`.

        Returns:
            ``1 / median(dt)`` over positive finite differences, or NaN if undefined.
        """
        if self.frame_count < 2:
            return float("nan")
        dt = np.diff(self.time_s)
        dt = dt[np.isfinite(dt) & (dt > 1e-9)]
        if dt.size == 0:
            return float("nan")
        return float(1.0 / np.median(dt))

    def finite_time_mask(self) -> BoolArray:
        """Mask frames with finite :attr:`time_s`.

        Returns:
            Boolean array of shape ``(frames,)``.
        """
        return np.isfinite(self.time_s)

    def keep_valid_frames(self) -> SyncClip[BodyT]:
        """Return a copy retaining only valid frames.

        Uses :attr:`valid` when set; otherwise keeps frames with finite :attr:`time_s`.

        Returns:
            New :class:`SyncClip` with sliced arrays and contact layers.
        """
        mask = (
            np.asarray(self.valid, dtype=bool)
            if self.valid is not None
            else self.finite_time_mask()
        )
        return self._subset(mask)

    @property
    def body_names(self) -> tuple[str, ...]:
        """Rigid-body names in this clip.

        Returns:
            Vicon subject strings (e.g. ``Left_Shoe``, ``Skateboard``).
        """
        return self.vicon.body_names

    @property
    def markers(self) -> MarkerTracks | None:
        """OptiTrack marker cloud when loaded.

        Returns:
            :class:`MarkerTracks`, or ``None`` if the synced file has no marker channel.
        """
        return self.vicon.markers

    @property
    def marker_names(self) -> tuple[str, ...]:
        """All marker names on the clip.

        Returns:
            Names from :attr:`markers`, or an empty tuple if markers are absent.
        """
        if self.vicon.markers is None:
            return ()
        return self.vicon.markers.names

    def _require_registered_bodies(self) -> type[BodyT]:
        if self.registered_bodies is None:
            raise RuntimeError(
                "Call register_bodies(YourBodiesEnum) before using body()."
            )
        return self.registered_bodies

    def _body_name_from_ref(self, ref: BodyT) -> str:
        enum_cls = self._require_registered_bodies()
        if not isinstance(ref, enum_cls):
            raise TypeError(
                f"Expected a member of {enum_cls.__name__}, got {type(ref).__name__}"
            )
        return ref.value

    def register_bodies(self, body_enum: type[BodyT]) -> SyncClip[BodyT]:
        """Attach a user-defined body :class:`StrEnum` for typed :meth:`body` access.

        Args:
            body_enum: Enum whose values match :attr:`body_names` exactly.

        Returns:
            ``self`` (for chaining).

        Raises:
            ValueError: If enum values do not match clip bodies.
            TypeError: If ``body_enum`` is not a :class:`StrEnum` subclass.
        """
        validate_body_enum(body_enum, self.body_names)
        self.registered_bodies = body_enum
        return cast(SyncClip[BodyT], self)

    def register_mocap(self, schema: MocapSchema[BodyT]) -> SyncClip[BodyT]:
        """Register bodies and one marker enum per body.

        Args:
            schema: :class:`~motion_sync.mocap_schema.MocapSchema` for this project.

        Returns:
            ``self`` with :attr:`body_marker_map` populated.

        Raises:
            ValueError: If the clip has no markers or schema validation fails.
        """
        if self.vicon.markers is None:
            raise ValueError(
                "clip has no marker tracks; load with vicon_mocap= so marker_names are available"
            )
        self.body_marker_map = schema.validate_against_clip(
            self.body_names,
            self.marker_names,
        )
        self.registered_bodies = schema.bodies
        self.registered_body_marker_enums = {
            body.value: schema.body_markers[body] for body in schema.bodies
        }
        return cast(SyncClip[BodyT], self)

    def has_body(self, ref: BodyT) -> bool:
        """Return whether a registered body is present on the clip.

        Args:
            ref: Member of the registered body enum.

        Returns:
            True if the body's Vicon name exists.

        Raises:
            RuntimeError: If bodies were not registered.
            TypeError: If ``ref`` is not an enum member.
        """
        return self.vicon.has_body(self._body_name_from_ref(ref))

    def body(self, ref: BodyT) -> RigidBodyTrack:
        """Return a rigid-body track for one registered enum member.

        Args:
            ref: Member of the enum passed to :meth:`register_bodies` or :meth:`register_mocap`.

        Returns:
            :class:`RigidBodyTrack` for that body.

        Raises:
            RuntimeError: If bodies were not registered.
            KeyError: If the body is missing from the clip.
            TypeError: If ``ref`` is not an enum member.
        """
        return self.vicon.body(self._body_name_from_ref(ref))

    def bodies(self) -> dict[BodyT, RigidBodyTrack]:
        """Map every registered body enum member to its track.

        Returns:
            Dict keyed by body enum members.

        Raises:
            RuntimeError: If bodies were not registered.
        """
        return {member: self.body(member) for member in self._require_registered_bodies()}

    def _require_registered_video(self) -> VideoSchema[Any]:
        if self.registered_video is None:
            raise RuntimeError(
                "Call register_video(VideoSchema(...)) before using joint()."
            )
        return self.registered_video

    def register_video(self, schema: VideoSchema[JointT]) -> SyncClip[BodyT]:
        """Register logical SMPL-X joints for :meth:`joint` and :meth:`core_joint_positions`.

        Args:
            schema: :class:`~motion_sync.video_schema.VideoSchema` for this project.

        Returns:
            New clip copy with :attr:`registered_video` and :attr:`video_joint_map` set.

        Raises:
            ValueError: If FK indices are out of range for this clip.
        """
        joint_map = schema.validate_against_clip(self.video.joint_count)
        return self.model_copy(
            update={
                "registered_video": schema,
                "video_joint_map": joint_map,
            }
        )

    def joint(self, ref: JointT) -> JointTrack:
        """Return trajectory for one registered video joint.

        Args:
            ref: Member of the joint enum in :attr:`registered_video`.

        Returns:
            :class:`JointTrack` with Y-up positions, shape ``(frames, 3)``.

        Raises:
            RuntimeError: If video was not registered.
            TypeError: If ``ref`` is not a joint enum member.
        """
        schema = self._require_registered_video()
        if not isinstance(ref, schema.joints):
            raise TypeError(
                f"Expected a member of {schema.joints.__name__}, got {type(ref).__name__}"
            )
        if self.video_joint_map is None:
            raise RuntimeError("video_joint_map missing; call register_video() first.")
        idx = self.video_joint_map[ref.value]
        return JointTrack(
            name=ref.value,
            positions=np.asarray(self.video.joints[:, idx, :], dtype=np.float64),
        )

    def core_joint_positions(self) -> FloatArray:
        """Stack registered core joints in schema order.

        Returns:
            Array with shape ``(frames, J, 3)`` in the video stream frame convention.

        Raises:
            RuntimeError: If video was not registered.
        """
        schema = self._require_registered_video()
        if self.video_joint_map is None:
            raise RuntimeError("video_joint_map missing; call register_video() first.")
        indices = [self.video_joint_map[m.value] for m in schema.joint_members()]
        return np.asarray(self.video.joints[:, indices, :], dtype=np.float64)

    def foot_speeds(
        self,
        left: BodyT,
        right: BodyT,
        *,
        time_s: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute foot speeds for two registered bodies.

        Args:
            left: Left shoe (or foot) body enum member.
            right: Right shoe body enum member.
            time_s: Times for differentiation; defaults to :attr:`time_s`.

        Returns:
            ``(left_speeds, right_speeds)`` in m/s, each shape ``(frames,)``.

        Raises:
            RuntimeError: If bodies were not registered.
        """
        if time_s is None:
            time_s = self.time_s
        return (
            self.body(left).speeds_at_times(time_s),
            self.body(right).speeds_at_times(time_s),
        )

    def _require_body_marker_enums(self) -> dict[str, type[StrEnum]]:
        if self.registered_body_marker_enums is None:
            raise RuntimeError(
                "Call register_mocap(MocapSchema(...)) before using marker accessors."
            )
        return self.registered_body_marker_enums

    def _marker_enum_for_body(self, body: BodyT) -> type[StrEnum]:
        body_value = self._body_name_from_ref(body)
        try:
            return self._require_body_marker_enums()[body_value]
        except KeyError as exc:
            raise KeyError(f"no marker enum registered for body {body_value!r}") from exc

    def _marker_name_from_ref(self, ref: StrEnum) -> str:
        for enum_cls in self._require_body_marker_enums().values():
            if isinstance(ref, enum_cls):
                return ref.value
        raise TypeError(
            f"{type(ref).__name__} is not a registered marker enum member "
            f"(expected one of: {', '.join(c.__name__ for c in self._require_body_marker_enums().values())})"
        )

    def marker(self, ref: StrEnum | MarkerRef) -> FloatArray:
        """Return one marker trajectory.

        Args:
            ref: Per-body marker enum member, or raw Vicon marker string if only bodies
                were registered.

        Returns:
            Positions with shape ``(frames, 3)``.

        Raises:
            ValueError: If the clip has no marker channel.
            RuntimeError: If an enum member is used without :meth:`register_mocap`.
            KeyError: If the marker name is unknown.
            TypeError: If ``ref`` has the wrong type.
        """
        if self.vicon.markers is None:
            raise ValueError("clip has no marker tracks; load with vicon_mocap= for names")
        if self.registered_body_marker_enums is not None:
            if isinstance(ref, StrEnum):
                name = self._marker_name_from_ref(ref)
            elif isinstance(ref, str):
                name = ref
            else:
                raise TypeError(f"Expected StrEnum marker member or str, got {type(ref).__name__}")
        else:
            if not isinstance(ref, str):
                raise RuntimeError(
                    "Call register_mocap(MocapSchema(...)) before marker() with an enum member."
                )
            name = ref
        return self.vicon.markers.marker(name)

    def marker_members_for_body(self, body: BodyT) -> tuple[StrEnum, ...]:
        """List marker enum members registered for one body.

        Args:
            body: Registered body enum member.

        Returns:
            Tuple of marker enum members for that body.

        Raises:
            RuntimeError: If mocap was not registered.
        """
        return tuple(self._marker_enum_for_body(body))

    def markers_for_body(self, body: BodyT) -> dict[StrEnum, FloatArray]:
        """Map marker enum members to trajectories for one body.

        Args:
            body: Registered body enum member.

        Returns:
            Dict keyed by that body's marker enum members.

        Raises:
            RuntimeError: If mocap was not registered.
        """
        return {member: self.marker(member) for member in self.marker_members_for_body(body)}

    def all_markers_visible_mask(
        self,
        body: BodyT,
        *,
        require_body: bool = False,
    ) -> BoolArray:
        """Build a mask where every marker on ``body`` has finite XYZ.

        Args:
            body: Registered body enum member.
            require_body: If True, also require finite rigid-body position.

        Returns:
            Boolean array of shape ``(frames,)``.

        Raises:
            RuntimeError: If mocap was not registered.
            ValueError: If markers are missing or the body has no markers in the schema.
        """
        if self.body_marker_map is None:
            raise RuntimeError("Call register_mocap(MocapSchema(...)) first.")
        if self.vicon.markers is None:
            raise ValueError("clip has no marker tracks")

        by_marker = self.markers_for_body(body)
        if not by_marker:
            raise ValueError(f"body {self._body_name_from_ref(body)!r} has no markers in the schema")

        n = self.frame_count
        visible = np.ones(n, dtype=bool)
        for traj in by_marker.values():
            visible &= np.isfinite(traj).all(axis=1)
        if require_body:
            visible &= np.isfinite(self.body(body).positions).all(axis=1)
        return visible

    def find_frame_all_markers_visible(
        self,
        body: BodyT,
        *,
        strategy: AllMarkersVisibleStrategy = "middle",
        require_body: bool = False,
    ) -> int:
        """Pick a frame index where every marker on ``body`` is visible.

        Args:
            body: Registered body enum member.
            strategy: Among qualifying frames, use ``first``, ``middle``, or ``last``.
            require_body: Forwarded to :meth:`all_markers_visible_mask`.

        Returns:
            Frame index.

        Raises:
            ValueError: If no frame has all markers visible (message includes best partial count).
        """
        mask = self.all_markers_visible_mask(body, require_body=require_body)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            by_marker = self.markers_for_body(body)
            per_frame = np.zeros(self.frame_count, dtype=int)
            for traj in by_marker.values():
                per_frame += np.isfinite(traj).all(axis=1).astype(int)
            best = int(per_frame.max())
            best_frames = np.flatnonzero(per_frame == best)
            raise ValueError(
                f"No frame has all {len(by_marker)} markers visible for "
                f"{self._body_name_from_ref(body)!r}; best was {best}/{len(by_marker)} "
                f"at frame(s) {best_frames[:5].tolist()}"
                f"{'...' if best_frames.size > 5 else ''}"
            )
        if strategy == "first":
            return int(indices[0])
        if strategy == "last":
            return int(indices[-1])
        return _center_of_longest_true_run(mask)

    def plot_body_markers(
        self,
        body: BodyT,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot this body and its markers in 3D (delegates to :func:`~motion_sync.body_marker_plot.plot_body_markers`).

        Args:
            body: Registered body enum member.
            **kwargs: Forwarded to :func:`~motion_sync.body_marker_plot.plot_body_markers`.

        Returns:
            ``(fig, ax)`` matplotlib handles.
        """
        from motion_sync.body_marker_plot import plot_body_markers

        return plot_body_markers(self, body, **kwargs)

    def has_contact(self, contact_type: ContactType[Any, Any]) -> bool:
        """Return whether a contact layer is stored on disk for this type.

        Args:
            contact_type: Registered contact type (uses :attr:`~ContactType.layer_id`).

        Returns:
            True if :attr:`contact_layers` contains the layer id.
        """
        return contact_type.layer_id in self.contact_layers

    def contact_is_fresh(self, contact_type: ContactType[Any, Any]) -> bool:
        """Return whether the stored layer matches this clip's detection metadata.

        Args:
            contact_type: Contact type to check.

        Returns:
            False if the layer is missing or stale.
        """
        if not self.has_contact(contact_type):
            return False
        return contact_layer_is_fresh(
            self.contact_layers[contact_type.layer_id],
            self,
        )

    def contact_layer(self, layer_id: str) -> ContactLayer:
        """Return a persisted contact layer by id.

        Args:
            layer_id: Storage key (no registration required).

        Returns:
            :class:`~motion_sync.contact_layer.ContactLayer`.

        Raises:
            KeyError: If ``layer_id`` is not in :attr:`contact_layers`.
        """
        try:
            return self.contact_layers[layer_id]
        except KeyError as exc:
            known = ", ".join(sorted(self.contact_layers)) or "(none)"
            raise KeyError(
                f"contact layer {layer_id!r} not found; known layers: {known}"
            ) from exc

    def _require_contact_type(self, contact_type: ContactType[Any, Any]) -> ContactType[Any, Any]:
        if self.registered_contacts is None:
            raise RuntimeError(
                f"Call register_contacts({contact_type.__class__.__name__}(...)) "
                f"before clip.contact(...)"
            )
        reg = self.registered_contacts.get(contact_type.layer_id)
        if reg is None:
            known = ", ".join(sorted(self.registered_contacts)) or "(none)"
            raise RuntimeError(
                f"contact type {contact_type.layer_id!r} is not registered; known: {known}"
            )
        return reg

    def contact(self, contact_type: ContactType[Any, Any]) -> Any:
        """Read a registered contact type as a typed view.

        Args:
            contact_type: Type previously passed to :meth:`register_contacts`.

        Returns:
            Project-specific view from :meth:`~ContactType.read`.

        Raises:
            RuntimeError: If the type was not registered.
            KeyError: If the layer is missing.
        """
        reg = self._require_contact_type(contact_type)
        layer = self.contact_layer(reg.layer_id)
        warn_if_stale_contact_layer(layer, self, layer_id=reg.layer_id)
        return reg.read(self, layer)

    def attach_contact(self, layer: ContactLayer) -> SyncClip[BodyT]:
        """Add or replace a contact layer on this clip.

        Args:
            layer: Layer aligned to :attr:`frame_count`.

        Returns:
            New clip copy with updated :attr:`contact_layers`.

        Raises:
            ValueError: If the layer length does not match the clip.
        """
        layer.validate_frame_count(self.frame_count)
        updated = dict(self.contact_layers)
        updated[layer.layer_id] = layer
        return self.model_copy(update={"contact_layers": updated})

    def register_contacts(
        self,
        schema_or_type: ContactSchema | ContactType[Any, Any],
        *more: ContactType[Any, Any],
    ) -> SyncClip[BodyT]:
        """Register contact types for :meth:`contact` and :meth:`detect`.

        Args:
            schema_or_type: :class:`~motion_sync.contact_registration.ContactSchema` or one
                :class:`~motion_sync.contact_registration.ContactType`.
            *more: Additional types when the first argument is a single type.

        Returns:
            New clip copy with :attr:`registered_contacts` merged.

        Raises:
            ValueError: If two types share the same ``layer_id``.
        """
        if isinstance(schema_or_type, ContactSchema):
            types = tuple(schema_or_type.types) + more
        else:
            types = (schema_or_type, *more)
        merged = merge_registered_contacts(self.registered_contacts, *types)
        return self.model_copy(update={"registered_contacts": merged})

    def detect(
        self,
        contact_type: ContactType[Any, Any],
        config: Any | None = None,
        *,
        attach: bool = True,
        force: bool = False,
    ) -> SyncClip[BodyT]:
        """Run a registered contact detector and optionally attach its layer.

        Skips work when a fresh layer is already attached unless ``force=True``.

        Args:
            contact_type: Registered type to run.
            config: Optional detector configuration.
            attach: If True, store the result in :attr:`contact_layers`.
            force: Re-run even when :meth:`contact_is_fresh` is True.

        Returns:
            Clip copy with layer attached when ``attach`` and detection ran.

        Raises:
            RuntimeError: If the type was not registered.
        """
        reg = self._require_contact_type(contact_type)
        if not force and self.has_contact(contact_type):
            existing = self.contact_layers[contact_type.layer_id]
            if contact_layer_is_fresh(existing, self):
                return self

        layer = stamp_detection_metadata(reg.detect(self, config), self, config=config)
        if not attach:
            return self
        return self.attach_contact(layer)

    def frame_index_at_time(self, time_s: float) -> int:
        """Find the nearest frame index for a video-clock time.

        Args:
            time_s: Time in seconds on the video clock.

        Returns:
            Frame index closest to ``time_s``.

        Raises:
            ValueError: If :attr:`time_s` is not monotonic on finite samples.
        """
        t = self.time_s
        if not np.all(np.diff(t[np.isfinite(t)]) >= -1e-9):
            raise ValueError("time_s must be monotonic for lookup")
        idx = int(np.searchsorted(t, time_s))
        if idx <= 0:
            return 0
        if idx >= len(t):
            return len(t) - 1
        if abs(t[idx] - time_s) < abs(time_s - t[idx - 1]):
            return idx
        return idx - 1

    @classmethod
    def load(
        cls,
        path: str | Path,
        name: str | None = None,
        *,
        mocap: MocapSchema[Any] | None = None,
        contacts: ContactSchema | ContactType[Any, Any] | None = None,
        video: VideoSchema[Any] | None = None,
        session: ClipSession[Any] | None = None,
    ) -> SyncClip:
        """Load a synced clip from a demo directory or ``synced.npz``.

        ``name`` defaults to the parent directory name (demo id). Marker names are
        attached automatically when the paired Vicon mocap export is present.

        Args:
            path: Demo folder or explicit NPZ path.
            name: Clip name override.
            mocap: Optional :class:`~motion_sync.mocap_schema.MocapSchema`.
            contacts: Optional contact schema or type.
            video: Optional :class:`~motion_sync.video_schema.VideoSchema`.
            session: Optional :class:`~motion_sync.session.ClipSession` (mutually exclusive
                with ``mocap`` / ``contacts`` / ``video``).

        Returns:
            Loaded :class:`SyncClip` with optional registration applied.

        Raises:
            ValueError: If ``session`` is combined with other registration kwargs.
            FileNotFoundError: If the path cannot be resolved.
            KeyError: If required NPZ keys are missing.
        """
        npz_path = _storage.resolve_synced_path(path)
        clip_name = name if name is not None else npz_path.parent.name
        vicon_npz = _storage.infer_vicon_mocap_for_synced(npz_path)
        clip = _storage.read_synced_clip(
            npz_path,
            name=clip_name,
            vicon_mocap=vicon_npz,
        )
        return apply_clip_registration(
            clip,
            mocap=mocap,
            contacts=contacts,
            video=video,
            session=session,
        )

    @classmethod
    def from_pipeline_aligned(
        cls,
        aligned: dict[str, Any],
        meta: dict[str, Any],
        *,
        name: str = "",
        path: Path | None = None,
    ) -> SyncClip:
        """Build a clip from in-memory sync pipeline output.

        Args:
            aligned: Column arrays from :func:`motion_sync.syncer.build_synced_dataset`.
            meta: Metadata dict (lag, correlation, …).
            name: Demo name stored on the clip.
            path: Optional source path recorded in :attr:`SyncMetadata.source_path`.

        Returns:
            New :class:`SyncClip` without registration.
        """
        return cls._from_storage(
            _storage.aligned_pipeline_to_storage_dict(aligned, meta),
            path=path,
            name=name,
        )

    @classmethod
    def _from_storage(
        cls,
        data: np.lib.npyio.NpzFile | dict[str, Any],
        *,
        path: Path | None = None,
        name: str = "",
    ) -> SyncClip:
        files = data.files if hasattr(data, "files") else list(data.keys())

        def get(key: str) -> Any:
            if key not in files:
                raise KeyError(f"synced dataset missing required key {key!r}")
            return data[key]  # type: ignore[index]

        time_s = as_float_array(get("t"), name="t")
        body_names = tuple(str(x) for x in get("vicon__body_names").tolist())

        vicon = ViconMocap(
            body_names=body_names,
            body_positions=get("vicon__body_pos"),
            body_orientations=get("vicon__body_quat") if "vicon__body_quat" in files else None,
            markers=_load_markers(data, files),
            frame=AxisConvention.Z_UP_RIGHT_HANDED,
            quaternion_order=QuaternionOrder.XYZW,
        )

        video = VideoSmplx(
            joints=get("video__joints"),
            transl=get("video__transl"),
            global_orient=get("video__global_orient"),
            body_pose=get("video__body_pose"),
            betas=get("video__betas"),
            vertices=get("video__vertices") if "video__vertices" in files else None,
        )

        lag = float(np.asarray(get("lag")).reshape(()))
        corr = None
        if "corr" in files:
            corr = float(np.asarray(get("corr")).reshape(()))

        valid = None
        if "valid" in files:
            valid = np.asarray(get("valid"), dtype=bool)

        meta = SyncMetadata(lag_s=lag, correlation=corr, source_path=path)

        def get_optional(key: str) -> Any:
            if key not in files:
                raise KeyError(key)
            return data[key]  # type: ignore[index]

        contact_layers = decode_contact_layers(files, get_optional)

        return cls(
            name=name,
            time_s=time_s,
            vicon=vicon,
            video=video,
            metadata=meta,
            valid=valid,
            contact_layers=contact_layers,
        )

    def export_vicon_bodies(
        self,
        *,
        zero_time: bool = True,
        apply_valid_mask: bool = True,
    ) -> tuple[FloatArray, tuple[str, ...], FloatArray]:
        """Export Vicon rigid-body arrays for legacy consumers.

        Args:
            zero_time: Subtract ``time_s[0]`` from the time axis.
            apply_valid_mask: Drop frames where :attr:`valid` is False when set.

        Returns:
            ``(time_s, body_names, positions)`` with positions shape ``(frames, bodies, 3)``.

        Raises:
            ValueError: If no frames remain after masking.
        """
        time_s = np.asarray(self.time_s, dtype=np.float64)
        positions = np.asarray(self.vicon.body_positions, dtype=np.float64)
        names = self.vicon.body_names
        if apply_valid_mask and self.valid is not None:
            mask = np.asarray(self.valid, dtype=bool)
            time_s = time_s[mask]
            positions = positions[mask]
        if time_s.size == 0:
            raise ValueError("clip contains no frames to export")
        if zero_time:
            time_s = time_s - float(time_s[0])
        return time_s, names, positions

    def _to_storage(self) -> dict[str, np.ndarray]:
        """Column-oriented on-disk layout (internal)."""
        out: dict[str, np.ndarray] = {
            "t": np.asarray(self.time_s, dtype=np.float64),
            "lag": np.array(self.metadata.lag_s),
            "vicon__body_names": np.array(self.vicon.body_names, dtype=object),
            "vicon__body_pos": np.asarray(self.vicon.body_positions, dtype=np.float64),
            "video__joints": np.asarray(self.video.joints, dtype=np.float64),
            "video__transl": np.asarray(self.video.transl, dtype=np.float64),
            "video__global_orient": np.asarray(self.video.global_orient, dtype=np.float64),
            "video__body_pose": np.asarray(self.video.body_pose, dtype=np.float64),
            "video__betas": np.asarray(self.video.betas, dtype=np.float64),
        }
        if self.metadata.correlation is not None:
            out["corr"] = np.array(self.metadata.correlation)
        if self.vicon.body_orientations is not None:
            out["vicon__body_quat"] = np.asarray(self.vicon.body_orientations, dtype=np.float64)
        if self.vicon.markers is not None:
            out["vicon__marker_pos"] = np.asarray(self.vicon.markers.positions, dtype=np.float64)
        if self.video.vertices is not None:
            out["video__vertices"] = np.asarray(self.video.vertices, dtype=np.float64)
        if self.valid is not None:
            out["valid"] = np.asarray(self.valid, dtype=bool)
        out.update(encode_contact_layers(self.contact_layers))
        return out

    def save(self, path: str | Path) -> Path:
        """Write the clip to ``synced.npz`` (demo directory or explicit file).

        Args:
            path: Output demo folder or NPZ file path.

        Returns:
            Resolved path of the written file.
        """
        return _storage.write_synced_clip(self, path)

    def _with_marker_names(self, names: tuple[str, ...]) -> SyncClip:
        if self.vicon.markers is None:
            return self
        markers = MarkerTracks(names=names, positions=self.vicon.markers.positions)
        vicon = self.vicon.model_copy(update={"markers": markers})
        return self.model_copy(update={"vicon": vicon})

    def _subset(self, mask: BoolArray) -> SyncClip[BodyT]:
        mask = np.asarray(mask, dtype=bool)
        m = MarkerTracks(
            names=self.vicon.markers.names,
            positions=self.vicon.markers.positions[mask],
        ) if self.vicon.markers is not None else None
        contact_layers = {
            layer_id: layer.subset_frames(mask)
            for layer_id, layer in self.contact_layers.items()
        }
        return SyncClip(
            name=self.name,
            time_s=self.time_s[mask],
            registered_bodies=self.registered_bodies,
            registered_body_marker_enums=self.registered_body_marker_enums,
            body_marker_map=self.body_marker_map,
            registered_contacts=self.registered_contacts,
            contact_layers=contact_layers,
            vicon=ViconMocap(
                body_names=self.vicon.body_names,
                body_positions=self.vicon.body_positions[mask],
                body_orientations=(
                    None
                    if self.vicon.body_orientations is None
                    else self.vicon.body_orientations[mask]
                ),
                markers=m,
                frame=self.vicon.frame,
                quaternion_order=self.vicon.quaternion_order,
            ),
            video=VideoSmplx(
                joints=self.video.joints[mask],
                transl=self.video.transl[mask],
                global_orient=self.video.global_orient[mask],
                body_pose=self.video.body_pose[mask],
                betas=self.video.betas[mask],
                vertices=None if self.video.vertices is None else self.video.vertices[mask],
                frame=self.video.frame,
            ),
            metadata=self.metadata,
            valid=None if self.valid is None else self.valid[mask],
        )


AllMarkersVisibleStrategy = Literal["first", "middle", "last"]
"""How :meth:`SyncClip.find_frame_all_markers_visible` picks among qualifying frames."""


def _center_of_longest_true_run(mask: BoolArray) -> int:
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("mask has no True entries")

    best_start = int(indices[0])
    best_len = 1
    run_start = int(indices[0])
    run_len = 1
    for idx in indices[1:]:
        if idx == run_start + run_len:
            run_len += 1
        else:
            if run_len > best_len:
                best_start = run_start
                best_len = run_len
            run_start = int(idx)
            run_len = 1
    if run_len > best_len:
        best_start = run_start
        best_len = run_len
    return best_start + best_len // 2


def scalar_speed_from_positions(
    positions: FloatArray,
    time_s: FloatArray,
    *,
    finite_mask: BoolArray | None = None,
) -> FloatArray:
    """Compute per-frame scalar speed from positions and times.

    Args:
        positions: ``(frames, 3)`` world positions.
        time_s: ``(frames,)`` times in seconds.
        finite_mask: Optional per-frame mask; defaults to finite positions and times.

    Returns:
        Speed in m/s with shape ``(frames,)``; NaN where differentiation is invalid.
    """
    positions = np.asarray(positions, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    n = positions.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    if finite_mask is None:
        finite_mask = np.isfinite(positions).all(axis=1) & np.isfinite(time_s)
    else:
        finite_mask = np.asarray(finite_mask, dtype=bool)
    dt = np.diff(time_s)
    dpos = np.diff(positions, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        step = np.linalg.norm(dpos, axis=1)
        speed = step / dt
    valid = finite_mask[:-1] & finite_mask[1:] & np.isfinite(dt) & (dt > 1e-9)
    out[1:][valid] = speed[valid]
    if n > 1 and finite_mask[0] and finite_mask[1] and np.isfinite(out[1]):
        out[0] = out[1]
    return out


def _load_markers(data: Any, files: list[str]) -> MarkerTracks | None:
    if "vicon__marker_pos" not in files:
        return None
  # marker names are not stored in synced.npz; callers can attach from vicon.npz
    positions = as_float_array(data["vicon__marker_pos"], name="vicon__marker_pos")
    names = tuple(f"marker_{i}" for i in range(positions.shape[1]))
    return MarkerTracks(names=names, positions=positions)
