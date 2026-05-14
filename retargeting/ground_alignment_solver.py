"""
This script estimates per-timestep transforms to align measured marker tracks to a
canonical model that makes the skateboard flush to the ground plane.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np


Point3 = Tuple[float, float, float]
MarkerTracks = Dict[str, List[Optional[Point3]]]
ModelPositions = Dict[str, np.ndarray]


@dataclass
class PoseEstimate:
    """
    Estimated skateboard pose in the measured coordinate system.

    Maps nominal model coordinates to measured coordinates:

        p_measured ~= R_model_to_measured @ q_model + t_model_to_measured
    """
    R_model_to_measured: np.ndarray
    t_model_to_measured: np.ndarray
    valid: bool
    used_previous: bool
    visible_markers: List[str]
    rms_error: float
    num_inliers: int


@dataclass
class GroundCorrectionTransform:
    """
    Pose-preserving corrective transform.

    Maps measured coordinates to restored coordinates:

        p_restored = R_correction @ p_measured + t_correction

    This should preserve board motion instead of snapping the board to a fixed pose.
    """
    R_correction: np.ndarray
    t_correction: np.ndarray
    valid: bool
    used_previous: bool
    source_pose_valid: bool
    visible_markers: List[str]
    lower_plane_height_after_correction: float
    rms_pose_error: float
    num_inliers: int


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _fit_plane_normal(points: np.ndarray) -> np.ndarray:
    """
    Fit a plane normal using SVD.

    points: shape (N, 3), N >= 3
    """
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    return _normalize(vh[-1])


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return rotation matrix R such that:

        R @ a ~= b
    """
    a = _normalize(a)
    b = _normalize(b)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        if c > 0:
            return np.eye(3)

        # 180-degree rotation. Pick any axis perpendicular to a.
        candidate = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(candidate, a)) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0])

        axis = _normalize(np.cross(a, candidate))

        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])

        return np.eye(3) + 2.0 * (K @ K)

    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _kabsch_model_to_measured(
    model_points: np.ndarray,
    measured_points: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve:

        measured_i ~= R @ model_i + t

    This estimates the skateboard pose in the measured coordinate system.
    """
    model_points = np.asarray(model_points, dtype=np.float64)
    measured_points = np.asarray(measured_points, dtype=np.float64)

    if model_points.shape != measured_points.shape:
        raise ValueError("model_points and measured_points must have the same shape.")

    if model_points.ndim != 2 or model_points.shape[1] != 3:
        raise ValueError("model_points and measured_points must have shape (N, 3).")

    n = model_points.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for a full 3D rigid transform.")

    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    weights = np.maximum(weights, 0.0)

    weight_sum = float(weights.sum())
    if weight_sum <= 1e-12:
        raise ValueError("All weights are zero.")

    weights = weights / weight_sum

    model_centroid = np.sum(model_points * weights[:, None], axis=0)
    measured_centroid = np.sum(measured_points * weights[:, None], axis=0)

    X = model_points - model_centroid
    Y = measured_points - measured_centroid

    H = X.T @ (Y * weights[:, None])

    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = measured_centroid - R @ model_centroid

    return R, t


def _transform_points(R: np.ndarray, t: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies:

        transformed = R @ point + t

    Vectorized for shape (N, 3).
    """
    return points @ R.T + t[None, :]


def _rms_error(
    R: np.ndarray,
    t: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    if source.shape[0] == 0:
        return np.inf

    pred = _transform_points(R, t, source)
    residuals = np.linalg.norm(pred - target, axis=1)
    return float(np.sqrt(np.mean(residuals ** 2)))


def _rank_ok(points: np.ndarray, min_singular_ratio: float = 1e-4) -> bool:
    """
    Checks whether points are non-collinear enough for a rigid 3D fit.

    You need at least 3 non-collinear points. More points can still be nearly
    degenerate if they lie close to a line.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] < 3:
        return False

    centered = points - points.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)

    if len(s) < 2:
        return False

    return bool(s[1] > min_singular_ratio * max(s[0], 1e-12))


def _extract_visible_model_measured_pairs(
    tracks: MarkerTracks,
    nominal_model: ModelPositions,
    timestep: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Returns:
        visible marker names
        model points q_i
        measured points p_i,t
    """
    names = []
    model_points = []
    measured_points = []

    for name, trajectory in tracks.items():
        if name not in nominal_model:
            continue

        if timestep >= len(trajectory):
            continue

        p = trajectory[timestep]

        if p is None:
            continue

        names.append(name)
        model_points.append(nominal_model[name])
        measured_points.append(p)

    if len(names) == 0:
        return names, np.empty((0, 3)), np.empty((0, 3))

    return (
        names,
        np.asarray(model_points, dtype=np.float64),
        np.asarray(measured_points, dtype=np.float64),
    )


def _robust_pose_fit_model_to_measured(
    model_points: np.ndarray,
    measured_points: np.ndarray,
    outlier_threshold: Optional[float] = None,
    min_inliers: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Robustly fit:

        measured ~= R @ model + t

    Returns:
        R, t, inlier_mask, rms_error
    """
    n = model_points.shape[0]

    if n < min_inliers:
        raise ValueError("Not enough visible markers.")

    if not _rank_ok(model_points) or not _rank_ok(measured_points):
        raise ValueError("Visible marker set is degenerate or nearly collinear.")

    R, t = _kabsch_model_to_measured(model_points, measured_points)

    pred = _transform_points(R, t, model_points)
    residuals = np.linalg.norm(pred - measured_points, axis=1)

    if outlier_threshold is None:
        med = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - med)))
        sigma = 1.4826 * mad

        if sigma < 1e-12:
            threshold = max(3.0 * med, 1e-9)
        else:
            threshold = med + 4.0 * sigma
    else:
        threshold = float(outlier_threshold)

    inlier_mask = residuals <= threshold

    if int(inlier_mask.sum()) >= min_inliers:
        model_in = model_points[inlier_mask]
        measured_in = measured_points[inlier_mask]

        if _rank_ok(model_in) and _rank_ok(measured_in):
            R, t = _kabsch_model_to_measured(model_in, measured_in)
            rms = _rms_error(R, t, model_in, measured_in)
            return R, t, inlier_mask, rms

    # If outlier rejection would make the frame unusable, use all points.
    rms = _rms_error(R, t, model_points, measured_points)
    return R, t, np.ones(n, dtype=bool), rms


def estimate_model_lower_plane_normal(
    nominal_model: ModelPositions,
    lower_markers: Sequence[str],
    upper_markers: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Estimate the lower-plane normal in nominal model coordinates.

    If upper_markers are provided, the normal is oriented from lower plane
    toward upper plane.
    """
    lower_markers = list(lower_markers)

    if len(lower_markers) < 3:
        raise ValueError("Need at least 3 lower-plane markers.")

    missing = [m for m in lower_markers if m not in nominal_model]
    if missing:
        raise ValueError(f"Lower markers missing from nominal_model: {missing}")

    lower_points = np.array([nominal_model[m] for m in lower_markers], dtype=np.float64)
    normal = _fit_plane_normal(lower_points)

    if upper_markers is not None and len(upper_markers) > 0:
        missing_upper = [m for m in upper_markers if m not in nominal_model]
        if missing_upper:
            raise ValueError(f"Upper markers missing from nominal_model: {missing_upper}")

        upper_points = np.array([nominal_model[m] for m in upper_markers], dtype=np.float64)

        lower_centroid = lower_points.mean(axis=0)
        upper_centroid = upper_points.mean(axis=0)

        lower_to_upper = upper_centroid - lower_centroid

        if np.dot(normal, lower_to_upper) < 0:
            normal = -normal

    return _normalize(normal)


def estimate_pose_sequence_model_to_measured(
    tracks: MarkerTracks,
    nominal_model: ModelPositions,
    min_visible_markers: int = 3,
    outlier_threshold: Optional[float] = None,
    carry_forward: bool = True,
) -> List[PoseEstimate]:
    """
    Estimate skateboard pose in measured coordinates for each timestep.

    This maps:

        nominal model -> measured frame

    so it preserves the skateboard's motion instead of cancelling it.
    """
    if not tracks:
        return []

    T = max(len(v) for v in tracks.values())

    poses: List[PoseEstimate] = []

    last_R: Optional[np.ndarray] = None
    last_t: Optional[np.ndarray] = None
    last_rms: float = np.inf

    for timestep in range(T):
        visible_names, model_points, measured_points = _extract_visible_model_measured_pairs(
            tracks=tracks,
            nominal_model=nominal_model,
            timestep=timestep,
        )

        can_attempt = (
            len(visible_names) >= min_visible_markers
            and _rank_ok(model_points)
            and _rank_ok(measured_points)
        )

        if can_attempt:
            try:
                R, t, inlier_mask, rms = _robust_pose_fit_model_to_measured(
                    model_points=model_points,
                    measured_points=measured_points,
                    outlier_threshold=outlier_threshold,
                    min_inliers=min_visible_markers,
                )

                inlier_names = [
                    name for name, keep in zip(visible_names, inlier_mask)
                    if keep
                ]

                pose = PoseEstimate(
                    R_model_to_measured=R,
                    t_model_to_measured=t,
                    valid=True,
                    used_previous=False,
                    visible_markers=inlier_names,
                    rms_error=rms,
                    num_inliers=len(inlier_names),
                )

                poses.append(pose)

                last_R = R
                last_t = t
                last_rms = rms
                continue

            except ValueError:
                pass

        if carry_forward and last_R is not None and last_t is not None:
            poses.append(
                PoseEstimate(
                    R_model_to_measured=last_R.copy(),
                    t_model_to_measured=last_t.copy(),
                    valid=True,
                    used_previous=True,
                    visible_markers=visible_names,
                    rms_error=last_rms,
                    num_inliers=0,
                )
            )
        else:
            poses.append(
                PoseEstimate(
                    R_model_to_measured=np.eye(3),
                    t_model_to_measured=np.zeros(3),
                    valid=False,
                    used_previous=False,
                    visible_markers=visible_names,
                    rms_error=np.inf,
                    num_inliers=0,
                )
            )

    return poses


def estimate_pose_preserving_ground_corrections(
    tracks: MarkerTracks,
    nominal_model: ModelPositions,
    lower_markers: Sequence[str],
    upper_markers: Optional[Sequence[str]],
    lower_plane_height: float,
    min_visible_markers: int = 3,
    outlier_threshold: Optional[float] = None,
    carry_forward_pose: bool = True,
    carry_forward_correction: bool = True,
    preserve_horizontal_position: bool = True,
) -> List[GroundCorrectionTransform]:
    """
    Estimate per-timestep ground correction transforms.

    This does NOT map the moving skateboard to a fixed canonical skateboard.

    Instead, for each timestep:

        1. Estimate skateboard pose:
               measured ~= R_pose @ nominal_model + t_pose

        2. Compute measured lower-plane normal:
               n_measured = R_pose @ n_model_lower

        3. Compute correction rotation:
               R_corr @ n_measured = world_up

        4. Rotate measured points by R_corr.

        5. Shift vertically so the lower plane sits at lower_plane_height.

    The final correction is:

        p_restored = R_corr @ p_measured + t_corr

    If preserve_horizontal_position=True, t_corr has only z translation.
    That preserves x/y skateboard movement.
    """
    lower_markers = list(lower_markers)
    upper_markers = list(upper_markers) if upper_markers is not None else None

    model_lower_normal = estimate_model_lower_plane_normal(
        nominal_model=nominal_model,
        lower_markers=lower_markers,
        upper_markers=upper_markers,
    )

    poses = estimate_pose_sequence_model_to_measured(
        tracks=tracks,
        nominal_model=nominal_model,
        min_visible_markers=min_visible_markers,
        outlier_threshold=outlier_threshold,
        carry_forward=carry_forward_pose,
    )

    corrections: List[GroundCorrectionTransform] = []

    world_up = np.array([0.0, 0.0, 1.0])

    last_R_corr: Optional[np.ndarray] = None
    last_t_corr: Optional[np.ndarray] = None
    last_lower_height: float = np.nan

    for pose in poses:
        if not pose.valid:
            if carry_forward_correction and last_R_corr is not None and last_t_corr is not None:
                corrections.append(
                    GroundCorrectionTransform(
                        R_correction=last_R_corr.copy(),
                        t_correction=last_t_corr.copy(),
                        valid=True,
                        used_previous=True,
                        source_pose_valid=False,
                        visible_markers=pose.visible_markers,
                        lower_plane_height_after_correction=last_lower_height,
                        rms_pose_error=pose.rms_error,
                        num_inliers=0,
                    )
                )
            else:
                corrections.append(
                    GroundCorrectionTransform(
                        R_correction=np.eye(3),
                        t_correction=np.zeros(3),
                        valid=False,
                        used_previous=False,
                        source_pose_valid=False,
                        visible_markers=pose.visible_markers,
                        lower_plane_height_after_correction=np.nan,
                        rms_pose_error=np.inf,
                        num_inliers=0,
                    )
                )
            continue

        # Lower-plane normal in the measured coordinate system.
        measured_lower_normal = pose.R_model_to_measured @ model_lower_normal
        measured_lower_normal = _normalize(measured_lower_normal)

        # Corrective rotation that makes the measured lower plane horizontal.
        R_corr = _rotation_from_a_to_b(measured_lower_normal, world_up)

        # Estimate lower-plane marker positions in measured coordinates using pose.
        # This works even if some lower markers are occluded in this frame.
        lower_model_points = np.array(
            [nominal_model[m] for m in lower_markers],
            dtype=np.float64,
        )

        lower_measured_pred = _transform_points(
            pose.R_model_to_measured,
            pose.t_model_to_measured,
            lower_model_points,
        )

        lower_after_rotation = _transform_points(
            R_corr,
            np.zeros(3),
            lower_measured_pred,
        )

        current_lower_z = float(np.mean(lower_after_rotation[:, 2]))

        dz = float(lower_plane_height - current_lower_z)

        if preserve_horizontal_position:
            t_corr = np.array([0.0, 0.0, dz], dtype=np.float64)
        else:
            # Usually not recommended for your skateboard case.
            # This would allow x/y recentering if you later decide you want it.
            t_corr = np.array([0.0, 0.0, dz], dtype=np.float64)

        corrected_lower = lower_after_rotation + t_corr[None, :]
        lower_height_after = float(np.mean(corrected_lower[:, 2]))

        correction = GroundCorrectionTransform(
            R_correction=R_corr,
            t_correction=t_corr,
            valid=True,
            used_previous=pose.used_previous,
            source_pose_valid=True,
            visible_markers=pose.visible_markers,
            lower_plane_height_after_correction=lower_height_after,
            rms_pose_error=pose.rms_error,
            num_inliers=pose.num_inliers,
        )

        corrections.append(correction)

        last_R_corr = R_corr
        last_t_corr = t_corr
        last_lower_height = lower_height_after

    return corrections


def apply_ground_corrections(
    tracks: MarkerTracks,
    corrections: Sequence[GroundCorrectionTransform],
) -> MarkerTracks:
    """
    Apply correction transforms to measured marker tracks.

    Returns restored tracks:

        p_restored = R_correction @ p_measured + t_correction
    """
    restored: MarkerTracks = {}

    for name, trajectory in tracks.items():
        corrected_trajectory: List[Optional[Point3]] = []

        for timestep, point in enumerate(trajectory):
            if point is None:
                corrected_trajectory.append(None)
                continue

            if timestep >= len(corrections) or not corrections[timestep].valid:
                corrected_trajectory.append(None)
                continue

            corr = corrections[timestep]
            p = np.asarray(point, dtype=np.float64)
            restored_p = corr.R_correction @ p + corr.t_correction

            corrected_trajectory.append(tuple(float(x) for x in restored_p))

        restored[name] = corrected_trajectory

    return restored


def apply_ground_correction_to_points(
    points_by_timestep: Sequence[np.ndarray],
    corrections: Sequence[GroundCorrectionTransform],
) -> List[Optional[np.ndarray]]:
    """
    Optional helper if you have other point clouds or trajectory points per timestep.

    points_by_timestep[t] should be shape (N, 3).

    Returns list where invalid timesteps are None.
    """
    output: List[Optional[np.ndarray]] = []

    for timestep, points in enumerate(points_by_timestep):
        if timestep >= len(corrections) or not corrections[timestep].valid:
            output.append(None)
            continue

        corr = corrections[timestep]
        points = np.asarray(points, dtype=np.float64)
        output.append(_transform_points(corr.R_correction, corr.t_correction, points))

    return output