from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.optimize import minimize_scalar

from smplx.joint_names import JOINT_NAMES

from retargeting.config import RetargetingConfig


InterpMode = Literal[
    "linear",
    "nearest",
    "copy",
    "repeat",
    "quat",
    "rotvec",
    "rotvec_flat",
]
QuatOrder = Literal["xyzw", "wxyz"]


CropMode = Literal["none", "valid", "support"]

DEFAULT_VIDEO_SCHEMA: Dict[str, Dict[str, Any]] = {
    "video/joints": {
        "array_key": "joints",
        "mode": "linear",
    },
    "video/vertices": {
        "array_key": "vertices",
        "mode": "linear",
    },
    "video/body_pose": {
        "array_key": "body_pose",
        "mode": "rotvec_flat",
    },
    "video/transl": {
        "array_key": "transl",
        "mode": "linear",
    },
    "video/global_orient": {
        "array_key": "global_orient",
        "mode": "rotvec",
    },
    "video/betas": {
        "array_key": "betas",
        "mode": "nearest",
    },
}
DEFAULT_VICON_SCHEMA: Dict[str, Dict[str, Any]] = {
    "vicon/body_pos": {
        "array_key": "body_pos",
        "mode": "linear",
    },
    "vicon/body_quat": {
        "array_key": "body_quat",
        "mode": "quat",
        "quat_order": "wxyz",
        "required": False,
    },
    "vicon/marker_pos": {
        "array_key": "marker_pos",
        "mode": "linear",
        "required": False,
    },
}


# Optional per-stream keys (used by ``resolve_vicon_schema_for_sync`` or callers):
#   validity_body_indices: tuple[int, ...] on ``vicon/body_pos`` — only these rigid-body
#   rows must be finite for crop="valid" (defaults to shoes from ``time_sync_solver``).


@dataclass
class FeatureSpec:
    key: str
    t: np.ndarray
    x: np.ndarray
    mode: InterpMode
    allow_extrapolate: bool = False
    quat_order: Optional[QuatOrder] = None
    required: bool = True
    # For (T, B, ...) rigid-body stacks: gate crop=valid finiteness on these body indices only
    # (e.g. shoes for sync while Skateboard TF may drop out).
    validity_body_indices: Optional[Tuple[int, ...]] = None


def _corr_demeaned_cosine(y1: np.ndarray, y2: np.ndarray) -> float:
    """Unweighted normalized cosine similarity between equal-length rows (after global demean)."""
    y1 = y1 - y1.mean(axis=0)
    y2 = y2 - y2.mean(axis=0)
    denom = np.linalg.norm(y1) * np.linalg.norm(y2)
    if denom == 0:
        return float("-inf")
    return float(np.sum(y1 * y2) / denom)


def _corr_motion_weighted(
    y1: np.ndarray,
    y2: np.ndarray,
    floor_quantile: float,
) -> float:
    """
    Foot-speed–weighted correlation: down-weights overlap samples where either modality is
    nearly static, which otherwise produce spurious NCC peaks when the wrong lag lines up
    two long quiet regions.
    """
    m = np.minimum(np.linalg.norm(y1, axis=1), np.linalg.norm(y2, axis=1))
    if len(m) < 3:
        return float("-inf")
    floor = float(np.quantile(m, floor_quantile))
    w = np.clip(m - floor, 0.0, None)
    sw = float(np.sum(w))
    if sw < 1e-12:
        return float("-inf")
    nch = y1.shape[1]
    acc = 0.0
    for d in range(nch):
        a = y1[:, d]
        b = y2[:, d]
        am = float(np.sum(w * a) / sw)
        bm = float(np.sum(w * b) / sw)
        ac = a - am
        bc = b - bm
        num = float(np.sum(w * ac * bc))
        den = float(np.sqrt(np.sum(w * ac * ac) * np.sum(w * bc * bc)))
        if den < 1e-18:
            return float("-inf")
        acc += num / den
    return acc / nch


def estimate_lag(
    t1,
    x1,
    t2,
    x2,
    dt=None,
    min_overlap=None,
    num_lags=2500,
    standardize_components=False,
    *,
    max_abs_lag_seconds: Optional[float] = None,
    motion_weighted: bool = False,
    motion_weight_floor_quantile: float = 0.12,
):
    """
    Estimate the time lag that maximizes normalized cross-correlation between
    two irregularly sampled 2D signals.

    Parameters
    ----------
    t1, t2 : array-like, shape (n,), (m,)
        Sample times for the two signals.

    x1, x2 : array-like, shape (n, d), (m, d)
        Vector-valued signal samples. For your case, d = 2.

    dt : float, optional
        Spacing of the common interpolation grid. If None, uses the smaller
        median sample spacing.

    min_overlap : float, optional
        Minimum time duration of overlap required for a lag to be considered.
        If None, uses 10% of the shorter signal duration.

    num_lags : int
        Number of lags to evaluate in the coarse search.

    standardize_components : bool
        If True, standardizes each coordinate globally before alignment.
        Useful if one coordinate has much larger scale than the other.

    max_abs_lag_seconds : float, optional
        If set, restricts the lag search to ``[-max_abs_lag_seconds, +max_abs_lag_seconds]``
        intersected with the physically valid overlap range.

    motion_weighted : bool
        If True, scores each lag with activity-weighted correlation (see module doc / config).

    motion_weight_floor_quantile : float
        Quantile of per-sample ``min(||y1||,||y2||)`` used as a floor when ``motion_weighted``.

    Returns
    -------
    best_lag : float
        Estimated lag. Positive means x2 should be shifted later to align
        with x1.

    best_corr : float
        Maximum normalized correlation.

    info : dict
        Extra diagnostic information: lag grid, correlation values, and bounds.
    """

    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if x1.ndim == 1:
        x1 = x1[:, None]
    if x2.ndim == 1:
        x2 = x2[:, None]

    if x1.shape[1] != x2.shape[1]:
        raise ValueError("x1 and x2 must have the same signal dimension.")

    if len(t1) != len(x1) or len(t2) != len(x2):
        raise ValueError("Time arrays and signal arrays must have matching lengths.")

    if len(t1) < 3 or len(t2) < 3:
        raise ValueError("Each signal needs at least 3 samples. Barely a signal, honestly.")

    # Sort by time
    i1 = np.argsort(t1)
    i2 = np.argsort(t2)

    t1, x1 = t1[i1], x1[i1]
    t2, x2 = t2[i2], x2[i2]

    # Remove duplicate timestamps by averaging samples at identical times
    t1, x1 = _merge_duplicate_times(t1, x1)
    t2, x2 = _merge_duplicate_times(t2, x2)

    if len(t1) < 3 or len(t2) < 3:
        raise ValueError("Each signal needs at least 3 unique time samples.")

    if standardize_components:
        x1, x2 = _standardize_pair(x1, x2)

    dur1 = t1[-1] - t1[0]
    dur2 = t2[-1] - t2[0]

    if dur1 <= 0 or dur2 <= 0:
        raise ValueError("Each signal must span a positive time duration.")

    if dt is None:
        dt = min(np.median(np.diff(t1)), np.median(np.diff(t2)))

    if dt <= 0:
        raise ValueError("dt must be positive.")

    if min_overlap is None:
        min_overlap = 0.10 * min(dur1, dur2)

    if min_overlap <= 0:
        raise ValueError("min_overlap must be positive.")

    # Positive lag means x2 is shifted later.
    # Compare x1(t) with x2(t - lag).
    lag_min = t1[0] - t2[-1] + min_overlap
    lag_max = t1[-1] - t2[0] - min_overlap

    if lag_min >= lag_max:
        raise ValueError("No valid lag range. Try reducing min_overlap.")

    lag_min_phys, lag_max_phys = float(lag_min), float(lag_max)
    lag_search_clamped = False
    if max_abs_lag_seconds is not None:
        m = float(max_abs_lag_seconds)
        nmin = max(lag_min_phys, -m)
        nmax = min(lag_max_phys, m)
        if nmin >= nmax:
            raise ValueError(
                f"No valid lag range after max_abs_lag_seconds={m}s clamp "
                f"(physical overlap allows [{lag_min_phys:.6g}, {lag_max_phys:.6g}] s). "
                "Increase max_abs_lag_seconds or check recording overlap."
            )
        lag_search_clamped = (nmin > lag_min_phys + 1e-9) or (nmax < lag_max_phys - 1e-9)
        lag_min, lag_max = nmin, nmax

    f1 = interp1d(t1, x1, axis=0, bounds_error=False, fill_value=np.nan)
    f2 = interp1d(t2, x2, axis=0, bounds_error=False, fill_value=np.nan)

    def corr_at_lag(lag):
        lo = max(t1[0], t2[0] + lag)
        hi = min(t1[-1], t2[-1] + lag)

        if hi - lo < min_overlap:
            return -np.inf

        grid = np.arange(lo, hi, dt)

        if len(grid) < 3:
            return -np.inf

        y1 = f1(grid)
        y2 = f2(grid - lag)

        mask = np.isfinite(y1).all(axis=1) & np.isfinite(y2).all(axis=1)

        if mask.sum() < 3:
            return -np.inf

        y1 = y1[mask]
        y2 = y2[mask]

        if motion_weighted:
            return _corr_motion_weighted(y1, y2, motion_weight_floor_quantile)
        return _corr_demeaned_cosine(y1, y2)

    # Coarse search first, because noisy/periodic signals love local optima.
    lags = np.linspace(lag_min, lag_max, num_lags)
    corrs = np.array([corr_at_lag(lag) for lag in lags])

    if not np.any(np.isfinite(corrs)):
        raise ValueError("No finite correlations found. Check dt, min_overlap, and signal coverage.")

    best_idx = int(np.nanargmax(corrs))

    # Local refinement around best coarse lag
    left_idx = max(best_idx - 1, 0)
    right_idx = min(best_idx + 1, len(lags) - 1)

    left = lags[left_idx]
    right = lags[right_idx]

    if left == right:
        best_lag = lags[best_idx]
        best_corr = corrs[best_idx]
    else:
        result = minimize_scalar(
            lambda lag: -corr_at_lag(lag),
            bounds=(left, right),
            method="bounded",
        )

        best_lag = result.x
        best_corr = -result.fun

    info = {
        "lags": lags,
        "corrs": corrs,
        "lag_min": lag_min,
        "lag_max": lag_max,
        "lag_min_phys": lag_min_phys,
        "lag_max_phys": lag_max_phys,
        "lag_search_clamped": lag_search_clamped,
        "dt": dt,
        "min_overlap": min_overlap,
        "motion_weighted": motion_weighted,
        "motion_weight_floor_quantile": motion_weight_floor_quantile,
    }

    return best_lag, best_corr, info


def _merge_duplicate_times(t, x):
    """
    Merge duplicate timestamps by averaging their signal values.
    """
    unique_t, inverse = np.unique(t, return_inverse=True)

    if len(unique_t) == len(t):
        return t, x

    x_merged = np.zeros((len(unique_t), x.shape[1]), dtype=float)
    counts = np.zeros(len(unique_t), dtype=float)

    for i, group in enumerate(inverse):
        x_merged[group] += x[i]
        counts[group] += 1

    x_merged /= counts[:, None]

    return unique_t, x_merged


def _standardize_pair(x1, x2):
    """
    Standardize components using pooled mean and std across both signals.
    """
    both = np.vstack([x1, x2])

    mean = np.nanmean(both, axis=0)
    std = np.nanstd(both, axis=0)

    std[std == 0] = 1.0

    return (x1 - mean) / std, (x2 - mean) / std


class TimelineStitcher:
    """
    Synchronize arbitrary time-indexed arrays onto one unified timeline.

    Each feature must have time as axis 0.

    Example output keys:
        "t"
        "video/joints"
        "video/vertices"
        "video/body_pose"
        "vicon/body_pos"
        "vicon/marker_pos"
        "valid"
    """

    def __init__(self):
        self.features: Dict[str, FeatureSpec] = {}
        self.source_times: Dict[str, np.ndarray] = {}

    def add_source_time(self, source: str, t: np.ndarray):
        self.source_times[source] = _as_time_array(t)

    def add(
        self,
        key: str,
        t: np.ndarray,
        x: Any,
        mode: InterpMode,
        *,
        allow_extrapolate: bool = False,
        quat_order: Optional[QuatOrder] = None,
        required: bool = True,
        validity_body_indices: Optional[Tuple[int, ...]] = None,
    ):
        t = _as_time_array(t)
        x = _as_numpy(x)
        if len(t) != len(x):
            raise ValueError(
                f"{key}: time length {len(t)} does not match array length {len(x)}."
            )
        if mode == "quat":
            if quat_order is None:
                raise ValueError(
                    f"{key}: quat_order must be specified for quaternion data. "
                    f"Use 'xyzw' for scipy order or 'wxyz' for scalar-first data."
                )
            if quat_order == "wxyz":
                x = wxyz_to_xyzw(x)
            x = sanitize_quaternions(x, replacement="identity")
        t, x = _sort_and_merge_duplicate_times(t, x)
        if key in self.features:
            raise ValueError(f"Feature already exists: {key}")
        self.features[key] = FeatureSpec(
            key=key,
            t=t,
            x=x,
            mode=mode,
            allow_extrapolate=allow_extrapolate,
            quat_order=quat_order,
            required=required,
            validity_body_indices=validity_body_indices,
        )

    def build(
        self,
        timeline: Union[str, np.ndarray] = "video",
        *,
        crop: CropMode = "valid",
        include_valid_masks: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(timeline, str):
            if timeline not in self.source_times:
                raise ValueError(
                    f"Unknown timeline source {timeline!r}. "
                    f"Known sources: {list(self.source_times)}"
                )
            t_out = self.source_times[timeline].copy()
        else:
            t_out = _as_time_array(timeline)

        out: Dict[str, Any] = {"t": t_out.copy()}
        valid_masks: Dict[str, np.ndarray] = {}

        for key, spec in self.features.items():
            print(
                f"[sync] resampling {key}: "
                f"shape={spec.x.shape}, dtype={spec.x.dtype}, mode={spec.mode}"
            )
            y = resample_feature(
                spec.t,
                spec.x,
                t_out,
                mode=spec.mode,
                allow_extrapolate=spec.allow_extrapolate,
            )

            out[key] = y
            if spec.validity_body_indices is not None:
                idx = spec.validity_body_indices
                if y.ndim < 3:
                    raise ValueError(
                        f"{key}: validity_body_indices requires ndim>=3 (T, bodies, ...), "
                        f"got shape {y.shape}."
                    )
                if max(idx) >= y.shape[1]:
                    raise ValueError(
                        f"{key}: validity_body_indices {idx} out of range for axis-1 size {y.shape[1]}."
                    )
                y_sel = y[:, idx, ...]
                valid_masks[key] = finite_time_mask(y_sel)
            else:
                valid_masks[key] = finite_time_mask(y)

        valid = np.ones(len(t_out), dtype=bool)

        if crop == "support":
            for spec in self.features.values():
                valid &= (t_out >= spec.t[0]) & (t_out <= spec.t[-1])

        elif crop == "valid":
            # Only *required* streams gate cropping. Optional features (e.g. markers
            # with partial tracking, optional quats) may have no row where every element
            # is finite; they must not zero out the whole timeline.
            for key, mask in valid_masks.items():
                if self.features[key].required:
                    valid &= mask

        elif crop == "none":
            pass

        else:
            raise ValueError(f"Unknown crop mode: {crop}")

        out["valid"] = valid

        if crop in {"valid", "support"}:
            out = {
                k: _crop_time_axis(v, valid)
                if _has_time_axis(v, len(t_out))
                else v
                for k, v in out.items()
            }

            valid_masks = {
                k: mask[valid]
                for k, mask in valid_masks.items()
            }

        if include_valid_masks:
            out["__valid_masks__"] = valid_masks

        return out


def resample_feature(
    t_source: np.ndarray,
    x_source: np.ndarray,
    t_target: np.ndarray,
    *,
    mode: InterpMode,
    allow_extrapolate: bool = False,
):
    t_source = _as_time_array(t_source)
    t_target = _as_time_array(t_target)
    x_source = _as_numpy(x_source)

    if mode in {"copy", "repeat"}:
        mode = "nearest"

    if mode == "linear":
        return _interp_linear(
            t_source,
            x_source,
            t_target,
            allow_extrapolate=allow_extrapolate,
        )

    if mode == "nearest":
        return _interp_nearest(
            t_source,
            x_source,
            t_target,
            allow_extrapolate=allow_extrapolate,
        )

    if mode == "quat":
        return _interp_quat(
            t_source,
            x_source,
            t_target,
            allow_extrapolate=allow_extrapolate,
        )

    if mode == "rotvec":
        return _interp_rotvec(
            t_source,
            x_source,
            t_target,
            allow_extrapolate=allow_extrapolate,
        )

    if mode == "rotvec_flat":
        if x_source.ndim != 2 or x_source.shape[1] % 3 != 0:
            raise ValueError(
                f"rotvec_flat expects shape (T, 3*K), got {x_source.shape}."
            )

        reshaped = x_source.reshape(len(t_source), -1, 3)

        y = _interp_rotvec(
            t_source,
            reshaped,
            t_target,
            allow_extrapolate=allow_extrapolate,
        )

        return y.reshape(len(t_target), x_source.shape[1])

    raise ValueError(f"Unknown interpolation mode: {mode}")


def _interp_linear(
    t_source: np.ndarray,
    x_source: np.ndarray,
    t_target: np.ndarray,
    *,
    allow_extrapolate: bool,
):
    if not np.issubdtype(x_source.dtype, np.number):
        return _interp_nearest(
            t_source,
            x_source,
            t_target,
            allow_extrapolate=allow_extrapolate,
        )

    fill_value = "extrapolate" if allow_extrapolate else np.nan

    f = interp1d(
        t_source,
        x_source,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=fill_value,
        assume_sorted=True,
    )

    return f(t_target)


def _interp_nearest(
    t_source: np.ndarray,
    x_source: np.ndarray,
    t_target: np.ndarray,
    *,
    allow_extrapolate: bool,
):
    idx = np.searchsorted(t_source, t_target)

    left = np.clip(idx - 1, 0, len(t_source) - 1)
    right = np.clip(idx, 0, len(t_source) - 1)

    choose_right = (
        np.abs(t_source[right] - t_target)
        <
        np.abs(t_target - t_source[left])
    )

    nearest = np.where(choose_right, right, left)
    y = x_source[nearest].copy()

    if not allow_extrapolate:
        outside = (t_target < t_source[0]) | (t_target > t_source[-1])
        y = _assign_nan_on_time_mask(y, outside)

    return y


def _interp_quat(
    t_source: np.ndarray,
    q_source: np.ndarray,
    t_target: np.ndarray,
    *,
    allow_extrapolate: bool,
):
    """
    Quaternion SLERP.

    Assumes scipy quaternion order:
        [x, y, z, w]

    Supports shapes:
        (T, 4)
        (T, N, 4)
        (T, A, B, 4)
    """

    q_source = np.asarray(q_source, dtype=float)
    if q_source.shape[-1] != 4:
        raise ValueError(
            f"Quaternion data must have last dimension 4, got {q_source.shape}."
        )
    q_source = sanitize_quaternions(q_source, replacement="identity")

    flat = q_source.reshape(len(t_source), -1, 4)
    out = np.full((len(t_target), flat.shape[1], 4), np.nan, dtype=float)

    if allow_extrapolate:
        query_mask = np.ones(len(t_target), dtype=bool)
        tq = np.clip(t_target, t_source[0], t_source[-1])
    else:
        query_mask = (t_target >= t_source[0]) & (t_target <= t_source[-1])
        tq = t_target[query_mask]

    if len(tq) == 0:
        return out.reshape((len(t_target),) + q_source.shape[1:])

    for j in range(flat.shape[1]):
        rotations = R.from_quat(flat[:, j, :])
        slerp = Slerp(t_source, rotations)
        out[query_mask, j, :] = slerp(tq).as_quat()

    return out.reshape((len(t_target),) + q_source.shape[1:])


def _interp_rotvec(
    t_source: np.ndarray,
    rv_source: np.ndarray,
    t_target: np.ndarray,
    *,
    allow_extrapolate: bool,
):
    """
    Rotation-vector interpolation by converting to rotations, doing SLERP,
    then converting back to rotation vectors.

    Supports:
        (T, 3)
        (T, J, 3)
        (T, A, B, 3)
    """

    rv_source = np.asarray(rv_source, dtype=float)

    if rv_source.shape[-1] != 3:
        raise ValueError(
            f"Rotation-vector data must have last dimension 3, got {rv_source.shape}."
        )

    flat = rv_source.reshape(len(t_source), -1, 3)
    out = np.full((len(t_target), flat.shape[1], 3), np.nan, dtype=float)

    if allow_extrapolate:
        query_mask = np.ones(len(t_target), dtype=bool)
        tq = np.clip(t_target, t_source[0], t_source[-1])
    else:
        query_mask = (t_target >= t_source[0]) & (t_target <= t_source[-1])
        tq = t_target[query_mask]

    if len(tq) == 0:
        return out.reshape((len(t_target),) + rv_source.shape[1:])

    for j in range(flat.shape[1]):
        rotations = R.from_rotvec(flat[:, j, :])
        slerp = Slerp(t_source, rotations)
        out[query_mask, j, :] = slerp(tq).as_rotvec()

    return out.reshape((len(t_target),) + rv_source.shape[1:])


def sanitize_quaternions(
    q: np.ndarray,
    *,
    eps: float = 1e-12,
    replacement: Literal["identity", "nan"] = "identity",
) -> np.ndarray:
    """
    Replace invalid zero-norm quaternions.
    Input and output are scipy quaternion order: [x, y, z, w].
    """
    q = np.asarray(q, dtype=float).copy()
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {q.shape}")
    norm = np.linalg.norm(q, axis=-1)
    bad = ~np.isfinite(norm) | (norm < eps)
    if not np.any(bad):
        return q
    print(f"[sync] warning: replacing {bad.sum()} invalid zero-norm quaternions")
    if replacement == "identity":
        q[bad] = np.array([0.0, 0.0, 0.0, 1.0])
    elif replacement == "nan":
        q[bad] = np.nan
    else:
        raise ValueError(f"Unknown replacement: {replacement}")
    return q


def load_vicon_data(
    path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Load a Vicon merged npz.

    Expected:
        stamp: (T,)
        body_names: object/list, optional
        body_pos: (T, B, 3), optional
        marker_pos: (T, M, 3), optional
        any other time-series arrays with first dimension T
    """

    data = np.load(path, allow_pickle=True)

    t = data["stamp"] - data["stamp"][0]

    out: Dict[str, Any] = {
        "t": t,
        "body_names": data["body_names"].tolist()
        if "body_names" in data
        else None,
    }

    for key in data.files:
        if key in {"stamp", "body_names"}:
            continue

        value = data[key]

        if hasattr(value, "shape") and value.shape[0] == len(t):
            out[key] = value

    return out


def load_gvhmr_data(
    gvhmr_output_dir: Union[str, Path],
) -> Dict[str, Any]:
    gvhmr_output_dir = Path(gvhmr_output_dir)

    joints = np.load(gvhmr_output_dir / "joints.npy")
    vertices = np.load(gvhmr_output_dir / "vertices.npy")

    hmr4d_results = torch.load(
        gvhmr_output_dir / "hmr4d_results.pt",
        map_location="cpu",
    )

    smpl = hmr4d_results["smpl_params_global"]

    body_pose = smpl["body_pose"]          # usually (T, 63), rotvec flat
    transl = smpl["transl"]                # usually (T, 3)
    global_orient = smpl["global_orient"]  # usually (T, 3), rotvec
    betas = smpl["betas"]                  # usually (T, 10) or (10,)

    return {
        "joints": joints,
        "vertices": vertices,
        "body_pose": body_pose,
        "transl": transl,
        "global_orient": global_orient,
        "betas": betas,
    }


def make_video_frame_times(num_frames: int, fps: float) -> np.ndarray:
    return np.arange(num_frames, dtype=float) / fps


def support_overlap_video_clock(
    t_video: np.ndarray,
    t_vicon: np.ndarray,
    lag: float,
) -> tuple[float, float]:
    """
    Video-clock time range where both video and shifted-Vicon sources have support.

    Matches ``TimelineStitcher.build(..., crop="support")`` overlap before row
    dropping; independent of ``target_timeline``.
    """
    t_video = np.asarray(t_video, dtype=float)
    t_vicon_shifted = np.asarray(t_vicon, dtype=float) - float(lag)
    return (
        float(max(t_video[0], t_vicon_shifted[0])),
        float(min(t_video[-1], t_vicon_shifted[-1])),
    )


def get_sync_signals(
    vicon: Dict[str, Any],
    gvhmr: Dict[str, Any],
    config: RetargetingConfig,
):
    """
    Build foot-speed sync signals from Vicon body positions and GVHMR joints.

    Returns
    -------
    t_mocap_vel : (Tm - 1,)
    x_mocap : (Tm - 1, 2)

    t_video_vel : (Tv - 1,)
    x_video : (Tv - 1, 2)
    """

    # ----- Vicon shoe velocities -----
    t_mocap = np.asarray(vicon["t"], dtype=float)
    body_names = vicon["body_names"]
    body_pos = np.asarray(vicon["body_pos"], dtype=float)

    if body_names is None:
        raise ValueError("Vicon data must contain body_names for foot sync.")

    right_shoe_index = body_names.index("Right_Shoe")
    left_shoe_index = body_names.index("Left_Shoe")

    right_shoe_pos = body_pos[:, right_shoe_index]
    left_shoe_pos = body_pos[:, left_shoe_index]

    dt_mocap = np.diff(t_mocap)

    if np.any(dt_mocap <= 0):
        raise ValueError("Vicon timestamps must be strictly increasing.")

    right_shoe_vel = (
        np.linalg.norm(np.diff(right_shoe_pos, axis=0), axis=-1)
        / dt_mocap
    )

    left_shoe_vel = (
        np.linalg.norm(np.diff(left_shoe_pos, axis=0), axis=-1)
        / dt_mocap
    )

    t_mocap_vel = 0.5 * (t_mocap[1:] + t_mocap[:-1])

    x_mocap = np.column_stack([left_shoe_vel, right_shoe_vel])

    # ----- Video foot velocities -----
    joints = np.asarray(gvhmr["joints"], dtype=float)

    left_foot_joint_names = config.time_sync_solver.smplx_joints[
        "vicon/Left_Shoe/Left_Shoe"
    ]

    right_foot_joint_names = config.time_sync_solver.smplx_joints[
        "vicon/Right_Shoe/Right_Shoe"
    ]

    left_foot_joint_indices = [
        JOINT_NAMES.index(joint_name)
        for joint_name in left_foot_joint_names
    ]

    right_foot_joint_indices = [
        JOINT_NAMES.index(joint_name)
        for joint_name in right_foot_joint_names
    ]

    left_foot_position = joints[:, left_foot_joint_indices].mean(axis=1)
    right_foot_position = joints[:, right_foot_joint_indices].mean(axis=1)

    fps = config.rate.video
    dt_video = 1.0 / fps

    left_foot_velocity = (
        np.linalg.norm(np.diff(left_foot_position, axis=0), axis=-1)
        / dt_video
    )

    right_foot_velocity = (
        np.linalg.norm(np.diff(right_foot_position, axis=0), axis=-1)
        / dt_video
    )

    t_video_vel = np.arange(len(left_foot_velocity), dtype=float) * dt_video + dt_video / 2

    x_video = np.column_stack([left_foot_velocity, right_foot_velocity])

    return t_mocap_vel, x_mocap, t_video_vel, x_video


def parse_extra_feature(spec: Any):
    """
    Accept only explicit extra feature specs:
        {
            "array": arr,
            "mode": mode,
            "quat_order": "xyzw" or "wxyz", optional
            "allow_extrapolate": bool, optional
            "required": bool, optional
        }
    """
    if not isinstance(spec, dict):
        raise ValueError(
            "Extra features must be dicts with at least 'array' and 'mode'. "
            "Implicit interpolation inference is intentionally not supported."
        )
    if "array" not in spec:
        raise ValueError("Extra feature spec must contain 'array'.")
    if "mode" not in spec:
        raise ValueError("Extra feature spec must contain explicit 'mode'.")
    return (
        _as_numpy(spec["array"]),
        spec["mode"],
        spec.get("quat_order"),
        spec.get("allow_extrapolate", False),
        spec.get("required", True),
    )


def register_schema_features(
    stitcher: TimelineStitcher,
    source_data: Dict[str, Any],
    t: np.ndarray,
    schema: Dict[str, Dict[str, Any]],
):
    for output_key, spec in schema.items():
        array_key = spec["array_key"]
        required = spec.get("required", True)
        if array_key not in source_data:
            if required:
                raise KeyError(
                    f"Required feature {output_key} expects source key {array_key!r}, "
                    f"but it was not found."
                )
            continue
        x = _as_numpy(source_data[array_key])
        if output_key == "video/betas" and x.ndim == 1:
            x = np.repeat(x[None, :], len(t), axis=0)
        vbi = spec.get("validity_body_indices")
        if vbi is not None:
            vbi = tuple(int(i) for i in vbi)
        stitcher.add(
            key=output_key,
            t=t,
            x=x,
            mode=spec["mode"],
            quat_order=spec.get("quat_order"),
            allow_extrapolate=spec.get("allow_extrapolate", False),
            required=required,
            validity_body_indices=vbi,
        )


def resolve_vicon_schema_for_sync(
    vicon: Dict[str, Any],
    config: RetargetingConfig,
    user_schema: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Clone the vicon schema and, unless the caller set ``validity_body_indices`` on
    ``vicon/body_pos``, restrict crop=valid finiteness to rigid bodies used for sync
    (typically Left_Shoe / Right_Shoe) so a missing third body does not empty the run.
    """
    schema = deepcopy(DEFAULT_VICON_SCHEMA if user_schema is None else user_schema)
    if "vicon/body_pos" not in schema:
        return schema
    if schema["vicon/body_pos"].get("validity_body_indices") is not None:
        return schema
    names = vicon.get("body_names")
    if not names:
        return schema

    shoes: list[str] = []
    for k in config.time_sync_solver.smplx_joints:
        parts = k.split("/")
        if len(parts) >= 2:
            b = parts[1]
            if b not in shoes and b in names:
                shoes.append(b)
    if not shoes:
        for b in ("Left_Shoe", "Right_Shoe"):
            if b in names:
                shoes.append(b)
    idx = tuple(names.index(b) for b in shoes)
    if not idx:
        return schema
    entry = dict(schema["vicon/body_pos"])
    entry["validity_body_indices"] = idx
    schema["vicon/body_pos"] = entry
    return schema


def register_extra_features(
    stitcher: TimelineStitcher,
    features: Optional[Dict[str, Any]],
    t_default: np.ndarray,
):
    if not features:
        return
    for key, spec in features.items():
        arr, mode, quat_order, allow_extrapolate, required = parse_extra_feature(spec)
        if len(arr) != len(t_default):
            raise ValueError(
                f"extra feature {key!r} has length {len(arr)}, expected {len(t_default)}."
            )
        stitcher.add(
            key=key,
            t=t_default,
            x=arr,
            mode=mode,
            quat_order=quat_order,
            allow_extrapolate=allow_extrapolate,
            required=required,
        )


def build_unified_dataset(
    *,
    gvhmr_dir: Union[str, Path],
    vicon_path: Union[str, Path],
    config: RetargetingConfig,
    target_timeline: Union[str, np.ndarray] = "video",
    crop: CropMode = "support",
    extra_video_features: Optional[Dict[str, Any]] = None,
    extra_vicon_features: Optional[Dict[str, Any]] = None,
    lag: Optional[float] = None,
    standardize_sync_components: bool = True,
    video_schema: Optional[Dict[str, Dict[str, Any]]] = None,
    vicon_schema: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Build a fully synchronized dataset.

    Parameters
    ----------
    gvhmr_dir:
        Directory containing:
            joints.npy
            vertices.npy
            hmr4d_results.pt

    vicon_path:
        Path to merged Vicon npz.

    config:
        RetargetingConfig.

    target_timeline:
        "video":
            one row per video frame.

        "vicon":
            one row per Vicon sample, shifted into the unified video clock.

        np.ndarray:
            explicit target times in unified/video-clock coordinates.

    crop:
        "support" (default):
            keep target rows inside every source time range (overlap in time).

        "valid":
            keep only rows where every *required* resampled feature is finite.
            Optional schema streams (`required: false`, e.g. partial markers) are
            still resampled but do not gate cropping. For ``vicon/body_pos``,
            finiteness is evaluated only on bodies used for foot sync (see
            ``resolve_vicon_schema_for_sync``) unless ``validity_body_indices`` is set
            explicitly in the schema.

        "none":
            keep full target timeline, with NaNs outside support.

    extra_video_features:
        Optional dict of explicit specs, each:
            {"array": arr, "mode": mode, ...}
        Arrays must match the video timeline length.

    extra_vicon_features:
        Same shape as extra_video_features; arrays must match Vicon length.

    video_schema, vicon_schema:
        Optional overrides for which source keys map to output feature keys
        and how each is interpolated. Defaults are DEFAULT_VIDEO_SCHEMA and
        DEFAULT_VICON_SCHEMA.

    lag:
        Optional precomputed lag.
        If None, lag is estimated from foot-speed signals.

    Returns
    -------
    aligned:
        Dict of synchronized arrays.

    meta:
        Metadata dict containing lag, corr, body_names, etc.
    """

    vicon = load_vicon_data(vicon_path)
    gvhmr = load_gvhmr_data(gvhmr_dir)

    t_mocap_vel, x_mocap, t_video_vel, x_video = get_sync_signals(
        vicon,
        gvhmr,
        config,
    )

    corr = None
    sync_info = None

    if lag is None:
        tss = config.time_sync_solver
        lag, corr, sync_info = estimate_lag(
            t_mocap_vel,
            x_mocap,
            t_video_vel,
            x_video,
            standardize_components=standardize_sync_components,
            max_abs_lag_seconds=tss.max_abs_lag_seconds,
            motion_weighted=tss.motion_weighted_sync,
            motion_weight_floor_quantile=tss.motion_weight_floor_quantile,
        )

        if tss.min_correlation > 0.0 and corr is not None and corr < tss.min_correlation:
            raise ValueError(
                f"Time sync rejected: correlation {corr:.4f} is below configured "
                f"time_sync_solver.min_correlation ({tss.min_correlation}). "
                "The estimator may be locking onto quiet overlap; try motion_weighted_sync, "
                "motion_weight_floor_quantile, max_abs_lag_seconds, or pass an explicit lag=."
            )
        if tss.max_abs_lag_seconds is not None and abs(float(lag)) > float(tss.max_abs_lag_seconds) + 1e-9:
            raise ValueError(
                f"Time sync rejected: |lag|={abs(float(lag)):.6g}s exceeds "
                f"time_sync_solver.max_abs_lag_seconds ({tss.max_abs_lag_seconds})."
            )

    n_video = len(gvhmr["joints"])
    t_video = make_video_frame_times(n_video, config.rate.video)

    t_vicon = np.asarray(vicon["t"], dtype=float)

    # Lag convention from estimate_lag(t_mocap, x_mocap, t_video, x_video):
    #
    #     x_mocap(t) ~= x_video(t - lag)
    #
    # Therefore Vicon timestamps expressed in video-clock coordinates are:
    #
    #     t_vicon_unified = t_vicon - lag
    #
    # Yes, this sign matters. Yes, it will ruin your day if flipped.
    t_vicon_unified = t_vicon - lag

    vicon_schema_resolved = resolve_vicon_schema_for_sync(vicon, config, vicon_schema)

    stitcher = TimelineStitcher()

    stitcher.add_source_time("video", t_video)
    stitcher.add_source_time("vicon", t_vicon_unified)

    register_schema_features(
        stitcher,
        gvhmr,
        t_video,
        DEFAULT_VIDEO_SCHEMA if video_schema is None else video_schema,
    )
    register_schema_features(
        stitcher,
        vicon,
        t_vicon_unified,
        vicon_schema_resolved,
    )
    register_extra_features(
        stitcher,
        extra_video_features,
        t_video,
    )
    register_extra_features(
        stitcher,
        extra_vicon_features,
        t_vicon_unified,
    )

    aligned = stitcher.build(
        timeline=target_timeline,
        crop=crop,
        include_valid_masks=True,
    )

    meta = {
        "lag": float(lag),
        "corr": None if corr is None else float(corr),
        "sync_info": sync_info,
        "body_names": vicon.get("body_names"),
        "body_pos_validity_indices": vicon_schema_resolved.get("vicon/body_pos", {}).get(
            "validity_body_indices"
        ),
        "target_timeline": target_timeline
        if isinstance(target_timeline, str)
        else "custom",
        "crop": crop,
        "video_fps": config.rate.video,
    }

    return aligned, meta


def save_aligned_npz(
    path: Union[str, Path],
    aligned: Dict[str, Any],
    meta: Dict[str, Any],
):
    """
    Save synchronized arrays to compressed npz.

    Slashes in keys are replaced with double underscores:
        video/joints -> video__joints
        vicon/body_pos -> vicon__body_pos
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, Any] = {}

    for key, value in aligned.items():
        if key == "__valid_masks__":
            continue

        safe_key = key.replace("/", "__")
        arrays[safe_key] = value

    arrays["lag"] = np.array(meta["lag"])

    if meta.get("corr") is not None:
        arrays["corr"] = np.array(meta["corr"])

    if meta.get("body_names") is not None:
        arrays["vicon__body_names"] = np.array(meta["body_names"], dtype=object)

    np.savez_compressed(path, **arrays)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion convention [w, x, y, z] to scipy convention [x, y, z, w].
    """
    q = np.asarray(q)
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """
    Convert scipy convention [x, y, z, w] to [w, x, y, z].
    """
    q = np.asarray(q)
    return np.concatenate([q[..., 3:], q[..., :3]], axis=-1)


def finite_time_mask(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim == 1:
        return np.isfinite(x)

    return np.isfinite(x).all(axis=tuple(range(1, x.ndim)))


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def _as_time_array(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)

    if t.ndim != 1:
        raise ValueError(f"Time array must be 1D, got shape {t.shape}.")

    if len(t) == 0:
        raise ValueError("Time array is empty.")

    return t


def _sort_and_merge_duplicate_times(
    t: np.ndarray,
    x: np.ndarray,
):
    order = np.argsort(t)
    t = t[order]
    x = x[order]

    unique_t, inverse = np.unique(t, return_inverse=True)

    if len(unique_t) == len(t):
        return t, x

    if not np.issubdtype(x.dtype, np.number):
        # For non-numeric data, keep the first value at each timestamp.
        keep = np.zeros(len(unique_t), dtype=int)
        seen = np.zeros(len(unique_t), dtype=bool)

        for i, group in enumerate(inverse):
            if not seen[group]:
                keep[group] = i
                seen[group] = True

        return unique_t, x[keep]

    x_flat = x.reshape(len(t), -1)
    merged = np.zeros((len(unique_t), x_flat.shape[1]), dtype=float)
    counts = np.zeros(len(unique_t), dtype=float)

    np.add.at(merged, inverse, x_flat)
    np.add.at(counts, inverse, 1)

    merged /= counts[:, None]
    merged = merged.reshape((len(unique_t),) + x.shape[1:])

    return unique_t, merged


def _assign_nan_on_time_mask(y: np.ndarray, mask: np.ndarray):
    y = np.asarray(y)

    if not np.any(mask):
        return y

    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(float)

    y[mask] = np.nan
    return y


def _has_time_axis(v: Any, expected_len: int) -> bool:
    return isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == expected_len


def _crop_time_axis(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return v[mask]