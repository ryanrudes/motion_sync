"""
This script estimates a rigid body model from partially-occluded samples
that are assumed to have constant pairwise relative distances between
markers over time.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

Point3 = Tuple[float, float, float]
MarkerTracks = Dict[str, List[Optional[Point3]]]


@dataclass
class Plane:
    coefficients: np.ndarray
    intercept: float
    stress: float


@dataclass
class RigidBodyModel:
    """Nominal rigid marker geometry plus diagnostics from the fit and optional session covisibility."""

    name: str
    marker_names: list[str]
    marker_positions: np.ndarray
    distance_matrix: np.ndarray
    weight_matrix: np.ndarray
    stress: float
    joint_covisibility_probs: np.ndarray | None = None
    marginal_covisibility_probs: np.ndarray | None = None
    planes: dict[str, Plane] = field(default_factory=dict)

    @property
    def positions(self) -> dict[str, np.ndarray]:
        """Per-marker 3D positions (rows of ``marker_positions``), as a mutable-friendly copy."""
        return {name: self.marker_positions[i].copy() for i, name in enumerate(self.marker_names)}


def fit_plane(
    marker_positions: np.ndarray, marker_names: list[str], markers: list[str]
) -> Plane:
    idx = [marker_names.index(m) for m in markers]
    X = marker_positions[idx]
    y = X[:, 2]
    X = X[:, :2]
    model = LinearRegression().fit(X, y)
    # Compute the stress of the plane
    residuals = y - model.predict(X)
    stress = np.mean(residuals ** 2)
    return Plane(coefficients=model.coef_, intercept=model.intercept_, stress=stress)


def fit_body_planes(
    marker_positions: np.ndarray,
    marker_names: list[str],
    planes: dict[str, list[str]] | None,
    *,
    body: str,
) -> dict[str, Plane]:
    """Fit each configured plane using markers that appear in ``marker_names`` (need at least 3)."""
    if not planes:
        return {}
    available = set(marker_names)
    fitted: dict[str, Plane] = {}
    for label, markers in planes.items():
        use = [m for m in markers if m in available]
        if len(use) < 3:
            missing = [m for m in markers if m not in available]
            warnings.warn(
                f"{body}: plane {label!r} was not fitted — need at least 3 markers that appear "
                f"in the rigid model, but only {len(use)} do ({use!r}). "
                f"Configured markers for this plane: {markers!r}; "
                f"configured but absent from the model: {missing!r}.",
                UserWarning,
                stacklevel=2,
            )
            continue
        fitted[label] = fit_plane(marker_positions, marker_names, use)
    return fitted


def _tracks_to_arrays(
    tracks: MarkerTracks,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not tracks:
        raise ValueError("tracks is empty.")

    names = list(tracks.keys())
    lengths = [len(tracks[name]) for name in names]

    if len(set(lengths)) != 1:
        raise ValueError("All marker tracks must have the same number of timesteps.")

    T = lengths[0]
    N = len(names)

    data = np.full((T, N, 3), np.nan, dtype=np.float64)
    visible = np.zeros((T, N), dtype=bool)

    for j, name in enumerate(names):
        for t, p in enumerate(tracks[name]):
            if p is None:
                continue

            arr = np.asarray(p, dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError(
                    f"Marker {name!r}, frame {t}: expected a 3-tuple, got shape {arr.shape}."
                )

            data[t, j] = arr
            visible[t, j] = True

    return names, data, visible


def _robust_distance_estimate(
    values: np.ndarray,
    mad_threshold: float = 4.0,
) -> tuple[float, float]:
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.inf

    med = np.median(values)
    abs_dev = np.abs(values - med)
    mad = np.median(abs_dev)

    sigma = 1.4826 * mad

    if sigma > 1e-12:
        keep = abs_dev <= mad_threshold * sigma
        if np.any(keep):
            filtered = values[keep]
            med = np.median(filtered)
            sigma = 1.4826 * np.median(np.abs(filtered - med))

    return float(med), float(sigma)


def estimate_pairwise_distances(
    tracks: MarkerTracks,
    min_common_frames: int = 10,
    mad_threshold: float = 4.0,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Robustly estimate inter-marker distances from co-visible frames.

    Returns:
        names
        D: pairwise distance matrix, NaN where unavailable
        W: confidence weight matrix
    """
    names, data, visible = _tracks_to_arrays(tracks)
    _, N, _ = data.shape

    D = np.full((N, N), np.nan, dtype=np.float64)
    W = np.zeros((N, N), dtype=np.float64)

    np.fill_diagonal(D, 0.0)

    for i in range(N):
        for j in range(i + 1, N):
            mask = visible[:, i] & visible[:, j]
            count = int(mask.sum())

            if count < min_common_frames:
                continue

            diffs = data[mask, i] - data[mask, j]
            dists = np.linalg.norm(diffs, axis=1)

            dij, sigma = _robust_distance_estimate(
                dists,
                mad_threshold=mad_threshold,
            )

            if not np.isfinite(dij):
                continue

            D[i, j] = D[j, i] = dij

            # Confidence increases with co-visibility and decreases with relative jitter.
            rel_sigma = sigma / max(dij, 1e-12)
            weight = count / (rel_sigma + 1e-3)

            # Cap so one ultra-stable pair does not bully the entire optimization.
            weight = min(weight, 1e5)

            W[i, j] = W[j, i] = weight

    return names, D, W


def _classical_mds_init(D: np.ndarray, W: np.ndarray, dim: int = 3) -> np.ndarray:
    """
    Initial coordinates from classical MDS.

    Missing distances are crudely filled using shortest-path distances on the
    observed distance graph, then remaining missing values are filled by median.
    This is just an initializer, not the final estimator.
    """
    N = D.shape[0]

    filled = np.array(D, copy=True)

    finite_positive = filled[np.isfinite(filled) & (filled > 0)]
    fallback = float(np.median(finite_positive)) if finite_positive.size else 1.0

    # Floyd-Warshall-style shortest path fill over observed distances.
    dist = np.full((N, N), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)

    observed = np.isfinite(D) & (W > 0)
    dist[observed] = D[observed]

    for k in range(N):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])

    filled[np.isfinite(dist)] = dist[np.isfinite(dist)]
    filled[~np.isfinite(filled)] = fallback
    np.fill_diagonal(filled, 0.0)

    D2 = filled ** 2

    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:dim]

    vals = np.maximum(eigvals[idx], 1e-12)
    X = eigvecs[:, idx] * np.sqrt(vals)

    return X


def _center_and_fix_orientation(X: np.ndarray) -> np.ndarray:
    """
    Remove arbitrary translation and choose a stable orientation convention.

    The output is still only defined up to reflection if the data itself does not
    distinguish chirality. That is unavoidable, because geometry enjoys having
    one last annoying loophole.
    """
    X = X - X.mean(axis=0, keepdims=True)

    # PCA-align for a stable coordinate basis.
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    X = X @ Vt.T

    # Deterministic sign convention.
    for axis in range(X.shape[1]):
        idx = np.argmax(np.abs(X[:, axis]))
        if X[idx, axis] < 0:
            X[:, axis] *= -1

    return X


def _pack_gauge_fixed(X: np.ndarray) -> np.ndarray:
    """
    Gauge fixing for nonlinear optimization.

    Fix:
        q0 = (0, 0, 0)
        q1 lies on +x axis
        q2 lies in xy plane with y >= 0

    Variables:
        q1_x
        q2_x, q2_y
        q3_x, q3_y, q3_z
        ...
    """
    N = X.shape[0]

    X = X.copy()
    X = X - X[0]

    # Build a local basis from first non-degenerate markers.
    e1 = X[1]
    n1 = np.linalg.norm(e1)
    if n1 < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = e1 / n1

    v2 = X[2] - np.dot(X[2], e1) * e1
    n2 = np.linalg.norm(v2)
    if n2 < 1e-12:
        # Pick arbitrary perpendicular vector.
        tmp = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(tmp, e1)) > 0.9:
            tmp = np.array([0.0, 0.0, 1.0])
        v2 = tmp - np.dot(tmp, e1) * e1
        n2 = np.linalg.norm(v2)

    e2 = v2 / n2
    e3 = np.cross(e1, e2)

    R = np.column_stack([e1, e2, e3])
    Xg = X @ R

    if Xg[2, 1] < 0:
        Xg[:, 1] *= -1
        Xg[:, 2] *= -1

    params = [max(Xg[1, 0], 1e-9), Xg[2, 0], max(Xg[2, 1], 1e-9)]

    for i in range(3, N):
        params.extend(Xg[i])

    return np.asarray(params, dtype=np.float64)


def _unpack_gauge_fixed(params: np.ndarray, N: int) -> np.ndarray:
    X = np.zeros((N, 3), dtype=np.float64)

    X[0] = [0.0, 0.0, 0.0]
    X[1] = [params[0], 0.0, 0.0]
    X[2] = [params[1], params[2], 0.0]

    k = 3
    for i in range(3, N):
        X[i] = params[k:k + 3]
        k += 3

    return X


def _distance_residuals_3d(
    params: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    N = D.shape[0]
    X = _unpack_gauge_fixed(params, N)

    residuals = []

    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] <= 0 or not np.isfinite(D[i, j]):
                continue

            pred = np.linalg.norm(X[i] - X[j])
            obs = D[i, j]

            residuals.append(np.sqrt(W[i, j]) * (pred - obs))

    return np.asarray(residuals, dtype=np.float64)


def estimate_rigid_body_model(
    tracks: MarkerTracks,
    body: str,
    *,
    planes: dict[str, list[str]] | None = None,
    min_common_frames: int = 10,
    mad_threshold: float = 4.0,
    n_restarts: int = 8,
    seed: int = 0,
) -> RigidBodyModel:
    """
    Estimate a nominal 3D rigid body marker model from occluded trajectories.

    This does NOT require:
        - plane separation
        - ground height
        - lower/upper marker labels
        - ground alignment

    The returned model is in an arbitrary canonical coordinate frame.
    It is suitable as the nominal rigid body model for later per-frame
    registration/restoration.

    ``planes``: map each plane label to the marker names used to fit that plane
    (markers must appear in ``tracks``; at least three per plane after filtering).
    """
    names, D, W = estimate_pairwise_distances(
        tracks,
        min_common_frames=min_common_frames,
        mad_threshold=mad_threshold,
    )

    N = len(names)

    if N < 4:
        raise ValueError("Need at least 4 markers for a stable 3D rigid model.")

    num_edges = int(np.sum(W > 0) // 2)
    if num_edges < 3 * N - 6:
        raise ValueError(
            f"Too few reliable pairwise distances: got {num_edges}, "
            f"roughly need at least {3 * N - 6} for a well-constrained 3D model."
        )

    rng = np.random.default_rng(seed)

    X0 = _classical_mds_init(D, W, dim=3)
    X0 = _center_and_fix_orientation(X0)

    finite_distances = D[np.isfinite(D) & (D > 0)]
    scale = float(np.median(finite_distances)) if finite_distances.size else 1.0

    best_X = None
    best_stress = np.inf

    for r in range(n_restarts):
        if r == 0:
            X_start = X0
        else:
            X_start = X0 + rng.normal(scale=0.05 * scale, size=X0.shape)

        p0 = _pack_gauge_fixed(X_start)

        result = least_squares(
            _distance_residuals_3d,
            p0,
            args=(D, W),
            loss="huber",
            f_scale=1.0,
            max_nfev=1000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )

        residuals = _distance_residuals_3d(result.x, D, W)
        stress = float(np.mean(residuals ** 2)) if residuals.size else np.inf

        if stress < best_stress:
            best_stress = stress
            best_X = _unpack_gauge_fixed(result.x, N)

    if best_X is None:
        raise RuntimeError("Rigid body model estimation failed.")

    best_X = _center_and_fix_orientation(best_X)

    marker_positions = best_X.copy()
    planes_fitted = fit_body_planes(marker_positions, names, planes, body=body)

    return RigidBodyModel(
        name=body,
        marker_names=names,
        marker_positions=marker_positions,
        distance_matrix=D,
        weight_matrix=W,
        stress=best_stress,
        planes=planes_fitted,
    )
