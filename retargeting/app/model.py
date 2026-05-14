from dataclasses import replace
from pathlib import Path
from re import M

import matplotlib.pyplot as plt
import numpy as np
import pickle
import typer

from rich.console import Console
from rich.pretty import pprint
from rich.table import Table

from retargeting.rigid_body_model_estimator import (
    MarkerTracks,
    RigidBodyModel,
    _tracks_to_arrays,
    estimate_rigid_body_model,
)

from retargeting.config import load_config
from retargeting import constants

model_app = typer.Typer(help="Commands for modeling the rigid body system.")

console = Console()


def _frame_segment_boundaries(frame_number: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start indices of each contiguous frame block and block lengths (preserves row order)."""
    change = np.flatnonzero(np.r_[True, frame_number[1:] != frame_number[:-1]])
    starts = change
    ends = np.r_[change[1:], len(frame_number)]
    lengths = ends - starts
    return starts, lengths


def _build_body_tracks(
    body: str,
    markers: list[str],
    starts: np.ndarray,
    lengths: np.ndarray,
    marker_name: np.ndarray,
    subject_name: np.ndarray,
    xyz: np.ndarray,
    occluded: np.ndarray,
) -> MarkerTracks:
    """One time series per marker (length = number of frames); None when missing or occluded."""
    t_count = len(starts)
    tracks: MarkerTracks = {m: [None] * t_count for m in markers}
    for t, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        end = int(start + length)
        for r in range(int(start), end):
            if str(subject_name[r]) != body:
                continue
            mn = str(marker_name[r])
            if mn not in tracks:
                continue
            if bool(np.asarray(occluded[r]).item()):
                tracks[mn][t] = None
            else:
                p = xyz[r]
                tracks[mn][t] = (float(p[0]), float(p[1]), float(p[2]))
    return tracks


def _joint_and_marginal_counts_from_visible(visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Co-visible frame counts and per-marker visible frame counts."""
    t, n = visible.shape
    joint = np.zeros((n, n), dtype=np.int64)
    for ti in range(t):
        v = visible[ti].astype(np.int64)
        joint += np.outer(v, v)
    vis_marginal = visible.sum(axis=0).astype(np.int64)
    return joint, vis_marginal


def _finalize_covisibility_probs(
    joint_counts: dict[str, np.ndarray],
    vis_marginal_counts: dict[str, np.ndarray],
    num_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Turn per-frame counts into empirical probabilities (same marker order as each body).

    ``joint_covisibility_probs[i, j]``: fraction of frames in which markers *i* and *j* are both
    visible (not occluded, finite position in that frame).

    ``marginal_covisibility_probs[i, j]``: among frames where marker *i* is visible, the fraction
    where *j* is also visible, i.e. P(j | i). Rows for markers *i* that never appear visible are NaN.
    """
    joint_probs: dict[str, np.ndarray] = {}
    marginal_probs: dict[str, np.ndarray] = {}
    if num_frames <= 0:
        for body, jc in joint_counts.items():
            shp = jc.shape
            joint_probs[body] = np.full(shp, np.nan, dtype=np.float64)
            marginal_probs[body] = np.full(shp, np.nan, dtype=np.float64)
        return joint_probs, marginal_probs
    nf = float(num_frames)
    for body, jc in joint_counts.items():
        jc_f = jc.astype(np.float64)
        joint_probs[body] = jc_f / nf
        vm = vis_marginal_counts[body].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            marginal_probs[body] = np.where(vm[:, None] > 0, jc_f / vm[:, None], np.nan)
    return joint_probs, marginal_probs


def _plot_nominal_body_markers(body: str, model: RigidBodyModel):
    """3D scatter of fitted nominal marker positions (estimator body frame)."""
    pts = model.marker_positions

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{body} (stress={model.stress:.3g})")

    for i, name in enumerate(model.marker_names):
        ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2], label=name)

    center = pts.mean(axis=0)
    half = float(np.max(np.linalg.norm(pts - center, axis=1)) * 1.05 + 1e-9)

    cmap = plt.get_cmap("tab10")
    for pi, (_, plane) in enumerate(model.planes.items()):
        color = cmap(pi % 10)
        xg = np.linspace(center[0] - half, center[0] + half, 10)
        yg = np.linspace(center[1] - half, center[1] + half, 10)
        Xg, Yg = np.meshgrid(xg, yg)
        Zg = plane.coefficients[0] * Xg + plane.coefficients[1] * Yg + plane.intercept
        ax.plot_surface(Xg, Yg, Zg, alpha=0.45, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    fig.tight_layout()


@model_app.command(help="Model the rigid body system.")
def bodies(
    demo_vicon_tables_dir: Path = typer.Argument(..., help="Path to the demo Vicon tables directory."),
    rigid_body_models_dir: Path = typer.Argument(..., help="Path to the output rigid body models directory."),
    config_path: Path = typer.Option(constants.DEFAULT_CONFIG_PATH, help="Path to the configuration file."),
    plot: bool = typer.Option(False, help="Plot the rigid body models."),
    verbose: bool = typer.Option(False, help="Print verbose output."),
):
    config = load_config(config_path)
    markers_path = demo_vicon_tables_dir / "vicon" / "markers.npz"
    data = np.load(markers_path, allow_pickle=True)

    subject_name = np.char.strip(np.asarray(data["subject_name"], dtype=str))
    marker_name = data["marker_name"]
    frame_number = data["frame_number"]
    xyz = np.asarray(data["xyz"], dtype=np.float64)
    occluded = data["occluded"]

    body_marker_names: dict[str, list[str]] = {}
    for subj in np.unique(subject_name):
        body = str(subj)
        if not body or body.lower() == "nan":
            continue
        rows = subject_name == subj
        markers = [str(m) for m in np.unique(marker_name[rows]).tolist()]
        body_marker_names[body] = markers

    if verbose:
        table = Table()
        table.add_column("Body")
        table.add_column("Marker Count")
        table.add_column("Marker Names")

        for body, markers in body_marker_names.items():
            table.add_row(body, str(len(markers)), ", ".join(markers))

        console.print(table)

    starts, lengths = _frame_segment_boundaries(np.asarray(frame_number))
    num_frames = len(starts)

    rigid_models: dict = {}
    joint_counts: dict[str, np.ndarray] = {}
    vis_marginal_counts: dict[str, np.ndarray] = {}

    for body, markers_all in body_marker_names.items():
        body_cfg = config.bodies.get(body)
        if body_cfg is not None:
            present = set(markers_all)
            markers = [m for m in body_cfg.markers if m in present]
            if len(markers) != len(body_cfg.markers) and verbose:
                missing = [m for m in body_cfg.markers if m not in present]
                console.print(
                    f"[yellow]{body}: config lists {len(body_cfg.markers)} markers, "
                    f"{len(missing)} absent from this trial ({', '.join(missing)}).[/yellow]"
                )
            planes_spec = body_cfg.planes
        else:
            markers = markers_all
            planes_spec = {}

        if len(markers) < 4:
            console.print(f"[yellow]Skip {body}: need at least 4 markers, got {len(markers)}.[/yellow]")
            continue

        tracks = _build_body_tracks(
            body, markers, starts, lengths, marker_name, subject_name, xyz, occluded
        )
        _, _, visible = _tracks_to_arrays(tracks)
        jc, vm = _joint_and_marginal_counts_from_visible(visible)
        joint_counts[body] = jc
        vis_marginal_counts[body] = vm

        try:
            rigid_models[body] = estimate_rigid_body_model(tracks, body, planes=planes_spec)
        except ValueError as e:
            console.print(f"[red]{body}: rigid fit failed: {e}[/red]")

    joint_covisibility_probs, marginal_covisibility_probs = _finalize_covisibility_probs(
        joint_counts, vis_marginal_counts, num_frames
    )

    for body in rigid_models:
        rigid_models[body] = replace(
            rigid_models[body],
            joint_covisibility_probs=joint_covisibility_probs[body],
            marginal_covisibility_probs=marginal_covisibility_probs[body],
        )

    distance_matrices = {b: m.distance_matrix for b, m in rigid_models.items()}

    for body, D in distance_matrices.items():
        assert np.allclose(np.nan_to_num(D - D.T), 0.0, atol=1e-9)

    for body, joint_covisibility_prob in joint_covisibility_probs.items():
        assert np.allclose(joint_covisibility_prob, joint_covisibility_prob.T)

    # It technically could be symmetric, but it is incredibly unlikely
    for body, marginal_covisibility_prob in marginal_covisibility_probs.items():
        if np.allclose(marginal_covisibility_prob, marginal_covisibility_prob.T):
            console.print(f"[yellow]Warning: {body} marginal covisibility matrix is symmetric. That's highly unlikely in real data.[/yellow]")

    if verbose:
        pprint(
            {
                "rigid_models": rigid_models,
                "joint_covisibility_probs": joint_covisibility_probs,
                "marginal_covisibility_probs": marginal_covisibility_probs,
            }
        )

    if verbose:
        for body, model in rigid_models.items():
            if not model.planes:
                console.print(f"[cyan]{body}[/cyan] stress={model.stress:.6g}")
            else:
                plane_s = ", ".join(f"{n}={p.stress:.3g}" for n, p in model.planes.items())
                console.print(f"[cyan]{body}[/cyan] stress={model.stress:.6g}  planes({plane_s})")

    if plot:
        for body, model in rigid_models.items():
            _plot_nominal_body_markers(body, model)

        if rigid_models:
            if plt.get_backend().lower() == "agg":
                plt.close("all")
            else:
                plt.show()
    
    # Check that all stresses are below the threshold
    max_plane_stress = config.rigid_body_solver.plane_solver.max_stress
    for body, model in rigid_models.items():
        if model.stress > config.rigid_body_solver.max_stress:
            console.print(
                f"[red]Warning: {body} stress is above the threshold "
                f"({model.stress:.6g} > {config.rigid_body_solver.max_stress:.6g}). "
                f"Demo: {demo_vicon_tables_dir.parts[-1]}[/red]"
            )

        for plane_name, plane in model.planes.items():
            if plane.stress > max_plane_stress:
                console.print(
                    f"[red]Warning: {body} plane {plane_name!r} stress is above the threshold "
                    f"({plane.stress:.6g} > {max_plane_stress:.6g}). "
                    f"Demo: {demo_vicon_tables_dir.parts[-1]}[/red]"
                )

    # Save the rigid body models to a file
    # Get the last part of demo_vicon_tables_dir
    demo_name = demo_vicon_tables_dir.parts[-1]
    rigid_body_models_path = rigid_body_models_dir / demo_name
    rigid_body_models_path.mkdir(parents=True, exist_ok=True)
    for body, model in rigid_models.items():
        with open(rigid_body_models_path / f"{body}.pkl", "wb") as file:
            pickle.dump(model, file)