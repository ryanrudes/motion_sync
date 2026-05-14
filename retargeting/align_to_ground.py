from rigid_body_model_estimator import estimate_rigid_body_model
from ground_alignment_solver import (
    # build_ground_aligned_canonical_model,
    # estimate_restorative_transforms,
    # apply_restorative_transforms,
    GroundCorrectionTransform,
    estimate_pose_preserving_ground_corrections,
    apply_ground_corrections,
)

from pathlib import Path

import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ground-align Vicon skateboard markers (flat lower plane) and write corrected "
            "marker tracks plus rigid Shoe/Skateboard poses."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Directory containing markers_long.csv, conversion_summary.csv, and rigid_bodies.csv "
            "(typically …/vicon_csvs/<trial>)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Directory to create/write ground_aligned.npz, ground_aligned.json, "
            "nominal_model.pkl, and corrections.pkl (e.g. …/ground_aligned/<trial>)"
        ),
    )
    parser.add_argument(
        "--marker-linear-scale",
        type=float,
        default=1e-3,
        metavar="S",
        help=(
            "Multiply marker x,y,z from markers_long.csv by S to get metres. "
            "Default 1e-3: Vicon-style millimetres in CSV, same space as corrections. "
            "Rigid bodies in rigid_bodies.csv are always metres. Use 1.0 if markers are already metres."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib diagnostics (non-interactive / batch runs).",
    )
    return parser


WHEEL_MARKER_NAMES = [
    "rear_left_wheel",
    "rear_right_wheel",
    "front_left_wheel",
    "front_right_wheel",
]

POLE_MARKER_NAMES = [
    "rear_left_pole_left",
    "rear_left_pole_right",
    "rear_right_pole_left",
    "rear_right_pole_right",
]


def get_marker_names(demo_path: Path) -> tuple[set[str], set[str]]:
    summary_path = demo_path / "conversion_summary.csv"
    summary_data = pd.read_csv(summary_path)

    names = summary_data["name"].tolist()
    skateboard_marker_names = []
    shoe_marker_names = []
    for name in names:
        if name.startswith("Skateboard/"):
            skateboard_marker_names.append(name.split("/")[-1])
        elif name.startswith("Shoe/"):
            shoe_marker_names.append(name.split("/")[-1])
    if not skateboard_marker_names:
        raise SystemExit("No Skateboard/ markers listed in conversion_summary.csv")
    if not shoe_marker_names:
        raise SystemExit("No Shoe/ markers listed in conversion_summary.csv")
    
    return set(skateboard_marker_names), set(shoe_marker_names)


def load_samples(demo_path: Path, *, marker_linear_scale: float) -> tuple[list[dict], set[str], set[str]]:
    """
    Load marker frames. Positions are converted to **metres** via ``marker_linear_scale``
    (default ``1e-3``: ``markers_long`` xyz in millimetres, matching typical Vicon exports).
    """
    if marker_linear_scale <= 0.0:
        raise ValueError("marker_linear_scale must be positive")
    skateboard_marker_names, shoe_marker_names = get_marker_names(demo_path)

    marker_data = pd.read_csv(demo_path / "markers_long.csv")

    samples = []
    sample = None
    for _, row in marker_data.iterrows():
        row_time = float(row["time_from_start_s"])

        if sample is None or row_time != sample["time"]:
            if sample is not None:
                samples.append(sample)
            skateboard_marker_positions = {name: None for name in skateboard_marker_names}
            shoe_marker_positions = {name: None for name in shoe_marker_names}
            sample = dict(time=row_time, Skateboard=skateboard_marker_positions, Shoe=shoe_marker_positions)
        
        subject_name = row["subject_name"]
        marker_name = row["marker_name"]

        if subject_name not in {"Skateboard", "Shoe"}:
            continue

        occluded = bool(row["occluded"])

        if occluded:
            position = None
        else:
            position = (
                np.array([row["x"], row["y"], row["z"]], dtype=np.float64) * marker_linear_scale
            )
        
        sample[subject_name][marker_name] = position

    if sample is not None:
        samples.append(sample)

    if not samples:
        raise SystemExit("No frames loaded from markers_long.csv")

    return samples, skateboard_marker_names, shoe_marker_names


RIGID_SHOE_ID = "vicon/Shoe/Shoe"
RIGID_SKATEBOARD_ID = "vicon/Skateboard/Skateboard"


def load_rigid_body_pose_table(rb_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rigid-body rows from ``rigid_bodies.csv`` on the **shared** Vicon rigid clock.

    Returns ``(time_from_start_s, shoe_pose, skateboard_pose)`` with each pose ``(T, 7)`` as
    ``[x, y, z, qw, qx, qy, qz]`` (same quaternion order as ``time_alignment_solver``).
    """
    df = pd.read_csv(rb_csv)
    need = {"child_frame_id", "time_from_start_s", "x", "y", "z", "qx", "qy", "qz", "qw"}
    if not need.issubset(df.columns):
        raise ValueError(f"rigid_bodies.csv missing columns {need}: {rb_csv}")

    shoe = df[df["child_frame_id"] == RIGID_SHOE_ID].copy()
    board = df[df["child_frame_id"] == RIGID_SKATEBOARD_ID].copy()
    if shoe.empty:
        raise SystemExit(f"No rows with child_frame_id={RIGID_SHOE_ID!r} in {rb_csv}")
    if board.empty:
        raise SystemExit(f"No rows with child_frame_id={RIGID_SKATEBOARD_ID!r} in {rb_csv}")

    shoe = shoe.sort_values("time_from_start_s").drop_duplicates("time_from_start_s", keep="last")
    board = board.sort_values("time_from_start_s").drop_duplicates("time_from_start_s", keep="last")

    merged = pd.merge(
        shoe[["time_from_start_s", "x", "y", "z", "qx", "qy", "qz", "qw"]].rename(
            columns={
                "x": "sx",
                "y": "sy",
                "z": "sz",
                "qx": "sqx",
                "qy": "sqy",
                "qz": "sqz",
                "qw": "sqw",
            }
        ),
        board[["time_from_start_s", "x", "y", "z", "qx", "qy", "qz", "qw"]].rename(
            columns={
                "x": "bx",
                "y": "by",
                "z": "bz",
                "qx": "bqx",
                "qy": "bqy",
                "qz": "bqz",
                "qw": "bqw",
            }
        ),
        on="time_from_start_s",
        how="inner",
    )
    if merged.empty:
        raise SystemExit("No overlapping rigid-body timestamps between Shoe and Skateboard.")

    t = merged["time_from_start_s"].to_numpy(dtype=np.float64)
    # CSV quaternions are xyzw; store as wxyz for downstream tools.
    shoe_pose = np.column_stack(
        [
            merged["sx"].to_numpy(dtype=np.float64),
            merged["sy"].to_numpy(dtype=np.float64),
            merged["sz"].to_numpy(dtype=np.float64),
            merged["sqw"].to_numpy(dtype=np.float64),
            merged["sqx"].to_numpy(dtype=np.float64),
            merged["sqy"].to_numpy(dtype=np.float64),
            merged["sqz"].to_numpy(dtype=np.float64),
        ]
    )
    board_pose = np.column_stack(
        [
            merged["bx"].to_numpy(dtype=np.float64),
            merged["by"].to_numpy(dtype=np.float64),
            merged["bz"].to_numpy(dtype=np.float64),
            merged["bqw"].to_numpy(dtype=np.float64),
            merged["bqx"].to_numpy(dtype=np.float64),
            merged["bqy"].to_numpy(dtype=np.float64),
            merged["bqz"].to_numpy(dtype=np.float64),
        ]
    )
    return t, shoe_pose, board_pose


def _forward_fill_corrections(
    corrections: list[GroundCorrectionTransform],
) -> tuple[np.ndarray, np.ndarray]:
    N = len(corrections)
    Rs = np.zeros((N, 3, 3), dtype=np.float64)
    ts = np.zeros((N, 3), dtype=np.float64)
    R_last = np.eye(3, dtype=np.float64)
    t_last = np.zeros(3, dtype=np.float64)
    for i, c in enumerate(corrections):
        if c.valid:
            R_last = np.asarray(c.R_correction, dtype=np.float64)
            t_last = np.asarray(c.t_correction, dtype=np.float64)
        Rs[i] = R_last
        ts[i] = t_last
    return Rs, ts


def _orthonormalize_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U = U.copy()
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def ground_correction_rt_at_times(
    times_marker: np.ndarray,
    corrections: list[GroundCorrectionTransform],
    times_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolated ground-correction rotation and translation at ``times_query`` (same math as
    rigid export in this module). Use to map GMR / video-world FK poses into the ground-aligned
    Vicon frame before comparing to ``rigid_*`` or markers in ``ground_aligned.npz``.
    """
    return _interp_corrections_at_times(times_marker, corrections, times_query)


def _interp_corrections_at_times(
    times_marker: np.ndarray,
    corrections: list[GroundCorrectionTransform],
    times_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate forward-filled ``R_correction`` / ``t_correction`` onto ``times_query``."""
    Rs, ts = _forward_fill_corrections(corrections)
    tm = np.asarray(times_marker, dtype=np.float64).ravel()
    if tm.shape[0] != Rs.shape[0]:
        raise ValueError("times_marker must match corrections length.")
    tq = np.asarray(times_query, dtype=np.float64).ravel()
    R_flat = Rs.reshape(-1, 9)
    Rq_flat = np.zeros((tq.shape[0], 9), dtype=np.float64)
    for j in range(9):
        f = interp1d(tm, R_flat[:, j], kind="linear", fill_value="extrapolate", assume_sorted=True)
        Rq_flat[:, j] = np.asarray(f(tq), dtype=np.float64)
    Rq = Rq_flat.reshape(-1, 3, 3)
    Rq = np.stack([_orthonormalize_rotation(Rq[k]) for k in range(Rq.shape[0])], axis=0)
    t_out = np.zeros((tq.shape[0], 3), dtype=np.float64)
    for j in range(3):
        f = interp1d(tm, ts[:, j], kind="linear", fill_value="extrapolate", assume_sorted=True)
        t_out[:, j] = np.asarray(f(tq), dtype=np.float64)
    return Rq, t_out


def _pose7_wxyz_to_Rt(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``pose`` (T,7) as ``x,y,z,qx,qy,qz,qw`` → ``R`` (T,3,3), ``t`` (T,3)."""
    t = pose[:, :3].astype(np.float64)
    qxyzw = pose[:, 3:7].astype(np.float64)
    R = Rotation.from_quat(qxyzw).as_matrix()
    return R, t


def _Rt_to_pose7_wxyz(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    qxyzw = Rotation.from_matrix(R).as_quat()
    qwxyz = np.column_stack([qxyzw[:, 3], qxyzw[:, 0], qxyzw[:, 1], qxyzw[:, 2]])
    return np.hstack([t.astype(np.float64), qwxyz.astype(np.float64)])


def apply_ground_correction_to_pose7(
    pose_meas: np.ndarray,
    R_corr: np.ndarray,
    t_corr: np.ndarray,
) -> np.ndarray:
    """
    Apply the same ground correction as markers: ``p' = R_corr @ p + t_corr`` on translation and
    ``R' = R_corr @ R`` on orientation.
    """
    Rb, tb = _pose7_wxyz_to_Rt(pose_meas)
    Rout = np.einsum("tij,tjk->tik", R_corr, Rb)
    tout = np.einsum("tij,tj->ti", R_corr, tb) + t_corr
    return _Rt_to_pose7_wxyz(Rout, tout)


def compute_ground_aligned_rigid_poses(
    rigid_bodies_csv: Path,
    times_marker: list[float] | np.ndarray,
    corrections: list[GroundCorrectionTransform],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_rigid, shoe_meas, board_meas = load_rigid_body_pose_table(rigid_bodies_csv)
    tm = np.asarray(times_marker, dtype=np.float64)
    Rq, tq = _interp_corrections_at_times(tm, corrections, t_rigid)
    shoe_ga = apply_ground_correction_to_pose7(shoe_meas, Rq, tq)
    board_ga = apply_ground_correction_to_pose7(board_meas, Rq, tq)
    return t_rigid, shoe_ga, board_ga


def plot_ground_aligned_tracks(tracks: dict[str, list[np.ndarray]]):
    # Plot the restored skateboard tracks
    rear_left_wheel = [tracks["Skateboard/rear_left_wheel"][i][2] if tracks["Skateboard/rear_left_wheel"][i] is not None else None for i in range(len(tracks["Skateboard/rear_left_wheel"]))]
    rear_right_wheel = [tracks["Skateboard/rear_right_wheel"][i][2] if tracks["Skateboard/rear_right_wheel"][i] is not None else None for i in range(len(tracks["Skateboard/rear_right_wheel"]))]
    front_left_wheel = [tracks["Skateboard/front_left_wheel"][i][2] if tracks["Skateboard/front_left_wheel"][i] is not None else None for i in range(len(tracks["Skateboard/front_left_wheel"]))]
    front_right_wheel = [tracks["Skateboard/front_right_wheel"][i][2] if tracks["Skateboard/front_right_wheel"][i] is not None else None for i in range(len(tracks["Skateboard/front_right_wheel"]))]
    rear_left_pole_left = [tracks["Skateboard/rear_left_pole_left"][i][2] if tracks["Skateboard/rear_left_pole_left"][i] is not None else None for i in range(len(tracks["Skateboard/rear_left_pole_left"]))]
    rear_left_pole_right = [tracks["Skateboard/rear_left_pole_right"][i][2] if tracks["Skateboard/rear_left_pole_right"][i] is not None else None for i in range(len(tracks["Skateboard/rear_left_pole_right"]))]
    rear_right_pole_left = [tracks["Skateboard/rear_right_pole_left"][i][2] if tracks["Skateboard/rear_right_pole_left"][i] is not None else None for i in range(len(tracks["Skateboard/rear_right_pole_left"]))]
    rear_right_pole_right = [tracks["Skateboard/rear_right_pole_right"][i][2] if tracks["Skateboard/rear_right_pole_right"][i] is not None else None for i in range(len(tracks["Skateboard/rear_right_pole_right"]))]

    plt.figure(figsize=(10, 8))

    plt.plot(rear_left_wheel, label="Rear left")
    plt.plot(rear_right_wheel, label="Rear right")
    plt.plot(front_left_wheel, label="Front left")
    plt.plot(front_right_wheel, label="Front right")
    plt.plot(rear_left_pole_left, label="Rear left pole left")
    plt.plot(rear_left_pole_right, label="Rear left pole right")
    plt.plot(rear_right_pole_left, label="Rear right pole left")
    plt.plot(rear_right_pole_right, label="Rear right pole right")

    # Shade all intervals of time where we have 2 or less wheels visible and 2 
    for i in range(len(rear_left_wheel)):
        num_visible_wheels = sum(1 for wheel in [rear_left_wheel[i], rear_right_wheel[i], front_left_wheel[i], front_right_wheel[i]] if wheel is not None)
        if num_visible_wheels <= 2:
            plt.axvspan(i, i+1, color="red", alpha=0.1)

    # Shade all intervals of time where we have 2 or less poles visible
    for i in range(len(rear_left_pole_left)):
        num_visible_poles = sum(1 for pole in [rear_left_pole_left[i], rear_left_pole_right[i], rear_right_pole_left[i], rear_right_pole_right[i]] if pole is not None)
        if num_visible_poles <= 2:
            plt.axvspan(i, i+1, color="blue", alpha=0.1)

    plt.legend()
    plt.show()


def main():
    args = build_arg_parser().parse_args()
    demo_path = args.input.resolve()
    if not demo_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {demo_path}")

    ground_aligned_path = args.output.resolve()
    ground_aligned_path.mkdir(parents=True, exist_ok=True)

    samples, skateboard_marker_names, shoe_marker_names = load_samples(
        demo_path, marker_linear_scale=float(args.marker_linear_scale)
    )

    skateboard_tracks = {"Skateboard/" + name: [] for name in skateboard_marker_names}
    shoe_tracks = {"Shoe/" + name: [] for name in shoe_marker_names}

    times = []
    for sample in samples:
        times.append(float(sample["time"]))
        board_data = sample["Skateboard"]
        shoe_data = sample["Shoe"]
        for track_name in skateboard_marker_names:
            skateboard_tracks["Skateboard/" + track_name].append(board_data[track_name])
        for track_name in shoe_marker_names:
            shoe_tracks["Shoe/" + track_name].append(shoe_data[track_name])

    nominal_model = estimate_rigid_body_model(skateboard_tracks, "Skateboard")

    # canonical = build_ground_aligned_canonical_model(
    #     nominal_model=nominal_model.positions,
    #     lower_markers=["Skateboard/" + name for name in WHEEL_MARKER_NAMES],
    #     upper_markers=["Skateboard/" + name for name in POLE_MARKER_NAMES],
    #     lower_plane_height=0.75,
    #     flatten_planes=True,
    # )

    # transforms = estimate_restorative_transforms(
    #     tracks=skateboard_tracks,
    #     canonical_model=canonical,
    #     min_visible_markers=3,
    #     outlier_threshold=None,
    #     carry_forward=True,
    # )

    # tracks = {**skateboard_tracks, **shoe_tracks}

    # restored_tracks = apply_restorative_transforms(
    #     tracks=tracks,
    #     transforms=transforms,
    # )

    nominal_model_path = ground_aligned_path / "nominal_model.pkl"
    with open(nominal_model_path, "wb") as file:
        pickle.dump(nominal_model, file)

    corrections = estimate_pose_preserving_ground_corrections(
        tracks=skateboard_tracks,
        nominal_model=nominal_model.positions,
        lower_markers=["Skateboard/" + name for name in WHEEL_MARKER_NAMES],
        upper_markers=["Skateboard/" + name for name in POLE_MARKER_NAMES],
        lower_plane_height=0.03,   # wheel radius is 3 cm
        min_visible_markers=3,
        outlier_threshold=None,
        carry_forward_pose=True,
        carry_forward_correction=True,
        preserve_horizontal_position=True,
    )

    corrections_path = ground_aligned_path / "corrections.pkl"
    with open(corrections_path, "wb") as file:
        pickle.dump(corrections, file)

    tracks = {**skateboard_tracks, **shoe_tracks}

    restored_tracks = apply_ground_corrections(
        tracks=tracks,
        corrections=corrections,
    )

    # Merge the time data with the restored tracks (marker / mocap marker clock).
    restored_tracks["time"] = times

    json_path = ground_aligned_path / "ground_aligned.json"
    with open(json_path, "w") as file:
        json.dump(restored_tracks, file)

    # Replace None with [np.nan, np.nan, np.nan]
    for key in restored_tracks:
        for i in range(len(restored_tracks[key])):
            if restored_tracks[key][i] is None:
                restored_tracks[key][i] = [np.nan, np.nan, np.nan]

    rb_csv = demo_path / "rigid_bodies.csv"
    if not rb_csv.is_file():
        raise SystemExit(
            f"Missing {rb_csv} — required to export ground-aligned rigid Shoe/Skateboard poses "
            "on the shared rigid-body clock."
        )
    t_rigid, shoe_pose_ga, board_pose_ga = compute_ground_aligned_rigid_poses(rb_csv, times, corrections)

    npz_path = ground_aligned_path / "ground_aligned.npz"
    npz_payload: dict[str, np.ndarray] = {
        "time": np.asarray(restored_tracks["time"], dtype=np.float64),
    }
    for key, val in restored_tracks.items():
        if key == "time":
            continue
        npz_payload[key] = np.asarray(val, dtype=np.float64)
    npz_payload["time_rigid_bodies_s"] = np.asarray(t_rigid, dtype=np.float64)
    npz_payload["rigid_Shoe_pose_wxyz"] = np.asarray(shoe_pose_ga, dtype=np.float64)
    npz_payload["rigid_Skateboard_pose_wxyz"] = np.asarray(board_pose_ga, dtype=np.float64)
    np.savez(npz_path, **npz_payload)

    if not args.no_plot:
        plot_ground_aligned_tracks(restored_tracks)


if __name__ == "__main__":
    main()