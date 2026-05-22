"""Shared helpers and synthetic data builders for unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
MINIMAL_CONFIG = FIXTURES_DIR / "retargeting_minimal.yaml"


def shifted_foot_speed_signals(
    *,
    n: int = 200,
    dt: float = 0.01,
    true_lag: float = 0.35,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build paired 2D signals where x2 trails x1 by ``true_lag`` seconds."""
    rng = np.random.default_rng(seed)
    t1 = np.arange(n, dtype=float) * dt
    t2 = np.arange(n, dtype=float) * dt
    phase = rng.uniform(0.0, 2.0 * np.pi)
    x1 = np.column_stack(
        [
            np.sin(2.0 * np.pi * 1.3 * t1 + phase),
            np.cos(2.0 * np.pi * 0.7 * t1 + phase * 0.5),
        ]
    )
    # Same motion on mocap clock shifted earlier by true_lag (see estimate_lag docstring).
    x2 = np.column_stack(
        [
            np.sin(2.0 * np.pi * 1.3 * (t2 + true_lag) + phase),
            np.cos(2.0 * np.pi * 0.7 * (t2 + true_lag) + phase * 0.5),
        ]
    )
    return t1, x1, t2, x2


def write_unified_npz(path: Path, t: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=np.asarray(t, dtype=float))


def synthetic_convert_inputs() -> tuple[dict, dict]:
    """Minimal TF/marker dicts satisfying merge_tf_and_marker_data constraints."""
    marker_names = np.array([f"Unlabeled{i:05d}" for i in range(28)])

    stamps = np.array([100, 101, 102, 103], dtype=np.int64)
    tf_stamps = stamps[[0, 2, 3]]
    child_frame_ids = np.array(
        [
            "vicon/Left_Shoe/Left_Shoe",
            "vicon/Right_Shoe/Right_Shoe",
            "vicon/Skateboard/Skateboard",
        ]
    )

    tf_data = {
        "header.stamp": tf_stamps,
        "child_frame_id": child_frame_ids,
        "xyz": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "wxyz": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }

    # merge_tf_and_marker_data requires all 28 unique marker names; TF stamps must
    # be a subset of marker stamps.
    marker_stamps = np.array([100] * 7 + [101] * 7 + [102] * 7 + [103] * 7, dtype=np.int64)
    subjects = np.array(["Left_Shoe"] * 10 + ["Right_Shoe"] * 9 + ["Skateboard"] * 9)
    marker_data = {
        "header.stamp": marker_stamps,
        "subject_name": subjects,
        "marker_name": marker_names,
        "xyz": np.tile([0.1, 0.0, 0.0], (28, 1)).astype(np.float32),
        "occluded": np.zeros(28, dtype=bool),
    }

    return tf_data, marker_data


def square_marker_tracks() -> dict[str, list[tuple[float, float, float] | None]]:
    """Four markers forming a unit square, fully visible for 20 frames."""
    tracks = {
        "a": [(0.0, 0.0, 0.0)] * 20,
        "b": [(1.0, 0.0, 0.0)] * 20,
        "c": [(1.0, 1.0, 0.0)] * 20,
        "d": [(0.0, 1.0, 0.0)] * 20,
    }
    return tracks
