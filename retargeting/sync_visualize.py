"""
Side-by-side playback: source video and OptiTrack markers from a cached unified.npz.

Timeline is taken from ``unified.npz`` (``t`` + ``vicon__marker_pos``). Video frames are
shown at the same unified (video-clock) times used when the dataset was built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer

from retargeting.config import RetargetingConfig, load_config


def _infer_merged_npz(unified_path: Path) -> Optional[Path]:
    """If ``output/synced/<demo>/unified.npz``, try ``output/vicon_tables/<demo>/merged.npz``."""
    demo = unified_path.parent.name
    out_root = unified_path.parent.parent.parent
    cand = out_root / "vicon_tables" / demo / "merged.npz"
    return cand if cand.is_file() else None


def _load_marker_names(merged_npz: Optional[Path]) -> Optional[np.ndarray]:
    if merged_npz is None or not merged_npz.is_file():
        return None
    m = np.load(merged_npz, allow_pickle=True)
    if "marker_names" not in m.files:
        return None
    return np.asarray(m["marker_names"])


def _sample_markers_at_time(
    tau: float,
    t_u: np.ndarray,
    markers: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of (M, 3) markers along unified time ``t_u`` (sorted)."""
    t_u = np.asarray(t_u, dtype=float)
    markers = np.asarray(markers, dtype=float)
    if len(t_u) == 0:
        raise ValueError("empty unified timeline")
    if len(t_u) == 1:
        return markers[0].copy()
    if tau <= t_u[0]:
        return markers[0].copy()
    if tau >= t_u[-1]:
        return markers[-1].copy()
    i = int(np.searchsorted(t_u, tau, side="right") - 1)
    i = max(0, min(i, len(t_u) - 2))
    t0, t1 = float(t_u[i]), float(t_u[i + 1])
    if t1 <= t0:
        return markers[i].copy()
    w = (tau - t0) / (t1 - t0)
    return (1.0 - w) * markers[i] + w * markers[i + 1]


def _world_xy_bounds(markers: np.ndarray, margin_frac: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Stable XY bounds from all marker samples (finite only)."""
    xy = markers[..., :2].reshape(-1, 2)
    ok = np.isfinite(xy).all(axis=1)
    xy = xy[ok]
    if xy.shape[0] == 0:
        lo = np.array([-0.5, -0.5], dtype=float)
        hi = np.array([0.5, 0.5], dtype=float)
        return lo, hi
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    pad = span * margin_frac
    return lo - pad, hi + pad


def _world_to_pixel(
    xy: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Map world XY to image pixel (u, v) with +y up in world, +v down in image."""
    x, y = float(xy[0]), float(xy[1])
    u = (x - lo[0]) / (hi[0] - lo[0]) * (width - 1)
    v = (height - 1) - (y - lo[1]) / (hi[1] - lo[1]) * (height - 1)
    return int(round(u)), int(round(v))


def _hsv_bgr(i: int, n: int) -> tuple[int, int, int]:
    import cv2

    if n <= 0:
        return 200, 200, 200
    h = int(180 * (i % max(n, 1)) / max(n, 1)) % 180
    color = cv2.cvtColor(np.uint8([[[h, 220, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def run_sync_visualize(
    *,
    unified_npz: Path,
    video_path: Path,
    config: RetargetingConfig,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    import cv2

    unified_npz = unified_npz.resolve()
    video_path = video_path.resolve()

    if not unified_npz.is_file():
        raise FileNotFoundError(f"Not a file: {unified_npz}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Not a file: {video_path}")

    data = np.load(unified_npz, allow_pickle=True)

    t_key = "t" if "t" in data.files else None
    if t_key is None:
        raise KeyError(f"{unified_npz} has no 't' array; expected a unified sync export.")

    t_u = np.asarray(data[t_key], dtype=float)
    mkey = "vicon__marker_pos"
    if mkey not in data.files:
        raise KeyError(
            f"{unified_npz} has no {mkey!r}. Re-run sync with marker data in merged.npz "
            "and ensure vicon/marker_pos is present in the stitcher output."
        )
    markers_u = np.asarray(data[mkey], dtype=float)
    if markers_u.shape[0] != len(t_u):
        raise ValueError(
            f"Length mismatch: t={len(t_u)} vs vicon__marker_pos rows={markers_u.shape[0]}"
        )

    lag = float(np.asarray(data["lag"]).reshape(())) if "lag" in data.files else float("nan")

    fps = float(config.rate.video or 0.0)
    if fps <= 0:
        raise ValueError("config.rate.video must be a positive fps for visualization timing.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_vid <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if vw <= 0 or vh <= 0:
        cap.release()
        raise RuntimeError("Could not read video width/height.")

    lo, hi = _world_xy_bounds(markers_u)
    n_markers = markers_u.shape[1]
    delay_ms = max(1, int(round(1000.0 / fps)))

    sf = max(0, start_frame)
    ef = n_vid if end_frame is None else min(n_vid, end_frame)
    if sf >= ef:
        cap.release()
        raise ValueError(f"Invalid frame range start={sf} end={ef} (n_vid={n_vid}).")

    win = "retargeting sync visualize (q=quit, space=pause)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    paused = False

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(sf))
    try:
        for fi in range(sf, ef):
            tau = fi / fps
            mpos = _sample_markers_at_time(tau, t_u, markers_u)

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            panel = np.full((vh, vw, 3), 30, dtype=np.uint8)
            for mi in range(n_markers):
                p = mpos[mi]
                if not np.isfinite(p).all():
                    continue
                u, v = _world_to_pixel(p[:2], lo, hi, vw, vh)
                u = int(np.clip(u, 0, vw - 1))
                v = int(np.clip(v, 0, vh - 1))
                color = _hsv_bgr(mi, n_markers)
                cv2.circle(panel, (u, v), 5, color, -1, lineType=cv2.LINE_AA)

            cv2.putText(
                frame,
                f"video t={tau:.3f}s  frame {fi}/{n_vid}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"markers (XY)  lag={lag:.4f}s" if np.isfinite(lag) else "markers (XY)",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

            combo = np.hstack([frame, panel])
            cv2.imshow(win, combo)

            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_sync_visualize_cli(
    unified_npz: Path,
    video_path: Path,
    *,
    config_path: Path,
    merged_npz: Optional[Path],
    start_frame: int,
    end_frame: Optional[int],
) -> None:
    config = load_config(config_path)
    merged = merged_npz or _infer_merged_npz(unified_npz)
    names = _load_marker_names(merged)
    if names is not None and len(names):
        typer.echo(f"Markers: {len(names)} tracks (names from {merged})")
    elif merged_npz is not None:
        typer.echo("Note: merged.npz has no marker_names array.", err=True)
    run_sync_visualize(
        unified_npz=unified_npz,
        video_path=video_path,
        config=config,
        start_frame=start_frame,
        end_frame=end_frame,
    )
