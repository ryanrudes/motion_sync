"""
Side-by-side playback: source video and OptiTrack markers from a synced clip.

Timeline and marker positions come from :class:`~motion_sync.synced_dataset.SyncClip`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer

from motion_sync.config import MotionSyncConfig, load_config
from motion_sync.synced_dataset import SyncClip


def _sample_markers_at_time(
    tau: float,
    t_u: np.ndarray,
    markers: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of (M, 3) markers along synced time ``t_u`` (sorted)."""
    t_u = np.asarray(t_u, dtype=float)
    markers = np.asarray(markers, dtype=float)
    if len(t_u) == 0:
        raise ValueError("empty synced timeline")
    if len(t_u) == 1:
        return markers[0].copy()
    if tau <= t_u[0]:
        return markers[0].copy()
    if tau >= t_u[-1]:
        return markers[-1].copy()

    i = int(np.searchsorted(t_u, tau, side="right") - 1)
    i = max(0, min(i, len(t_u) - 2))
    if t_u[i + 1] <= t_u[i]:
        return markers[i].copy()
    w = (tau - t_u[i]) / (t_u[i + 1] - t_u[i])
    return (1.0 - w) * markers[i] + w * markers[i + 1]


def _world_xy_bounds(markers: np.ndarray, margin_frac: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Stable XY bounds from all marker samples (finite only)."""
    xy = markers[..., :2].reshape(-1, 2)
    ok = np.isfinite(xy).all(axis=1)
    if not ok.any():
        return np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    xy = xy[ok]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 0.5)
    pad = span * margin_frac
    return lo - pad, hi + pad


def _world_to_pixel(
    xy: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float]:
    span = hi - lo
    span = np.where(span < 1e-6, 1.0, span)
    u = (xy[0] - lo[0]) / span[0] * (width - 1)
    v = (1.0 - (xy[1] - lo[1]) / span[1]) * (height - 1)
    return u, v


def _hsv_bgr(index: int, count: int) -> tuple[int, int, int]:
    import cv2

    hue = int(180 * index / max(count, 1)) % 180
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def run_sync_visualize(
    *,
    synced_path: Path,
    video_path: Path,
    config: MotionSyncConfig,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    import cv2

    synced_path = synced_path.resolve()
    video_path = video_path.resolve()

    clip = SyncClip.load(synced_path)
    if clip.vicon.markers is None:
        raise ValueError(
            "synced clip has no marker tracks; re-run sync with Vicon marker data present"
        )

    t_u = np.asarray(clip.time_s, dtype=float)
    markers_u = np.asarray(clip.vicon.markers.positions, dtype=float)
    lag = clip.metadata.lag_s

    fps = float(config.rate.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sf = max(0, start_frame)
    ef = n_vid if end_frame is None else min(end_frame, n_vid)
    if sf >= ef:
        cap.release()
        raise ValueError(f"Invalid frame range: start={sf} end={ef} (video has {n_vid} frames)")

    delay_ms = max(1, int(round(1000.0 / fps)))

    lo, hi = _world_xy_bounds(markers_u)
    n_markers = markers_u.shape[1]

    win = "sync visualize (q quit, space pause)"
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
                f"markers (XY)  lag={lag:.4f}s",
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
    synced_path: Path,
    video_path: Path,
    *,
    config_path: Path,
    start_frame: int,
    end_frame: Optional[int],
) -> None:
    config = load_config(config_path)
    clip = SyncClip.load(synced_path)
    n_markers = 0 if clip.vicon.markers is None else clip.vicon.markers.marker_count
    typer.echo(f"Markers: {n_markers} tracks from synced clip {clip.name!r}")
    run_sync_visualize(
        synced_path=synced_path,
        video_path=video_path,
        config=config,
        start_frame=start_frame,
        end_frame=end_frame,
    )
