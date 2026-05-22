"""
Trim a demo video to the synced time window stored in ``unified.npz``.

The kept range is ``t[0]`` … ``t[-1]`` on the video-clock axis (same as ``sync visualize``).
Frame indices use ``frame_i / fps`` with ``fps`` from config ``rate.video``.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from retargeting.config import RetargetingConfig, load_config


def synced_time_range_from_unified(unified_npz: Path) -> tuple[float, float]:
    """Return ``(t_start, t_end)`` in video-clock seconds from ``unified.npz`` ``t``."""
    unified_npz = Path(unified_npz).resolve()
    if not unified_npz.is_file():
        raise FileNotFoundError(f"Not a file: {unified_npz}")

    data = np.load(unified_npz, allow_pickle=True)
    if "t" not in data.files:
        raise KeyError(f"{unified_npz} has no 't' array; expected a unified sync export.")

    t_u = np.asarray(data["t"], dtype=float)
    if t_u.size == 0:
        raise ValueError(f"{unified_npz}: empty timeline 't' (re-run sync with a non-empty crop).")

    ok = np.isfinite(t_u)
    if not ok.any():
        raise ValueError(f"{unified_npz}: no finite samples in 't'.")

    t_u = t_u[ok]
    if t_u.size == 1:
        return float(t_u[0]), float(t_u[0])

    if np.any(np.diff(t_u) < -1e-9):
        typer.echo(
            "Warning: unified 't' is not monotonic; using min/max for trim window.",
            err=True,
        )
    return float(t_u.min()), float(t_u.max())


def time_range_to_frame_range(
    t_start: float,
    t_end: float,
    fps: float,
    n_frames: int,
) -> tuple[int, int]:
    """
    Map a closed video-clock interval to half-open frame indices ``[start, end)``.

    Frame ``i`` is sampled at time ``i / fps`` (matches ``sync_visualize``).
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if t_end < t_start:
        raise ValueError(f"t_end ({t_end}) < t_start ({t_start}).")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive.")

    start = int(math.ceil(float(t_start) * fps - 1e-9))
    end = int(math.floor(float(t_end) * fps + 1e-9)) + 1
    start = max(0, min(start, n_frames))
    end = max(start, min(end, n_frames))
    if start >= end:
        raise ValueError(
            f"Trim window collapses to zero frames: t=[{t_start}, {t_end}] s, "
            f"fps={fps}, frames=[{start}, {end}) of {n_frames}."
        )
    return start, end


def _trim_with_ffmpeg(
    *,
    video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame_exclusive: int,
    fps: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found on PATH")

    start_sec = start_frame / fps
    duration_sec = (end_frame_exclusive - start_frame) / fps
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        f"{start_sec:.9f}",
        "-t",
        f"{duration_sec:.9f}",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _trim_with_opencv(
    *,
    video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame_exclusive: int,
    fps: float,
) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video width/height.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    try:
        for _ in range(start_frame, end_frame_exclusive):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def trim_video_to_synced_range(
    *,
    unified_npz: Path,
    video_path: Path,
    output_path: Path,
    config: RetargetingConfig,
    prefer_ffmpeg: bool = True,
) -> dict[str, float | int]:
    """
    Trim ``video_path`` to the ``unified.npz`` synced window and write ``output_path``.

    Returns a small summary dict (times, frames, fps).
    """
    import cv2

    unified_npz = Path(unified_npz).resolve()
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Not a file: {video_path}")

    t_start, t_end = synced_time_range_from_unified(unified_npz)
    fps = float(config.rate.video or 0.0)
    if fps <= 0:
        raise ValueError("config.rate.video must be a positive fps for video trim.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_frames <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    start_frame, end_frame_exclusive = time_range_to_frame_range(
        t_start, t_end, fps, n_frames
    )

    if prefer_ffmpeg and shutil.which("ffmpeg") is not None:
        try:
            _trim_with_ffmpeg(
                video_path=video_path,
                output_path=output_path,
                start_frame=start_frame,
                end_frame_exclusive=end_frame_exclusive,
                fps=fps,
            )
        except subprocess.CalledProcessError as exc:
            typer.echo(
                f"ffmpeg failed ({exc}); falling back to OpenCV VideoWriter.",
                err=True,
            )
            _trim_with_opencv(
                video_path=video_path,
                output_path=output_path,
                start_frame=start_frame,
                end_frame_exclusive=end_frame_exclusive,
                fps=fps,
            )
    else:
        if prefer_ffmpeg:
            typer.echo("ffmpeg not on PATH; using OpenCV VideoWriter.", err=True)
        _trim_with_opencv(
            video_path=video_path,
            output_path=output_path,
            start_frame=start_frame,
            end_frame_exclusive=end_frame_exclusive,
            fps=fps,
        )

    return {
        "t_start": t_start,
        "t_end": t_end,
        "fps": fps,
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame_exclusive,
        "n_frames_out": end_frame_exclusive - start_frame,
    }


def run_sync_trim_video_cli(
    unified_npz: Path,
    video_path: Path,
    output_path: Path,
    *,
    config_path: Path,
    prefer_ffmpeg: bool,
) -> None:
    config = load_config(config_path)
    summary = trim_video_to_synced_range(
        unified_npz=unified_npz,
        video_path=video_path,
        output_path=output_path,
        config=config,
        prefer_ffmpeg=prefer_ffmpeg,
    )
    typer.echo(
        f"Wrote {output_path}  "
        f"t=[{summary['t_start']:.6g}, {summary['t_end']:.6g}] s  "
        f"frames=[{summary['start_frame']}, {summary['end_frame_exclusive']})  "
        f"({summary['n_frames_out']} frames @ {summary['fps']} Hz)"
    )
