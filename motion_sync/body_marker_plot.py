"""Matplotlib 3D plots of a rigid body and its registered markers."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from motion_sync.synced_dataset import AllMarkersVisibleStrategy

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from motion_sync.synced_dataset import BodyT, SyncClip


def _marker_label(marker: StrEnum, label: Literal["name", "value"]) -> str:
    return marker.name if label == "name" else marker.value


def _finite_at(traj: np.ndarray, frame: int) -> np.ndarray | None:
    if frame < 0 or frame >= traj.shape[0]:
        return None
    point = np.asarray(traj[frame], dtype=np.float64)
    return point if np.isfinite(point).all() else None


def _axis_limits(
    clip: SyncClip[BodyT],
    body: BodyT,
    *,
    margin_frac: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds over the full clip (stable view while animating)."""
    track = clip.body(body)
    chunks = [np.ma.masked_invalid(track.positions).compressed().reshape(-1, 3)]
    for traj in clip.markers_for_body(body).values():
        chunks.append(np.ma.masked_invalid(traj).compressed().reshape(-1, 3))
    if not chunks or all(c.size == 0 for c in chunks):
        return _square_axis_limits([], margin_frac=margin_frac)
    points = [row for c in chunks if c.size for row in np.vstack(c)]
    return _square_axis_limits(points, margin_frac=margin_frac)


def _points_at_frame(
    clip: SyncClip[BodyT],
    body: BodyT,
    frame: int,
) -> list[np.ndarray]:
    track = clip.body(body)
    points: list[np.ndarray] = []
    body_point = _finite_at(track.positions, frame)
    if body_point is not None:
        points.append(body_point)
    for traj in clip.markers_for_body(body).values():
        point = _finite_at(traj, frame)
        if point is not None:
            points.append(point)
    return points


def _square_axis_limits(
    points: list[np.ndarray],
    *,
    margin_frac: float = 0.12,
    min_half_span_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal half-span on X/Y/Z so the 3D view is a cube centered on the data."""
    if not points:
        half = 0.5
        lo = np.array([-half, -half, -half])
        hi = np.array([half, half, half])
        return lo, hi

    stack = np.stack(points, axis=0)
    lo = stack.min(axis=0)
    hi = stack.max(axis=0)
    center = 0.5 * (lo + hi)
    half_span = 0.5 * float(np.max(hi - lo))
    half_span = max(half_span, min_half_span_m)
    half_span *= 1.0 + margin_frac
    lo = center - half_span
    hi = center + half_span
    return lo, hi


def _apply_limits(ax: Axes, limits: tuple[np.ndarray, np.ndarray]) -> None:
    lo, hi = limits
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1, 1, 1))


def draw_body_markers_frame(
    ax: Axes,
    clip: SyncClip[BodyT],
    body: BodyT,
    frame: int,
    *,
    label: Literal["name", "value"] = "name",
    connect_to_body: bool = True,
    body_color: str = "black",
    marker_color: str | None = None,
    limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Draw one frame on ``ax`` (clears the axes first)."""
    ax.cla()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    track = clip.body(body)
    body_point = _finite_at(track.positions, frame)
    if body_point is not None:
        ax.scatter(
            *body_point,
            s=140,
            c=body_color,
            marker="*",
            zorder=5,
            label=f"{body.value} (rigid body)",
        )

    for marker, traj in clip.markers_for_body(body).items():
        point = _finite_at(traj, frame)
        if point is None:
            continue
        ax.scatter(*point, s=55, c=marker_color, zorder=4)
        ax.text(*point, _marker_label(marker, label), fontsize=7)
        if connect_to_body and body_point is not None:
            ax.plot(
                [body_point[0], point[0]],
                [body_point[1], point[1]],
                [body_point[2], point[2]],
                c="0.65",
                linewidth=0.8,
                zorder=2,
            )

    if limits is not None:
        _apply_limits(ax, limits)


FrameSelect = int | None | Literal["all_visible"]


def _resolve_plot_frame(
    clip: SyncClip[BodyT],
    body: BodyT,
    frame: FrameSelect,
    *,
    all_visible_strategy: AllMarkersVisibleStrategy,
    require_body: bool,
) -> int:
    if frame == "all_visible":
        return clip.find_frame_all_markers_visible(
            body,
            strategy=all_visible_strategy,
            require_body=require_body,
        )
    return 0 if frame is None else int(frame)


def plot_body_markers(
    clip: SyncClip[BodyT],
    body: BodyT,
    *,
    frame: FrameSelect = 0,
    animate: bool = False,
    frame_step: int = 1,
    all_visible_strategy: AllMarkersVisibleStrategy = "middle",
    require_body: bool = False,
    label: Literal["name", "value"] = "name",
    connect_to_body: bool = True,
    show: bool = True,
    block: bool | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a rigid body (star) and its markers (labeled) in 3D.

    Requires :meth:`SyncClip.register_mocap` on ``clip``. For a single frame, pass
    ``frame`` (default ``0``) or ``frame="all_visible"`` to auto-pick a frame where every
    marker on ``body`` is finite. For playback, set ``animate=True`` (uses ``frame_step``
    and ``clip.time_s`` for pacing).

    Returns ``(fig, ax)`` for further customization.
    """
    import matplotlib.pyplot as plt

    if clip.body_marker_map is None:
        raise RuntimeError("Call clip.register_mocap(MocapSchema(...)) before plotting.")
    if clip.vicon.markers is None:
        raise ValueError("clip has no marker tracks")

    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    clip_limits = _axis_limits(clip, body)
    owned_fig = ax is None
    if owned_fig:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    draw_kwargs = {
        "label": label,
        "connect_to_body": connect_to_body,
    }

    if not animate:
        idx = _resolve_plot_frame(
            clip,
            body,
            frame,
            all_visible_strategy=all_visible_strategy,
            require_body=require_body,
        )
        frame_limits = _square_axis_limits(_points_at_frame(clip, body, idx))
        draw_body_markers_frame(
            ax,
            clip,
            body,
            idx,
            limits=frame_limits,
            **draw_kwargs,
        )
        t = float(clip.time_s[idx]) if idx < clip.frame_count else float("nan")
        ax.set_title(f"{body.value}  frame {idx + 1}/{clip.frame_count}  t={t:.3f} s")
        if show:
            plt.show(block=block if block is not None else True)
        return fig, ax

    plt.ion()
    if show and owned_fig:
        fig.show()

    start = _resolve_plot_frame(
        clip,
        body,
        frame,
        all_visible_strategy=all_visible_strategy,
        require_body=require_body,
    )
    indices = range(start, clip.frame_count, frame_step)
    t_axis = clip.time_s
    for idx in indices:
        draw_body_markers_frame(
            ax,
            clip,
            body,
            idx,
            limits=clip_limits,
            **draw_kwargs,
        )
        t = float(t_axis[idx])
        ax.set_title(f"{body.value}  frame {idx + 1}/{clip.frame_count}  t={t:.3f} s")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if idx + frame_step < clip.frame_count:
            dt = float(t_axis[idx + frame_step] - t_axis[idx])
        else:
            dt = 1.0 / clip.mean_fps()
        plt.pause(max(dt, 1e-5))

    if show:
        plt.ioff()
        plt.show(block=block if block is not None else True)
    return fig, ax
