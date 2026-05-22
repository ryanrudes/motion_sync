from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import numpy as np
import typer

from retargeting.config import RetargetingConfig, load_config
from retargeting.constants import DEFAULT_CONFIG_PATH
from retargeting.sync_trim_video import run_sync_trim_video_cli
from retargeting.sync_visualize import run_sync_visualize_cli
from retargeting.syncer import (
    CropMode,
    build_unified_dataset,
    get_sync_signals,
    load_gvhmr_data,
    load_vicon_data,
    make_video_frame_times,
    save_aligned_npz,
    support_overlap_video_clock,
)

sync_app = typer.Typer(help="Commands for performing time synchronization of the Vicon data with the GVHMR data.")


def _plot_foot_speed_sync(
    *,
    config: RetargetingConfig,
    gvhmr_output_dir: Path,
    vicon_tables_dir: Path,
    lag: float,
    corr: Optional[float],
    crop: str,
    t_keep: Optional[tuple[float, float]],
    show: bool,
    plot_file: Optional[Path],
) -> None:
    """
    Overlay Vicon vs video foot-speed signals in video-clock time.

    Matches ``build_unified_dataset`` lag convention:
    ``t_vicon_unified = t_vicon - lag``, so mocap native time ``t`` is drawn at ``t - lag``.
    """
    if not show and plot_file is not None:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    vicon = load_vicon_data(vicon_tables_dir / "merged.npz")
    gvhmr = load_gvhmr_data(gvhmr_output_dir)
    t_mocap, x_mocap, t_video, x_video = get_sync_signals(vicon, gvhmr, config)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    title = f"{gvhmr_output_dir.name}  lag={lag:.4f}s"
    if corr is not None:
        title += f"  corr={corr:.4f}"
    fig.suptitle(title)

    # Video-clock abscissa for mocap: unified time = mocap_time - lag
    t_m_shifted = t_mocap - lag

    ax1.plot(t_video, x_video[:, 0], label="video (left)", alpha=0.85)
    ax1.plot(t_m_shifted, x_mocap[:, 0], label="vicon (left)", alpha=0.85)
    ax1.set_ylabel("left foot speed")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_video, x_video[:, 1], label="video (right)", alpha=0.85)
    ax2.plot(t_m_shifted, x_mocap[:, 1], label="vicon (right)", alpha=0.85)
    ax2.set_ylabel("right foot speed")
    ax2.set_xlabel("time (s, video clock)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    if t_keep is not None:
        t_lo, t_hi = t_keep
        vline_kw = dict(linestyle=":", color="0.35", linewidth=1.5, zorder=5)
        ax1.axvline(t_lo, label=f"kept window ({crop})", **vline_kw)
        ax1.axvline(t_hi, **vline_kw)
        ax2.axvline(t_lo, **vline_kw)
        ax2.axvline(t_hi, **vline_kw)

    fig.tight_layout()

    if plot_file is not None:
        plot_file = Path(plot_file)
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_file, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


@sync_app.command(help="Perform time synchronization of the Vicon data with the GVHMR data.")
def time(
    vicon_tables_dir: Path = typer.Argument(..., help="Path to the Vicon tables directory."),
    gvhmr_output_dir: Path = typer.Argument(..., help="Path to the GVHMR output directory."),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to the configuration file."),
    target_timeline: str = typer.Option(
        "vicon",
        "--target-timeline",
        help="Resample onto the vicon clock (shifted by lag) or the video frame clock.",
    ),
    crop: str = typer.Option(
        "support",
        "--crop",
        help="support (default): time overlap of all sources; valid: also require finite "
        "required channels (shoe bodies only for body_pos); none: full timeline with NaNs.",
    ),
    plot: bool = typer.Option(False, "--plot", help="Show foot-speed alignment figure (matplotlib)."),
    plot_file: Optional[Path] = typer.Option(
        None,
        "--plot-file",
        help="Save the same figure to this path (png or other matplotlib-supported format).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="If set, write aligned dataset to unified.npz under this directory.",
    ),
):
    if target_timeline not in {"vicon", "video"}:
        raise typer.BadParameter("target_timeline must be 'vicon' or 'video'.")
    if crop not in {"valid", "support", "none"}:
        raise typer.BadParameter("crop must be 'valid', 'support', or 'none'.")

    config = load_config(config_path)

    aligned_vicon, meta_vicon = build_unified_dataset(
        gvhmr_dir=gvhmr_output_dir,
        vicon_path=vicon_tables_dir / "merged.npz",
        config=config,
        target_timeline=target_timeline,
        crop=cast(CropMode, crop),
    )

    lag = float(meta_vicon["lag"])
    corr = meta_vicon.get("corr")
    corr_f = float(corr) if corr is not None else None

    n_t = int(aligned_vicon["t"].shape[0])
    typer.echo(
        f"lag={lag:.6g}s  corr={corr_f if corr_f is not None else 'n/a'}  "
        f"t={aligned_vicon['t'].shape}  timeline={target_timeline}  crop={crop}"
    )

    if n_t == 0:
        typer.secho(
            "Warning: zero rows after crop — try --crop none or --target-timeline video, "
            "or pass a manual lag= via the Python API.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    elif crop == "valid" and n_t < 200:
        typer.secho(
            "Warning: very few rows after crop=valid. Often mocap/video duration mismatch "
            "or sparse video samples on this timeline; shoe-only finiteness is already used "
            "for body_pos. Try --crop support (default in build_unified_dataset) or --crop none.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    if plot or plot_file is not None:
        t_keep: Optional[tuple[float, float]] = None
        if crop != "none":
            t_aligned = aligned_vicon["t"]
            if n_t > 0:
                t_keep = (float(t_aligned[0]), float(t_aligned[-1]))
            else:
                vicon = load_vicon_data(vicon_tables_dir / "merged.npz")
                gvhmr = load_gvhmr_data(gvhmr_output_dir)
                t_video = make_video_frame_times(
                    len(gvhmr["joints"]),
                    config.rate.video,
                )
                t_keep = support_overlap_video_clock(
                    t_video,
                    np.asarray(vicon["t"], dtype=float),
                    lag,
                )

        _plot_foot_speed_sync(
            config=config,
            gvhmr_output_dir=gvhmr_output_dir,
            vicon_tables_dir=vicon_tables_dir,
            lag=lag,
            corr=corr_f,
            crop=crop,
            t_keep=t_keep,
            show=plot,
            plot_file=plot_file,
        )
        if plot_file is not None:
            typer.echo(f"Wrote plot to {plot_file}")

    if output_dir is not None:
        save_aligned_npz(output_dir / "unified.npz", aligned_vicon, meta_vicon)
        typer.echo(f"Wrote aligned npz to {output_dir / 'unified.npz'}")


@sync_app.command(help="Trim source video to the synced time window in unified.npz.")
def video(
    unified_npz: Path = typer.Argument(
        ...,
        help="Path to unified.npz (e.g. output/synced/<demo>/unified.npz).",
    ),
    video_path: Path = typer.Argument(..., help="Original demo video (mp4/mov)."),
    output_path: Path = typer.Argument(..., help="Output video path (e.g. .mp4)."),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to the configuration file (for fps)."),
    no_ffmpeg: bool = typer.Option(
        False,
        "--no-ffmpeg",
        help="Do not use ffmpeg; encode with OpenCV VideoWriter only.",
    ),
) -> None:
    run_sync_trim_video_cli(
        unified_npz,
        video_path,
        output_path,
        config_path=config_path,
        prefer_ffmpeg=not no_ffmpeg,
    )


@sync_app.command(help="Play source video beside OptiTrack markers using cached unified.npz times.")
def visualize(
    unified_npz: Path = typer.Argument(..., help="Path to unified.npz (e.g. output/synced/<demo>/unified.npz)."),
    video_path: Path = typer.Argument(..., help="Original demo video (mp4/mov)."),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to the configuration file (for fps)."),
    merged_npz: Optional[Path] = typer.Option(
        None,
        "--merged-npz",
        help="Optional merged.npz for marker names; if omitted, tries output/vicon_tables/<demo>/merged.npz.",
    ),
    start_frame: int = typer.Option(0, "--start-frame", help="First video frame index."),
    end_frame: Optional[int] = typer.Option(
        None,
        "--end-frame",
        help="Exclusive end frame index (default: video length).",
    ),
) -> None:
    run_sync_visualize_cli(
        unified_npz,
        video_path,
        config_path=config_path,
        merged_npz=merged_npz,
        start_frame=start_frame,
        end_frame=end_frame,
    )
