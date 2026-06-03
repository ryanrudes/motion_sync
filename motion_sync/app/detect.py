"""Contact detection on synced clips."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from motion_sync.contacts.foot_support import foot_support_config
from motion_sync.schemas.skateboarding import SKATE_FOOT_SUPPORT, SKATE_SESSION
from motion_sync.synced_dataset import SyncClip

detect_app = typer.Typer(help="Contact and support-state detection on synced demos.")


def _load_foot_support_config(path: Path | None) -> object:
    if path is None:
        return foot_support_config(
            SKATE_FOOT_SUPPORT.left,
            SKATE_FOOT_SUPPORT.right,
            SKATE_FOOT_SUPPORT.board,
        )
    data = yaml.safe_load(path.read_text())
    section = data.get("foot_support", data) if isinstance(data, dict) else {}
    try:
        from contact_detection import FootSupportConfig
    except ImportError as exc:
        raise typer.BadParameter(
            "Install contact-detection (e.g. uv pip install -e ../event_detection)"
        ) from exc
    return FootSupportConfig(**section)


@detect_app.command("foot-support")
def foot_support(
    demo_path: Path = typer.Argument(..., help="Demo directory or synced.npz path."),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="YAML with a foot_support section (event_detection-style).",
    ),
    plot: bool = typer.Option(False, "--plot", help="Write a diagnostic PNG next to the demo."),
    show: bool = typer.Option(False, "--show", help="Open matplotlib window after plotting."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-run detection even if a fresh foot_support layer is already on the clip.",
    ),
) -> None:
    """Classify per-foot air / ground / skateboard and save on the synced clip."""
    clip = SyncClip.load(demo_path, session=SKATE_SESSION)
    if clip.contact_is_fresh(SKATE_FOOT_SUPPORT) and not force:
        typer.echo(f"Skipping detect: fresh {SKATE_FOOT_SUPPORT.layer_id!r} layer already present")
    else:
        clip = clip.detect(SKATE_FOOT_SUPPORT, _load_foot_support_config(config), force=force)
        out_path = clip.save(demo_path)
        typer.echo(f"Wrote {SKATE_FOOT_SUPPORT.layer_id!r} layer on {out_path}")

    if not clip.has_contact(SKATE_FOOT_SUPPORT):
        raise typer.Exit("foot_support layer missing after detect")

    if plot:
        try:
            from contact_detection.debug import plot_foot_support_states
        except ImportError as exc:
            raise typer.BadParameter(
                "Plotting requires contact-detection with matplotlib"
            ) from exc

        data = clip.contact(SKATE_FOOT_SUPPORT)
        png_path = Path(demo_path)
        if png_path.is_file():
            png_path = png_path.parent
        png_path = png_path / "foot_support.png"
        plot_foot_support_states(
            data.classification(),
            output_path=png_path,
            title=f"{clip.name}: foot support",
        )
        typer.echo(f"Wrote {png_path}")
        if show:
            import matplotlib.pyplot as plt

            plt.show()
