"""Export Holosoma ``object_interaction`` motion NPZ from ``unified.npz``."""

from pathlib import Path

import typer

from retargeting.config import load_config
from retargeting.constants import DEFAULT_CONFIG_PATH
from retargeting.holosoma_export import export_holosoma_object_npz

holosoma_export_app = typer.Typer(help="Export Holosoma object-interaction NPZ from unified data.")


@holosoma_export_app.command("object-npz")
def object_npz(
    unified_npz: Path = typer.Argument(..., help="Path to unified.npz from retargeting sync."),
    out: Path = typer.Argument(..., help="Output .npz path (e.g. demo_seq.npz)."),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Retargeting YAML (for smplx_models path)."),
    swap_yz: bool = typer.Option(
        False,
        help="Apply [x,y,z] to [x,z,y] swap on mocap and SMPL transl before FK; re-synthesize board quaternion from shoes.",
    ),
    default_height_m: float = typer.Option(
        1.72,
        help="Fallback human height (m) if heuristic from head–pelvis is unusable.",
    ),
) -> None:
    cfg = load_config(config_path)
    export_holosoma_object_npz(
        unified_npz,
        out,
        smplx_model_root=cfg.paths.smplx_models,
        swap_y_up_to_z_up=swap_yz,
        default_height_m=default_height_m,
    )
    typer.echo(f"Wrote Holosoma package motion to {out}")
