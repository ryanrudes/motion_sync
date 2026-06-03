import numpy as np
import typer
import torch
import smplx

from pathlib import Path

from motion_sync.config import load_config, MotionSyncConfig
from motion_sync.constants import DEFAULT_CONFIG_PATH

fkin_app = typer.Typer(help="Commands for performing forward kinematics on SMPL-X models.")

def run_smplx_fkin(
    body_pose: torch.Tensor,
    transl: torch.Tensor,
    orient: torch.Tensor,
    betas: torch.Tensor,
    config: MotionSyncConfig,
) -> dict:
    seq_len = body_pose.shape[0]
    num_betas = betas.shape[1]

    model_root = config.paths.smplx_models
    smplx_weights = model_root / "smplx" / "SMPLX_NEUTRAL.npz"
    smplx_weights_pkl = model_root / "smplx" / "SMPLX_NEUTRAL.pkl"
    if not smplx_weights.exists() and not smplx_weights_pkl.exists():
        raise FileNotFoundError(
            "GVHMR `smpl_params_global` uses SMPL-X layout (body_pose is 21 joints × 3, not "
            "SMPL's 23×3). Add SMPL-X neutral weights, e.g. "
            f"{smplx_weights} (.npz or .pkl). Download from https://smpl-x.is.tue.mpg.de/ "
            "(same parent folder layout as smplx: `.../smplx/SMPLX_NEUTRAL.npz`)."
        )

    model = smplx.create(
        model_path=model_root,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=num_betas,
        batch_size=seq_len,
        use_face_contour=True,
    )

    output = model(
        betas=betas,
        body_pose=body_pose,
        transl=transl,
        global_orient=orient, 
        return_verts=True,
    )

    joints = output.joints
    vertices = output.vertices
    return joints, vertices


@fkin_app.command(help="Perform forward kinematics on a SMPL-X model.")
def run(
    gvhmr_output_dir: Path = typer.Argument(..., help="Path to the GVHMR output directory."),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to the configuration file."),
):
    config = load_config(config_path)
    hmr4d_results = torch.load(gvhmr_output_dir / "hmr4d_results.pt")

    body_pose = hmr4d_results["smpl_params_global"]["body_pose"]  # (T, 63) = SMPL-X 21×3
    transl = hmr4d_results["smpl_params_global"]["transl"]  # (T, 3)
    orient = hmr4d_results["smpl_params_global"]["global_orient"]  # (T, 3)
    betas = hmr4d_results["smpl_params_global"]["betas"]  # (T, 10)

    joints, vertices = run_smplx_fkin(body_pose, transl, orient, betas, config)
    joints = joints.detach().numpy()
    vertices = vertices.detach().numpy()

    np.save(gvhmr_output_dir / "joints.npy", joints)
    np.save(gvhmr_output_dir / "vertices.npy", vertices)
