"""
Build Holosoma ``object_interaction`` NPZ from ``unified.npz`` (synced video + mocap).

Output contract matches ``holosoma_retargeting.src.utils.load_object_interaction_npz``:
  - ``global_joint_positions``: (T, 52, 3), Holosoma ``SMPLH_DEMO_JOINTS`` order
  - ``object_poses``: (T, 7) as ``[qw, qx, qy, qz, x, y, z]`` (same as InterMimic loader)
  - ``height``: scalar meters (for robot / human scale)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R

# Holosoma SMPL-H joint order (must match holosoma ``config_types/data_type.py``).
SMPLH_DEMO_JOINTS: list[str] = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

# Same axis swap as Holosoma ``transform_y_up_to_z_up``: [x,y,z] -> [x,z,y]
COORD_SWAP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def smplh_name_to_smplx_joint_name(smplh: str) -> str:
    """Map Holosoma SMPL-H demo joint label to ``smplx.joint_names.JOINT_NAMES`` entry."""
    if smplh == "Torso":
        return "spine1"
    if smplh == "Spine":
        return "spine2"
    if smplh == "Chest":
        return "spine3"
    if smplh == "L_Thorax":
        return "left_collar"
    if smplh == "R_Thorax":
        return "right_collar"
    if smplh == "L_Toe":
        return "left_foot"
    if smplh == "R_Toe":
        return "right_foot"
    if smplh == "Pelvis":
        return "pelvis"
    if smplh.startswith("L_"):
        return "left_" + smplh[2:].lower()
    if smplh.startswith("R_"):
        return "right_" + smplh[2:].lower()
    return smplh.lower()


def _build_smplh_index_into_smplx() -> np.ndarray:
    from smplx.joint_names import JOINT_NAMES as SX_NAMES

    smplx_to_i = {n: i for i, n in enumerate(SX_NAMES)}
    indices: list[int] = []
    for name in SMPLH_DEMO_JOINTS:
        sx = smplh_name_to_smplx_joint_name(name)
        if sx not in smplx_to_i:
            raise KeyError(f"No SMPL-X joint for SMPL-H '{name}' -> '{sx}'")
        indices.append(smplx_to_i[sx])
    return np.asarray(indices, dtype=np.int64)


def _decode_body_names(raw: Any) -> list[str]:
    if raw is None:
        raise ValueError("unified npz missing vicon__body_names (required for board / shoes).")
    names = [str(x) for x in np.asarray(raw).tolist()]
    return names


def _interp_nans_along_time_xyz(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate non-finite samples along time; first axis is T, last is xyz."""
    a = np.asarray(x, dtype=np.float64).copy()
    if a.ndim < 3 or a.shape[-1] != 3:
        raise ValueError(f"Expected (T, ..., 3) array, got shape {a.shape}")
    t = a.shape[0]
    flat = a.reshape(t, -1, 3)
    _, ntail, _ = flat.shape
    for j in range(ntail):
        for k in range(3):
            col = flat[:, j, k]
            mask = np.isfinite(col)
            if bool(np.all(mask)):
                continue
            if not bool(np.any(mask)):
                col[:] = 0.0
                continue
            xi = np.arange(t, dtype=np.float64)
            col[~mask] = np.interp(xi[~mask], xi[mask], col[mask])
    return flat.reshape(a.shape)


def _sanitize_object_poses_wxyz_xyz(op: np.ndarray) -> np.ndarray:
    """Interpolate NaN translations; invalid quaternions -> identity, then normalize."""
    out = np.asarray(op, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 7:
        raise ValueError(f"object_poses must be (T, 7), got {out.shape}")
    t = out.shape[0]
    for k in (4, 5, 6):
        col = out[:, k]
        mask = np.isfinite(col)
        if bool(np.all(mask)):
            continue
        if not bool(np.any(mask)):
            col[:] = 0.0
            continue
        xi = np.arange(t, dtype=np.float64)
        col[~mask] = np.interp(xi[~mask], xi[mask], col[mask])
    ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for i in range(t):
        q = out[i, :4]
        if (not np.all(np.isfinite(q))) or float(np.linalg.norm(q)) < 1e-12:
            out[i, :4] = ident
        else:
            out[i, :4] = q / np.linalg.norm(q)
    return out


def _swap_y_up_to_z_up_points(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != 3:
        raise ValueError(f"Expected ...x3 positions, got shape {x.shape}")
    return (COORD_SWAP @ x.reshape(-1, 3).T).T.reshape(x.shape)




def _quat_wxyz_from_matrix(mat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = R.from_matrix(mat).as_quat()
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _synthesize_board_pose_wxyz(
    left_shoe: np.ndarray,
    right_shoe: np.ndarray,
    board_pos: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Board orientation: local x along length, y across width, z = deck normal (world up)."""
    w = np.asarray(world_up, dtype=np.float64).reshape(3)
    w = w / (np.linalg.norm(w) + 1e-12)
    ls = np.asarray(left_shoe, dtype=np.float64).reshape(3)
    rs = np.asarray(right_shoe, dtype=np.float64).reshape(3)
    width0 = ls - rs
    width_flat = width0 - np.dot(width0, w) * w
    nw = np.linalg.norm(width_flat)
    if nw < 1e-6:
        width_flat = np.cross(np.array([1.0, 0.0, 0.0]), w)
        nw = np.linalg.norm(width_flat)
    width_flat = width_flat / (nw + 1e-12)
    x_axis = np.cross(w, width_flat)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    y_axis = np.cross(w, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    r_mat = np.stack([x_axis, y_axis, w], axis=1)
    quat = _quat_wxyz_from_matrix(r_mat)
    pos = np.asarray(board_pos, dtype=np.float64).reshape(3)
    return np.concatenate([quat, pos])


def _forward_smplx_joints(
    model_root: Path,
    body_pose: np.ndarray,
    transl: np.ndarray,
    global_orient: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Return (T, J_smplx, 3) joint positions from SMPL-X parameters."""
    body_pose = np.asarray(body_pose, dtype=np.float64)
    transl = np.asarray(transl, dtype=np.float64)
    global_orient = np.asarray(global_orient, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    if betas.ndim == 1:
        betas = np.broadcast_to(betas, (body_pose.shape[0], betas.shape[0]))
    t = body_pose.shape[0]
    num_betas = betas.shape[1]

    model = smplx.create(
        str(model_root),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=num_betas,
        batch_size=t,
        use_face_contour=True,
    )
    out = model(
        betas=torch.as_tensor(betas, dtype=torch.float32),
        body_pose=torch.as_tensor(body_pose, dtype=torch.float32),
        transl=torch.as_tensor(transl, dtype=torch.float32),
        global_orient=torch.as_tensor(global_orient, dtype=torch.float32),
    )
    joints = out.joints.detach().cpu().numpy()
    return joints


def export_holosoma_object_npz(
    unified_npz: Path,
    out_path: Path,
    *,
    smplx_model_root: Path,
    swap_y_up_to_z_up: bool = False,
    default_height_m: float = 1.72,
) -> None:
    """
    Read ``unified.npz`` and write Holosoma-ready ``.npz`` at ``out_path``.

    Required unified keys:
      ``video__body_pose``, ``video__transl``, ``video__global_orient``, ``video__betas``
      ``vicon__body_pos``, ``vicon__body_names``
    Optional:
      ``vicon__body_quat`` (wxyz per row), else board orientation is synthesized from shoes.
    """
    unified_npz = Path(unified_npz)
    z = np.load(unified_npz, allow_pickle=True)

    need = (
        "video__body_pose",
        "video__transl",
        "video__global_orient",
        "video__betas",
        "vicon__body_pos",
        "vicon__body_names",
    )
    missing = [k for k in need if k not in z.files]
    if missing:
        raise KeyError(f"{unified_npz} missing keys {missing}; have {sorted(z.files)}")

    body_pose = z["video__body_pose"]
    transl = np.asarray(z["video__transl"], dtype=np.float64)
    global_orient = z["video__global_orient"]
    betas = z["video__betas"]
    body_pos = np.asarray(z["vicon__body_pos"], dtype=np.float64)
    body_pos = _interp_nans_along_time_xyz(body_pos)
    names = _decode_body_names(z["vicon__body_names"])
    body_quat = z["vicon__body_quat"] if "vicon__body_quat" in z.files else None

    t = body_pose.shape[0]
    if body_pos.shape[0] != t:
        raise ValueError(
            f"Time length mismatch: video {t} vs vicon body_pos {body_pos.shape[0]} "
            f"(re-build unified with matching timeline)."
        )

    if swap_y_up_to_z_up:
        transl = _swap_y_up_to_z_up_points(transl)
        body_pos = _swap_y_up_to_z_up_points(body_pos)

    try:
        i_board = names.index("Skateboard")
        i_left = names.index("Left_Shoe")
        i_right = names.index("Right_Shoe")
    except ValueError as e:
        raise ValueError(
            f"vicon__body_names must include Skateboard, Left_Shoe, Right_Shoe; got {names}"
        ) from e

    smplx_j = _forward_smplx_joints(Path(smplx_model_root), body_pose, transl, global_orient, betas)
    idx_map = _build_smplh_index_into_smplx()
    human = np.zeros((t, 52, 3), dtype=np.float64)
    human[:] = smplx_j[:, idx_map, :]

    if swap_y_up_to_z_up:
        human = _swap_y_up_to_z_up_points(human)

    # Mocap feet on deck: override ankles + toes (Holosoma foot sticking + interaction mesh).
    human[:, SMPLH_DEMO_JOINTS.index("L_Ankle"), :] = body_pos[:, i_left, :]
    human[:, SMPLH_DEMO_JOINTS.index("R_Ankle"), :] = body_pos[:, i_right, :]
    toe_off = np.array([0.0, 0.0, 0.02], dtype=np.float64)
    human[:, SMPLH_DEMO_JOINTS.index("L_Toe"), :] = body_pos[:, i_left, :] + toe_off
    human[:, SMPLH_DEMO_JOINTS.index("R_Toe"), :] = body_pos[:, i_right, :] + toe_off

    object_poses = np.zeros((t, 7), dtype=np.float64)
    for i in range(t):
        bpos = body_pos[i, i_board, :]
        lsh = body_pos[i, i_left, :]
        rsh = body_pos[i, i_right, :]
        use_synth = swap_y_up_to_z_up or body_quat is None
        if not use_synth and body_quat is not None:
            q = np.asarray(body_quat[i, i_board, :], dtype=np.float64).reshape(4)
            if np.all(np.isfinite(q)) and np.linalg.norm(q) > 1e-6:
                w, x, y, z = q[0], q[1], q[2], q[3]
                n = np.linalg.norm([w, x, y, z])
                w, x, y, z = w / n, x / n, y / n, z / n
                row = np.array([w, x, y, z, bpos[0], bpos[1], bpos[2]], dtype=np.float64)
            else:
                row = _synthesize_board_pose_wxyz(lsh, rsh, bpos)
        else:
            row = _synthesize_board_pose_wxyz(lsh, rsh, bpos)
        object_poses[i] = row

    human = _interp_nans_along_time_xyz(human)
    object_poses = _sanitize_object_poses_wxyz_xyz(object_poses)

    pelvis = human[:, 0, :]
    head = human[:, SMPLH_DEMO_JOINTS.index("Head"), :]
    height_est = float(np.nanmean(np.linalg.norm(head - pelvis, axis=1)) * 2.1)
    height = height_est if np.isfinite(height_est) and height_est > 1.0 else default_height_m

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        global_joint_positions=human.astype(np.float32),
        object_poses=object_poses.astype(np.float32),
        height=np.float32(height),
        source_unified=str(unified_npz),
    )
