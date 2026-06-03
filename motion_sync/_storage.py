"""Internal on-disk persistence for synced clips and Vicon mocap exports.

Application code should use :class:`~motion_sync.synced_dataset.SyncClip` and
:class:`~motion_sync.vicon_recording.ViconRecording` only — not this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from motion_sync.synced_dataset import SyncClip

_SYNCED_BASENAME = "synced.npz"
_VICON_BASENAME = "vicon.npz"


def synced_dataset_path(sync_demo_dir: Path) -> Path:
    """Return ``synced.npz`` path under a demo output directory."""
    return Path(sync_demo_dir) / _SYNCED_BASENAME


def vicon_mocap_path(demo_vicon_dir: Path) -> Path:
    """Return ``vicon.npz`` path under a demo Vicon tables directory."""
    return Path(demo_vicon_dir) / _VICON_BASENAME


def resolve_synced_path(path: str | Path) -> Path:
    """Resolve a demo directory or file path to an existing ``synced.npz``.

    Args:
        path: Demo folder, ``synced.npz`` file, or path without suffix.

    Returns:
        Absolute path to the synced NPZ file.

    Raises:
        FileNotFoundError: No synced clip at the resolved location.
    """
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir() or path.suffix == "":
        path = synced_dataset_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"synced clip not found: {path}")
    return path


def resolve_vicon_path(demo_vicon_dir: Path) -> Path:
    """Resolve a demo directory to an existing ``vicon.npz``.

    Raises:
        FileNotFoundError: No Vicon mocap at the resolved location.
    """
    path = vicon_mocap_path(demo_vicon_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Vicon mocap not found: {path}")
    return path


def infer_vicon_mocap_for_synced(synced_path: Path) -> Path | None:
    """Guess sibling ``vicon_tables/<demo>/vicon.npz`` for marker names.

    Returns:
        Path when the standard layout exists, else ``None``.
    """
    synced_path = resolve_synced_path(synced_path)
    demo = synced_path.parent.name
    root = synced_path.parent.parent.parent
    candidate = root / "vicon_tables" / demo / _VICON_BASENAME
    return candidate if candidate.is_file() else None


def read_vicon_mocap(path: str | Path) -> dict[str, Any]:
    """Load ``vicon.npz`` into the dict shape expected by the sync pipeline."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    t = data["stamp"] - data["stamp"][0]
    out: dict[str, Any] = {
        "t": t,
        "body_names": data["body_names"].tolist() if "body_names" in data.files else None,
    }
    for key in data.files:
        if key in {"stamp", "body_names"}:
            continue
        value = data[key]
        if hasattr(value, "shape") and value.shape[0] == len(t):
            out[key] = value
    return out


def read_vicon_marker_names(path: str | Path) -> tuple[str, ...]:
    """Load the ``marker_names`` array from a Vicon export NPZ.

    Raises:
        KeyError: File has no ``marker_names`` array.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if "marker_names" not in data.files:
        raise KeyError(f"{path} has no marker_names array")
    return tuple(str(x) for x in data["marker_names"].tolist())


def read_vicon_markers_table(path: str | Path) -> dict[str, np.ndarray]:
    """Load per-frame marker table from convert output (``vicon/markers.npz``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def write_vicon_mocap(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a Vicon mocap dict to ``vicon.npz`` (creates parent dirs).

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


def aligned_pipeline_to_storage_dict(
    aligned: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Convert stitcher output (slash keys) to the synced on-disk layout."""
    arrays: dict[str, Any] = {}
    for key, value in aligned.items():
        if key == "__valid_masks__":
            continue
        arrays[key.replace("/", "__")] = value
    arrays["lag"] = np.array(meta["lag"])
    if meta.get("corr") is not None:
        arrays["corr"] = np.array(meta["corr"])
    if meta.get("body_names") is not None:
        arrays["vicon__body_names"] = np.array(meta["body_names"], dtype=object)
    return arrays


def read_synced_clip(
    path: str | Path,
    *,
    name: str,
    vicon_mocap: Path | None,
) -> SyncClip:
    """Load a :class:`~motion_sync.synced_dataset.SyncClip` from disk.

    Args:
        path: Demo directory or ``synced.npz`` file.
        name: Demo label stored on the clip.
        vicon_mocap: Optional ``vicon.npz`` used to attach marker name strings.

    Returns:
        Hydrated sync clip (marker names attached when ``vicon_mocap`` is set).
    """
    from motion_sync.synced_dataset import SyncClip

    npz_path = resolve_synced_path(path)
    raw = np.load(npz_path, allow_pickle=True)
    clip = SyncClip._from_storage(raw, path=npz_path, name=name)
    if vicon_mocap is not None and clip.vicon.markers is not None:
        clip = clip._with_marker_names(read_vicon_marker_names(vicon_mocap))
    return clip


def write_synced_clip(clip: SyncClip, path: str | Path) -> Path:
    """Persist a sync clip to ``synced.npz`` (creates parent dirs).

    Returns:
        Path to the written file.
    """
    path = Path(path)
    if not path.is_file() and (path.is_dir() or path.suffix == ""):
        path = synced_dataset_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **clip._to_storage())
    return path
