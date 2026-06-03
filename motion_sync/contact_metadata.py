"""Detection provenance and stale-layer checks for contact layers."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from motion_sync.contact_layer import ContactLayer

META_SOURCE_FRAME_COUNT = "source_frame_count"
META_TIME_FINGERPRINT = "time_fingerprint"
META_CONFIG_HASH = "config_hash"


def fingerprint_timeline(time_s: np.ndarray) -> str:
    """Short hash of clip timeline shape and endpoints (detects sync regen / crop)."""
    t = np.asarray(time_s, dtype=np.float64)
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return "empty"
    digest = hashlib.sha256()
    digest.update(str(int(finite.shape[0])).encode())
    digest.update(np.asarray(finite[0], dtype=np.float64).tobytes())
    digest.update(np.asarray(finite[-1], dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


def hash_detection_config(config: Any) -> str | None:
    """Stable short hash of a detector config dataclass (or None)."""
    if config is None:
        return None
    if is_dataclass(config):
        payload = asdict(config)
    elif isinstance(config, dict):
        payload = config
    else:
        payload = {"repr": repr(config)}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def stamp_detection_metadata(
    layer: ContactLayer,
    clip: Any,
    *,
    config: Any = None,
) -> ContactLayer:
    """Attach provenance fields so stale layers can be detected after sync changes."""
    meta = dict(layer.metadata)
    meta[META_SOURCE_FRAME_COUNT] = int(clip.frame_count)
    meta[META_TIME_FINGERPRINT] = fingerprint_timeline(clip.time_s)
    config_hash = hash_detection_config(config)
    if config_hash is not None:
        meta[META_CONFIG_HASH] = config_hash
    return layer.model_copy(update={"metadata": meta})


def contact_layer_is_fresh(layer: ContactLayer, clip: Any) -> bool:
    """True if layer metadata matches the clip timeline (or metadata is absent)."""
    stored_frames = layer.metadata.get(META_SOURCE_FRAME_COUNT)
    if stored_frames is None:
        return True

    if int(stored_frames) != int(clip.frame_count):
        return False

    stored_fp = layer.metadata.get(META_TIME_FINGERPRINT)
    if stored_fp is None:
        return True

    return str(stored_fp) == fingerprint_timeline(clip.time_s)


def warn_if_stale_contact_layer(
    layer: ContactLayer,
    clip: Any,
    *,
    layer_id: str | None = None,
) -> bool:
    """Emit a warning when contact data may be out of date; return False if stale."""
    if contact_layer_is_fresh(layer, clip):
        return True

    label = layer_id or layer.layer_id
    stored = layer.metadata.get(META_SOURCE_FRAME_COUNT)
    warnings.warn(
        f"Contact layer {label!r} may be stale: recorded for {stored!r} frames but "
        f"clip has {clip.frame_count} (timeline changed after detection). "
        f"Re-run clip.detect(...) or detect with --force.",
        stacklevel=3,
    )
    return False
