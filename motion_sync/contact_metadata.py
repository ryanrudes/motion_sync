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
"""Layer metadata key: frame count when detection ran."""

META_TIME_FINGERPRINT = "time_fingerprint"
"""Layer metadata key: :func:`fingerprint_timeline` of ``clip.time_s`` at detection."""

META_CONFIG_HASH = "config_hash"
"""Layer metadata key: :func:`hash_detection_config` of the detector config."""


def fingerprint_timeline(time_s: np.ndarray) -> str:
    """Hash clip timeline shape and endpoints (detects sync regen / crop).

    Args:
        time_s: Clip time axis in seconds.

    Returns:
        16-character hex digest, or ``"empty"`` when no finite samples exist.
    """
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
    """Stable short hash of a detector config dataclass or dict.

    Args:
        config: Dataclass, dict, or other object serialized for hashing.

    Returns:
        16-character hex digest, or ``None`` when ``config`` is ``None``.
    """
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
    """Attach provenance fields so stale layers can be detected after sync changes.

    Args:
        layer: Contact layer to stamp (copied on return).
        clip: Clip supplying ``frame_count`` and ``time_s``.
        config: Optional detector config hashed into metadata.

    Returns:
        Copy of ``layer`` with :data:`META_SOURCE_FRAME_COUNT`, :data:`META_TIME_FINGERPRINT`,
        and optionally :data:`META_CONFIG_HASH` set.
    """
    meta = dict(layer.metadata)
    meta[META_SOURCE_FRAME_COUNT] = int(clip.frame_count)
    meta[META_TIME_FINGERPRINT] = fingerprint_timeline(clip.time_s)
    config_hash = hash_detection_config(config)
    if config_hash is not None:
        meta[META_CONFIG_HASH] = config_hash
    return layer.model_copy(update={"metadata": meta})


def contact_layer_is_fresh(layer: ContactLayer, clip: Any) -> bool:
    """Return whether layer metadata matches the clip timeline.

    Missing provenance fields are treated as fresh (backward compatible).

    Args:
        layer: Stored contact layer.
        clip: Current clip to compare against.

    Returns:
        ``True`` if frame count and time fingerprint match (or metadata absent).
    """
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
    """Emit a warning when contact data may be out of date.

    Args:
        layer: Stored contact layer.
        clip: Current clip to compare against.
        layer_id: Optional label for the warning (defaults to ``layer.layer_id``).

    Returns:
        ``True`` if the layer is fresh; ``False`` if stale (after emitting a warning).
    """
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
