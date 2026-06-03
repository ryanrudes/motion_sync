"""Contact annotations on a synced clip timeline."""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from motion_sync.types import BoolArray

ContactLayerKind = Literal["binary", "categorical"]
CONTACT_STORAGE_VERSION = 1

_LAYER_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ContactLayer(BaseModel):
    """One detector output aligned to a clip timeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    layer_id: str
    kind: ContactLayerKind
    subjects: tuple[str, ...]
    labels: tuple[str, ...] = ()
    states: np.ndarray | None = None
    mask: np.ndarray | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("layer_id")
    @classmethod
    def _layer_id(cls, value: str) -> str:
        if not _LAYER_ID_RE.match(value):
            raise ValueError(
                f"layer_id must match {_LAYER_ID_RE.pattern!r}, got {value!r}"
            )
        return value

    @field_validator("states", mode="before")
    @classmethod
    def _states(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.int8)
        if arr.ndim != 2:
            raise ValueError("states must have shape (frames, subjects)")
        return arr

    @field_validator("mask", mode="before")
    @classmethod
    def _mask(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=bool)
        if arr.ndim != 2:
            raise ValueError("mask must have shape (frames, subjects)")
        return arr

    @model_validator(mode="after")
    def _consistent_kind(self) -> ContactLayer:
        n_subjects = len(self.subjects)
        if self.kind == "categorical":
            if not self.labels:
                raise ValueError("categorical layers require labels")
            if self.states is None:
                raise ValueError("categorical layers require states")
            if self.mask is not None:
                raise ValueError("categorical layers must not set mask")
        elif self.kind == "binary":
            if self.mask is None:
                raise ValueError("binary layers require mask")
            if self.states is not None:
                raise ValueError("binary layers must not set states")
        if self.states is not None and self.states.shape[1] != n_subjects:
            raise ValueError("states subject axis must match subjects")
        if self.mask is not None and self.mask.shape[1] != n_subjects:
            raise ValueError("mask subject axis must match subjects")
        return self

    @property
    def frame_count(self) -> int:
        if self.states is not None:
            return int(self.states.shape[0])
        if self.mask is not None:
            return int(self.mask.shape[0])
        return 0

    def validate_frame_count(self, n_frames: int) -> None:
        if self.frame_count != n_frames:
            raise ValueError(
                f"contact layer {self.layer_id!r} has {self.frame_count} frames, "
                f"expected {n_frames}"
            )

    def subset_frames(self, mask: BoolArray) -> ContactLayer:
        mask = np.asarray(mask, dtype=bool)
        states = None if self.states is None else self.states[mask]
        layer_mask = None if self.mask is None else self.mask[mask]
        return self.model_copy(update={"states": states, "mask": layer_mask})


def _storage_prefix(layer_id: str) -> str:
    return f"contact__{layer_id}__"


def encode_contact_layers(layers: dict[str, ContactLayer]) -> dict[str, np.ndarray]:
    """Serialize contact layers into synced.npz keys."""
    out: dict[str, np.ndarray] = {}
    for layer in layers.values():
        p = _storage_prefix(layer.layer_id)
        out[f"{p}version"] = np.array(CONTACT_STORAGE_VERSION, dtype=np.int32)
        out[f"{p}kind"] = np.array(layer.kind, dtype=object)
        out[f"{p}subjects"] = np.array(layer.subjects, dtype=object)
        if layer.labels:
            out[f"{p}labels"] = np.array(layer.labels, dtype=object)
        if layer.states is not None:
            out[f"{p}states"] = np.asarray(layer.states, dtype=np.int8)
        if layer.mask is not None:
            out[f"{p}mask"] = np.asarray(layer.mask, dtype=bool)
        for key, value in layer.metadata.items():
            meta_key = f"{p}meta__{key}"
            if isinstance(value, (bool, int, float, np.floating, np.integer)):
                out[meta_key] = np.asarray(value)
            elif isinstance(value, str):
                out[meta_key] = np.array(value, dtype=object)
            elif isinstance(value, np.ndarray):
                out[meta_key] = np.asarray(value)
            elif isinstance(value, dict):
                names = np.array(list(value.keys()), dtype=object)
                vals = np.array([float(value[k]) for k in value.keys()], dtype=np.float64)
                out[f"{p}meta__{key}__names"] = names
                out[f"{p}meta__{key}__values"] = vals
            else:
                raise TypeError(
                    f"unsupported metadata type for {layer.layer_id}.{key}: {type(value)}"
                )
    return out


def decode_contact_layers(
    files: list[str],
    get: Any,
) -> dict[str, ContactLayer]:
    """Load contact layers from synced.npz keys."""
    layer_ids: set[str] = set()
    for key in files:
        if key.startswith("contact__") and key.endswith("__version"):
            rest = key[len("contact__") : -len("__version")]
            layer_ids.add(rest)

    layers: dict[str, ContactLayer] = {}
    for layer_id in sorted(layer_ids):
        p = _storage_prefix(layer_id)
        version = int(np.asarray(get(f"{p}version")).reshape(()))
        if version != CONTACT_STORAGE_VERSION:
            raise ValueError(
                f"unsupported contact layer version {version} for {layer_id!r}"
            )
        kind = str(np.asarray(get(f"{p}kind")).reshape(()))
        subjects = tuple(str(x) for x in get(f"{p}subjects").tolist())
        labels: tuple[str, ...] = ()
        if f"{p}labels" in files:
            labels = tuple(str(x) for x in get(f"{p}labels").tolist())
        states = None
        if f"{p}states" in files:
            states = np.asarray(get(f"{p}states"), dtype=np.int8)
        mask = None
        if f"{p}mask" in files:
            mask = np.asarray(get(f"{p}mask"), dtype=bool)

        metadata: dict[str, Any] = {}
        meta_prefix = f"{p}meta__"
        dict_keys: set[str] = set()
        for key in files:
            if not key.startswith(meta_prefix):
                continue
            suffix = key[len(meta_prefix) :]
            if suffix.endswith("__names"):
                dict_keys.add(suffix[: -len("__names")])
                continue
            if suffix.endswith("__values"):
                continue
            value = get(key)
            if hasattr(value, "shape") and value.shape == ():
                metadata[suffix] = value.item()
            elif hasattr(value, "shape") and value.dtype == object and value.shape == ():
                metadata[suffix] = str(value.item())
            else:
                metadata[suffix] = np.asarray(value)

        for dict_key in dict_keys:
            names = [str(x) for x in get(f"{meta_prefix}{dict_key}__names").tolist()]
            vals = np.asarray(get(f"{meta_prefix}{dict_key}__values"), dtype=float)
            metadata[dict_key] = {n: float(v) for n, v in zip(names, vals, strict=True)}

        layers[layer_id] = ContactLayer(
            layer_id=layer_id,
            kind=kind,  # type: ignore[arg-type]
            subjects=subjects,
            labels=labels,
            states=states,
            mask=mask,
            metadata=metadata,
        )
    return layers
