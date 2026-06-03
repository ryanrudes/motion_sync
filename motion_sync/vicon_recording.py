"""Vicon-only mocap export (pre-sync)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from motion_sync import _storage


class ViconRecording:
    """One demo's Vicon mocap timeline from ``vicon.npz``."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: str | Path) -> ViconRecording:
        """Load from a demo directory or ``vicon.npz`` file."""
        path = Path(path)
        if path.is_dir():
            path = _storage.resolve_vicon_path(path)
        return cls(_storage.read_vicon_mocap(path))

    def to_syncer_dict(self) -> dict[str, Any]:
        """Dict consumed by :mod:`motion_sync.syncer`."""
        return dict(self._data)
