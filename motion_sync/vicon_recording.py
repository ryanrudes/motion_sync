"""Vicon-only mocap export (pre-sync)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from motion_sync import _storage


class ViconRecording:
    """Vicon mocap timeline loaded from ``vicon.npz`` (mocap clock, pre-time-sync).

    Use :meth:`load` to read a demo directory or explicit NPZ path, then pass
    :meth:`to_syncer_dict` into the sync pipeline.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Wrap a column-oriented Vicon dict produced by :mod:`motion_sync._storage`.

        Args:
            data: Keys such as ``t``, ``body_names``, ``body_pos``, and optional markers.
        """
        self._data = data

    @classmethod
    def load(cls, path: str | Path) -> ViconRecording:
        """Load from a demo directory or ``vicon.npz`` file.

        Args:
            path: Demo folder containing ``vicon.npz``, or the NPZ path itself.

        Returns:
            Recording ready for :meth:`to_syncer_dict`.
        """
        path = Path(path)
        if path.is_dir():
            path = _storage.resolve_vicon_path(path)
        return cls(_storage.read_vicon_mocap(path))

    def to_syncer_dict(self) -> dict[str, Any]:
        """Export the internal layout for :mod:`motion_sync.syncer`.

        Returns:
            Shallow copy of the storage dict (times, bodies, optional markers).
        """
        return dict(self._data)
