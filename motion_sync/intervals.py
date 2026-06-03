"""Interval helpers for contact and motion masks (video-clock ``time_s``)."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

IntervalList: TypeAlias = list[tuple[float, float]]


def intervals_from_mask(
    t: ArrayLike,
    mask: ArrayLike,
    min_duration: float = 0.0,
) -> IntervalList:
    """Contiguous ``(t_start, t_end)`` runs where ``mask`` is True."""
    t_arr = np.asarray(t, dtype=float)
    mask_arr = np.asarray(mask, dtype=bool)

    if t_arr.ndim != 1 or mask_arr.ndim != 1:
        raise ValueError("t and mask must be 1D arrays.")
    if len(t_arr) != len(mask_arr):
        raise ValueError("t and mask must have the same length.")
    if len(t_arr) == 0:
        return []

    padded = np.r_[False, mask_arr, False]
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    intervals: IntervalList = []
    for start, end in zip(starts, ends, strict=True):
        start_time = float(t_arr[start])
        end_time = float(t_arr[end - 1])
        if end_time - start_time >= min_duration:
            intervals.append((start_time, end_time))
    return intervals
