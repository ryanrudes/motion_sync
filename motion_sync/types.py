"""Small shared typing helpers for ndarray-backed models."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def as_float_array(
    value: Any,
    *,
    name: str = "array",
    allow_nan: bool = True,
) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if not allow_nan and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr
