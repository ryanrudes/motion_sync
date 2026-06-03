"""Small shared typing helpers for ndarray-backed models."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray1D = np.ndarray[tuple[int], np.dtype[np.integer]]

def as_float_array(
    value: Any,
    *,
    name: str = "array",
    allow_nan: bool = True,
) -> FloatArray:
    """Coerce ``value`` to a float64 ndarray.

    Args:
        value: Array-like input.
        name: Label used in validation error messages.
        allow_nan: If False, reject arrays containing non-finite values.

    Returns:
        ``float64`` ndarray view or copy of ``value``.

    Raises:
        ValueError: Non-finite values when ``allow_nan`` is False.
    """
    arr = np.asarray(value, dtype=np.float64)
    if not allow_nan and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr
