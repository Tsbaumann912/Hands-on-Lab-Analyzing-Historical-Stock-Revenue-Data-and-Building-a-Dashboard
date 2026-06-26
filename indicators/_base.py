"""Shared helpers for the indicators layer."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _validate_warmup(arr: np.ndarray, period: int, name: str) -> Optional[np.ndarray]:
    """
    Return ``None`` when the array is too short to compute the indicator.

    All indicator functions should call this guard before computation.
    """
    if len(arr) < period:
        return None  # signal: insufficient data
    return arr


def _fill_warmup(out: np.ndarray, period: int) -> np.ndarray:
    """Fill the first (period-1) values with NaN to reflect the warm-up gap."""
    out[:period - 1] = np.nan
    return out
