"""Vectorised volume indicators: OBV, VWAP, Volume Oscillator."""

from __future__ import annotations

from typing import Optional

import numpy as np



def obv(close: np.ndarray, volume: np.ndarray) -> Optional[np.ndarray]:
    """
    On-Balance Volume (OBV).

    Returns ``None`` if arrays are empty.
    """
    if len(close) == 0:
        return None

    close_f = close.astype(np.float64)
    volume_f = volume.astype(np.float64)

    direction = np.sign(np.diff(close_f, prepend=close_f[0]))
    direction[0] = 0  # first bar has no prior

    return np.cumsum(direction * volume_f)


def vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Volume-Weighted Average Price (session-rolling VWAP).

    For futures, this is typically reset each trading session.
    Here we return the cumulative intraday VWAP starting from index 0.
    """
    if len(close) == 0:
        return None

    typical_price = (high.astype(np.float64) + low.astype(np.float64) + close.astype(np.float64)) / 3.0
    volume_f = volume.astype(np.float64)

    cum_tpv = np.cumsum(typical_price * volume_f)
    cum_vol = np.cumsum(volume_f)

    return np.where(cum_vol > 0, cum_tpv / cum_vol, np.nan)


def volume_oscillator(
    volume: np.ndarray,
    fast_period: int = 5,
    slow_period: int = 20,
) -> Optional[np.ndarray]:
    """
    Volume Oscillator — percentage difference between fast and slow volume SMAs.

    Returns ``None`` if insufficient data.
    """
    from indicators.trend import sma as _sma

    fast_sma = _sma(volume.astype(np.float64), fast_period)
    slow_sma = _sma(volume.astype(np.float64), slow_period)

    if fast_sma is None or slow_sma is None:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(slow_sma != 0, (fast_sma - slow_sma) / slow_sma * 100.0, np.nan)
