"""Vectorised trend indicators: SMA, EMA, WMA, SuperTrend."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from indicators._base import _fill_warmup, _validate_warmup
from indicators.volatility import atr


def sma(close: np.ndarray, period: int) -> Optional[np.ndarray]:
    """
    Simple Moving Average.

    Returns ``None`` if ``len(close) < period``.
    """
    if _validate_warmup(close, period, "SMA") is None:
        return None

    kernel = np.ones(period, dtype=np.float64) / period
    raw = np.convolve(close.astype(np.float64), kernel, mode="full")[period - 1: len(close)]
    out = np.full(len(close), np.nan, dtype=np.float64)
    out[period - 1:] = raw[:len(close) - period + 1]
    return out


def ema(close: np.ndarray, period: int) -> Optional[np.ndarray]:
    """
    Exponential Moving Average (standard 2/(n+1) smoothing factor).

    Returns ``None`` if ``len(close) < period``.
    """
    if _validate_warmup(close, period, "EMA") is None:
        return None

    alpha = 2.0 / (period + 1)
    close_f = close.astype(np.float64)
    out = np.full(len(close_f), np.nan, dtype=np.float64)
    out[period - 1] = close_f[:period].mean()

    for i in range(period, len(close_f)):
        out[i] = out[i - 1] + alpha * (close_f[i] - out[i - 1])

    return out


def wma(close: np.ndarray, period: int) -> Optional[np.ndarray]:
    """
    Weighted Moving Average (linearly weighted).

    Returns ``None`` if ``len(close) < period``.
    """
    if _validate_warmup(close, period, "WMA") is None:
        return None

    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = weights.sum()
    close_f = close.astype(np.float64)

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(close_f, period)
    wma_vals = (windows * weights).sum(axis=-1) / weight_sum

    out = np.full(len(close_f), np.nan, dtype=np.float64)
    out[period - 1:] = wma_vals
    return out


class SuperTrendResult(NamedTuple):
    supertrend: np.ndarray
    direction: np.ndarray    # +1 = bullish, -1 = bearish


def supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> Optional[SuperTrendResult]:
    """
    SuperTrend indicator (ATR-based dynamic support/resistance).

    Returns ``None`` if insufficient data.
    """
    atr_vals = atr(high, low, close, period)
    if atr_vals is None:
        return None

    high_f = high.astype(np.float64)
    low_f = low.astype(np.float64)
    close_f = close.astype(np.float64)
    n = len(close_f)

    hl2 = (high_f + low_f) / 2.0
    upper_band = hl2 + multiplier * atr_vals
    lower_band = hl2 - multiplier * atr_vals

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    trend = np.ones(n, dtype=np.float64)  # +1 bullish
    supertrend_line = np.full(n, np.nan, dtype=np.float64)

    for i in range(1, n):
        if np.isnan(atr_vals[i]):
            continue
        final_upper[i] = (
            upper_band[i]
            if upper_band[i] < final_upper[i - 1] or close_f[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower_band[i]
            if lower_band[i] > final_lower[i - 1] or close_f[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )

        if trend[i - 1] == -1 and close_f[i] > final_upper[i]:
            trend[i] = 1
        elif trend[i - 1] == 1 and close_f[i] < final_lower[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

        supertrend_line[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

    return SuperTrendResult(supertrend_line, trend)
