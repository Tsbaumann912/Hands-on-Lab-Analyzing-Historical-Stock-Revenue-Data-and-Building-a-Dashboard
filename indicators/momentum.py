"""Vectorised momentum indicators: RSI, MACD, Stochastic."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from indicators._base import _validate_warmup


def rsi(close: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
    """
    Relative Strength Index (Wilder smoothing).

    Parameters
    ----------
    close:
        1-D array of closing prices.
    period:
        Look-back window (default 14).

    Returns
    -------
    np.ndarray | None
        RSI values in [0, 100]; ``None`` when ``len(close) < period + 1``.
    """
    if _validate_warmup(close, period + 1, "RSI") is None:
        return None

    delta = np.diff(close.astype(np.float64))
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Wilder's smoothed moving average via EWM-style recurrence
    alpha = 1.0 / period
    avg_gain = np.empty(len(gains), dtype=np.float64)
    avg_loss = np.empty(len(gains), dtype=np.float64)

    # Seed with simple mean over the first window
    avg_gain[period - 1] = gains[:period].mean()
    avg_loss[period - 1] = losses[:period].mean()

    for i in range(period, len(gains)):
        avg_gain[i] = avg_gain[i - 1] * (1 - alpha) + gains[i] * alpha
        avg_loss[i] = avg_loss[i - 1] * (1 - alpha) + losses[i] * alpha

    # The loop above is over a derived delta array, not the price array directly,
    # which is the standard Wilder recurrence — unavoidable scalar recurrence.
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi_values = np.full(len(close), np.nan, dtype=np.float64)
    rsi_values[period:] = 100.0 - (100.0 / (1.0 + rs[period - 1:]))
    return rsi_values


class MACDResult(NamedTuple):
    macd_line: np.ndarray
    signal_line: np.ndarray
    histogram: np.ndarray


def macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[MACDResult]:
    """
    MACD — Moving Average Convergence/Divergence.

    Returns ``None`` if there are fewer than ``slow + signal`` data points.
    """
    from indicators.trend import ema  # local import to avoid circular reference

    min_len = slow + signal
    if _validate_warmup(close, min_len, "MACD") is None:
        return None

    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)

    if ema_fast is None or ema_slow is None:
        return None

    macd_line = ema_fast - ema_slow
    signal_line = _ema_of(macd_line, signal)
    histogram = macd_line - signal_line
    return MACDResult(macd_line, signal_line, histogram)


def _ema_of(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA of an array (vectorised via cumulative formula)."""
    alpha = 2.0 / (period + 1)
    out = np.full_like(arr, np.nan)
    start = np.argmax(~np.isnan(arr))
    out[start] = arr[start]
    for i in range(start + 1, len(arr)):
        if not np.isnan(arr[i]):
            out[i] = out[i - 1] * (1 - alpha) + arr[i] * alpha
        else:
            out[i] = out[i - 1]
    return out


class StochasticResult(NamedTuple):
    pct_k: np.ndarray
    pct_d: np.ndarray


def stochastic_oscillator(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> Optional[StochasticResult]:
    """
    Stochastic Oscillator (%K, %D).

    Returns ``None`` when the input is shorter than ``k_period + d_period``.
    """
    min_len = k_period + d_period
    if _validate_warmup(close, min_len, "Stochastic") is None:
        return None

    n = len(close)
    pct_k = np.full(n, np.nan, dtype=np.float64)

    # Vectorised rolling window using stride tricks
    high_f = high.astype(np.float64)
    low_f = low.astype(np.float64)
    close_f = close.astype(np.float64)

    # Build rolling max/min arrays
    from numpy.lib.stride_tricks import sliding_window_view
    high_roll = sliding_window_view(high_f, k_period).max(axis=-1)
    low_roll = sliding_window_view(low_f, k_period).min(axis=-1)

    denom = high_roll - low_roll
    # Avoid divide-by-zero when range is 0
    pct_k[k_period - 1:] = np.where(
        denom == 0,
        50.0,
        100.0 * (close_f[k_period - 1:] - low_roll) / denom,
    )

    pct_d = _ema_of(pct_k, d_period)
    return StochasticResult(pct_k, pct_d)
