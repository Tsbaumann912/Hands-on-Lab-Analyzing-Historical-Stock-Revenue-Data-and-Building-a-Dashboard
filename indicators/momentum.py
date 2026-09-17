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

    # Wilder recurrence — unavoidable scalar loop; suppress overflow for
    # extreme/synthetic price series (values are capped before next step).
    _MAX_AVG = np.finfo(np.float64).max / 2.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for i in range(period, len(gains)):
            avg_gain[i] = avg_gain[i - 1] * (1 - alpha) + gains[i] * alpha
            avg_loss[i] = avg_loss[i - 1] * (1 - alpha) + losses[i] * alpha
            if avg_gain[i] > _MAX_AVG:
                avg_gain[i] = _MAX_AVG
            if avg_loss[i] > _MAX_AVG:
                avg_loss[i] = _MAX_AVG

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


class StochasticRSIResult(NamedTuple):
    stoch_rsi: np.ndarray
    pct_k: np.ndarray
    pct_d: np.ndarray


def stochastic_rsi(
    close: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> Optional[StochasticRSIResult]:
    """
    Stochastic RSI — stochastic applied to RSI values (0–100 scale).

    ``StochRSI = 100 * (RSI - min_n(RSI)) / (max_n(RSI) - min_n(RSI))``,
    then %K / %D are SMA smooths of StochRSI.

    Returns ``None`` when the series is shorter than the warm-up window.
    """
    min_len = rsi_period + stoch_period + k_period + d_period
    if _validate_warmup(close, min_len, "StochasticRSI") is None:
        return None

    rsi_vals = rsi(close, rsi_period)
    if rsi_vals is None:
        return None

    n = len(close)
    stoch = np.full(n, np.nan, dtype=np.float64)
    from numpy.lib.stride_tricks import sliding_window_view

    # Only windows where RSI is finite contribute; leading RSI NaNs stay NaN.
    rsi_f = rsi_vals.astype(np.float64)
    # Replace leading NaNs with a fill that won't enter windows until warm-up.
    first_valid = int(np.argmax(np.isfinite(rsi_f)))
    if not np.isfinite(rsi_f[first_valid]):
        return None

    usable = rsi_f[first_valid:]
    if len(usable) < stoch_period:
        return None

    windows = sliding_window_view(usable, stoch_period)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_max = np.nanmax(windows, axis=-1)
        r_min = np.nanmin(windows, axis=-1)
        denom = r_max - r_min
        raw = np.where(
            (~np.isfinite(denom)) | (denom == 0.0),
            50.0,
            100.0 * (usable[stoch_period - 1 :] - r_min) / denom,
        )
    # Align raw StochRSI onto full length (index of last bar in each window).
    start_idx = first_valid + stoch_period - 1
    stoch[start_idx : start_idx + len(raw)] = raw

    pct_k = _sma_of(stoch, k_period)
    pct_d = _sma_of(pct_k, d_period)
    return StochasticRSIResult(stoch_rsi=stoch, pct_k=pct_k, pct_d=pct_d)


def _sma_of(arr: np.ndarray, period: int) -> np.ndarray:
    """Trailing SMA that preserves NaNs until a full finite window exists."""
    import warnings

    out = np.full_like(arr, np.nan, dtype=np.float64)
    if period < 1 or len(arr) < period:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(arr.astype(np.float64), period)
    finite = np.sum(np.isfinite(windows), axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(windows, axis=-1)
    means = np.where(finite >= period, means, np.nan)
    out[period - 1 :] = means
    return out
