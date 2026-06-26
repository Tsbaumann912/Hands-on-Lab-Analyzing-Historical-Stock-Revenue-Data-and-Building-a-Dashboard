"""Vectorised volatility indicators: ATR, Bollinger Bands, HV, Keltner Channels."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from indicators._base import _validate_warmup


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> Optional[np.ndarray]:
    """
    Average True Range (Wilder smoothing).

    Returns ``None`` if ``len(close) < period + 1``.
    """
    if _validate_warmup(close, period + 1, "ATR") is None:
        return None

    high_f = high.astype(np.float64)
    low_f = low.astype(np.float64)
    close_f = close.astype(np.float64)

    prev_close = np.empty_like(close_f)
    prev_close[0] = close_f[0]
    prev_close[1:] = close_f[:-1]

    tr = np.maximum(
        high_f - low_f,
        np.maximum(np.abs(high_f - prev_close), np.abs(low_f - prev_close)),
    )

    alpha = 1.0 / period
    out = np.full(len(close_f), np.nan, dtype=np.float64)
    out[period] = tr[1 : period + 1].mean()

    for i in range(period + 1, len(tr)):
        out[i] = out[i - 1] * (1 - alpha) + tr[i] * alpha

    return out


class BollingerBandsResult(NamedTuple):
    middle: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    bandwidth: np.ndarray
    percent_b: np.ndarray


def bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> Optional[BollingerBandsResult]:
    """
    Bollinger Bands (SMA ± num_std × rolling σ).

    Returns ``None`` if ``len(close) < period``.
    """
    if _validate_warmup(close, period, "BollingerBands") is None:
        return None

    close_f = close.astype(np.float64)
    n = len(close_f)

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(close_f, period)

    middle_vals = windows.mean(axis=-1)
    std_vals = windows.std(axis=-1, ddof=1)

    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    middle[period - 1:] = middle_vals
    upper[period - 1:] = middle_vals + num_std * std_vals
    lower[period - 1:] = middle_vals - num_std * std_vals

    bandwidth = np.where(middle != 0, (upper - lower) / middle, np.nan)
    percent_b = np.where(
        (upper - lower) != 0,
        (close_f - lower) / (upper - lower),
        np.nan,
    )

    return BollingerBandsResult(middle, upper, lower, bandwidth, percent_b)


def historical_volatility(
    close: np.ndarray,
    period: int = 21,
    annualisation_factor: float = 252.0,
) -> Optional[np.ndarray]:
    """
    Close-to-close historical (realised) volatility, annualised.

    Returns ``None`` if there are fewer than ``period + 1`` data points.
    """
    if _validate_warmup(close, period + 1, "HV") is None:
        return None

    log_returns = np.diff(np.log(close.astype(np.float64)))

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(log_returns, period)
    std_vals = windows.std(axis=-1, ddof=1)

    out = np.full(len(close), np.nan, dtype=np.float64)
    out[period:] = std_vals * np.sqrt(annualisation_factor)
    return out


class KeltnerChannelsResult(NamedTuple):
    middle: np.ndarray
    upper: np.ndarray
    lower: np.ndarray


def keltner_channels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 14,
    multiplier: float = 2.0,
) -> Optional[KeltnerChannelsResult]:
    """
    Keltner Channels (EMA ± multiplier × ATR).

    Returns ``None`` if either EMA or ATR cannot be computed.
    """
    from indicators.trend import ema as _ema

    ema_vals = _ema(close, ema_period)
    atr_vals = atr(high, low, close, atr_period)

    if ema_vals is None or atr_vals is None:
        return None

    upper = ema_vals + multiplier * atr_vals
    lower = ema_vals - multiplier * atr_vals
    return KeltnerChannelsResult(ema_vals, upper, lower)
