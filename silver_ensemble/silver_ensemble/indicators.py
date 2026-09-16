"""Standalone vectorised indicators for the silver ensemble (no QuantTerminal imports)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average; leading values NaN until warm-up."""
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n < period:
        return out
    s = pd.Series(np.asarray(close, dtype=np.float64))
    return s.rolling(window=period, min_periods=period).mean().to_numpy(dtype=np.float64)


def wilder_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI in [0, 100]; NaN until warm-up. Returns NaN array if too short."""
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n < period + 1:
        return out
    c = np.asarray(close, dtype=np.float64)
    delta = np.diff(c)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    # Wilder smoothing via EWM (alpha = 1/period, adjust=False)
    avg_gain = pd.Series(gains).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(losses).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain.to_numpy() / np.maximum(avg_loss.to_numpy(), 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    out[1:] = rsi
    # Invalidate where either average is still NaN
    out[1:] = np.where(np.isfinite(avg_gain.to_numpy()) & np.isfinite(avg_loss.to_numpy()), out[1:], np.nan)
    return out


def stochastic_rsi(
    close: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
) -> np.ndarray:
    """
    Stochastic RSI (%K) in [0, 1].

    StochRSI = (RSI - min(RSI, L)) / (max(RSI, L) - min(RSI, L)), then SMA-smoothed.
    """
    rsi = wilder_rsi(close, rsi_period)
    n = rsi.shape[0]
    raw = np.full(n, np.nan, dtype=np.float64)
    if stoch_period <= 0 or n < stoch_period:
        return raw
    s = pd.Series(rsi)
    roll_min = s.rolling(stoch_period, min_periods=stoch_period).min()
    roll_max = s.rolling(stoch_period, min_periods=stoch_period).max()
    denom = (roll_max - roll_min).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (rsi - roll_min.to_numpy()) / np.where(np.abs(denom) < 1e-12, np.nan, denom)
    if smooth_k > 1:
        raw = pd.Series(raw).rolling(smooth_k, min_periods=smooth_k).mean().to_numpy(dtype=np.float64)
    return raw.astype(np.float64)
