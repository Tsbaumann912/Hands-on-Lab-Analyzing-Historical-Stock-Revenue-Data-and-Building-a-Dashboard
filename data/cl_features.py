"""
Vectorised feature builders for Light Crude Oil (CL / WTI) strategies.

Carry prefers a true front/back curve. When only continuous front prices are
available, ``implied_back_month`` builds a research proxy (documented in
``docs/CL_RESEARCH.md``) — not a substitute for production multi-expiry data.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def annualized_carry(
    front: np.ndarray,
    back: np.ndarray,
    months_apart: int,
) -> np.ndarray:
    """
    AQR-style carry from two futures prices.

    ``Carry ≈ (F1 - Fn) / Fn * (12 / months_apart)`` annualised.
    Positive → backwardation (scarce inventory proxy).
    """
    front_f = front.astype(np.float64)
    back_f = back.astype(np.float64)
    months = max(int(months_apart), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (front_f - back_f) / back_f
    out = raw * (12.0 / float(months))
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def spread_carry(front: np.ndarray, back: np.ndarray) -> np.ndarray:
    """Simple dollar spread carry ``F1 - Fn`` (Rebellion F1-Fn style)."""
    return front.astype(np.float64) - back.astype(np.float64)


def implied_back_month(
    front: np.ndarray,
    back_month: int,
    basis_lookback: int,
) -> np.ndarray:
    """
    Research proxy for Fn when the true curve is unavailable.

    Uses a rolling mean of log-returns as an implied basis (copper_ensemble
    Yahoo path). ``next ≈ front * (1 - basis * back_month/12)``.
    """
    front_f = front.astype(np.float64)
    n = front_f.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2 or basis_lookback < 2:
        return out

    log_px = np.log(np.clip(front_f, 1e-12, None))
    rets = np.empty(n, dtype=np.float64)
    rets[0] = np.nan
    rets[1:] = log_px[1:] - log_px[:-1]

    # Rolling mean of returns via cumulative sum (vectorised).
    valid = np.where(np.isfinite(rets), rets, 0.0)
    csum = np.cumsum(valid)
    count = np.cumsum(np.isfinite(rets).astype(np.float64))
    basis = np.full(n, np.nan, dtype=np.float64)
    L = int(basis_lookback)
    idx = np.arange(n)
    start = idx - L + 1
    mask = start >= 0
    sum_win = np.empty(n, dtype=np.float64)
    cnt_win = np.empty(n, dtype=np.float64)
    sum_win[mask] = csum[idx[mask]] - np.where(
        start[mask] > 0, csum[start[mask] - 1], 0.0
    )
    cnt_win[mask] = count[idx[mask]] - np.where(
        start[mask] > 0, count[start[mask] - 1], 0.0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        basis[mask] = sum_win[mask] / np.maximum(cnt_win[mask], 1.0)

    scale = float(max(back_month, 1)) / 12.0
    out = front_f * (1.0 - basis * scale)
    out = np.where(np.isfinite(out) & (out > 0.0), out, np.nan)
    return out


def resolve_curve(
    front: np.ndarray,
    back: Optional[np.ndarray],
    back_month: int,
    basis_lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(front, back)``, synthesising back when missing."""
    front_f = front.astype(np.float64)
    if back is not None and len(back) == len(front_f) and np.isfinite(back).any():
        return front_f, back.astype(np.float64)
    return front_f, implied_back_month(front_f, back_month, basis_lookback)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean; NaN until warm-up."""
    x_f = x.astype(np.float64)
    n = x_f.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1 or n < window:
        return out
    # Replace NaNs with 0 for cumsum, track finite counts.
    finite = np.isfinite(x_f)
    filled = np.where(finite, x_f, 0.0)
    csum = np.cumsum(filled)
    ccnt = np.cumsum(finite.astype(np.float64))
    idx = np.arange(n)
    start = idx - window + 1
    ok = start >= 0
    s = np.empty(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    s[ok] = csum[idx[ok]] - np.where(start[ok] > 0, csum[start[ok] - 1], 0.0)
    c[ok] = ccnt[idx[ok]] - np.where(start[ok] > 0, ccnt[start[ok] - 1], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = np.where(c[ok] >= window * 0.8, s[ok] / np.maximum(c[ok], 1.0), np.nan)
    return out


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing sample std; NaN until warm-up."""
    x_f = x.astype(np.float64)
    n = x_f.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2 or n < window:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    import warnings

    windows = sliding_window_view(x_f, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stds = np.nanstd(windows, axis=-1, ddof=1)
    finite_counts = np.sum(np.isfinite(windows), axis=-1)
    stds = np.where(finite_counts >= 2, stds, np.nan)
    out[window - 1 :] = stds
    return out


def carry_momentum_z(
    carry: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Carry-momentum z-score: ``(C - MA_N(C)) / σ(C)``.

    Positive z → carry rising (tightening / toward backwardation).
    """
    mu = rolling_mean(carry, lookback)
    sd = rolling_std(carry, lookback)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (carry - mu) / sd
    return np.where(np.isfinite(z), z, np.nan)


def ewma_volatility(returns: np.ndarray, com: float) -> np.ndarray:
    """EWMA volatility of returns (pandas ewm — vectorised filter)."""
    import pandas as pd

    r = returns.astype(np.float64)
    if r.shape[0] == 0 or com <= 0:
        return np.full(r.shape[0], np.nan, dtype=np.float64)
    var = pd.Series(r * r).ewm(com=float(com), min_periods=max(2, int(com) // 4)).mean()
    out = np.sqrt(np.clip(var.to_numpy(dtype=np.float64), 0.0, None))
    return out


def reaction_function(z: np.ndarray, b: float = 1.0) -> np.ndarray:
    """
    Nonlinear reaction: ``R(z) = z * exp(0.5 * b * (1 - z^2))``.

    Extreme |z| shrinks size (crowding / mean-reversion risk).
    """
    z_f = z.astype(np.float64)
    b_f = float(b)
    with np.errstate(over="ignore", invalid="ignore"):
        r = z_f * np.exp(0.5 * b_f * (1.0 - z_f * z_f))
    return np.where(np.isfinite(r), r, np.nan)


def multi_horizon_tsmom(
    closes: np.ndarray,
    horizons: Tuple[int, ...],
    vol: np.ndarray,
) -> np.ndarray:
    """
    Average of vol-scaled returns over multiple lookbacks.

    ``z_h = (close_t / close_{t-h} - 1) / σ_t`` then mean across horizons.
    """
    c = closes.astype(np.float64)
    n = c.shape[0]
    acc = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.float64)
    for h in horizons:
        if h < 1 or h >= n:
            continue
        past = np.full(n, np.nan, dtype=np.float64)
        past[h:] = c[:-h]
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (c / past) - 1.0
            z = raw / vol
        ok = np.isfinite(z)
        acc = np.where(ok, acc + z, acc)
        count = np.where(ok, count + 1.0, count)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = acc / count
    return np.where(count > 0, out, np.nan)


def inventory_proxy_from_price(close: np.ndarray, sma_window: int) -> np.ndarray:
    """
    Price-implied inventory proxy when EIA stocks are unavailable.

    High price z → scarce proxy (low inventory). Units are arbitrary.
    """
    mu = rolling_mean(close, sma_window)
    sd = rolling_std(close, sma_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (close - mu) / sd
    z = np.where(np.isfinite(z), z, 0.0)
    return 100_000.0 - 10_000.0 * z


def inventory_surprise(
    inventory: np.ndarray,
    expect_window: int,
) -> np.ndarray:
    """
    ``s_t = ΔI_t - E[ΔI_t]`` with E from trailing mean of changes.

    Positive surprise (build) → bearish for nearby crude.
    """
    inv = inventory.astype(np.float64)
    n = inv.shape[0]
    delta = np.full(n, np.nan, dtype=np.float64)
    delta[1:] = inv[1:] - inv[:-1]
    expected = rolling_mean(delta, expect_window)
    return delta - expected


def clip01(x: float) -> float:
    """Clamp a scalar into [0, 1]."""
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))
