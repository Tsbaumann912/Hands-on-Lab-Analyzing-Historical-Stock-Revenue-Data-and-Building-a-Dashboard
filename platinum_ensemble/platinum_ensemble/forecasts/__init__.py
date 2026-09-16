"""Five platinum forecast sleeves (A–E), vectorised."""

from __future__ import annotations

from typing import Dict

import numpy as np

from platinum_ensemble.models import EnsembleConfig


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or n < window:
        return out
    c = np.cumsum(np.nan_to_num(x, nan=0.0))
    out[window - 1 :] = c[window - 1 :] - np.concatenate(([0.0], c[:-window]))
    nan_c = np.cumsum(np.isnan(x).astype(np.float64))
    nan_win = np.full(n, np.nan)
    nan_win[window - 1 :] = nan_c[window - 1 :] - np.concatenate(([0.0], nan_c[:-window]))
    out[nan_win > 0] = np.nan
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    s = _rolling_sum(x, window)
    return s / float(window)


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    x0 = np.nan_to_num(x, nan=0.0)
    c1 = np.cumsum(x0)
    c2 = np.cumsum(x0 * x0)
    sum1 = c1[window - 1 :] - np.concatenate(([0.0], c1[:-window]))
    sum2 = c2[window - 1 :] - np.concatenate(([0.0], c2[:-window]))
    mean = sum1 / float(window)
    var = np.maximum(sum2 / float(window) - mean * mean, 0.0)
    out[window - 1 :] = np.sqrt(var)
    return out


def _zscore(x: np.ndarray, window: int) -> np.ndarray:
    mu = _rolling_mean(x, window)
    sd = _rolling_std(x, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - mu) / sd
    z = np.where(sd < 1e-12, np.nan, z)
    return z


def _clip(x: np.ndarray, cap: float) -> np.ndarray:
    return np.clip(x, -cap, cap)


def _rolling_beta(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """Rolling OLS slope of y on x (vectorised via rolling moments)."""
    n = y.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    y0 = np.nan_to_num(y, nan=0.0)
    x0 = np.nan_to_num(x, nan=0.0)
    cy = np.cumsum(y0)
    cx = np.cumsum(x0)
    cxy = np.cumsum(y0 * x0)
    cxx = np.cumsum(x0 * x0)
    sum_y = cy[window - 1 :] - np.concatenate(([0.0], cy[:-window]))
    sum_x = cx[window - 1 :] - np.concatenate(([0.0], cx[:-window]))
    sum_xy = cxy[window - 1 :] - np.concatenate(([0.0], cxy[:-window]))
    sum_xx = cxx[window - 1 :] - np.concatenate(([0.0], cxx[:-window]))
    w = float(window)
    cov = sum_xy / w - (sum_x / w) * (sum_y / w)
    var_x = sum_xx / w - (sum_x / w) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(var_x > 1e-12, cov / var_x, np.nan)
    out[window - 1 :] = beta
    return out


def forecast_tsmom(returns: np.ndarray, cfg: EnsembleConfig) -> np.ndarray:
    """A: equal-weight multi-horizon sign of excess returns → forecast in [-20, 20]."""
    n = returns.shape[0]
    acc = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=np.float64)
    for h in cfg.horizons_days:
        s = _rolling_sum(returns, h)
        sig = np.sign(s)
        m = ~np.isnan(s)
        acc = np.where(m, acc + sig, acc)
        valid = np.where(m, valid + 1.0, valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(valid > 0, acc / valid, np.nan)
    return _clip(10.0 * avg, cfg.forecast_cap)


def forecast_carry(carry: np.ndarray, cfg: EnsembleConfig) -> np.ndarray:
    """B: z-scored curve carry (roll yield)."""
    z = _zscore(carry, window=max(63, cfg.rv_lookback))
    return _clip(10.0 * z, cfg.forecast_cap)


def forecast_pl_gc_rv(
    pl_close: np.ndarray,
    gold_close: np.ndarray,
    cfg: EnsembleConfig,
) -> np.ndarray:
    """
    C: Gold–Platinum relative value.

    Spread s = ln(PL) - β ln(GC); forecast = -10 * z(s) when |z| > entry
    (long PL when cheap vs gold).
    """
    L = cfg.rv_lookback
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_pl = np.log(np.clip(pl_close, 1e-12, None))
        ln_gc = np.log(np.clip(gold_close, 1e-12, None))
    beta = _rolling_beta(ln_pl, ln_gc, L)
    spread = ln_pl - beta * ln_gc
    z = _zscore(spread, L)
    # Convergent: fade rich PL / buy cheap PL
    raw = -10.0 * z
    # Soft gate: only trade when |z| exceeds entry; otherwise damp toward 0
    strength = np.clip(np.abs(z) / max(cfg.rv_entry_z, 1e-6), 0.0, 1.0)
    out = raw * strength
    out = np.where(np.isnan(z) | np.isnan(beta), np.nan, out)
    return _clip(out, cfg.forecast_cap)


def forecast_inventory_trend(
    returns: np.ndarray,
    inventory: np.ndarray,
    cfg: EnsembleConfig,
) -> np.ndarray:
    """D: inventory draw/build confirming 63d price trend."""
    n = returns.shape[0]
    out = np.zeros(n, dtype=np.float64)
    r63 = _rolling_sum(returns, 63)
    d = cfg.inventory_delta_days
    d_inv = np.full(n, np.nan, dtype=np.float64)
    if n > d:
        d_inv[d:] = inventory[d:] - inventory[:-d]
    long_m = (r63 > 0) & (d_inv < 0)
    short_m = (r63 < 0) & (d_inv > 0)
    out = np.where(long_m, 10.0, out)
    out = np.where(short_m, -10.0, out)
    out = np.where(np.isnan(r63) | np.isnan(d_inv), np.nan, out)
    return _clip(out, cfg.forecast_cap)


def forecast_macro_fade(
    close: np.ndarray,
    returns: np.ndarray,
    inventory: np.ndarray,
    gold_ret_20d: np.ndarray,
    usd_ret_20d: np.ndarray,
    cfg: EnsembleConfig,
) -> np.ndarray:
    """
    E: fade stretched PL moves unless gold/USD (macro) confirm;
    disabled in structural inventory-aligned trends.
    """
    z_px = _zscore(close, cfg.price_z_lookback)
    out = np.zeros(close.shape[0], dtype=np.float64)

    # Macro confirmation for PGMs: gold bid + USD weakness = bullish spillover
    bull_macro = (gold_ret_20d > 0.0) & (usd_ret_20d <= 0.0)
    bear_macro = (gold_ret_20d < 0.0) & (usd_ret_20d > 0.0)

    fade_short = (z_px > cfg.fade_z) & (~bull_macro)
    fade_long = (z_px < -cfg.fade_z) & (~bear_macro)
    out = np.where(fade_short, -10.0, out)
    out = np.where(fade_long, 10.0, out)

    r252 = _rolling_sum(returns, 252)
    inv_mu = _rolling_mean(inventory, cfg.inventory_sma)
    inv_sd = _rolling_std(inventory, cfg.inventory_sma)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_inv = -(inventory - inv_mu) / inv_sd  # high => tightness
    structural = (
        (~np.isnan(r252))
        & (~np.isnan(z_inv))
        & (np.sign(r252) == np.sign(z_inv))
        & (np.sign(r252) != 0)
        & (np.abs(z_inv) > 0.5)
    )
    out = np.where(structural, 0.0, out)
    out = np.where(np.isnan(z_px), np.nan, out)
    return _clip(out, cfg.forecast_cap)


def compute_all_forecasts(features: Dict[str, np.ndarray], cfg: EnsembleConfig) -> Dict[str, np.ndarray]:
    """Run all five sleeves; keys match config weight names."""
    tsmom = forecast_tsmom(features["returns"], cfg)
    inventory = forecast_inventory_trend(features["returns"], features["inventory"], cfg)
    fade = forecast_macro_fade(
        features["close"],
        features["returns"],
        features["inventory"],
        features["gold_ret_20d"],
        features["usd_ret_20d"],
        cfg,
    )
    aligned = (
        (np.sign(tsmom) == np.sign(inventory))
        & (np.abs(tsmom) > 5.0)
        & (np.abs(inventory) > 5.0)
        & np.isfinite(tsmom)
        & np.isfinite(inventory)
    )
    fade = np.where(aligned, 0.0, fade)
    return {
        "tsmom": tsmom,
        "carry": forecast_carry(features["carry"], cfg),
        "pl_gc_rv": forecast_pl_gc_rv(features["close"], features["gold_close"], cfg),
        "inventory": inventory,
        "fade": fade,
    }
