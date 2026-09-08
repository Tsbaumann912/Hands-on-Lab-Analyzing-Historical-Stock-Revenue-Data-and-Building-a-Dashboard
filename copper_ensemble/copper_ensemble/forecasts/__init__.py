"""Five copper forecast sleeves (A–E), vectorised."""

from __future__ import annotations

from typing import Dict

import numpy as np

from copper_ensemble.models import EnsembleConfig


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or n < window:
        return out
    c = np.cumsum(np.nan_to_num(x, nan=0.0))
    out[window - 1 :] = c[window - 1 :] - np.concatenate(([0.0], c[:-window]))
    # invalidate where any NaN in window
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
    # vectorised via cumsums of x and x^2
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
    """B: z-scored curve carry."""
    z = _zscore(carry, window=max(63, cfg.basis_mom_lookback))
    return _clip(10.0 * z, cfg.forecast_cap)


def forecast_basis_momentum(basis: np.ndarray, cfg: EnsembleConfig) -> np.ndarray:
    """C: z-scored change in log basis."""
    L = cfg.basis_mom_lookback
    n = basis.shape[0]
    bm = np.full(n, np.nan, dtype=np.float64)
    if n > L:
        bm[L:] = basis[L:] - basis[:-L]
    z = _zscore(bm, window=L)
    return _clip(10.0 * z, cfg.forecast_cap)


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
    china_pmi: np.ndarray,
    usd_ret_20d: np.ndarray,
    cfg: EnsembleConfig,
) -> np.ndarray:
    """E: fade stretched moves unless macro/inventory confirm; disabled in structural trends."""
    n = close.shape[0]
    z_px = _zscore(close, cfg.price_z_lookback)
    out = np.zeros(n, dtype=np.float64)

    # Macro confirmation
    bull_macro = (china_pmi >= 50.0) & (usd_ret_20d <= 0.0)
    bear_macro = (china_pmi < 50.0) & (usd_ret_20d > 0.0)

    fade_short = (z_px > cfg.fade_z) & (~bull_macro)
    fade_long = (z_px < -cfg.fade_z) & (~bear_macro)
    out = np.where(fade_short, -10.0, out)
    out = np.where(fade_long, 10.0, out)

    # Hard gate: structural TSMOM 252d aligned with inventory tightness → disable fade
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
        features["china_pmi"],
        features["usd_ret_20d"],
        cfg,
    )
    # Extra hard gate: when trend + inventory *forecasts* agree, never fade
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
        "basis_mom": forecast_basis_momentum(features["basis"], cfg),
        "inventory": inventory,
        "fade": fade,
    }
