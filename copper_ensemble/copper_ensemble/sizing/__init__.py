"""Volatility targeting and futures contract sizing."""

from __future__ import annotations

import numpy as np

from copper_ensemble.models import Config


def forecast_to_contracts(
    forecast: np.ndarray,
    price: np.ndarray,
    vol_annual: np.ndarray,
    equity: float,
    config: Config,
) -> np.ndarray:
    """
    Map forecast F* ∈ [-20, 20] to signed contract counts.

    Notional_t = (F*/10) * (σ_target / σ_t) * Equity
    N = round(Notional / (P * multiplier))
    """
    cfg = config.ensemble
    mult = config.contract.multiplier
    strength = forecast / 10.0
    # Avoid div by zero
    vol = np.where((vol_annual is None) | (vol_annual <= 1e-8), np.nan, vol_annual)
    with np.errstate(divide="ignore", invalid="ignore"):
        # kelly_fraction=0.5 ⇒ use full vol_target; 0.25 ⇒ half-Kelly exposure
        scale = cfg.kelly_fraction / 0.5
        notional = strength * (cfg.vol_target_annual / vol) * equity * scale
        contracts = notional / (price * mult)

    # hard caps
    max_n = float(config.risk.max_contracts)
    max_by_pct = (config.risk.max_position_size_pct * equity) / (price * mult)
    max_by_lev = (config.risk.max_leverage * equity) / (price * mult)
    cap = np.minimum(max_n, np.minimum(np.abs(max_by_pct), np.abs(max_by_lev)))
    contracts = np.clip(contracts, -cap, cap)
    contracts = np.where(np.isnan(contracts) | np.isnan(forecast), 0.0, contracts)
    # Round toward nearest contract; keep a 1-lot minimum when |N|≥0.5 and cap allows
    rounded = np.rint(contracts)
    needs_min = (np.abs(contracts) >= 0.5) & (np.abs(rounded) < 1.0) & (cap >= 1.0)
    rounded = np.where(needs_min, np.sign(contracts), rounded)
    return rounded.astype(np.float64)


def apply_forecast_buffer(
    forecast: np.ndarray,
    buffer: float,
) -> np.ndarray:
    """Suppress tiny forecast flips: keep previous forecast until change > buffer."""
    n = forecast.shape[0]
    out = np.zeros(n, dtype=np.float64)
    last = 0.0
    for i in range(n):
        f = forecast[i]
        if np.isnan(f):
            out[i] = 0.0
            continue
        if abs(f - last) >= buffer or (last == 0.0 and abs(f) >= buffer):
            last = float(f)
        out[i] = last
    return out
