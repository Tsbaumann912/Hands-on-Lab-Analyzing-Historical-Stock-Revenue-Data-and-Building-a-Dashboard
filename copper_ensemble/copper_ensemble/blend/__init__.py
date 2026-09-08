"""Forecast blending: weights, disagreement gate, FDM."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

from copper_ensemble.models import EnsembleConfig

SLEEVE_ORDER = ("tsmom", "carry", "basis_mom", "inventory", "fade")


def _weight_vector(weights: Mapping[str, float]) -> np.ndarray:
    w = np.array([float(weights[k]) for k in SLEEVE_ORDER], dtype=np.float64)
    s = float(np.sum(w))
    if s <= 0:
        raise ValueError("ensemble weights must sum to a positive number")
    return w / s


def agreement_ratio(sleeve_matrix: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """
    Per-row sign agreement in [0, 1].

    ``sleeve_matrix`` shape (n_bars, n_sleeves).
    """
    active = np.abs(sleeve_matrix) > eps
    signs = np.sign(sleeve_matrix)
    signed = np.where(active, signs, 0.0)
    n_active = active.sum(axis=1).astype(np.float64)
    agree = np.abs(signed.sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_active > 0, agree / n_active, 0.0)
    return ratio


def forecast_diversification_multiplier(
    sleeve_matrix: np.ndarray,
    weights: np.ndarray,
    lookback: int = 126,
    cap: float = 2.5,
) -> np.ndarray:
    """
    Expanding/rolling FDM = 1 / sqrt(w' C w) from sleeve forecast changes.

    Uses a rolling correlation of sleeve forecast *changes* as a simple proxy.
    """
    n, k = sleeve_matrix.shape
    out = np.ones(n, dtype=np.float64)
    # fill nan with 0 for corr estimation
    filled = np.nan_to_num(sleeve_matrix, nan=0.0)
    dF = np.diff(filled, axis=0, prepend=filled[:1] * 0.0)

    for i in range(n):
        start = max(0, i - lookback + 1)
        window = dF[start : i + 1]
        if window.shape[0] < max(20, k + 2):
            out[i] = 1.0
            continue
        # correlation matrix
        c = np.corrcoef(window, rowvar=False)
        if not np.all(np.isfinite(c)):
            out[i] = 1.0
            continue
        # numerical floor on diagonal
        c = np.nan_to_num(c, nan=0.0)
        c = 0.5 * (c + c.T)
        np.fill_diagonal(c, 1.0)
        port_var = float(weights @ c @ weights)
        if port_var <= 1e-12:
            out[i] = 1.0
        else:
            out[i] = min(cap, 1.0 / np.sqrt(port_var))
    return out


def blend_forecasts(
    forecasts: Dict[str, np.ndarray],
    cfg: EnsembleConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Blend sleeves → final capped forecast series.

    Robust rule: after disagreement flatten, rebuild the blend from
    **majority-sign sleeves only** so opposing forecasts do not cancel edge.

    Returns
    -------
    F_star, agreement, fdm, raw_blend
    """
    mat = np.column_stack([forecasts[k] for k in SLEEVE_ORDER])
    w = _weight_vector(cfg.weights)
    n, k = mat.shape
    mask = np.isnan(mat)
    mat0 = np.where(mask, 0.0, mat)

    a = agreement_ratio(mat0, eps=1.0)

    # Majority sign from active sleeves
    active = np.abs(mat0) > 1.0
    signed = np.where(active, np.sign(mat0), 0.0)
    maj = np.sign(signed.sum(axis=1))

    # Keep only sleeves aligned with majority; renormalise weights per row
    keep = active & (np.sign(mat0) == maj.reshape(-1, 1)) & (maj.reshape(-1, 1) != 0)
    w_row = np.where(keep, w.reshape(1, -1), 0.0)
    w_sum = w_row.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(w_sum > 0, (mat0 * w_row).sum(axis=1) / np.maximum(w_sum, 1e-12), 0.0)

    # Flatten on low agreement
    scaled = np.where(a <= cfg.agreement_min, 0.0, a * raw)
    scaled = np.clip(scaled, -cfg.forecast_cap, cfg.forecast_cap)

    fdm = forecast_diversification_multiplier(mat, w, cap=cfg.fdm_cap)
    f_star = np.clip(scaled * fdm, -cfg.forecast_cap, cfg.forecast_cap)
    # Preserve NaN warm-up where all sleeves NaN
    all_nan = mask.all(axis=1)
    f_star = np.where(all_nan, np.nan, f_star)
    raw = np.where(all_nan, np.nan, raw)
    return f_star, a, fdm, raw
