"""
Vectorised performance metric calculations.

All functions operate on NumPy arrays of returns or equity values.
No loops over individual return observations.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_metrics(
    equity_curve: np.ndarray,
    risk_free_rate: float = 0.05,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve:
        1-D array of portfolio equity values (one observation per period).
    risk_free_rate:
        Annual risk-free rate (default 5 %).
    periods_per_year:
        Number of periods in a trading year (252 for daily, 252*390 for minute).

    Returns
    -------
    Dict mapping metric name → float value.
    """
    if len(equity_curve) < 2:
        return {}

    eq = equity_curve.astype(np.float64)
    returns = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], np.nan)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {}

    # ── Basic return stats ─────────────────────────────────────────────────
    total_return = (eq[-1] - eq[0]) / eq[0]
    n_years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # ── Risk-adjusted return ───────────────────────────────────────────────
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_period

    sharpe = (
        excess_returns.mean() / (excess_returns.std(ddof=1) + 1e-9) * np.sqrt(periods_per_year)
    )

    downside_returns = returns[returns < rf_period]
    downside_std = downside_returns.std(ddof=1) if len(downside_returns) > 1 else 1e-9
    sortino = excess_returns.mean() / (downside_std + 1e-9) * np.sqrt(periods_per_year)

    # ── Drawdown ───────────────────────────────────────────────────────────
    cummax = np.maximum.accumulate(eq)
    drawdowns = np.where(cummax > 0, (eq - cummax) / cummax, 0.0)
    max_drawdown = drawdowns.min()

    calmar = cagr / (abs(max_drawdown) + 1e-9)

    # ── Win / loss statistics ──────────────────────────────────────────────
    win_mask = returns > 0
    loss_mask = returns < 0
    win_rate = win_mask.mean()
    avg_win = returns[win_mask].mean() if win_mask.any() else 0.0
    avg_loss = returns[loss_mask].mean() if loss_mask.any() else 0.0
    profit_factor = (
        abs(returns[win_mask].sum()) / (abs(returns[loss_mask].sum()) + 1e-9)
    )

    # ── Value at Risk (historical, 95 %) ──────────────────────────────────
    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

    return {
        "total_return": round(float(total_return), 6),
        "cagr": round(float(cagr), 6),
        "sharpe_ratio": round(float(sharpe), 4),
        "sortino_ratio": round(float(sortino), 4),
        "max_drawdown": round(float(max_drawdown), 6),
        "calmar_ratio": round(float(calmar), 4),
        "win_rate": round(float(win_rate), 4),
        "avg_win": round(float(avg_win), 6),
        "avg_loss": round(float(avg_loss), 6),
        "profit_factor": round(float(profit_factor), 4),
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
        "n_periods": int(len(returns)),
    }


def max_consecutive_losses(returns: np.ndarray) -> int:
    """Return the maximum streak of consecutive negative returns."""
    if len(returns) == 0:
        return 0
    loss = (returns < 0).astype(np.int8)
    # Vectorised streak computation
    streaks = np.diff(np.concatenate(([0], loss, [0])))
    starts = np.where(streaks == 1)[0]
    ends = np.where(streaks == -1)[0]
    if len(starts) == 0:
        return 0
    return int((ends - starts).max())
