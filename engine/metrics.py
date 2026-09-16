"""
Vectorised performance metric calculations.

All functions operate on NumPy arrays of returns or equity values.
No loops over individual return observations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Sequence, Union

import numpy as np


TimestampLike = Union[datetime, np.datetime64]


def _year_of(ts: TimestampLike) -> int:
    if isinstance(ts, np.datetime64):
        return int(str(ts.astype("datetime64[Y]")))
    return int(ts.year)


def calendar_year_returns(
    equity_curve: np.ndarray,
    timestamps: Sequence[TimestampLike],
    *,
    min_bars: int = 2,
) -> Dict[int, float]:
    """
    Calendar-year total returns from a stamped equity path.

    For each calendar year with at least ``min_bars`` equity points, return is
    ``(year_end - year_start_basis) / year_start_basis`` where the basis is the
    equity immediately before the year's first point when available (true
    calendar P&L), otherwise the first point in the year.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2 or len(timestamps) != len(eq):
        return {}

    years = np.asarray([_year_of(t) for t in timestamps], dtype=np.int32)
    out: Dict[int, float] = {}
    for y in np.unique(years):
        idx = np.flatnonzero(years == y)
        if len(idx) < min_bars:
            continue
        start_i = int(idx[0])
        end_i = int(idx[-1])
        start_eq = float(eq[start_i - 1]) if start_i > 0 else float(eq[start_i])
        end_eq = float(eq[end_i])
        if not np.isfinite(start_eq) or start_eq == 0.0:
            continue
        out[int(y)] = float((end_eq - start_eq) / start_eq)
    return out


def all_calendar_years_profitable(
    equity_curve: np.ndarray,
    timestamps: Sequence[TimestampLike],
    *,
    min_bars: int = 2,
    min_year_return: float = 0.0,
) -> tuple[bool, Dict[int, float], list[str]]:
    """
    Return ``(ok, year_returns, failures)``.

    ``ok`` is True iff every evaluated calendar year has return ``> min_year_return``.
    """
    year_rets = calendar_year_returns(
        equity_curve, timestamps, min_bars=min_bars
    )
    failures: list[str] = []
    if not year_rets:
        failures.append("no_calendar_years_evaluable")
        return False, year_rets, failures
    for y, r in sorted(year_rets.items()):
        # Strict profitability: require strictly positive calendar-year return.
        if not np.isfinite(r) or r <= min_year_return:
            failures.append(f"year_{y}_return={r:.6f} <= {min_year_return}")
    return len(failures) == 0, year_rets, failures


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
    if n_years > 0 and (1.0 + total_return) > 0.0:
        cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0
    elif n_years > 0 and total_return <= -1.0:
        cagr = -1.0
    else:
        cagr = 0.0

    # ── Risk-adjusted return ───────────────────────────────────────────────
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_period

    excess_std_raw = excess_returns.std(ddof=1) if len(excess_returns) > 1 else 0.0
    excess_std = 0.0 if (np.isnan(excess_std_raw) or np.isinf(excess_std_raw)) else float(excess_std_raw)
    sharpe = (
        excess_returns.mean() / (excess_std + 1e-9) * np.sqrt(periods_per_year)
    )

    downside_returns = returns[returns < rf_period]
    with np.errstate(invalid="ignore", divide="ignore"):
        downside_std_raw = downside_returns.std(ddof=1) if len(downside_returns) > 1 else np.nan
    downside_std = 0.0 if (np.isnan(downside_std_raw) or np.isinf(downside_std_raw)) else float(downside_std_raw)
    sortino = excess_returns.mean() / (downside_std + 1e-9) * np.sqrt(periods_per_year)

    # ── Drawdown ───────────────────────────────────────────────────────────
    cummax = np.maximum.accumulate(eq)
    drawdowns = np.where(cummax > 0, (eq - cummax) / cummax, 0.0)
    max_drawdown = drawdowns.min()

    # Ulcer Index: RMS of percentage drawdowns (Martin & McCann).
    # Drawdowns are ≤ 0; square removes the sign.
    ulcer_index = float(np.sqrt(np.mean(drawdowns * drawdowns)))
    ulcer_performance_index = float(cagr / (ulcer_index + 1e-9))

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
        "ulcer_index": round(ulcer_index, 6),
        "ulcer_performance_index": round(ulcer_performance_index, 4),
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
