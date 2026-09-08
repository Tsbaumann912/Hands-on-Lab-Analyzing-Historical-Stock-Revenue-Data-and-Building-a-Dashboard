"""Simple event-style backtest for the copper ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from copper_ensemble.models import Bar, Config, Direction, Signal
from copper_ensemble.risk import RiskManager
from copper_ensemble.strategy import CopperEnsembleStrategy


@dataclass
class Fill:
    timestamp: object
    direction: Direction
    quantity: float
    price: float
    commission: float


@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    fills: List[Fill] = field(default_factory=list)
    signals: List[Signal] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def ulcer_index(equity: np.ndarray) -> float:
    """
    Peter Martin Ulcer Index from an equity curve.

    Percentage drawdowns from running peak → RMS. Returned in **percent** units
    (e.g. 5.0 means 5%).
    """
    if equity.size < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    # Percent drawdown series (0 at peaks, positive when underwater)
    dd_pct = 100.0 * (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.sqrt(np.mean(np.square(dd_pct))))


def ulcer_performance_index(
    equity: np.ndarray,
    *,
    periods_per_year: float = 252.0,
    risk_free_annual: float = 0.0,
) -> float:
    """
    Ulcer Performance Index = (ann. return % − R_f %) / Ulcer Index.

    Higher is better: reward for return per unit of drawdown pain.
    """
    if equity.size < 3 or equity[0] <= 0:
        return 0.0
    n = float(equity.size - 1)
    total = float(equity[-1] / equity[0])
    if total <= 0:
        return -10.0
    ann = total ** (periods_per_year / max(n, 1.0)) - 1.0
    ann_pct = 100.0 * (ann - risk_free_annual)
    ui = ulcer_index(equity)
    if ui < 1e-9:
        return 10.0 if ann_pct > 0 else (-10.0 if ann_pct < 0 else 0.0)
    return float(ann_pct / ui)


def calendar_year_returns(
    equity: np.ndarray,
    timestamps: Sequence[object],
) -> Dict[int, float]:
    """
    Calendar-year equity returns (year-end / prior year-end − 1).

    First year uses the first equity observation as the start mark.
    """
    if equity.size == 0 or len(timestamps) == 0:
        return {}
    n = min(int(equity.size), len(timestamps))
    import pandas as pd

    s = pd.Series(np.asarray(equity[:n], dtype=float), index=pd.to_datetime(list(timestamps[:n])))
    out: Dict[int, float] = {}
    prev: Optional[float] = None
    for year in sorted(int(y) for y in s.index.year.unique()):
        sy = s[s.index.year == year]
        if sy.empty:
            continue
        start = float(prev if prev is not None else sy.iloc[0])
        end = float(sy.iloc[-1])
        out[year] = end / start - 1.0 if start > 0 else 0.0
        prev = end
    return out


def _compute_metrics(equity: np.ndarray) -> Dict[str, float]:
    if equity.size < 3:
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "ulcer_index": 0.0,
            "upi": 0.0,
            "cagr": 0.0,
        }
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]
    mu = float(np.mean(rets)) if rets.size else 0.0
    sd = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
    sharpe = (mu / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity / np.maximum(peak, 1e-12)
    total_return = float(equity[-1] / equity[0] - 1.0)
    n = float(equity.size - 1)
    cagr = float((equity[-1] / equity[0]) ** (252.0 / max(n, 1.0)) - 1.0) if equity[0] > 0 else 0.0
    ui = ulcer_index(equity)
    upi = ulcer_performance_index(equity)
    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(np.max(dd)),
        "total_return": total_return,
        "end_equity": float(equity[-1]),
        "n_bars": float(equity.size),
        "ulcer_index": float(ui),
        "upi": float(upi),
        "cagr": float(cagr),
    }


class BacktestEngine:
    """Bar-by-bar mark-to-market backtest with costs."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, bars: List[Bar], sleeve_mask: Optional[Dict[str, bool]] = None) -> BacktestResult:
        strat = CopperEnsembleStrategy(self.config)
        strat.prepare(bars)

        # Optional ablation: zero out sleeves before regenerating
        if sleeve_mask is not None:
            for name, enabled in sleeve_mask.items():
                if not enabled and name in strat._forecasts:
                    strat._forecasts[name] = np.zeros_like(strat._forecasts[name])
            from copper_ensemble.blend import blend_forecasts
            from copper_ensemble.sizing import apply_forecast_buffer, forecast_to_contracts

            f_star, agreement, fdm, _ = blend_forecasts(strat._forecasts, self.config.ensemble)
            buffered = apply_forecast_buffer(f_star, self.config.ensemble.buffer_forecast)
            strat._f_star = buffered
            strat._agreement = agreement
            strat._fdm = fdm
            strat._contracts = forecast_to_contracts(
                buffered,
                strat._features["close"],
                strat._features["vol"],
                strat.equity,
                self.config,
            )

        risk = RiskManager(self.config)
        equity = self.config.portfolio.initial_cash
        position = 0.0  # signed contracts
        mult = self.config.contract.multiplier
        tick = self.config.contract.tick_size
        slip = self.config.contract.slippage_ticks * tick
        commission = self.config.contract.commission_per_contract

        eq_curve = np.zeros(len(bars), dtype=np.float64)
        pos_curve = np.zeros(len(bars), dtype=np.float64)
        fills: List[Fill] = []
        signals: List[Signal] = []
        prev_close = bars[0].close

        for i, bar in enumerate(bars):
            # MTM
            prior_equity = equity
            if i > 0:
                equity += position * mult * (bar.close - prev_close)
            risk.update_equity(equity, bar.timestamp, prior_equity=prior_equity)

            raw_sig = strat.signal_at(i)
            decision = risk.evaluate(raw_sig, bar.close)
            sig = decision.adjusted_signal
            signals.append(sig)

            target = 0.0
            if sig.direction == Direction.LONG:
                target = float(sig.suggested_size or 0.0)
            elif sig.direction == Direction.SHORT:
                target = -float(sig.suggested_size or 0.0)

            delta = target - position
            if abs(delta) >= 1.0:
                trade_px = bar.close + slip if delta > 0 else bar.close - slip
                cost = abs(delta) * commission
                # cash impact of paying commission; entry mark at close already in MTM path
                equity -= cost
                fills.append(
                    Fill(
                        timestamp=bar.timestamp,
                        direction=Direction.LONG if delta > 0 else Direction.SHORT,
                        quantity=abs(delta),
                        price=trade_px,
                        commission=cost,
                    )
                )
                # slippage cash vs mid
                equity -= abs(delta) * mult * abs(trade_px - bar.close)
                position = target

            eq_curve[i] = equity
            pos_curve[i] = position
            prev_close = bar.close
            strat.equity = equity

        rets = np.zeros(len(bars), dtype=np.float64)
        rets[1:] = np.diff(eq_curve) / np.maximum(eq_curve[:-1], 1e-12)
        metrics = _compute_metrics(eq_curve)
        metrics["n_fills"] = float(len(fills))
        yearly = calendar_year_returns(eq_curve, [b.timestamp for b in bars])
        if yearly:
            metrics["n_profitable_years"] = float(sum(1 for r in yearly.values() if r > 0))
            metrics["n_losing_years"] = float(sum(1 for r in yearly.values() if r < 0))
            metrics["min_year_return"] = float(min(yearly.values()))
            metrics["n_calendar_years"] = float(len(yearly))
        return BacktestResult(
            equity_curve=eq_curve,
            returns=rets,
            positions=pos_curve,
            fills=fills,
            signals=signals,
            metrics=metrics,
        )
