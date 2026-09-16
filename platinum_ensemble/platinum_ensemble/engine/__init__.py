"""Simple event-style backtest for the platinum ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from platinum_ensemble.models import Bar, Config, Direction, Signal
from platinum_ensemble.risk import RiskManager
from platinum_ensemble.strategy import PlatinumEnsembleStrategy


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


def _compute_metrics(equity: np.ndarray) -> Dict[str, float]:
    """Sharpe, Max DD, CAGR, Ulcer Index, and Ulcer Performance Index."""
    empty = {
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "total_return": 0.0,
        "cagr": 0.0,
        "ulcer_index": 0.0,
        "upi": 0.0,
        "end_equity": float(equity[-1]) if equity.size else 0.0,
        "n_bars": float(equity.size),
    }
    if equity.size < 3:
        return empty

    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    rets = rets[np.isfinite(rets)]
    mu = float(np.mean(rets)) if rets.size else 0.0
    sd = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
    sharpe = (mu / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity / np.maximum(peak, 1e-12)
    max_dd = float(np.max(dd))

    n = float(equity.size)
    total_return = float(equity[-1] / equity[0] - 1.0)
    # CAGR annualised on trading-day count
    if equity[0] > 0 and n > 1 and equity[-1] > 0:
        cagr = float((equity[-1] / equity[0]) ** (252.0 / (n - 1.0)) - 1.0)
    elif equity[0] > 0 and n > 1 and equity[-1] <= 0:
        cagr = -1.0
    else:
        cagr = 0.0

    # Ulcer Index as fraction (same units as max_drawdown)
    ulcer = float(np.sqrt(np.mean(dd * dd)))
    if ulcer > 1e-12:
        upi = float(cagr / ulcer)
    else:
        upi = float(cagr / 1e-12) if cagr > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "total_return": total_return,
        "cagr": cagr,
        "ulcer_index": ulcer,
        "upi": upi,
        "end_equity": float(equity[-1]),
        "n_bars": n,
    }


class BacktestEngine:
    """Bar-by-bar mark-to-market backtest with costs."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, bars: List[Bar], sleeve_mask: Optional[Dict[str, bool]] = None) -> BacktestResult:
        strat = PlatinumEnsembleStrategy(self.config)
        strat.prepare(bars)

        # Optional ablation: zero out sleeves before regenerating
        if sleeve_mask is not None:
            for name, enabled in sleeve_mask.items():
                if not enabled and name in strat._forecasts:
                    strat._forecasts[name] = np.zeros_like(strat._forecasts[name])
            from platinum_ensemble.blend import blend_forecasts
            from platinum_ensemble.sizing import apply_forecast_buffer, forecast_to_contracts

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
            if i > 0:
                equity += position * mult * (bar.close - prev_close)
            risk.update_equity(equity)

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
        return BacktestResult(
            equity_curve=eq_curve,
            returns=rets,
            positions=pos_curve,
            fills=fills,
            signals=signals,
            metrics=metrics,
        )
