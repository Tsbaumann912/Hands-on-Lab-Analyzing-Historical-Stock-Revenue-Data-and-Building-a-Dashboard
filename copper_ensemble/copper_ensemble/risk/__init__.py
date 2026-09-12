"""Independent risk layer for the copper ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd

from copper_ensemble.models import Config, Direction, Signal


@dataclass
class RiskDecision:
    approved: bool
    adjusted_signal: Signal
    reasons: List[str]
    quantity: float


class RiskManager:
    """Circuit breaker + size/leverage caps + optional calendar-year profit lock."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._halted: bool = False
        self._halt_reason: Optional[str] = None
        self._peak_equity: float = config.portfolio.initial_cash
        self._equity: float = config.portfolio.initial_cash
        self._bars_since_halt: int = 0
        # Yearly profit lock state
        self._year_start_equity: float = config.portfolio.initial_cash
        self._current_year: Optional[int] = None
        self._days_in_year: int = 0
        self._yearly_locked: bool = False

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def yearly_locked(self) -> bool:
        return self._yearly_locked

    def update_equity(
        self,
        equity: float,
        timestamp: Any = None,
        *,
        prior_equity: Optional[float] = None,
    ) -> None:
        self._equity = float(equity)
        self._peak_equity = max(self._peak_equity, self._equity)
        dd = 0.0 if self._peak_equity <= 0 else 1.0 - self._equity / self._peak_equity
        cap = self._cfg.risk.max_daily_drawdown_pct
        cooldown = int(self._cfg.risk.halt_cooldown_bars)

        if timestamp is not None and self._cfg.risk.yearly_profit_lock_enabled:
            self._update_yearly_lock(timestamp, prior_equity=prior_equity)

        if self._halted:
            self._bars_since_halt += 1
            # Resume when DD recovers, OR after a timed cooldown (flat books never
            # recover vs peak, which previously killed multi-year backtests).
            if dd <= 0.5 * cap or self._bars_since_halt >= max(cooldown, 1):
                self._halted = False
                self._halt_reason = None
                self._bars_since_halt = 0
                # Reset high-water mark so the breaker can re-arm cleanly.
                self._peak_equity = self._equity
            return

        if dd >= cap and self._cfg.risk.halt_on_breach:
            self._halted = True
            self._halt_reason = f"drawdown {dd:.2%} breached cap {cap:.2%}"
            self._bars_since_halt = 0

    def _update_yearly_lock(
        self,
        timestamp: Any,
        *,
        prior_equity: Optional[float] = None,
    ) -> None:
        ts = pd.Timestamp(timestamp)
        year = int(ts.year)
        month = int(ts.month)
        if self._current_year is None:
            self._current_year = year
            # Year-start mark is equity before today's MTM when available.
            self._year_start_equity = float(
                prior_equity if prior_equity is not None else self._equity
            )
            self._days_in_year = 0
            self._yearly_locked = False
        elif year != self._current_year:
            self._current_year = year
            self._year_start_equity = float(
                prior_equity if prior_equity is not None else self._equity
            )
            self._days_in_year = 0
            self._yearly_locked = False

        self._days_in_year += 1
        if self._yearly_locked:
            return

        ytd = (
            self._equity / self._year_start_equity - 1.0
            if self._year_start_equity > 0
            else 0.0
        )
        lock_pct = float(self._cfg.risk.yearly_profit_lock_pct)
        min_days = int(self._cfg.risk.yearly_profit_lock_min_days)
        if self._days_in_year >= max(min_days, 1) and ytd >= lock_pct:
            self._yearly_locked = True
            return
        if self._cfg.risk.yearly_nov_protect and month >= 11 and ytd > 0.0:
            self._yearly_locked = True

    def resume(self) -> None:
        self._halted = False
        self._halt_reason = None
        self._bars_since_halt = 0

    def evaluate(self, signal: Signal, price: float) -> RiskDecision:
        reasons: List[str] = []
        if self._halted:
            flat = Signal(
                symbol=signal.symbol,
                direction=Direction.FLAT,
                strength=0.0,
                timestamp=signal.timestamp,
                strategy_name=signal.strategy_name,
                metadata={**signal.metadata, "halt": self._halt_reason},
            )
            return RiskDecision(False, flat, [self._halt_reason or "halted"], 0.0)

        if self._yearly_locked:
            flat = Signal(
                symbol=signal.symbol,
                direction=Direction.FLAT,
                strength=0.0,
                timestamp=signal.timestamp,
                strategy_name=signal.strategy_name,
                suggested_size=0.0,
                metadata={**signal.metadata, "yearly_lock": True},
            )
            return RiskDecision(False, flat, ["yearly_profit_lock"], 0.0)

        qty = abs(float(signal.suggested_size or 0.0))
        mult = self._cfg.contract.multiplier
        notional = qty * price * mult
        if self._equity > 0 and notional / self._equity > self._cfg.risk.max_position_size_pct:
            qty = (self._cfg.risk.max_position_size_pct * self._equity) / (price * mult)
            reasons.append("resized to max_position_size_pct")
        if qty > self._cfg.risk.max_contracts:
            qty = float(self._cfg.risk.max_contracts)
            reasons.append("capped at max_contracts")

        qty = float(int(round(qty)))
        if signal.direction == Direction.FLAT or qty <= 0:
            adj = Signal(
                symbol=signal.symbol,
                direction=Direction.FLAT,
                strength=0.0,
                timestamp=signal.timestamp,
                strategy_name=signal.strategy_name,
                suggested_size=0.0,
                metadata=signal.metadata,
            )
            return RiskDecision(True, adj, reasons, 0.0)

        adj = Signal(
            symbol=signal.symbol,
            direction=signal.direction,
            strength=signal.strength,
            timestamp=signal.timestamp,
            strategy_name=signal.strategy_name,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            suggested_size=qty,
            metadata=signal.metadata,
        )
        return RiskDecision(True, adj, reasons, qty)
