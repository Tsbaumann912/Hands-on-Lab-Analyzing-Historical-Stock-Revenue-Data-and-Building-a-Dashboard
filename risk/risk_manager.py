"""
RiskManager — the administrative firewall between strategy signals and execution.

Enforces hard limits:
  1. Drawdown Circuit Breaker — halts all trading if equity falls below a
     configurable percentage of the peak equity.
  2. Position Sizing Limit — caps the notional of any single trade to a
     maximum percentage of current equity.
  3. Maximum Open Positions — rejects new entries when the position count
     exceeds the configured limit.
  4. Stop-Loss / Take-Profit Application — applies default ATR-based targets
     to any signal that does not include its own levels.

All logic is pure-function; no network I/O occurs here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from core.config import Config
from core.enums import Direction
from core.models import Signal
from portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)


class ViolationType(str, Enum):
    DRAWDOWN_BREACH = "DRAWDOWN_BREACH"
    POSITION_SIZE_EXCEEDED = "POSITION_SIZE_EXCEEDED"
    MAX_POSITIONS_EXCEEDED = "MAX_POSITIONS_EXCEEDED"
    LEVERAGE_EXCEEDED = "LEVERAGE_EXCEEDED"
    TRADING_HALTED = "TRADING_HALTED"


@dataclass
class RiskViolation:
    violation_type: ViolationType
    message: str
    original_signal: Signal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RiskDecision:
    """
    Output of the RiskManager.evaluate() call.

    ``approved`` indicates whether the signal may proceed.
    ``adjusted_signal`` carries any modifications (e.g., reduced size, added stops).
    ``violations`` lists any soft/hard rule breaches detected.
    """

    approved: bool
    adjusted_signal: Signal
    violations: List[RiskViolation] = field(default_factory=list)
    suggested_quantity: float = 0.0


class RiskManager:
    """
    Intercepts strategy signals and enforces capital-defence rules.

    Parameters
    ----------
    config:
        Terminal-wide configuration; risk limits come from ``config.risk``.
    portfolio:
        Live portfolio state used to read current equity, drawdown, and
        open positions.
    """

    def __init__(self, config: Config, portfolio: Portfolio) -> None:
        self._cfg = config.risk
        self._portfolio_cfg = config.portfolio
        self._portfolio = portfolio
        self._trading_halted: bool = False
        self._halt_reason: Optional[str] = None
        self._atr_cache: Dict[str, float] = {}   # symbol → latest ATR

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, signal: Signal) -> RiskDecision:
        """
        Evaluate a strategy signal against all risk rules.

        Returns a ``RiskDecision`` with ``approved=True`` only if *all*
        hard limits are satisfied. Soft violations are noted but do not
        block execution.
        """
        violations: List[RiskViolation] = []

        # ── Rule 1: Trading halt circuit breaker ──────────────────────────────
        if self._trading_halted:
            v = RiskViolation(
                violation_type=ViolationType.TRADING_HALTED,
                message=f"Trading halted: {self._halt_reason}",
                original_signal=signal,
            )
            return RiskDecision(approved=False, adjusted_signal=signal, violations=[v])

        # Flat signals always pass — they are exit / neutral intents
        if signal.direction == Direction.FLAT:
            return RiskDecision(approved=True, adjusted_signal=signal, suggested_quantity=0.0)

        # ── Rule 2: Drawdown circuit breaker ──────────────────────────────────
        drawdown = self._portfolio.current_drawdown
        if drawdown >= self._cfg.max_daily_drawdown_pct:
            msg = (
                f"Drawdown {drawdown:.2%} ≥ limit {self._cfg.max_daily_drawdown_pct:.2%}. "
                f"Trading halted."
            )
            logger.warning(msg)
            if self._cfg.halt_on_breach:
                self._trading_halted = True
                self._halt_reason = msg
            v = RiskViolation(
                violation_type=ViolationType.DRAWDOWN_BREACH,
                message=msg,
                original_signal=signal,
            )
            violations.append(v)
            return RiskDecision(
                approved=False, adjusted_signal=signal, violations=violations
            )

        # ── Rule 3: Maximum open positions ────────────────────────────────────
        open_count = len(self._portfolio.open_positions)
        symbol_already_open = signal.symbol in self._portfolio.open_positions

        if not symbol_already_open and open_count >= self._cfg.max_open_positions:
            msg = (
                f"Open positions ({open_count}) at limit "
                f"({self._cfg.max_open_positions}). Signal rejected."
            )
            logger.warning(msg)
            v = RiskViolation(
                violation_type=ViolationType.MAX_POSITIONS_EXCEEDED,
                message=msg,
                original_signal=signal,
            )
            return RiskDecision(
                approved=False, adjusted_signal=signal, violations=[v]
            )

        # ── Rule 4: Position sizing ────────────────────────────────────────────
        equity = self._portfolio.total_equity
        max_notional = equity * self._cfg.max_position_size_pct
        contract_multiplier = self._portfolio_cfg.contract_multiplier

        # Compute ATR-based contract quantity if signal has no suggested size
        suggested_qty = signal.suggested_size or self._kelly_size(
            signal, equity, contract_multiplier
        )
        notional = suggested_qty * self._current_price(signal) * contract_multiplier

        if notional > max_notional:
            # Scale down to the maximum allowed notional
            suggested_qty = max_notional / (
                self._current_price(signal) * contract_multiplier + 1e-9
            )
            msg = (
                f"Notional {notional:.2f} > max {max_notional:.2f}. "
                f"Qty scaled to {suggested_qty:.2f}."
            )
            logger.info(msg)

        if suggested_qty < 1.0:
            suggested_qty = 1.0         # minimum 1 contract

        # ── Apply default stop / take-profit if missing ────────────────────────
        adjusted = self._apply_default_stops(signal, suggested_qty)

        return RiskDecision(
            approved=True,
            adjusted_signal=adjusted,
            violations=violations,
            suggested_quantity=suggested_qty,
        )

    def resume_trading(self) -> None:
        """Manually lift a trading halt (e.g. at the start of a new session)."""
        self._trading_halted = False
        self._halt_reason = None
        logger.info("Trading halt lifted.")

    def update_atr(self, symbol: str, atr_value: float) -> None:
        """Inform the risk manager of the latest ATR for a symbol."""
        self._atr_cache[symbol] = atr_value

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _kelly_size(
        self, signal: Signal, equity: float, contract_multiplier: float
    ) -> float:
        """
        Conservative fixed-fractional position sizing.

        Uses 1% of equity as default risk per trade, sized by ATR stop distance.
        """
        atr_val = self._atr_cache.get(signal.symbol, 10.0)
        risk_per_trade = equity * 0.01  # 1 % risk

        stop_distance = (
            abs(signal.stop_loss - self._current_price(signal))
            if signal.stop_loss
            else self._cfg.default_stop_loss_atr_mult * atr_val
        )

        if stop_distance <= 0:
            return 1.0

        qty = risk_per_trade / (stop_distance * contract_multiplier)
        return max(1.0, round(qty, 0))

    @staticmethod
    def _current_price(signal: Signal) -> float:
        """Extract a reference price from the signal metadata, defaulting to 0."""
        return float(signal.metadata.get("price", signal.metadata.get("close", 1.0)))

    def _apply_default_stops(self, signal: Signal, qty: float) -> Signal:
        """Return a new Signal with stop/TP populated from ATR defaults if absent."""
        if signal.stop_loss is not None and signal.take_profit is not None:
            return signal

        atr_val = self._atr_cache.get(signal.symbol, 10.0)
        ref_price = self._current_price(signal)

        sl = signal.stop_loss
        tp = signal.take_profit

        if sl is None:
            mult = self._cfg.default_stop_loss_atr_mult
            sl = (
                ref_price - mult * atr_val
                if signal.direction == Direction.LONG
                else ref_price + mult * atr_val
            )

        if tp is None:
            mult = self._cfg.default_take_profit_atr_mult
            tp = (
                ref_price + mult * atr_val
                if signal.direction == Direction.LONG
                else ref_price - mult * atr_val
            )

        # Return a new Signal with updated levels (Signal is mutable dataclass)
        return Signal(
            symbol=signal.symbol,
            direction=signal.direction,
            strength=signal.strength,
            timestamp=signal.timestamp,
            strategy_name=signal.strategy_name,
            stop_loss=sl,
            take_profit=tp,
            suggested_size=qty,
            metadata=signal.metadata,
        )
