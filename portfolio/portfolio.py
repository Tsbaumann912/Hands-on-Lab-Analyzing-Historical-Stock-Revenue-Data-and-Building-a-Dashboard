"""
Deterministic portfolio state tracker for futures positions.

Design principles:
- Immutable snapshots for equity curve; mutable intra-bar state
- All arithmetic on **notional value** (price × multiplier × contracts)
- Commission and slippage are deducted at fill time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from core.config import Config, PortfolioConfig
from core.enums import Direction
from core.models import Fill

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Open futures position for one symbol."""

    symbol: str
    direction: Direction
    quantity: float                        # signed: positive = long
    avg_entry_price: float
    contract_multiplier: float = 50.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    open_time: Optional[datetime] = None

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.avg_entry_price * self.contract_multiplier

    def mark_to_market(self, current_price: float) -> float:
        """Update unrealised P&L and return the new value."""
        price_diff = current_price - self.avg_entry_price
        sign = 1.0 if self.direction == Direction.LONG else -1.0
        self.unrealised_pnl = (
            sign * price_diff * abs(self.quantity) * self.contract_multiplier
        )
        return self.unrealised_pnl


@dataclass
class EquitySnapshot:
    timestamp: datetime
    cash: float
    unrealised_pnl: float
    realised_pnl: float
    total_equity: float
    open_positions: int


class Portfolio:
    """
    Tracks cash, open positions, margin usage, and the equity curve.

    Parameters
    ----------
    config:
        Terminal-wide config; portfolio params come from ``config.portfolio``.
    """

    def __init__(self, config: Config) -> None:
        self._cfg: PortfolioConfig = config.portfolio
        self._cash: float = self._cfg.initial_cash
        self._peak_equity: float = self._cfg.initial_cash
        self._positions: Dict[str, Position] = {}
        self._equity_curve: List[EquitySnapshot] = []
        self._fills: List[Fill] = []
        self._total_commission: float = 0.0
        self._total_slippage: float = 0.0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self._positions.values())

    @property
    def realised_pnl(self) -> float:
        return sum(p.realised_pnl for p in self._positions.values())

    @property
    def total_equity(self) -> float:
        return self._cash + self.unrealised_pnl

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a fraction of peak equity (0.0 – 1.0)."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self.total_equity) / self._peak_equity

    @property
    def open_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    @property
    def equity_curve(self) -> List[EquitySnapshot]:
        return list(self._equity_curve)

    @property
    def fills(self) -> List[Fill]:
        return list(self._fills)

    # ── Fill processing ───────────────────────────────────────────────────────

    def process_fill(self, fill: Fill) -> None:
        """
        Apply an executed fill to the portfolio state.

        Uses futures-style margin accounting:
        - Opening a position does NOT deduct the full notional from cash.
          Instead, realised P&L accrues to cash on close (variation margin model).
        - Commission and slippage are deducted from cash immediately.
        - Unrealised P&L is tracked separately via mark-to-market.
        """
        self._fills.append(fill)

        # Deduct transaction costs immediately
        self._cash -= fill.commission + fill.slippage * self._cfg.tick_value
        self._total_commission += fill.commission
        self._total_slippage += fill.slippage * self._cfg.tick_value

        symbol = fill.symbol
        qty_signed = fill.filled_quantity if fill.direction == Direction.LONG else -fill.filled_quantity

        if symbol not in self._positions:
            self._open_position(fill, qty_signed)
        else:
            self._update_position(fill, qty_signed)

    def _open_position(self, fill: Fill, qty_signed: float) -> None:
        direction = Direction.LONG if qty_signed > 0 else Direction.SHORT
        self._positions[fill.symbol] = Position(
            symbol=fill.symbol,
            direction=direction,
            quantity=qty_signed,
            avg_entry_price=fill.fill_price,
            contract_multiplier=self._cfg.contract_multiplier,
            open_time=fill.timestamp,
        )
        logger.info(
            "OPEN %s %s @ %.4f qty=%.0f",
            direction.value,
            fill.symbol,
            fill.fill_price,
            abs(qty_signed),
        )

    def _update_position(self, fill: Fill, qty_signed: float) -> None:
        pos = self._positions[fill.symbol]
        new_qty = pos.quantity + qty_signed

        if abs(new_qty) < 1e-9:
            # Position fully closed — credit realised P&L to cash
            realised = self._compute_realised_pnl(pos, fill.fill_price, abs(pos.quantity))
            pos.realised_pnl += realised
            self._cash += realised
            del self._positions[fill.symbol]
            logger.info(
                "CLOSE %s @ %.4f realised_pnl=%.2f",
                fill.symbol,
                fill.fill_price,
                realised,
            )

        elif (qty_signed > 0) == (pos.quantity > 0):
            # Adding to existing position — update average entry price
            total_qty = abs(pos.quantity) + abs(qty_signed)
            pos.avg_entry_price = (
                pos.avg_entry_price * abs(pos.quantity)
                + fill.fill_price * abs(qty_signed)
            ) / total_qty
            pos.quantity = new_qty

        else:
            # Partial close — realise P&L on the closed portion
            close_qty = min(abs(pos.quantity), abs(qty_signed))
            realised = self._compute_realised_pnl(pos, fill.fill_price, close_qty)
            pos.realised_pnl += realised
            self._cash += realised
            pos.quantity = new_qty
            if abs(new_qty) < 1e-9:
                del self._positions[fill.symbol]

    @staticmethod
    def _compute_realised_pnl(pos: Position, exit_price: float, qty: float) -> float:
        price_diff = exit_price - pos.avg_entry_price
        sign = 1.0 if pos.direction == Direction.LONG else -1.0
        return sign * price_diff * qty * pos.contract_multiplier

    # ── Mark-to-market ────────────────────────────────────────────────────────

    def mark_to_market(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """Update unrealised P&L for all open positions and record equity snapshot."""
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].mark_to_market(price)

        equity = self.total_equity
        self._peak_equity = max(self._peak_equity, equity)

        self._equity_curve.append(
            EquitySnapshot(
                timestamp=timestamp,
                cash=self._cash,
                unrealised_pnl=self.unrealised_pnl,
                realised_pnl=self.realised_pnl,
                total_equity=equity,
                open_positions=len(self._positions),
            )
        )

    # ── Margin assessment ─────────────────────────────────────────────────────

    def available_margin(self, initial_margin_per_contract: float = 12_000.0) -> float:
        """
        Available margin after accounting for existing position requirements.

        Simplified: assumes a fixed initial margin per contract (e.g., CME ES ≈ $12k).
        Available = total_equity - used_margin
        """
        used_margin = sum(
            abs(p.quantity) * initial_margin_per_contract
            for p in self._positions.values()
        )
        return max(0.0, self.total_equity - used_margin)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, float]:
        return {
            "cash": self._cash,
            "unrealised_pnl": self.unrealised_pnl,
            "realised_pnl": self.realised_pnl,
            "total_equity": self.total_equity,
            "drawdown": self.current_drawdown,
            "total_commission": self._total_commission,
            "total_slippage": self._total_slippage,
            "open_positions": float(len(self._positions)),
        }
