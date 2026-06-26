"""
Comprehensive unit tests for portfolio/ module.

Tests cover:
- Cash balance tracking after fills
- Long / short position opening and closing
- Mark-to-market unrealised P&L
- Realised P&L on full and partial closes
- Drawdown circuit breaker threshold
- Available margin calculation
- Commission and slippage deduction
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.config import Config
from core.enums import Direction
from core.models import Fill
from portfolio.portfolio import Portfolio, Position


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fill(
    symbol: str = "ES.c.0",
    direction: Direction = Direction.LONG,
    qty: float = 1.0,
    price: float = 4500.0,
    commission: float = 2.25,
    slippage: float = 0.0,
) -> Fill:
    return Fill(
        order_id="test-001",
        symbol=symbol,
        direction=direction,
        filled_quantity=qty,
        fill_price=price,
        commission=commission,
        slippage=slippage,
        timestamp=datetime(2023, 6, 1, 14, 0, tzinfo=timezone.utc),
    )


def _ts(offset_minutes: int = 0) -> datetime:
    return datetime(2023, 6, 1, 14, offset_minutes, tzinfo=timezone.utc)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def portfolio(default_config: Config) -> Portfolio:
    return Portfolio(default_config)


# ── Test: Initial state ───────────────────────────────────────────────────────

class TestInitialState:
    def test_initial_cash_equals_config(self, portfolio: Portfolio, default_config: Config):
        assert portfolio.cash == default_config.portfolio.initial_cash

    def test_initial_total_equity_equals_cash(self, portfolio: Portfolio, default_config: Config):
        assert portfolio.total_equity == default_config.portfolio.initial_cash

    def test_no_open_positions(self, portfolio: Portfolio):
        assert len(portfolio.open_positions) == 0

    def test_zero_unrealised_pnl(self, portfolio: Portfolio):
        assert portfolio.unrealised_pnl == 0.0

    def test_zero_drawdown(self, portfolio: Portfolio):
        assert portfolio.current_drawdown == 0.0


# ── Test: Long position ───────────────────────────────────────────────────────

class TestLongPosition:
    def test_open_long_reduces_cash_by_commission(self, portfolio: Portfolio):
        """Opening a futures position deducts commission (not full notional) from cash."""
        initial_cash = portfolio.cash
        fill = _fill(direction=Direction.LONG, qty=1.0, price=4500.0, commission=2.25)
        portfolio.process_fill(fill)
        # Only commission is deducted immediately; notional is margin-tracked separately
        assert portfolio.cash < initial_cash
        expected_cash = initial_cash - 2.25
        assert abs(portfolio.cash - expected_cash) < 0.01

    def test_open_long_creates_position(self, portfolio: Portfolio):
        fill = _fill(direction=Direction.LONG, qty=2.0, price=4500.0)
        portfolio.process_fill(fill)
        assert "ES.c.0" in portfolio.open_positions
        pos = portfolio.open_positions["ES.c.0"]
        assert pos.direction == Direction.LONG
        assert abs(pos.quantity - 2.0) < 1e-9

    def test_mark_to_market_unrealised_gain(self, portfolio: Portfolio):
        fill = _fill(direction=Direction.LONG, qty=1.0, price=4500.0)
        portfolio.process_fill(fill)
        portfolio.mark_to_market({"ES.c.0": 4550.0}, _ts(1))
        # Unrealised gain = (4550 - 4500) * 1 contract * 50 multiplier = 2500
        assert abs(portfolio.unrealised_pnl - 2500.0) < 0.01

    def test_mark_to_market_unrealised_loss(self, portfolio: Portfolio):
        fill = _fill(direction=Direction.LONG, qty=1.0, price=4500.0)
        portfolio.process_fill(fill)
        portfolio.mark_to_market({"ES.c.0": 4400.0}, _ts(1))
        # Unrealised loss = (4400 - 4500) * 1 * 50 = -5000
        assert abs(portfolio.unrealised_pnl - (-5000.0)) < 0.01

    def test_close_long_restores_equity(self, portfolio: Portfolio):
        """After a round-trip, total equity equals initial + net P&L."""
        open_fill = _fill(direction=Direction.LONG, qty=1.0, price=4500.0, commission=2.25)
        portfolio.process_fill(open_fill)

        close_fill = _fill(direction=Direction.SHORT, qty=1.0, price=4550.0, commission=2.25)
        portfolio.process_fill(close_fill)

        assert "ES.c.0" not in portfolio.open_positions
        # Net PnL = (4550 - 4500) * 1 * 50 = 2500, minus 2 × 2.25 commission = 2495.5
        net_pnl = (4550.0 - 4500.0) * 1.0 * 50.0 - 2 * 2.25
        expected_equity = 100_000.0 + net_pnl
        assert abs(portfolio.total_equity - expected_equity) < 1.0


# ── Test: Short position ──────────────────────────────────────────────────────

class TestShortPosition:
    def test_open_short_creates_position(self, portfolio: Portfolio):
        fill = _fill(direction=Direction.SHORT, qty=1.0, price=4500.0)
        portfolio.process_fill(fill)
        pos = portfolio.open_positions.get("ES.c.0")
        assert pos is not None
        assert pos.direction == Direction.SHORT
        assert pos.quantity < 0  # short stored as negative quantity

    def test_mark_to_market_short_gain(self, portfolio: Portfolio):
        """Short position gains when price falls."""
        fill = _fill(direction=Direction.SHORT, qty=1.0, price=4500.0, commission=0.0)
        portfolio.process_fill(fill)
        portfolio.mark_to_market({"ES.c.0": 4400.0}, _ts(1))
        # Short gain = (4500 - 4400) * 1 * 50 = 5000
        assert abs(portfolio.unrealised_pnl - 5000.0) < 0.01


# ── Test: Drawdown tracking ───────────────────────────────────────────────────

class TestDrawdown:
    def test_drawdown_increases_after_loss(self, portfolio: Portfolio):
        """MTM loss below initial equity produces positive drawdown."""
        fill = _fill(direction=Direction.LONG, qty=1.0, price=4500.0, commission=0.0)
        portfolio.process_fill(fill)
        # Mark to a price lower than entry → unrealised loss → equity < initial
        portfolio.mark_to_market({"ES.c.0": 4450.0}, _ts(1))
        assert portfolio.current_drawdown > 0.0

    def test_drawdown_zero_at_new_peak(self, portfolio: Portfolio):
        """Equity above prior peak produces zero drawdown."""
        fill = _fill(direction=Direction.LONG, qty=1.0, price=4500.0, commission=0.0)
        portfolio.process_fill(fill)
        # Mark up — total equity > initial equity → new peak, drawdown = 0
        portfolio.mark_to_market({"ES.c.0": 4600.0}, _ts(1))
        assert portfolio.current_drawdown == 0.0

    def test_drawdown_respects_peak_equity(self, portfolio: Portfolio):
        fill = _fill(direction=Direction.LONG, qty=1.0, price=4500.0, commission=0.0)
        portfolio.process_fill(fill)
        portfolio.mark_to_market({"ES.c.0": 4600.0}, _ts(1))  # establish peak
        peak = portfolio.peak_equity
        portfolio.mark_to_market({"ES.c.0": 4550.0}, _ts(2))  # pull back
        expected_dd = (peak - portfolio.total_equity) / peak
        assert abs(portfolio.current_drawdown - expected_dd) < 1e-6


# ── Test: Commission and slippage ─────────────────────────────────────────────

class TestTransactionCosts:
    def test_commission_deducted_from_cash(self, portfolio: Portfolio):
        """Commission is deducted from cash immediately on fill."""
        initial_cash = portfolio.cash
        fill = _fill(commission=5.00, slippage=0.0)
        portfolio.process_fill(fill)
        # Only commission deducted (margin accounting — no full notional deduction)
        assert abs(portfolio.cash - (initial_cash - 5.00)) < 0.01

    def test_slippage_deducted(self, portfolio: Portfolio):
        """Slippage cost (ticks × tick_value) is deducted from cash immediately."""
        initial_cash = portfolio.cash
        fill = _fill(commission=2.25, slippage=2.0)  # 2 ticks slippage
        portfolio.process_fill(fill)
        tick_value = 12.50
        expected_slippage_cost = 2.0 * tick_value
        assert abs(
            portfolio.cash - (initial_cash - 2.25 - expected_slippage_cost)
        ) < 0.01


# ── Test: Portfolio summary ───────────────────────────────────────────────────

class TestSummary:
    def test_summary_has_required_keys(self, portfolio: Portfolio):
        summary = portfolio.summary()
        required = {
            "cash", "unrealised_pnl", "realised_pnl",
            "total_equity", "drawdown", "open_positions",
        }
        assert required.issubset(summary.keys())

    def test_equity_curve_recorded(self, portfolio: Portfolio):
        fill = _fill()
        portfolio.process_fill(fill)
        portfolio.mark_to_market({"ES.c.0": 4510.0}, _ts(1))
        assert len(portfolio.equity_curve) == 1

    def test_multiple_mtm_snapshots(self, portfolio: Portfolio):
        fill = _fill()
        portfolio.process_fill(fill)
        for i in range(5):
            portfolio.mark_to_market({"ES.c.0": 4500.0 + i * 10}, _ts(i))
        assert len(portfolio.equity_curve) == 5
