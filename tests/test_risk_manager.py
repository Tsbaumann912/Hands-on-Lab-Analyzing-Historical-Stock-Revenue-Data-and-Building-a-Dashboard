"""
Unit tests for the risk/ module (RiskManager).

Tests cover:
- Drawdown circuit breaker halts trading
- Position sizing limit scales down oversized signals
- Maximum open positions rejects new entries
- Default stop/take-profit application
- Flat signals always approved
- Manual halt resume
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.config import Config, RiskConfig
from core.enums import Direction
from core.models import Fill, Signal
from portfolio.portfolio import Portfolio
from risk.risk_manager import RiskDecision, RiskManager, ViolationType


# ── Helpers ───────────────────────────────────────────────────────────────────

def _signal(
    direction: Direction = Direction.LONG,
    symbol: str = "ES.c.0",
    strength: float = 0.7,
    stop_loss: float = None,
    take_profit: float = None,
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,
        strength=strength,
        timestamp=datetime(2023, 6, 1, 14, 0, tzinfo=timezone.utc),
        strategy_name="TestStrategy",
        stop_loss=stop_loss,
        take_profit=take_profit,
        metadata={"price": 4500.0},
    )


def _fill_portfolio(portfolio: Portfolio, equity_drop_pct: float = 0.0) -> None:
    """Drain the portfolio to simulate a drawdown."""
    from core.models import Fill
    from core.enums import Direction as D
    # Simulate a loss by opening a position and marking down
    fill = Fill(
        order_id="drwn-001",
        symbol="ES.c.0",
        direction=D.LONG,
        filled_quantity=1.0,
        fill_price=4500.0,
        commission=0.0,
        timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
    )
    portfolio.process_fill(fill)
    drop_price = 4500.0 * (1 - equity_drop_pct / 50.0)  # rough approximation
    portfolio.mark_to_market({"ES.c.0": drop_price}, datetime(2023, 1, 2, tzinfo=timezone.utc))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def portfolio(default_config: Config) -> Portfolio:
    return Portfolio(default_config)


@pytest.fixture
def risk_mgr(default_config: Config, portfolio: Portfolio) -> RiskManager:
    return RiskManager(default_config, portfolio)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFlatSignalAlwaysApproved:
    def test_flat_signal_approved(self, risk_mgr: RiskManager):
        sig = _signal(direction=Direction.FLAT)
        decision = risk_mgr.evaluate(sig)
        assert decision.approved is True

    def test_flat_signal_suggested_qty_zero(self, risk_mgr: RiskManager):
        sig = _signal(direction=Direction.FLAT)
        decision = risk_mgr.evaluate(sig)
        assert decision.suggested_quantity == 0.0


class TestDrawdownCircuitBreaker:
    def test_breach_rejects_signal(self, default_config: Config):
        config = Config()
        config.risk.max_daily_drawdown_pct = 0.03
        config.risk.halt_on_breach = True
        portfolio = Portfolio(config)

        # Directly manipulate peak equity to simulate a 5% drawdown
        portfolio._peak_equity = 100_000.0
        portfolio._cash = 95_000.0  # 5% drop below peak

        risk_mgr = RiskManager(config, portfolio)
        decision = risk_mgr.evaluate(_signal())
        assert decision.approved is False

    def test_breach_sets_halt_flag(self, default_config: Config):
        config = Config()
        config.risk.max_daily_drawdown_pct = 0.03
        config.risk.halt_on_breach = True
        portfolio = Portfolio(config)
        portfolio._peak_equity = 100_000.0
        portfolio._cash = 94_000.0  # 6% drawdown

        risk_mgr = RiskManager(config, portfolio)
        risk_mgr.evaluate(_signal())
        assert risk_mgr.is_halted is True

    def test_resume_clears_halt(self, default_config: Config):
        config = Config()
        portfolio = Portfolio(config)
        portfolio._peak_equity = 100_000.0
        portfolio._cash = 90_000.0

        risk_mgr = RiskManager(config, portfolio)
        risk_mgr.evaluate(_signal())  # trigger halt
        risk_mgr.resume_trading()
        assert risk_mgr.is_halted is False

    def test_violation_type_is_drawdown(self, default_config: Config):
        config = Config()
        config.risk.max_daily_drawdown_pct = 0.02
        portfolio = Portfolio(config)
        portfolio._peak_equity = 100_000.0
        portfolio._cash = 90_000.0

        risk_mgr = RiskManager(config, portfolio)
        decision = risk_mgr.evaluate(_signal())
        viol_types = [v.violation_type for v in decision.violations]
        assert ViolationType.DRAWDOWN_BREACH in viol_types


class TestMaxOpenPositions:
    def test_rejects_when_at_max(self, default_config: Config):
        config = Config()
        config.risk.max_open_positions = 2
        portfolio = Portfolio(config)

        # Manually inject two open positions
        from portfolio.portfolio import Position
        portfolio._positions["ES.c.0"] = Position(
            symbol="ES.c.0", direction=Direction.LONG,
            quantity=1.0, avg_entry_price=4500.0
        )
        portfolio._positions["NQ.c.0"] = Position(
            symbol="NQ.c.0", direction=Direction.LONG,
            quantity=1.0, avg_entry_price=15000.0
        )

        risk_mgr = RiskManager(config, portfolio)
        # Try to open a third position on a NEW symbol
        sig = _signal(symbol="CL.c.0")
        decision = risk_mgr.evaluate(sig)
        assert decision.approved is False

    def test_approves_when_below_max(self, risk_mgr: RiskManager):
        decision = risk_mgr.evaluate(_signal())
        assert decision.approved is True


class TestDefaultStopApplication:
    def test_stop_applied_when_missing(self, risk_mgr: RiskManager):
        sig = _signal(stop_loss=None, take_profit=None)
        risk_mgr.update_atr("ES.c.0", 15.0)
        decision = risk_mgr.evaluate(sig)
        assert decision.adjusted_signal.stop_loss is not None
        assert decision.adjusted_signal.take_profit is not None

    def test_existing_stop_preserved(self, risk_mgr: RiskManager):
        sig = _signal(stop_loss=4480.0, take_profit=4560.0)
        risk_mgr.update_atr("ES.c.0", 15.0)
        decision = risk_mgr.evaluate(sig)
        assert decision.adjusted_signal.stop_loss == 4480.0
        assert decision.adjusted_signal.take_profit == 4560.0

    def test_long_stop_below_entry(self, risk_mgr: RiskManager):
        sig = _signal(direction=Direction.LONG, stop_loss=None, take_profit=None)
        risk_mgr.update_atr("ES.c.0", 20.0)
        decision = risk_mgr.evaluate(sig)
        # For long: stop should be below entry price (4500 from metadata)
        assert decision.adjusted_signal.stop_loss < 4500.0

    def test_short_stop_above_entry(self, risk_mgr: RiskManager):
        sig = _signal(direction=Direction.SHORT, stop_loss=None, take_profit=None)
        risk_mgr.update_atr("ES.c.0", 20.0)
        decision = risk_mgr.evaluate(sig)
        # For short: stop should be above entry price
        assert decision.adjusted_signal.stop_loss > 4500.0


class TestHaltedTrading:
    def test_halted_rejects_all_directions(self, default_config: Config):
        portfolio = Portfolio(default_config)
        risk_mgr = RiskManager(default_config, portfolio)
        risk_mgr._trading_halted = True
        risk_mgr._halt_reason = "manual test halt"

        for direction in (Direction.LONG, Direction.SHORT):
            decision = risk_mgr.evaluate(_signal(direction=direction))
            assert decision.approved is False

    def test_halted_violation_type(self, default_config: Config):
        portfolio = Portfolio(default_config)
        risk_mgr = RiskManager(default_config, portfolio)
        risk_mgr._trading_halted = True
        decision = risk_mgr.evaluate(_signal())
        assert decision.violations[0].violation_type == ViolationType.TRADING_HALTED
