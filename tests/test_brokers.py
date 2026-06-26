"""Unit tests for the brokers/ module (PaperBroker)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from brokers.paper import PaperBroker
from core.config import Config
from core.enums import Direction, OrderStatus, OrderType
from core.models import Fill, Order


def _market_order(
    direction: Direction = Direction.LONG,
    qty: float = 1.0,
    price: float = 4500.0,
    symbol: str = "ES.c.0",
) -> Order:
    return Order(
        symbol=symbol,
        direction=direction,
        order_type=OrderType.MARKET,
        quantity=qty,
        timestamp=datetime(2023, 6, 1, 14, 0, tzinfo=timezone.utc),
        limit_price=price,
    )


def _limit_order(price: float = 4500.0) -> Order:
    return Order(
        symbol="ES.c.0",
        direction=Direction.LONG,
        order_type=OrderType.LIMIT,
        quantity=2.0,
        timestamp=datetime(2023, 6, 1, tzinfo=timezone.utc),
        limit_price=price,
    )


@pytest.fixture
def broker(default_config: Config) -> PaperBroker:
    return PaperBroker(default_config)


class TestPaperBrokerSubmit:
    def test_market_order_returns_fill(self, broker: PaperBroker):
        order = _market_order()
        fill = broker.submit_order(order)
        assert fill is not None
        assert isinstance(fill, Fill)

    def test_fill_direction_matches_order(self, broker: PaperBroker):
        order = _market_order(direction=Direction.SHORT)
        fill = broker.submit_order(order)
        assert fill.direction == Direction.SHORT

    def test_fill_quantity_matches_order(self, broker: PaperBroker):
        order = _market_order(qty=3.0)
        fill = broker.submit_order(order)
        assert fill.filled_quantity == 3.0

    def test_long_market_order_adds_slippage(self, broker: PaperBroker, default_config: Config):
        order = _market_order(direction=Direction.LONG, price=4500.0)
        fill = broker.submit_order(order)
        expected_slip = default_config.portfolio.slippage_ticks * default_config.portfolio.tick_size
        assert abs(fill.fill_price - (4500.0 + expected_slip)) < 1e-9

    def test_short_market_order_subtracts_slippage(self, broker: PaperBroker, default_config: Config):
        order = _market_order(direction=Direction.SHORT, price=4500.0)
        fill = broker.submit_order(order)
        expected_slip = default_config.portfolio.slippage_ticks * default_config.portfolio.tick_size
        assert abs(fill.fill_price - (4500.0 - expected_slip)) < 1e-9

    def test_limit_order_fills_at_limit_price(self, broker: PaperBroker):
        order = _limit_order(price=4498.0)
        fill = broker.submit_order(order)
        assert fill is not None
        assert fill.fill_price == 4498.0

    def test_commission_applied(self, broker: PaperBroker, default_config: Config):
        order = _market_order(qty=2.0)
        fill = broker.submit_order(order)
        expected_comm = default_config.portfolio.commission_per_contract * 2.0
        assert abs(fill.commission - expected_comm) < 1e-9


class TestPaperBrokerCancel:
    def test_cancel_all_returns_list(self, broker: PaperBroker):
        result = broker.cancel_all_orders()
        assert isinstance(result, list)

    def test_cancel_clears_open_orders(self, broker: PaperBroker):
        # Manually add a pending order to open_orders dict
        from core.models import Order as Ord
        o = _market_order()
        o.order_id = "test-open-001"
        broker._open_orders["test-open-001"] = o
        broker.cancel_all_orders()
        assert len(broker._open_orders) == 0


class TestEmergencyShutdown:
    def test_shutdown_does_not_raise(self, broker: PaperBroker):
        # Should complete without errors even with empty state
        broker.emergency_shutdown()


class TestAccountInfo:
    def test_returns_dict(self, broker: PaperBroker):
        info = broker.get_account_info()
        assert isinstance(info, dict)
        assert info.get("broker") == "paper"
