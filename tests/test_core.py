"""Unit tests for the core/ module: enums, events, config, models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.config import Config, DataConfig, RiskConfig
from core.enums import Direction, EventType, OrderStatus, OrderType
from core.events import Event, EventBus
from core.models import Bar, Fill, Order, Signal, Tick


# ── Config tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config_has_initial_cash(self):
        cfg = Config()
        assert cfg.portfolio.initial_cash == 100_000.0

    def test_default_symbols_non_empty(self):
        cfg = Config()
        assert len(cfg.data.symbols) > 0

    def test_config_from_yaml_fallback_to_defaults(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        cfg = Config.from_yaml(missing)
        assert cfg.portfolio.initial_cash == 100_000.0

    def test_config_to_yaml_round_trip(self, tmp_path):
        cfg = Config()
        cfg.portfolio.initial_cash = 250_000.0
        out_path = tmp_path / "config.yaml"
        cfg.to_yaml(out_path)
        loaded = Config.from_yaml(out_path)
        assert loaded.portfolio.initial_cash == 250_000.0

    def test_risk_config_drawdown_default(self):
        cfg = Config()
        assert 0 < cfg.risk.max_daily_drawdown_pct < 1.0


# ── EventBus tests ────────────────────────────────────────────────────────────

class TestEventBus:
    def _make_event(self, event_type: EventType = EventType.BAR) -> Event:
        return Event(
            event_type=event_type,
            timestamp=datetime(2023, 6, 1, tzinfo=timezone.utc),
            payload={"test": True},
            source="test",
        )

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.BAR, received.append)
        event = self._make_event(EventType.BAR)
        bus.publish(event)
        assert len(received) == 1
        assert received[0] is event

    def test_multiple_subscribers(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe(EventType.SIGNAL, a.append)
        bus.subscribe(EventType.SIGNAL, b.append)
        bus.publish(self._make_event(EventType.SIGNAL))
        assert len(a) == 1 and len(b) == 1

    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.TICK, received.append)
        bus.unsubscribe(EventType.TICK, received.append)
        bus.publish(self._make_event(EventType.TICK))
        assert len(received) == 0

    def test_wrong_type_not_delivered(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.FILL, received.append)
        bus.publish(self._make_event(EventType.BAR))
        assert len(received) == 0

    def test_event_log_when_enabled(self):
        bus = EventBus()
        bus.enable_event_log()
        bus.publish(self._make_event(EventType.HEARTBEAT))
        assert len(bus.event_log) == 1

    def test_handler_exception_doesnt_stop_others(self):
        bus = EventBus()
        results = []

        def bad_handler(e: Event) -> None:
            raise RuntimeError("boom")

        def good_handler(e: Event) -> None:
            results.append(e)

        bus.subscribe(EventType.BAR, bad_handler)
        bus.subscribe(EventType.BAR, good_handler)
        bus.publish(self._make_event(EventType.BAR))
        assert len(results) == 1

    def test_subscriber_count(self):
        bus = EventBus()
        bus.subscribe(EventType.ORDER, lambda e: None)
        bus.subscribe(EventType.ORDER, lambda e: None)
        assert bus.subscriber_count(EventType.ORDER) == 2


# ── Model tests ───────────────────────────────────────────────────────────────

class TestBar:
    def test_mid_price(self):
        bar = Bar(
            symbol="ES.c.0",
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            open=4500.0,
            high=4510.0,
            low=4490.0,
            close=4505.0,
            volume=1000.0,
        )
        assert bar.mid == (4510.0 + 4490.0) / 2.0

    def test_range(self):
        bar = Bar(
            symbol="ES.c.0",
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            open=4500.0,
            high=4510.0,
            low=4490.0,
            close=4505.0,
            volume=1000.0,
        )
        assert bar.range == 20.0


class TestTick:
    def test_mid_and_spread(self):
        tick = Tick(
            symbol="ES.c.0",
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            bid=4499.75,
            ask=4500.25,
            bid_size=10.0,
            ask_size=8.0,
            last=4500.0,
            last_size=5.0,
        )
        assert abs(tick.mid - 4500.0) < 1e-9
        assert abs(tick.spread - 0.5) < 1e-9


class TestSignalValidation:
    def test_valid_signal(self):
        sig = Signal(
            symbol="ES.c.0",
            direction=Direction.LONG,
            strength=0.8,
            timestamp=datetime.utcnow(),
            strategy_name="test",
        )
        assert sig.strength == 0.8

    def test_strength_below_zero_raises(self):
        with pytest.raises(ValueError):
            Signal(
                symbol="ES.c.0",
                direction=Direction.LONG,
                strength=-0.1,
                timestamp=datetime.utcnow(),
                strategy_name="test",
            )

    def test_strength_above_one_raises(self):
        with pytest.raises(ValueError):
            Signal(
                symbol="ES.c.0",
                direction=Direction.LONG,
                strength=1.01,
                timestamp=datetime.utcnow(),
                strategy_name="test",
            )


class TestOrder:
    def test_is_active_pending(self):
        order = Order(
            symbol="ES.c.0",
            direction=Direction.LONG,
            order_type=OrderType.MARKET,
            quantity=1.0,
            timestamp=datetime.utcnow(),
        )
        assert order.is_active() is True

    def test_filled_order_not_active(self):
        order = Order(
            symbol="ES.c.0",
            direction=Direction.LONG,
            order_type=OrderType.MARKET,
            quantity=1.0,
            timestamp=datetime.utcnow(),
            status=OrderStatus.FILLED,
        )
        assert order.is_active() is False
