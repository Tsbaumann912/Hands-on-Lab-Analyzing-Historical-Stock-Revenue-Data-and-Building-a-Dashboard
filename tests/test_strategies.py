"""Unit tests for the strategies/ module."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import numpy as np
import pytest

from core.config import Config
from core.enums import Direction
from core.models import Bar, Signal
from strategies.base import BarBuffer, Strategy
from strategies.mean_reversion import MeanReversionRSI
from strategies.momentum import MomentumBreakout
from strategies.trend_following import TrendFollowingMACD

from tests.conftest import make_bars


# ── BarBuffer tests ───────────────────────────────────────────────────────────

class TestBarBuffer:
    def test_initially_not_full(self):
        buf = BarBuffer(maxlen=50)
        assert not buf.is_full

    def test_full_after_maxlen_pushes(self):
        buf = BarBuffer(maxlen=10)
        bars = make_bars(n=10)
        for b in bars:
            buf.push(b)
        assert buf.is_full

    def test_closes_returns_correct_length(self):
        buf = BarBuffer(maxlen=20)
        bars = make_bars(n=20)
        for b in bars:
            buf.push(b)
        closes = buf.closes()
        assert len(closes) == 20

    def test_arrays_are_float64(self):
        buf = BarBuffer(maxlen=10)
        bars = make_bars(n=10)
        for b in bars:
            buf.push(b)
        assert buf.closes().dtype == np.float64
        assert buf.volumes().dtype == np.float64

    def test_latest_returns_most_recent_bar(self):
        buf = BarBuffer(maxlen=5)
        bars = make_bars(n=5)
        for b in bars:
            buf.push(b)
        assert buf.latest() == bars[-1]

    def test_maxlen_rolling(self):
        buf = BarBuffer(maxlen=3)
        bars = make_bars(n=5)
        for b in bars:
            buf.push(b)
        assert len(buf) == 3
        # Latest bar should be the last one pushed
        assert buf.latest() == bars[-1]


# ── Warm-up guard tests ───────────────────────────────────────────────────────

class TestWarmUpGuard:
    """Verify strategies emit FLAT signals during warm-up period."""

    @pytest.mark.parametrize("strategy_cls", [
        MeanReversionRSI,
        MomentumBreakout,
        TrendFollowingMACD,
    ])
    def test_flat_during_warmup(self, default_config: Config, strategy_cls):
        strategy = strategy_cls(default_config)
        bars = make_bars(n=10)  # far fewer than any strategy's warm-up window
        for bar in bars:
            signal = strategy.update(bar)
            assert signal.direction == Direction.FLAT, (
                f"{strategy_cls.__name__} should be FLAT during warm-up"
            )

    @pytest.mark.parametrize("strategy_cls", [
        MeanReversionRSI,
        MomentumBreakout,
        TrendFollowingMACD,
    ])
    def test_signal_strength_in_range(self, default_config: Config, strategy_cls):
        strategy = strategy_cls(default_config)
        bars = make_bars(n=300)
        for bar in bars:
            signal = strategy.update(bar)
            assert 0.0 <= signal.strength <= 1.0, (
                f"strength={signal.strength} out of [0,1] for {strategy_cls.__name__}"
            )


# ── Signal contract tests ─────────────────────────────────────────────────────

class TestSignalContract:
    """Verify returned Signal instances have required fields."""

    def test_signal_has_symbol(self, default_config: Config):
        strategy = MeanReversionRSI(default_config)
        bars = make_bars(n=300)
        signals = [strategy.update(b) for b in bars]
        for sig in signals:
            assert sig.symbol == "ES.c.0"

    def test_signal_has_timestamp(self, default_config: Config):
        strategy = MeanReversionRSI(default_config)
        bars = make_bars(n=300)
        signals = [strategy.update(b) for b in bars]
        for sig in signals:
            assert isinstance(sig.timestamp, datetime)

    def test_signal_has_strategy_name(self, default_config: Config):
        strategy = MeanReversionRSI(default_config)
        bars = make_bars(n=50)
        for bar in bars:
            sig = strategy.update(bar)
            assert sig.strategy_name == "MeanReversionRSI"

    def test_invalid_strength_raises(self):
        with pytest.raises(ValueError, match="strength"):
            Signal(
                symbol="ES.c.0",
                direction=Direction.LONG,
                strength=1.5,  # invalid: > 1.0
                timestamp=datetime.now(timezone.utc),
                strategy_name="test",
            )


# ── MeanReversionRSI specific ─────────────────────────────────────────────────

class TestMeanReversionRSI:
    def test_produces_at_least_one_non_flat_on_volatile_series(
        self, default_config: Config
    ):
        """With 300 bars and default params, expect at least one entry signal."""
        strategy = MeanReversionRSI(default_config)
        bars = make_bars(n=300, volatility=20.0, seed=123)  # high vol
        non_flat = [s for b in bars if (s := strategy.update(b)).direction != Direction.FLAT]
        assert len(non_flat) > 0, "Expected at least one non-FLAT signal on volatile data"

    def test_stop_loss_below_entry_for_long(self, default_config: Config):
        strategy = MeanReversionRSI(default_config)
        bars = make_bars(n=300, volatility=20.0, seed=42)
        for bar in bars:
            sig = strategy.update(bar)
            if sig.direction == Direction.LONG and sig.stop_loss is not None:
                assert sig.stop_loss < bar.close, "Long stop should be below close price"
                break

    def test_stop_loss_above_entry_for_short(self, default_config: Config):
        strategy = MeanReversionRSI(default_config)
        bars = make_bars(n=300, volatility=20.0, seed=99)
        for bar in bars:
            sig = strategy.update(bar)
            if sig.direction == Direction.SHORT and sig.stop_loss is not None:
                assert sig.stop_loss > bar.close, "Short stop should be above close price"
                break


# ── Position state tracking ───────────────────────────────────────────────────

class TestPositionTracking:
    def test_set_and_get_position(self, default_config: Config):
        strategy = MeanReversionRSI(default_config)
        strategy.set_position("ES.c.0", Direction.LONG)
        assert strategy.current_position("ES.c.0") == Direction.LONG

    def test_unknown_symbol_returns_flat(self, default_config: Config):
        strategy = MeanReversionRSI(default_config)
        assert strategy.current_position("UNKNOWN") == Direction.FLAT
