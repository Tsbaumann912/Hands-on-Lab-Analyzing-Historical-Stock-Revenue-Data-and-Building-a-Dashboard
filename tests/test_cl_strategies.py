"""Unit tests for CL energy risk-premia strategies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.config import Config
from core.enums import AssetClass, Direction
from core.models import Bar
from strategies.cl_carry_curve import CLCarryCurve
from strategies.cl_carry_momentum import CLCarryMomentum
from strategies.cl_inventory_confirm import CLInventoryConfirm
from strategies.cl_vol_target_tsmom import CLVolTargetTSMOM

CL_STRATEGIES = [
    CLCarryCurve,
    CLCarryMomentum,
    CLVolTargetTSMOM,
    CLInventoryConfirm,
]


def make_cl_bars(
    n: int = 400,
    start_price: float = 75.0,
    volatility: float = 1.5,
    seed: int = 42,
    drift: float = 0.0,
) -> list[Bar]:
    """CL bars without the ES-oriented $100 price floor in ``make_bars``."""
    rng = np.random.default_rng(seed)
    prices = start_price + np.cumsum(rng.normal(drift, volatility, n))
    prices = np.clip(prices, 10.0, None)
    base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    bars: list[Bar] = []
    for i in range(n):
        close = float(prices[i])
        spread = float(rng.uniform(0.2, 0.8))
        bars.append(
            Bar(
                symbol="CL.c.0",
                timestamp=base_ts + timedelta(days=i),
                open=float(prices[i - 1]) if i > 0 else close,
                high=close + spread,
                low=close - spread,
                close=close,
                volume=float(rng.uniform(500, 5000)),
                asset_class=AssetClass.FUTURES,
                contract_multiplier=1000.0,
            )
        )
    return bars


@pytest.fixture
def cl_config() -> Config:
    cfg = Config()
    cfg.portfolio.contract_multiplier = 1000.0
    cfg.portfolio.tick_size = 0.01
    cfg.portfolio.tick_value = 10.0
    return cfg


class TestCLWarmUp:
    @pytest.mark.parametrize("strategy_cls", CL_STRATEGIES)
    def test_flat_during_warmup(self, cl_config: Config, strategy_cls):
        strategy = strategy_cls(cl_config, symbols=["CL.c.0"])
        bars = make_cl_bars(n=10, volatility=0.5)
        for bar in bars:
            signal = strategy.update(bar)
            assert signal.direction == Direction.FLAT

    @pytest.mark.parametrize("strategy_cls", CL_STRATEGIES)
    def test_strength_in_range(self, cl_config: Config, strategy_cls):
        strategy = strategy_cls(cl_config, symbols=["CL.c.0"])
        bars = make_cl_bars(n=400, volatility=1.0, seed=7)
        for bar in bars:
            sig = strategy.update(bar)
            assert 0.0 <= sig.strength <= 1.0
            assert sig.strategy_name == strategy_cls.__name__
            assert sig.symbol == "CL.c.0"


class TestCLCarryCurve:
    def test_emits_directional_signal(self, cl_config: Config):
        strategy = CLCarryCurve(cl_config, symbols=["CL.c.0"])
        bars = make_cl_bars(n=400, volatility=1.2, seed=11)
        dirs = {strategy.update(b).direction for b in bars}
        assert Direction.LONG in dirs or Direction.SHORT in dirs

    def test_metadata_has_carry(self, cl_config: Config):
        strategy = CLCarryCurve(cl_config, symbols=["CL.c.0"])
        bars = make_cl_bars(n=400, volatility=1.0, seed=3)
        for bar in bars:
            sig = strategy.update(bar)
            if sig.direction != Direction.FLAT:
                assert "carry_spread" in sig.metadata
                break
        else:
            pytest.fail("expected a non-flat carry signal")


class TestCLCarryMomentum:
    def test_emits_directional_signal(self, cl_config: Config):
        strategy = CLCarryMomentum(cl_config, symbols=["CL.c.0"])
        bars = make_cl_bars(n=400, volatility=1.5, seed=21)
        non_flat = [
            s
            for b in bars
            if (s := strategy.update(b)).direction != Direction.FLAT
        ]
        assert len(non_flat) > 0


class TestCLVolTargetTSMOM:
    def test_trending_series_goes_long(self, cl_config: Config):
        strategy = CLVolTargetTSMOM(cl_config, symbols=["CL.c.0"])
        bars = make_cl_bars(n=400, volatility=0.15, drift=0.08, seed=0)
        longs = sum(
            1 for b in bars if strategy.update(b).direction == Direction.LONG
        )
        assert longs > 0


class TestCLInventoryConfirm:
    def test_external_inventory_runs(self, cl_config: Config):
        strategy = CLInventoryConfirm(cl_config, symbols=["CL.c.0"])
        bars = make_cl_bars(n=400, volatility=0.8, seed=5)
        inv = np.linspace(400_000.0, 500_000.0, len(bars))
        strategy.set_inventory_series("CL.c.0", inv)
        for bar in bars:
            sig = strategy.update(bar)
            assert 0.0 <= sig.strength <= 1.0

    def test_disabled_always_flat(self, cl_config: Config):
        cl_config.strategy.inventory_enabled = False
        strategy = CLInventoryConfirm(cl_config, symbols=["CL.c.0"])
        bars = make_cl_bars(n=300)
        for bar in bars:
            assert strategy.update(bar).direction == Direction.FLAT


class TestCLRegistry:
    def test_param_spaces_registered(self):
        from engine.optimizer import STRATEGY_PARAM_SPACES

        for name in (
            "CLCarryCurve",
            "CLCarryMomentum",
            "CLVolTargetTSMOM",
            "CLInventoryConfirm",
        ):
            assert name in STRATEGY_PARAM_SPACES
            assert len(STRATEGY_PARAM_SPACES[name]) > 0

    def test_strategy_registry(self):
        from app.data_service import STRATEGY_REGISTRY

        for name in (
            "CLCarryCurve",
            "CLCarryMomentum",
            "CLVolTargetTSMOM",
            "CLInventoryConfirm",
        ):
            assert name in STRATEGY_REGISTRY
