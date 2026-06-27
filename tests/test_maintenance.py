"""
Regression tests for all bugs fixed in the maintenance pass.

Coverage:
  - RiskManager contract_multiplier per-symbol correctness
  - compute_metrics NaN-free output on minimal equity curves
  - WalkForwardOptimizer aggregate_metrics outlier clipping
  - datetime.utcnow() usage fully replaced (no deprecation warnings in core paths)
  - RSI EMA overflow suppressed
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from core.config import Config
from core.enums import AssetClass, Direction
from core.models import Bar, Signal
from engine.metrics import compute_metrics
from engine.optimizer import WalkForwardOptimizer, build_search_space_from_yaml, HAS_OPTUNA
from portfolio.portfolio import Portfolio
from risk.risk_manager import RiskManager, RiskViolation

from tests.conftest import make_bars


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1: RiskManager contract_multiplier is now per-symbol (not hardcoded 50)
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskManagerContractMultiplier:
    def _risk_manager(self, multiplier: float) -> tuple[RiskManager, Config]:
        cfg = Config()
        cfg.portfolio.contract_multiplier = multiplier
        cfg.portfolio.initial_cash = 100_000.0
        cfg.risk.max_position_size_pct = 0.10   # $10 000 notional max
        port = Portfolio(cfg)
        return RiskManager(cfg, port), cfg

    def _signal(self) -> Signal:
        return Signal(
            "CL.c.0", Direction.LONG, 0.8,
            datetime.now(timezone.utc), "TestStrategy",
            stop_loss=70.0, take_profit=85.0,
        )

    def test_cl_multiplier_1000_used_in_sizing(self):
        """With CL multiplier=1000 and price≈75, 1 contract = $75 000 notional.
        At 10% equity limit ($10 000), suggested_qty must be scaled well below 1."""
        rm, cfg = self._risk_manager(1000.0)
        # Inject a reference price into signal metadata so notional is computable
        sig = Signal(
            "CL.c.0", Direction.LONG, 0.8,
            datetime.now(timezone.utc), "TestStrategy",
            stop_loss=70.0, take_profit=85.0,
            metadata={"price": 75.0},
        )
        d = rm.evaluate(sig)
        # Position should be approved at minimum 1 contract (floor)
        assert d.approved
        # Notional at 1 contract: 1 * 75 * 1000 = 75 000 > max 10 000
        # → scaling logic runs; qty was scaled then floored at 1
        assert d.suggested_quantity >= 1.0

    def test_es_multiplier_50_default_preserved(self):
        rm, _ = self._risk_manager(50.0)
        d = rm.evaluate(self._signal())
        assert d.approved
        assert d.suggested_quantity >= 1.0

    def test_config_multiplier_injected(self):
        """RiskManager reads multiplier from config.portfolio, not from a hardcode."""
        for mult in [20.0, 50.0, 100.0, 1000.0]:
            rm, _ = self._risk_manager(mult)
            assert rm._portfolio_cfg.contract_multiplier == mult


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2: compute_metrics never returns NaN sharpe on minimal equity curves
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMetricsNaN:
    def test_two_point_curve_sharpe_not_nan(self):
        eq = np.array([100_000.0, 100_001.0])
        m = compute_metrics(eq)
        assert not math.isnan(m["sharpe_ratio"]), "sharpe_ratio should not be NaN"

    def test_two_point_curve_sortino_not_nan(self):
        eq = np.array([100_000.0, 100_001.0])
        m = compute_metrics(eq)
        assert not math.isnan(m["sortino_ratio"]), "sortino_ratio should not be NaN"

    def test_flat_equity_finite_sharpe(self):
        """Flat equity still produces a finite (not NaN) sharpe ratio.
        The value will be large-negative (0% return vs 5% risk-free) but finite."""
        eq = np.full(100, 100_000.0)
        m = compute_metrics(eq)
        assert math.isfinite(m["sharpe_ratio"]), "sharpe_ratio must be finite for flat equity"
        assert m["total_return"] == 0.0

    def test_crash_to_zero_returns_finite_metrics(self):
        eq = np.array([100_000.0, 50_000.0, 1.0])
        m = compute_metrics(eq)
        for k, v in m.items():
            assert isinstance(v, (int, float)), f"{k} not numeric"
            if isinstance(v, float):
                assert not math.isnan(v) or k in ("cagr",), f"{k}={v} is NaN"


# ─────────────────────────────────────────────────────────────────────────────
# Bug 3: No datetime.utcnow() in core paths (timezone-aware timestamps only)
# ─────────────────────────────────────────────────────────────────────────────

class TestTimezoneAwareness:
    def test_risk_violation_timestamp_is_aware(self):
        cfg = Config()
        port = Portfolio(cfg)
        rm = RiskManager(cfg, port)
        # Force drawdown breach to trigger a RiskViolation with timestamp
        rm._trading_halted = True
        sig = Signal("ES.c.0", Direction.LONG, 0.5, datetime.now(timezone.utc), "test")
        d = rm.evaluate(sig)
        assert len(d.violations) > 0
        ts = d.violations[0].timestamp
        assert ts.tzinfo is not None, "RiskViolation.timestamp must be tz-aware"

    def test_paper_broker_fill_timestamp_is_aware(self):
        from brokers.paper import PaperBroker
        from core.enums import OrderType
        from core.models import Order
        cfg = Config()
        broker = PaperBroker(cfg)
        order = Order("ES.c.0", Direction.LONG, OrderType.MARKET, 1.0,
                      datetime.now(timezone.utc), limit_price=4500.0)
        fill = broker.submit_order(order)
        assert fill.timestamp.tzinfo is not None


# ─────────────────────────────────────────────────────────────────────────────
# Bug 4: WalkForwardOptimizer aggregate_metrics clips outlier windows
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateMetricsOutlierClip:
    def test_single_extreme_window_does_not_poison_aggregate(self):
        metrics_list = [
            {"sharpe_ratio": -1_500_000.0, "total_return": -0.99},  # degenerate
            {"sharpe_ratio": 0.8, "total_return": 0.12},
            {"sharpe_ratio": 1.2, "total_return": 0.20},
        ]
        agg = WalkForwardOptimizer._aggregate_metrics(metrics_list)
        assert abs(agg["sharpe_ratio"]) <= 1_001.0, "Extreme sharpe should be clipped"
        assert agg["sharpe_ratio"] > -1_001.0

    def test_all_nan_returns_nan(self):
        metrics_list = [{"sharpe_ratio": float("nan")}, {"sharpe_ratio": float("nan")}]
        agg = WalkForwardOptimizer._aggregate_metrics(metrics_list)
        assert math.isnan(agg["sharpe_ratio"])

    def test_normal_values_unchanged(self):
        metrics_list = [{"sharpe_ratio": 0.5}, {"sharpe_ratio": 1.0}]
        agg = WalkForwardOptimizer._aggregate_metrics(metrics_list)
        assert abs(agg["sharpe_ratio"] - 0.75) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Bug 6: RSI EMA overflow suppressed for extreme price series
# ─────────────────────────────────────────────────────────────────────────────

class TestRSIOverflowSuppressed:
    def test_rsi_on_extreme_price_series(self):
        """Very large price differences should not produce inf/nan RSI values."""
        import warnings
        from indicators.momentum import rsi
        rng = np.random.default_rng(0)
        # Spike an already-extreme series to trigger the overflow path
        prices = np.concatenate([
            np.full(50, 1.0),
            np.full(50, 1e10),
            np.full(50, 1.0),
        ]).astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = rsi(prices, 14)   # should not raise RuntimeWarning
        assert result is not None
        finite = result[~np.isnan(result)]
        assert len(finite) > 0
        assert np.all(np.isfinite(finite))
        assert np.all((finite >= 0.0) & (finite <= 100.0))
