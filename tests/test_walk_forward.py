"""Tests for anchored/rolling WFO builders, Ulcer metrics, and gates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import pytest

from core.config import Config
from core.enums import AssetClass
from core.models import Bar
from engine.metrics import compute_metrics
from engine.optimizer import WFOWindow
from engine.walk_forward import (
    GateThresholds,
    build_anchored_windows,
    build_rolling_windows,
    evaluate_gates,
    stitch_oos_equity,
    years_to_bars,
)


def _bars(n: int, symbol: str = "CL.c.0") -> Dict[str, List[Bar]]:
    base = datetime(2008, 1, 1, tzinfo=timezone.utc)
    out: List[Bar] = []
    for i in range(n):
        px = 70.0 + i * 0.01
        out.append(
            Bar(
                symbol=symbol,
                timestamp=base + timedelta(days=i),
                open=px,
                high=px + 0.2,
                low=px - 0.2,
                close=px,
                volume=1000.0,
                asset_class=AssetClass.FUTURES,
                contract_multiplier=1000.0,
            )
        )
    return {symbol: out}


class TestUlcerMetrics:
    def test_ulcer_keys_present(self):
        eq = np.linspace(100_000, 120_000, 252)
        m = compute_metrics(eq)
        assert "ulcer_index" in m
        assert "ulcer_performance_index" in m

    def test_flat_equity_ulcer_near_zero(self):
        eq = np.full(100, 100_000.0)
        m = compute_metrics(eq)
        assert m["ulcer_index"] == pytest.approx(0.0, abs=1e-9)

    def test_drawdown_path_positive_ulcer(self):
        # Rise then 20% drawdown then recover
        up = np.linspace(100, 120, 50)
        down = np.linspace(120, 96, 30)
        rec = np.linspace(96, 110, 40)
        eq = np.concatenate([up, down, rec])
        m = compute_metrics(eq)
        assert m["ulcer_index"] > 0.0
        assert m["max_drawdown"] < 0.0

    def test_upi_positive_on_uptrend(self):
        eq = np.linspace(100_000, 130_000, 252)
        m = compute_metrics(eq)
        # Mild path → small ulcer; CAGR > 0 ⇒ UPI > 0 when UI ~ 0 uses eps
        assert m["cagr"] > 0.0
        assert m["ulcer_performance_index"] > 0.0


class TestWindowBuilders:
    def test_anchored_start_fixed(self):
        bars = _bars(2000)
        windows = build_anchored_windows(bars, min_is_bars=500, oos_bars=100, step_bars=100)
        assert len(windows) >= 2
        # Every IS starts at index 0 of the original series
        first_ts = bars["CL.c.0"][0].timestamp
        for w in windows:
            assert w.is_bars["CL.c.0"][0].timestamp == first_ts
            assert len(w.oos_bars["CL.c.0"]) == 100
            # Expanding IS
        assert len(windows[1].is_bars["CL.c.0"]) > len(windows[0].is_bars["CL.c.0"])

    def test_rolling_fixed_is_and_purge(self):
        bars = _bars(2500)
        windows = build_rolling_windows(
            bars, is_bars=500, oos_bars=100, step_bars=100, purge_bars=5
        )
        assert len(windows) >= 2
        for w in windows:
            assert len(w.is_bars["CL.c.0"]) == 500
            assert len(w.oos_bars["CL.c.0"]) == 100
            # Purge: OOS starts 5 bars after IS end
            is_end = w.is_bars["CL.c.0"][-1].timestamp
            oos_start = w.oos_bars["CL.c.0"][0].timestamp
            assert oos_start > is_end

    def test_years_to_bars(self):
        assert years_to_bars(1) == 252
        assert years_to_bars(3) == 756


class TestGatesAndStitch:
    def test_evaluate_gates_pass(self):
        metrics = {
            "sharpe_ratio": 0.5,
            "ulcer_performance_index": 1.2,
            "cagr": 0.08,
            "max_drawdown": -0.12,
        }
        g = evaluate_gates(metrics, GateThresholds(max_dd_limit=0.30))
        assert g.passed
        assert g.failures == []

    def test_evaluate_gates_fail_dd(self):
        metrics = {
            "sharpe_ratio": 1.0,
            "ulcer_performance_index": 1.0,
            "cagr": 0.1,
            "max_drawdown": -0.45,
        }
        g = evaluate_gates(metrics, GateThresholds(max_dd_limit=0.30))
        assert not g.passed
        assert any("max_drawdown" in f for f in g.failures)

    def test_evaluate_gates_require_yearly_profit(self):
        from engine.metrics import calendar_year_returns

        base = datetime(2010, 1, 1, tzinfo=timezone.utc)
        # 2010 up, 2011 down → fail year gate at 100% fraction
        n = 400
        eq = np.concatenate(
            [np.linspace(100, 120, 200), np.linspace(120, 90, 200)]
        )
        ts = np.array([base + timedelta(days=i) for i in range(n)], dtype=object)
        years = calendar_year_returns(eq, ts, min_bars=2)
        assert any(r <= 0 for r in years.values())
        metrics = {
            "sharpe_ratio": 0.5,
            "ulcer_performance_index": 0.5,
            "cagr": 0.05,
            "max_drawdown": -0.1,
        }
        g = evaluate_gates(
            metrics,
            GateThresholds(
                require_all_years_profitable=True,
                min_year_bars=2,
                min_year_pass_fraction=1.0,
                max_year_loss=-0.05,
            ),
            equity_curve=eq,
            equity_timestamps=ts,
        )
        assert not g.passed
        assert g.failures

    def test_calendar_year_returns_all_positive(self):
        from engine.metrics import all_calendar_years_profitable

        base = datetime(2012, 1, 1, tzinfo=timezone.utc)
        eq = np.linspace(100, 150, 500)
        ts = np.array([base + timedelta(days=i) for i in range(500)], dtype=object)
        ok, years, failures = all_calendar_years_profitable(
            eq, ts, min_bars=2, min_year_return=0.0
        )
        assert ok
        assert failures == []
        assert all(r > 0 for r in years.values())

    def test_stitch_oos_equity_compounds(self):
        from engine.backtest import BacktestResult

        w1 = WFOWindow(window_id=0, is_bars={}, oos_bars={})
        w1.oos_result = BacktestResult(
            metrics={},
            equity_curve=np.array([100.0, 110.0, 121.0]),
            trade_log=[],
            fills=[],
        )
        w2 = WFOWindow(window_id=1, is_bars={}, oos_bars={})
        w2.oos_result = BacktestResult(
            metrics={},
            equity_curve=np.array([50.0, 55.0]),  # +10%
            trade_log=[],
            fills=[],
        )
        path = stitch_oos_equity([w1, w2], initial_cash=1000.0)
        # First window: 1000 * 1.1 * 1.1 = 1210; then *1.1 = 1331
        assert path[0] == pytest.approx(1000.0)
        assert path[-1] == pytest.approx(1331.0, rel=1e-6)


class TestWalkForwardSmoke:
    def test_validator_synthetic_smoke(self):
        from engine.optimizer import STRATEGY_PARAM_SPACES, build_search_space_from_yaml
        from engine.walk_forward import WalkForwardValidator
        from strategies.cl_carry_momentum import CLCarryMomentum

        cfg = Config()
        cfg.portfolio.initial_cash = 350_000_000.0
        cfg.portfolio.contract_multiplier = 1000.0
        cfg.risk.halt_on_breach = False
        cfg.risk.max_daily_drawdown_pct = 0.99
        cfg.backtest.risk_free_rate = 0.0
        cfg.cl_validation.oos_warmup_bars = 80

        bars = _bars(2200)
        space = build_search_space_from_yaml(
            STRATEGY_PARAM_SPACES["CLCarryMomentum"]
        )
        validator = WalkForwardValidator(
            cfg, CLCarryMomentum, space, thresholds=GateThresholds()
        )
        report = validator.run_mode(
            bars,
            "rolling",
            n_trials=2,
            timeout=30,
            is_bars=500,
            oos_bars=100,
            step_bars=400,
            purge_bars=5,
        )
        assert len(report.wfo.windows) >= 1
        assert "passed" in report.gates.to_dict()
        # With warm-up, OOS equity should have length ~= oos_bars
        assert report.wfo.windows[0].oos_result is not None
        assert len(report.wfo.windows[0].oos_result.equity_curve) >= 50
