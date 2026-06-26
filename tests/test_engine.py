"""Unit tests for the engine/ module: metrics and backtesting."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from core.config import Config
from core.models import Bar
from engine.backtest import BacktestEngine
from engine.metrics import compute_metrics, max_consecutive_losses
from strategies.mean_reversion import MeanReversionRSI

from tests.conftest import make_bars


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_empty_on_short_series(self):
        result = compute_metrics(np.array([100.0]))
        assert result == {}

    def test_flat_equity_has_zero_return(self):
        eq = np.full(100, 100_000.0)
        metrics = compute_metrics(eq)
        assert abs(metrics["total_return"]) < 1e-6

    def test_growing_equity_positive_return(self):
        eq = np.linspace(100_000, 150_000, 252)
        metrics = compute_metrics(eq)
        assert metrics["total_return"] > 0.0

    def test_declining_equity_negative_return(self):
        eq = np.linspace(100_000, 80_000, 100)
        metrics = compute_metrics(eq)
        assert metrics["total_return"] < 0.0

    def test_max_drawdown_non_positive(self):
        eq = np.array([100, 110, 105, 90, 95, 100], dtype=float)
        metrics = compute_metrics(eq)
        assert metrics["max_drawdown"] <= 0.0

    def test_sharpe_ratio_positive_on_strong_uptrend(self):
        """A deterministic linear uptrend should always yield a positive Sharpe."""
        eq = np.linspace(100_000, 130_000, 252)  # steady 30% gain, zero volatility
        metrics = compute_metrics(eq)
        # With zero volatility the Sharpe approaches +inf; confirm it's clearly positive
        assert metrics["sharpe_ratio"] > 0.0

    def test_all_required_keys_present(self):
        eq = np.linspace(100, 200, 300)
        metrics = compute_metrics(eq)
        required_keys = {
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
            "var_95", "cvar_95",
        }
        assert required_keys.issubset(metrics.keys())

    def test_win_rate_in_range(self):
        eq = np.linspace(100_000, 120_000, 252)
        metrics = compute_metrics(eq)
        assert 0.0 <= metrics["win_rate"] <= 1.0


class TestMaxConsecutiveLosses:
    def test_all_gains_returns_zero(self):
        returns = np.array([0.01, 0.02, 0.005])
        assert max_consecutive_losses(returns) == 0

    def test_consecutive_losses(self):
        returns = np.array([0.01, -0.01, -0.02, -0.03, 0.01])
        assert max_consecutive_losses(returns) == 3

    def test_empty_returns_zero(self):
        assert max_consecutive_losses(np.array([])) == 0


# ── BacktestEngine tests ──────────────────────────────────────────────────────

class TestBacktestEngine:
    def test_run_returns_backtest_result(self, default_config: Config):
        engine = BacktestEngine(default_config, MeanReversionRSI)
        bars = make_bars(n=300)
        result = engine.run({"ES.c.0": bars})
        assert result is not None
        assert isinstance(result.equity_curve, np.ndarray)

    def test_equity_curve_starts_near_initial_cash(self, default_config: Config):
        engine = BacktestEngine(default_config, MeanReversionRSI)
        bars = make_bars(n=300)
        result = engine.run({"ES.c.0": bars})
        # First equity snapshot should be near initial cash
        if len(result.equity_curve) > 0:
            assert abs(result.equity_curve[0] - 100_000.0) / 100_000.0 < 0.5

    def test_metrics_populated(self, default_config: Config):
        engine = BacktestEngine(default_config, MeanReversionRSI)
        bars = make_bars(n=300)
        result = engine.run({"ES.c.0": bars})
        assert len(result.metrics) > 0

    def test_parameter_override(self, default_config: Config):
        engine = BacktestEngine(default_config, MeanReversionRSI)
        bars = make_bars(n=300)
        # Override RSI period and verify no crash
        result = engine.run({"ES.c.0": bars}, strategy_params={"rsi_period": 10})
        assert result is not None

    def test_multi_symbol_run(self, default_config: Config):
        engine = BacktestEngine(default_config, MeanReversionRSI)
        bars_es = make_bars(n=300, symbol="ES.c.0")
        bars_nq = make_bars(n=300, symbol="NQ.c.0", start_price=15000.0)
        result = engine.run({"ES.c.0": bars_es, "NQ.c.0": bars_nq})
        assert result is not None
        assert len(result.equity_curve) > 0
