"""Unit tests for engine/optimizer.py — search space, grid, walk-forward."""

from __future__ import annotations

import pytest

from core.config import Config
from engine.optimizer import (
    GridOptimizer,
    STRATEGY_PARAM_SPACES,
    WalkForwardOptimizer,
    build_param_grid_from_yaml,
    build_search_space_from_yaml,
    load_optuna_bounds,
    HAS_OPTUNA,
)
from strategies.mean_reversion import MeanReversionRSI

from tests.conftest import make_bars

OPTUNA_PATH = "config/optuna.yaml"


class TestSearchSpaceLoaders:
    def test_load_optuna_bounds(self):
        bounds = load_optuna_bounds(OPTUNA_PATH)
        assert "rsi_period" in bounds
        assert bounds["rsi_period"]["type"] == "int"

    def test_build_search_space_skips_unknown(self):
        space = build_search_space_from_yaml(["rsi_period", "does_not_exist"], OPTUNA_PATH)
        assert "rsi_period" in space
        assert "does_not_exist" not in space

    def test_build_param_grid_respects_steps(self):
        grid = build_param_grid_from_yaml(["rsi_period", "bb_std"], steps=4, path=OPTUNA_PATH)
        assert len(grid["bb_std"]) == 4
        # int grid is de-duplicated but should not exceed steps
        assert len(grid["rsi_period"]) <= 4
        assert all(isinstance(v, int) for v in grid["rsi_period"])

    def test_param_registry_known_strategies(self):
        assert "MeanReversionRSI" in STRATEGY_PARAM_SPACES
        assert "rsi_period" in STRATEGY_PARAM_SPACES["MeanReversionRSI"]


class TestGridOptimizer:
    def test_combinations_cartesian_product(self):
        cfg = Config()
        grid = {"rsi_period": [10, 20], "rsi_oversold": [25.0, 35.0]}
        opt = GridOptimizer(cfg, MeanReversionRSI, grid)
        combos = opt.combinations()
        assert len(combos) == 4

    def test_run_returns_ranked_trials(self):
        cfg = Config()
        bars = {"ES.c.0": make_bars(n=300, seed=7)}
        grid = {"rsi_period": [10, 20], "rsi_oversold": [25.0, 35.0]}
        opt = GridOptimizer(cfg, MeanReversionRSI, grid, objective_metric="sharpe_ratio")
        result = opt.run(bars)
        assert result.n_combinations == 4
        assert len(result.trials) == 4
        assert result.best_params in [t.params for t in result.trials]
        # Trials sorted descending by objective (NaNs last)
        import numpy as np
        finite = [t.objective_value for t in result.trials if not np.isnan(t.objective_value)]
        assert finite == sorted(finite, reverse=True)

    def test_run_raises_when_grid_too_large(self):
        cfg = Config()
        bars = {"ES.c.0": make_bars(n=100)}
        grid = {"rsi_period": list(range(10, 20)), "rsi_oversold": [20.0, 30.0, 40.0]}
        opt = GridOptimizer(cfg, MeanReversionRSI, grid, max_combinations=5)
        with pytest.raises(ValueError):
            opt.run(bars)


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
class TestWalkForwardOptimizer:
    def test_run_produces_windows_and_aggregate(self):
        cfg = Config()
        bars = {"ES.c.0": make_bars(n=600, seed=11)}
        space = build_search_space_from_yaml(["rsi_period", "rsi_oversold"], OPTUNA_PATH)
        wfo = WalkForwardOptimizer(cfg, MeanReversionRSI, space, objective_metric="sharpe_ratio")
        result = wfo.run(bars, n_windows=2, in_sample_ratio=0.7, n_trials=3, timeout=30)
        assert len(result.windows) == 2
        assert len(result.best_params_per_window) == 2
        for window in result.windows:
            assert window.oos_result is not None
        assert isinstance(result.aggregated_oos_metrics, dict)
