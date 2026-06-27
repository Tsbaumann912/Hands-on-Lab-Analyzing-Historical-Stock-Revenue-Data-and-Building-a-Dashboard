"""Integration tests for the Strategies-tab analysis bridge (data_service)."""

from __future__ import annotations

from app.data_service import (
    get_analysis_config,
    load_futures_bars_for_ui,
    run_strategy_analysis,
)


class TestAnalysisConfig:
    def test_contracts_loaded(self):
        cfg = get_analysis_config()
        assert "ES" in cfg.contracts
        assert cfg.contracts["CL"].contract_multiplier == 1000.0

    def test_analysis_defaults(self):
        cfg = get_analysis_config()
        assert cfg.analysis.history_start_date == "2000-01-01"
        assert "sharpe_ratio" in cfg.analysis.objective_metrics


class TestLoadFuturesBars:
    def test_synthetic_source_spans_range(self):
        df, meta = load_futures_bars_for_ui(
            "ES", timeframe="1d", start_date="2010-01-01",
            end_date="2015-01-01", data_source="synthetic",
        )
        assert not df.empty
        assert meta["data_source"] == "synthetic"
        assert meta["range_from"] >= "2010-01-01"
        assert meta["range_to"] <= "2015-01-02"

    def test_bar_cap_applied(self):
        df, meta = load_futures_bars_for_ui(
            "ES", timeframe="1d", start_date="2000-01-01",
            end_date="2020-01-01", data_source="synthetic", max_bars=500,
        )
        assert len(df) == 500
        assert meta["bar_cap_applied"] == 500


class TestRunStrategyAnalysis:
    def test_backtest_mode(self):
        result = run_strategy_analysis(
            mode="backtest", symbol="ES", strategy_name="MeanReversionRSI",
            timeframe="1d", start_date="2012-01-01", end_date="2018-01-01",
            data_source="synthetic", initial_cash=100_000,
            params={"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20},
        )
        assert result.get("error") is None
        assert result["mode"] == "backtest"
        assert "sharpe_ratio" in result["metrics"]
        assert not result["equity_curve"].empty

    def test_optimize_mode(self):
        result = run_strategy_analysis(
            mode="optimize", symbol="GC", strategy_name="MeanReversionRSI",
            timeframe="1d", start_date="2012-01-01", end_date="2017-01-01",
            data_source="synthetic", optimize_params=["rsi_period", "rsi_oversold"],
            objective_metric="sharpe_ratio", grid_steps=2,
        )
        assert result.get("error") is None
        assert result["mode"] == "optimize"
        assert result["n_combinations"] == 4
        assert len(result["trials"]) == 4
        assert result["best_params"]

    def test_walk_forward_mode(self):
        result = run_strategy_analysis(
            mode="walk_forward", symbol="ES", strategy_name="MeanReversionRSI",
            timeframe="1d", start_date="2008-01-01", end_date="2018-01-01",
            data_source="synthetic", optimize_params=["rsi_period", "rsi_oversold"],
            objective_metric="sharpe_ratio", n_windows=2, n_trials=4,
        )
        assert result.get("error") is None
        assert result["mode"] == "walk_forward"
        assert len(result["windows"]) == 2
        assert "sharpe_ratio" in result["aggregated_oos_metrics"]

    def test_insufficient_data_returns_error(self):
        result = run_strategy_analysis(
            mode="backtest", symbol="ES", strategy_name="MeanReversionRSI",
            timeframe="1d", start_date="2020-01-01", end_date="2020-01-15",
            data_source="synthetic",
        )
        assert "error" in result
