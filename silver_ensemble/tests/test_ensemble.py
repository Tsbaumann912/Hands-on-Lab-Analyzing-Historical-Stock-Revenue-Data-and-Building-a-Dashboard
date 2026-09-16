"""Tests for the standalone silver ensemble (no QuantTerminal imports)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silver_ensemble.blend import agreement_ratio, blend_forecasts
from silver_ensemble.data import build_feature_matrix, dataframe_to_bars, make_synthetic_si
from silver_ensemble.engine import BacktestEngine, _compute_metrics
from silver_ensemble.forecasts import compute_all_forecasts, forecast_tsmom
from silver_ensemble.models import Direction, load_config
from silver_ensemble.strategy import SilverEnsembleStrategy
from silver_ensemble.validation import (
    build_anchored_windows,
    build_rolling_windows,
    deflated_sharpe_ratio,
    run_institutional_wfo,
    stitch_oos_equity,
    validate_ensemble,
)


def test_no_quantterminal_import() -> None:
    """Ensure this project does not depend on QuantTerminal packages."""
    mods = [
        "silver_ensemble.models",
        "silver_ensemble.data",
        "silver_ensemble.forecasts",
        "silver_ensemble.blend",
        "silver_ensemble.sizing",
        "silver_ensemble.risk",
        "silver_ensemble.strategy",
        "silver_ensemble.engine",
        "silver_ensemble.validation",
        "silver_ensemble.cli",
        "silver_ensemble.indicators",
        "silver_ensemble.optimize",
    ]
    for m in mods:
        mod = importlib.import_module(m)
        assert mod.__name__.startswith("silver_ensemble")
        assert "silver_ensemble" in (mod.__file__ or "")


def test_config_loads() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    assert cfg.contract.symbol == "SI"
    assert cfg.contract.multiplier == 5000.0
    assert abs(sum(cfg.ensemble.weights.values()) - 1.0) < 1e-9
    assert 126 in cfg.ensemble.horizons_days
    assert cfg.portfolio.initial_cash == 350_000_000.0
    assert cfg.risk.max_contracts == 2500
    assert cfg.risk.max_position_size_pct >= 100.0
    assert cfg.validation.max_drawdown_gate == 0.30
    assert cfg.backtest.data_start == "2008-01-01"
    assert cfg.backtest.is_years == 3
    assert "stoch_ma" in cfg.ensemble.weights
    assert cfg.ensemble.ma_fast < cfg.ensemble.ma_slow


def test_tsmom_signs_on_trending_series() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    rets = np.full(400, 0.01)
    f = forecast_tsmom(rets, cfg.ensemble)
    assert np.nanmean(f[300:]) > 5.0


def test_disagreement_flattens() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    n = 50
    forecasts = {
        "tsmom": np.full(n, 10.0),
        "carry": np.full(n, -10.0),
        "basis_mom": np.full(n, 10.0),
        "inventory": np.full(n, -10.0),
        "fade": np.full(n, 10.0),
        "stoch_ma": np.full(n, -10.0),
    }
    f_star, agreement, _, _ = blend_forecasts(forecasts, cfg.ensemble)
    assert float(np.nanmean(agreement)) < cfg.ensemble.agreement_min + 0.05
    assert float(np.nanmean(np.abs(f_star))) < 1.0


def test_agreement_perfect() -> None:
    mat = np.ones((10, 6)) * 10.0
    a = agreement_ratio(mat)
    assert np.allclose(a, 1.0)


def test_feature_matrix_has_gs_ratio() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_si(300, seed=5), "SI")
    feats = build_feature_matrix(bars, cfg)
    assert "gs_ratio" in feats
    assert "real_yield_chg" in feats
    assert "gold_close" in feats
    assert "ma_fast" in feats
    assert "ma_slow" in feats
    assert "stoch_rsi" in feats
    assert "rsi" in feats
    assert feats["gs_ratio"].shape[0] == len(bars)
    assert np.isfinite(feats["gs_ratio"][50:]).any()
    assert np.isfinite(feats["stoch_rsi"][80:]).any()


def test_backtest_synthetic_runs() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    df = make_synthetic_si(n_days=800, seed=7)
    bars = dataframe_to_bars(df, "SI")
    result = BacktestEngine(cfg).run(bars)
    assert result.equity_curve.shape[0] == len(bars)
    assert "sharpe" in result.metrics
    assert "cagr" in result.metrics
    assert "ulcer_index" in result.metrics
    assert "upi" in result.metrics
    assert result.metrics["n_bars"] == float(len(bars))


def test_strategy_emits_signal_contract() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_si(900, seed=1), "SI")
    strat = SilverEnsembleStrategy(cfg)
    strat.prepare(bars)
    sig = strat.signal_at(len(bars) - 1)
    assert sig.symbol == "SI"
    assert isinstance(sig.direction, Direction)
    assert 0.0 <= sig.strength <= 1.0
    assert "sleeve_tsmom" in sig.metadata
    assert sig.strategy_name == "SilverEnsemble"


def test_fade_disabled_in_structural_trend() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_si(1000, seed=2), "SI")
    strat = SilverEnsembleStrategy(cfg)
    strat.prepare(bars)
    fade = strat._forecasts["fade"]
    tsmom = strat._forecasts["tsmom"]
    inv = strat._forecasts["inventory"]
    aligned = (np.sign(tsmom) == np.sign(inv)) & (np.abs(tsmom) > 5) & (np.abs(inv) > 5)
    if np.any(aligned):
        assert np.nanmax(np.abs(fade[aligned])) < 1e-9


def test_validation_report() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_si(1200, seed=3), "SI")
    report = validate_ensemble(cfg, bars)
    assert "all" in report.ablations
    assert "only_tsmom" in report.ablations
    assert 0.0 <= report.deflated_sharpe <= 1.0


def test_dsr_monotonic_in_trials() -> None:
    dsr_few = deflated_sharpe_ratio(1.5, n_obs=500, n_trials=2)
    dsr_many = deflated_sharpe_ratio(1.5, n_obs=500, n_trials=200)
    assert dsr_few >= dsr_many


def test_notional_uses_si_multiplier() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    assert cfg.contract.multiplier == 5000.0
    assert abs(cfg.contract.tick_value - 25.0) < 1e-9


def test_metrics_cagr_ulcer_on_monotone_equity() -> None:
    # Steady growth: positive CAGR, near-zero Ulcer, finite UPI
    eq = 1_000_000.0 * np.cumprod(np.full(504, 1.001))
    eq = np.concatenate([[1_000_000.0], eq])
    m = _compute_metrics(eq)
    assert m["cagr"] > 0.0
    assert m["ulcer_index"] < 1.0
    assert m["max_drawdown"] < 0.01
    assert m["upi"] >= 0.0


def test_metrics_drawdown_raises_ulcer() -> None:
    eq = np.array([100.0, 110.0, 90.0, 95.0, 100.0], dtype=np.float64)
    m = _compute_metrics(eq)
    assert m["max_drawdown"] > 0.15
    assert m["ulcer_index"] > 0.0


def test_rolling_window_indices_purge() -> None:
    wins = build_rolling_windows(n_bars=3000, is_bars=756, oos_bars=252, step_bars=252, purge=5)
    assert len(wins) >= 5
    for is_s, is_e, oos_s, oos_e in wins:
        assert is_e - is_s == 756
        assert oos_e - oos_s == 252
        assert oos_s == is_e + 5
        assert is_s >= 0
        assert oos_e <= 3000


def test_anchored_window_indices_expand() -> None:
    wins = build_anchored_windows(n_bars=3000, min_is_bars=756, oos_bars=252, step_bars=252, purge=5)
    assert len(wins) >= 5
    for is_s, is_e, oos_s, oos_e in wins:
        assert is_s == 0
        assert oos_s == is_e + 5
        assert oos_e - oos_s == 252
        assert is_e >= 756
    # IS expands over windows
    assert wins[-1][1] > wins[0][1]


def test_stitch_oos_equity_compounds() -> None:
    a = np.array([100.0, 110.0, 121.0])
    b = np.array([50.0, 55.0])
    st = stitch_oos_equity([a, b], initial_cash=100.0)
    assert st[0] == pytest.approx(100.0)
    # After first segment ends at 121; second grows 10% → 133.1
    assert st[-1] == pytest.approx(133.1)


def test_institutional_wfo_synthetic_schema() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    # ≥ 3y IS + several OOS years
    bars = dataframe_to_bars(make_synthetic_si(n_days=2800, seed=11), "SI")
    report = run_institutional_wfo(cfg, bars)
    assert report.account_size == 350_000_000.0
    assert report.n_bars == len(bars)
    assert len(report.rolling.windows) >= 1
    assert len(report.anchored.windows) >= 1
    for key in ("sharpe", "cagr", "upi", "max_drawdown", "ulcer_index"):
        assert key in report.rolling.stitched_metrics
        assert key in report.anchored.stitched_metrics
    # Anchored IS always starts at 0
    assert all(w.is_start == 0 for w in report.anchored.windows)
    # Gates evaluated (passed may be True or False on synthetic)
    assert isinstance(report.passed, bool)
    assert isinstance(report.rolling.passed, bool)
    assert isinstance(report.anchored.passed, bool)


def test_calendar_year_returns_and_optimize_apply() -> None:
    from silver_ensemble.optimize import (
        apply_trial_params,
        calendar_year_returns,
        evaluate_config,
        mean_calendar_year_return,
    )

    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_si(800, seed=9), "SI")
    res = BacktestEngine(cfg).run(bars)
    yrs = calendar_year_returns(bars, res.equity_curve)
    assert len(yrs) >= 2
    assert abs(mean_calendar_year_return(bars, res.equity_curve) - float(np.mean(list(yrs.values())))) < 1e-12

    params = {
        "w_tsmom": 0.3,
        "w_carry": 0.15,
        "w_basis_mom": 0.1,
        "w_inventory": 0.15,
        "w_fade": 0.1,
        "w_stoch_ma": 0.2,
        "vol_target_annual": 0.08,
        "kelly_fraction": 0.25,
    }
    cfg2 = apply_trial_params(cfg, params)
    assert abs(sum(cfg2.ensemble.weights.values()) - 1.0) < 1e-9
    assert cfg2.ensemble.vol_target_annual == 0.08
    assert cfg2.risk.max_position_size_pct >= 100.0
    ev = evaluate_config(cfg2, bars)
    assert "avg_calendar_year_return" in ev
    assert "min_year_return" in ev
    assert "all_years_positive" in ev
    assert "total_return" in ev
    assert ev["max_drawdown"] >= 0.0
    assert ev["pct_years_positive"] >= 0.0
    assert ev["pct_years_positive"] <= 1.0


def test_stoch_ma_sleeve_signs() -> None:
    from silver_ensemble.forecasts import forecast_stoch_ma

    cfg = load_config(ROOT / "config" / "default.yaml")
    n = 200
    ma_fast = np.linspace(20.0, 30.0, n)  # rising above slow
    ma_slow = np.full(n, 22.0)
    stoch = np.full(n, 0.15)  # oversold
    f = forecast_stoch_ma(ma_fast, ma_slow, stoch, cfg.ensemble)
    assert np.nanmean(f[50:]) > 5.0
