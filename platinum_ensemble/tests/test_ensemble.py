"""Tests for the standalone platinum ensemble (no QuantTerminal imports)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platinum_ensemble.blend import agreement_ratio, blend_forecasts
from platinum_ensemble.data import dataframe_to_bars, make_synthetic_pl
from platinum_ensemble.engine import BacktestEngine
from platinum_ensemble.forecasts import compute_all_forecasts, forecast_pl_gc_rv, forecast_tsmom
from platinum_ensemble.models import Direction, load_config
from platinum_ensemble.strategy import PlatinumEnsembleStrategy
from platinum_ensemble.validation import deflated_sharpe_ratio, validate_ensemble


def test_no_quantterminal_import() -> None:
    """Ensure this project does not depend on QuantTerminal packages."""
    mods = [
        "platinum_ensemble.models",
        "platinum_ensemble.data",
        "platinum_ensemble.forecasts",
        "platinum_ensemble.blend",
        "platinum_ensemble.sizing",
        "platinum_ensemble.risk",
        "platinum_ensemble.strategy",
        "platinum_ensemble.engine",
        "platinum_ensemble.validation",
        "platinum_ensemble.cli",
    ]
    for m in mods:
        mod = importlib.import_module(m)
        assert mod.__name__.startswith("platinum_ensemble")
        assert "platinum_ensemble" in (mod.__file__ or "")


def test_config_loads() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    assert cfg.contract.symbol == "PL"
    assert cfg.contract.multiplier == 50.0
    assert abs(sum(cfg.ensemble.weights.values()) - 1.0) < 1e-9
    assert "pl_gc_rv" in cfg.ensemble.weights
    assert cfg.portfolio.initial_cash == 350_000_000.0
    assert cfg.risk.max_contracts == 10_000
    assert cfg.backtest.anchored_initial_is_bars == 756
    assert cfg.validation.max_oos_drawdown == 0.30
    assert cfg.validation.data_start == "2008-01-01"


def test_tsmom_signs_on_trending_series() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    rets = np.full(400, 0.01)
    f = forecast_tsmom(rets, cfg.ensemble)
    assert np.nanmean(f[300:]) > 5.0


def test_pl_gc_rv_buys_cheap_platinum() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    n = 400
    gold = np.full(n, 2000.0)
    # PL collapses vs gold in the second half → should go long PL (positive forecast)
    pl = np.concatenate([np.full(200, 1000.0), np.linspace(1000.0, 700.0, 200)])
    f = forecast_pl_gc_rv(pl, gold, cfg.ensemble)
    assert np.nanmean(f[320:]) > 0.0


def test_disagreement_flattens() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    n = 50
    forecasts = {
        "tsmom": np.full(n, 10.0),
        "carry": np.full(n, -10.0),
        "pl_gc_rv": np.full(n, 10.0),
        "inventory": np.full(n, -10.0),
        "fade": np.full(n, 10.0),
    }
    f_star, agreement, _, _ = blend_forecasts(forecasts, cfg.ensemble)
    assert float(np.nanmean(agreement)) < cfg.ensemble.agreement_min + 0.05
    assert float(np.nanmean(np.abs(f_star))) < 1.0


def test_agreement_perfect() -> None:
    mat = np.ones((10, 5)) * 10.0
    a = agreement_ratio(mat)
    assert np.allclose(a, 1.0)


def test_backtest_synthetic_runs() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    df = make_synthetic_pl(n_days=800, seed=7)
    bars = dataframe_to_bars(df, "PL")
    result = BacktestEngine(cfg).run(bars)
    assert result.equity_curve.shape[0] == len(bars)
    assert "sharpe" in result.metrics
    assert result.metrics["n_bars"] == float(len(bars))


def test_strategy_emits_signal_contract() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_pl(900, seed=1), "PL")
    strat = PlatinumEnsembleStrategy(cfg)
    strat.prepare(bars)
    sig = strat.signal_at(len(bars) - 1)
    assert sig.symbol == "PL"
    assert isinstance(sig.direction, Direction)
    assert 0.0 <= sig.strength <= 1.0
    assert "sleeve_tsmom" in sig.metadata
    assert "sleeve_pl_gc_rv" in sig.metadata


def test_fade_disabled_in_structural_trend() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_pl(1000, seed=2), "PL")
    strat = PlatinumEnsembleStrategy(cfg)
    strat.prepare(bars)
    fade = strat._forecasts["fade"]
    tsmom = strat._forecasts["tsmom"]
    inv = strat._forecasts["inventory"]
    aligned = (np.sign(tsmom) == np.sign(inv)) & (np.abs(tsmom) > 5) & (np.abs(inv) > 5)
    if np.any(aligned):
        assert np.nanmax(np.abs(fade[aligned])) < 1e-9


def test_validation_report() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_pl(1200, seed=3), "PL")
    report = validate_ensemble(cfg, bars)
    assert "all" in report.ablations
    assert "only_tsmom" in report.ablations
    assert "only_pl_gc_rv" in report.ablations
    assert 0.0 <= report.deflated_sharpe <= 1.0


def test_dsr_monotonic_in_trials() -> None:
    dsr_few = deflated_sharpe_ratio(1.5, n_obs=500, n_trials=2)
    dsr_many = deflated_sharpe_ratio(1.5, n_obs=500, n_trials=200)
    assert dsr_few >= dsr_many


def test_notional_uses_pl_multiplier() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    assert cfg.contract.multiplier == 50.0
    assert cfg.contract.tick_value == 5.0


def test_metrics_include_cagr_ulcer_upi() -> None:
    from platinum_ensemble.engine import _compute_metrics

    equity = np.cumprod(1.0 + np.full(500, 0.001)) * 350_000_000.0
    m = _compute_metrics(equity)
    assert "cagr" in m and m["cagr"] > 0
    assert "ulcer_index" in m
    assert "upi" in m
    assert m["max_drawdown"] >= 0.0


def test_anchored_wfo_expands_from_origin() -> None:
    from platinum_ensemble.engine import BacktestEngine
    from platinum_ensemble.validation import anchored_walk_forward

    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_pl(2200, seed=11), "PL")
    engine = BacktestEngine(cfg)
    windows, eqs = anchored_walk_forward(
        engine,
        bars,
        initial_is_bars=756,
        oos_bars=252,
        purge=5,
    )
    assert len(windows) >= 2
    assert all(w.is_start == 0 for w in windows)
    # Expanding IS ends
    is_ends = [w.is_end for w in windows]
    assert is_ends == sorted(is_ends)
    assert is_ends[0] == 756
    assert is_ends[1] == 756 + 252
    assert len(eqs) == len(windows)


def test_rolling_wfo_fixed_is_length() -> None:
    from platinum_ensemble.engine import BacktestEngine
    from platinum_ensemble.validation import rolling_walk_forward

    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_pl(2200, seed=12), "PL")
    engine = BacktestEngine(cfg)
    windows, eqs = rolling_walk_forward(
        engine,
        bars,
        is_bars=756,
        oos_bars=252,
        step_bars=252,
        purge=5,
    )
    assert len(windows) >= 2
    assert all((w.is_end - w.is_start) == 756 for w in windows)
    assert windows[1].is_start == windows[0].is_start + 252
    assert len(eqs) == len(windows)


def test_max_dd_gate_fails_when_breached() -> None:
    from platinum_ensemble.validation import evaluate_stitched_gates

    cfg = load_config(ROOT / "config" / "default.yaml")
    metrics = {
        "sharpe": 1.0,
        "cagr": 0.10,
        "upi": 1.0,
        "max_drawdown": 0.35,
    }
    passed, notes = evaluate_stitched_gates(metrics, cfg)
    assert passed is False
    assert any("Max DD" in n for n in notes)


def test_institutional_wfo_uses_350m() -> None:
    from platinum_ensemble.validation import run_institutional_wfo

    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_pl(2200, seed=13), "PL")
    report = run_institutional_wfo(cfg, bars, cash=350_000_000.0)
    assert report.account_size == 350_000_000.0
    assert report.anchored.stitched_equity[0] == pytest.approx(350_000_000.0)
    assert "sharpe" in report.anchored.stitched_metrics
    assert "upi" in report.anchored.stitched_metrics
    assert "cagr" in report.rolling.stitched_metrics
    assert report.anchored.mode == "anchored"
    assert report.rolling.mode == "rolling"
