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
from silver_ensemble.engine import BacktestEngine
from silver_ensemble.forecasts import compute_all_forecasts, forecast_tsmom
from silver_ensemble.models import Direction, load_config
from silver_ensemble.strategy import SilverEnsembleStrategy
from silver_ensemble.validation import deflated_sharpe_ratio, validate_ensemble


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
    }
    f_star, agreement, _, _ = blend_forecasts(forecasts, cfg.ensemble)
    assert float(np.nanmean(agreement)) < cfg.ensemble.agreement_min + 0.05
    assert float(np.nanmean(np.abs(f_star))) < 1.0


def test_agreement_perfect() -> None:
    mat = np.ones((10, 5)) * 10.0
    a = agreement_ratio(mat)
    assert np.allclose(a, 1.0)


def test_feature_matrix_has_gs_ratio() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_si(300, seed=5), "SI")
    feats = build_feature_matrix(bars, cfg)
    assert "gs_ratio" in feats
    assert "real_yield_chg" in feats
    assert "gold_close" in feats
    assert feats["gs_ratio"].shape[0] == len(bars)
    assert np.isfinite(feats["gs_ratio"][50:]).any()


def test_backtest_synthetic_runs() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    df = make_synthetic_si(n_days=800, seed=7)
    bars = dataframe_to_bars(df, "SI")
    result = BacktestEngine(cfg).run(bars)
    assert result.equity_curve.shape[0] == len(bars)
    assert "sharpe" in result.metrics
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
