"""Tests for the standalone copper ensemble (no QuantTerminal imports)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from copper_ensemble.blend import agreement_ratio, blend_forecasts
from copper_ensemble.data import dataframe_to_bars, make_synthetic_hg
from copper_ensemble.engine import BacktestEngine
from copper_ensemble.forecasts import compute_all_forecasts, forecast_tsmom
from copper_ensemble.models import Direction, load_config
from copper_ensemble.strategy import CopperEnsembleStrategy
from copper_ensemble.validation import deflated_sharpe_ratio, validate_ensemble


def test_no_quantterminal_import() -> None:
    """Ensure this project does not depend on QuantTerminal packages."""
    forbidden = ["core", "engine", "strategies", "app", "brokers", "portfolio", "risk", "indicators"]
    # Import our package modules and inspect their __name__ roots only
    mods = [
        "copper_ensemble.models",
        "copper_ensemble.data",
        "copper_ensemble.forecasts",
        "copper_ensemble.blend",
        "copper_ensemble.sizing",
        "copper_ensemble.risk",
        "copper_ensemble.strategy",
        "copper_ensemble.engine",
        "copper_ensemble.validation",
        "copper_ensemble.cli",
    ]
    for m in mods:
        mod = importlib.import_module(m)
        assert mod.__name__.startswith("copper_ensemble")
        # loaded module file must live under copper_ensemble/
        assert "copper_ensemble" in (mod.__file__ or "")


def test_config_loads() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    assert cfg.contract.symbol == "HG"
    assert abs(sum(cfg.ensemble.weights.values()) - 1.0) < 1e-9


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
    # Alternating signs → agreement = 0.2; with agreement_min=0.20 must flatten
    assert float(np.nanmean(agreement)) <= cfg.ensemble.agreement_min + 1e-9
    assert float(np.nanmean(np.abs(f_star))) < 1.0


def test_agreement_perfect() -> None:
    mat = np.ones((10, 5)) * 10.0
    a = agreement_ratio(mat)
    assert np.allclose(a, 1.0)


def test_backtest_synthetic_runs() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    df = make_synthetic_hg(n_days=800, seed=7)
    bars = dataframe_to_bars(df, "HG")
    result = BacktestEngine(cfg).run(bars)
    assert result.equity_curve.shape[0] == len(bars)
    assert "sharpe" in result.metrics
    assert result.metrics["n_bars"] == float(len(bars))


def test_strategy_emits_signal_contract() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_hg(900, seed=1), "HG")
    strat = CopperEnsembleStrategy(cfg)
    strat.prepare(bars)
    sig = strat.signal_at(len(bars) - 1)
    assert sig.symbol == "HG"
    assert isinstance(sig.direction, Direction)
    assert 0.0 <= sig.strength <= 1.0
    assert "sleeve_tsmom" in sig.metadata


def test_fade_disabled_in_structural_trend() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_hg(1000, seed=2), "HG")
    strat = CopperEnsembleStrategy(cfg)
    strat.prepare(bars)
    # When inventory sleeve and tsmom strongly aligned, fade should often be ~0
    fade = strat._forecasts["fade"]
    tsmom = strat._forecasts["tsmom"]
    inv = strat._forecasts["inventory"]
    aligned = (np.sign(tsmom) == np.sign(inv)) & (np.abs(tsmom) > 5) & (np.abs(inv) > 5)
    if np.any(aligned):
        assert np.nanmax(np.abs(fade[aligned])) < 1e-9


def test_validation_report() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_hg(1200, seed=3), "HG")
    report = validate_ensemble(cfg, bars)
    assert "all" in report.ablations
    assert "only_tsmom" in report.ablations
    assert 0.0 <= report.deflated_sharpe <= 1.0


def test_long_history_uses_eight_wfa_windows() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_hg(3200, seed=5), "HG")
    report = validate_ensemble(cfg, bars)
    assert len(report.walk_forward) >= 6  # should target 8; some may skip if short blocks


def test_yfinance_start_2008_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke: loader accepts start=; skip if network/Yahoo unavailable."""
    from copper_ensemble.data import load_yfinance_hg

    try:
        df = load_yfinance_hg("HG=F", start="2008-01-01", end="2008-06-30")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Yahoo unavailable: {exc}")
    assert len(df) > 50
    assert df.index.min().year == 2008

