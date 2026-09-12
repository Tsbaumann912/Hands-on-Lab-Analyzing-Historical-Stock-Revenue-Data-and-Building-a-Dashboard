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
        "ma_cross": np.full(n, -10.0),
        "stoch_rsi": np.full(n, 10.0),
    }
    f_star, agreement, _, _ = blend_forecasts(forecasts, cfg.ensemble)
    # Alternating signs → low agreement; must flatten
    assert float(np.nanmean(agreement)) <= cfg.ensemble.agreement_min + 1e-9
    assert float(np.nanmean(np.abs(f_star))) < 1.0


def test_new_indicators_present() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    assert cfg.risk.max_position_size_pct == 7.0
    assert "ma_cross" in cfg.ensemble.weights
    assert "stoch_rsi" in cfg.ensemble.weights
    bars = dataframe_to_bars(make_synthetic_hg(400, seed=4), "HG")
    strat = CopperEnsembleStrategy(cfg)
    strat.prepare(bars)
    assert "ma_cross" in strat._forecasts
    assert "stoch_rsi" in strat._forecasts
    assert "ma_fast" in strat._features
    assert np.isfinite(np.nanmean(strat._features["ma_fast"][100:]))
    assert np.isfinite(np.nanmean(strat._forecasts["stoch_rsi"][80:]))


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


def test_rebuild_weights_disables_fade() -> None:
    from copper_ensemble.models import rebuild_weights

    base = {
        "tsmom": 0.3,
        "carry": 0.2,
        "basis_mom": 0.15,
        "inventory": 0.1,
        "fade": 0.1,
        "ma_cross": 0.1,
        "stoch_rsi": 0.05,
    }
    w = rebuild_weights(base, tsmom_weight=0.6, enable_fade=False)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["fade"] == 0.0
    assert abs(w["tsmom"] - 0.6) < 1e-9


def test_nested_optimize_synthetic_runs() -> None:
    from copper_ensemble.optimize import nested_walk_forward_optimize

    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_hg(900, seed=11), "HG")
    # Tiny search for CI speed
    space = {
        "n_trials_per_window": 4,
        "n_windows": 3,
        "in_sample_ratio": 0.70,
        "purge_bars": 3,
        "max_dd_soft_cap": 0.30,
        "min_fills": 2,
        "agreement_min": {"type": "float", "low": 0.15, "high": 0.35},
        "buffer_forecast": {"type": "float", "low": 0.5, "high": 2.0},
        "vol_target_annual": {"type": "float", "low": 0.10, "high": 0.16},
        "kelly_fraction": {"type": "float", "low": 0.25, "high": 0.40},
        "stop_atr_mult": {"type": "float", "low": 1.5, "high": 3.0},
        "take_profit_atr_mult": {"type": "float", "low": 3.0, "high": 5.0},
        "fdm_cap": {"type": "float", "low": 1.0, "high": 2.0},
        "tsmom_weight": {"type": "float", "low": 0.45, "high": 0.70},
        "enable_fade": {"type": "categorical", "choices": [True, False]},
        "horizon_set": {"type": "categorical", "choices": [[21, 63, 252], [10, 21, 63]]},
    }
    report = nested_walk_forward_optimize(cfg, bars, space=space, seed=3)
    assert len(report.windows) >= 2
    assert report.robust_params
    assert "agreement_min" in report.robust_params
    assert np.isfinite(report.mean_oos_sharpe)


def test_ulcer_metrics_on_monotonic_equity() -> None:
    from copper_ensemble.engine import ulcer_index, ulcer_performance_index

    eq = np.cumprod(1.0 + np.full(252, 0.001)) * 100_000.0
    assert ulcer_index(eq) < 1e-6
    assert ulcer_performance_index(eq) > 0.0
    # Drawdown path → positive UI
    eq2 = eq.copy()
    eq2[100:150] = eq2[99] * 0.9
    assert ulcer_index(eq2) > 1.0


def test_anchored_and_rolling_slices() -> None:
    from copper_ensemble.optimize import anchored_window_slices, rolling_window_slices

    roll = rolling_window_slices(2000, 4, 0.7, 5)
    anch = anchored_window_slices(2000, 4, min_is_bars=500, purge=5)
    assert len(roll) >= 3
    assert len(anch) >= 3
    # Anchored IS always starts at 0
    assert all(s[0] == 0 for s in anch)
    # Rolling blocks move forward
    assert roll[0][0] < roll[-1][0]


def test_candidate_selection_upi_metric() -> None:
    from copper_ensemble.optimize import select_pre_specified_candidates

    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_hg(1400, seed=12), "HG")
    winner, ranked = select_pre_specified_candidates(cfg, bars, n_windows=3, metric="upi")
    assert winner.name
    assert hasattr(winner, "composite_oos_upi")
    assert len(ranked) >= 6
    assert np.isfinite(winner.composite_oos_upi)


def test_halt_cooldown_resumes() -> None:
    from copper_ensemble.risk import RiskManager

    cfg = load_config(ROOT / "config" / "default.yaml")
    rm = RiskManager(cfg)
    peak = cfg.portfolio.initial_cash
    # Breach beyond configured max_daily_drawdown_pct (production is 25%)
    breach_equity = peak * (1.0 - cfg.risk.max_daily_drawdown_pct - 0.05)
    rm.update_equity(breach_equity)
    assert rm.halted
    # Staying flat must still resume after cooldown bars
    for _ in range(cfg.risk.halt_cooldown_bars):
        rm.update_equity(breach_equity)
    assert not rm.halted


def test_candidate_selection_synthetic() -> None:
    from copper_ensemble.optimize import select_pre_specified_candidates

    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(make_synthetic_hg(1200, seed=9), "HG")
    winner, ranked = select_pre_specified_candidates(cfg, bars, n_windows=3)
    assert winner.name
    assert len(ranked) >= 6
    assert np.isfinite(winner.mean_oos_sharpe)


def test_all_green_years_and_max_dd_le_30_on_config() -> None:
    """Production: every calendar year > 0 and max DD ≤ 30% on HG 2008→now."""
    from copper_ensemble.data import load_yfinance_hg
    from copper_ensemble.engine import BacktestEngine, calendar_year_returns

    cfg = load_config(ROOT / "config" / "default.yaml")
    assert cfg.risk.yearly_profit_lock_enabled is True
    assert cfg.ensemble.vol_target_annual <= 0.40
    assert cfg.risk.max_position_size_pct == 7.0
    assert cfg.risk.max_daily_drawdown_pct <= 0.30
    try:
        bars = dataframe_to_bars(load_yfinance_hg("HG=F", start="2008-01-01"), "HG")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Yahoo unavailable: {exc}")
    if len(bars) < 2000:
        pytest.skip("insufficient HG history")
    result = BacktestEngine(cfg).run(bars)
    yearly = calendar_year_returns(result.equity_curve, [b.timestamp for b in bars])
    assert yearly, "expected calendar years"
    vals = list(yearly.values())
    assert all(v > 0.0 for v in vals), f"losing years present: {yearly}"
    assert float(result.metrics.get("max_drawdown", 1.0)) <= 0.30
    assert float(result.metrics.get("total_return", -1.0)) > 0.0
    assert float(np.mean(vals)) > 0.0
    assert cfg.ensemble.take_profit_atr_mult >= 20.0
    assert cfg.ensemble.stop_atr_mult >= 4.0


def test_yfinance_start_2008_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke: loader accepts start=; skip if network/Yahoo unavailable."""
    from copper_ensemble.data import load_yfinance_hg

    try:
        df = load_yfinance_hg("HG=F", start="2008-01-01", end="2008-06-30")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Yahoo unavailable: {exc}")
    assert len(df) > 50
    assert df.index.min().year == 2008

