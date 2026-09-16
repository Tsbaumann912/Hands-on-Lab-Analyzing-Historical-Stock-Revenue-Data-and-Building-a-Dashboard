"""Constrained Optuna re-optimisation for calendar-year profitability."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from platinum_ensemble.engine import BacktestEngine, BacktestResult, _compute_metrics
from platinum_ensemble.models import Bar, Config, EnsembleConfig


SLEEVE_WEIGHT_KEYS = ("tsmom", "carry", "pl_gc_rv", "inventory", "fade")


@dataclass
class CalendarYearStats:
    years: Dict[int, float]
    mean_return: float
    median_return: float
    pct_positive: float
    n_years: int


@dataclass
class OptimizeResult:
    best_params: Dict[str, float]
    best_weights: Dict[str, float]
    is_metrics: Dict[str, float]
    is_calendar: CalendarYearStats
    oos_metrics: Dict[str, float]
    oos_calendar: CalendarYearStats
    full_metrics: Dict[str, float]
    full_calendar: CalendarYearStats
    n_trials: int
    constraint_max_dd: float
    notes: List[str] = field(default_factory=list)


def calendar_year_returns(
    equity: np.ndarray,
    timestamps: Sequence[Any],
) -> CalendarYearStats:
    """Compute simple return per calendar year from an equity curve."""
    if equity.size < 2 or len(timestamps) != equity.size:
        return CalendarYearStats({}, 0.0, 0.0, 0.0, 0)

    years = np.array([_year_of(ts) for ts in timestamps], dtype=np.int32)
    uniq = np.unique(years)
    out: Dict[int, float] = {}
    for y in uniq:
        mask = years == y
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            continue
        e0 = float(equity[idx[0]])
        e1 = float(equity[idx[-1]])
        if e0 > 1e-12:
            out[int(y)] = e1 / e0 - 1.0
    if not out:
        return CalendarYearStats({}, 0.0, 0.0, 0.0, 0)
    vals = np.array(list(out.values()), dtype=np.float64)
    return CalendarYearStats(
        years=out,
        mean_return=float(np.mean(vals)),
        median_return=float(np.median(vals)),
        pct_positive=float(np.mean(vals > 0.0)),
        n_years=int(vals.size),
    )


def _year_of(ts: Any) -> int:
    if isinstance(ts, datetime):
        return int(ts.year)
    if hasattr(ts, "year"):
        return int(ts.year)
    return int(str(ts)[:4])


def load_optuna_bounds(path: str | Path | None = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "optuna.yaml"
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


def _normalize_weights(raw: Mapping[str, float]) -> Dict[str, float]:
    vals = {k: max(0.0, float(raw.get(k, 0.0))) for k in SLEEVE_WEIGHT_KEYS}
    s = sum(vals.values())
    if s <= 1e-12:
        return {k: 1.0 / len(SLEEVE_WEIGHT_KEYS) for k in SLEEVE_WEIGHT_KEYS}
    return {k: v / s for k, v in vals.items()}


def apply_trial_params(config: Config, params: Mapping[str, float]) -> Config:
    """Return a new Config with ensemble overrides from an Optuna trial dict."""
    weights = _normalize_weights(
        {
            "tsmom": float(params.get("weight_tsmom", config.ensemble.weights["tsmom"])),
            "carry": float(params.get("weight_carry", config.ensemble.weights["carry"])),
            "pl_gc_rv": float(params.get("weight_pl_gc_rv", config.ensemble.weights["pl_gc_rv"])),
            "inventory": float(params.get("weight_inventory", config.ensemble.weights["inventory"])),
            "fade": float(params.get("weight_fade", config.ensemble.weights["fade"])),
        }
    )
    ens = EnsembleConfig(
        horizons_days=list(config.ensemble.horizons_days),
        weights=weights,
        forecast_cap=config.ensemble.forecast_cap,
        agreement_min=float(params.get("agreement_min", config.ensemble.agreement_min)),
        fdm_cap=float(params.get("fdm_cap", config.ensemble.fdm_cap)),
        vol_target_annual=float(params.get("vol_target_annual", config.ensemble.vol_target_annual)),
        ewma_vol_com_days=config.ensemble.ewma_vol_com_days,
        fade_z=float(params.get("fade_z", config.ensemble.fade_z)),
        buffer_forecast=float(params.get("buffer_forecast", config.ensemble.buffer_forecast)),
        kelly_fraction=float(params.get("kelly_fraction", config.ensemble.kelly_fraction)),
        inventory_sma=config.ensemble.inventory_sma,
        inventory_delta_days=config.ensemble.inventory_delta_days,
        rv_lookback=config.ensemble.rv_lookback,
        rv_entry_z=float(params.get("rv_entry_z", config.ensemble.rv_entry_z)),
        price_z_lookback=config.ensemble.price_z_lookback,
        atr_period=config.ensemble.atr_period,
        stop_atr_mult=float(params.get("stop_atr_mult", config.ensemble.stop_atr_mult)),
        take_profit_atr_mult=config.ensemble.take_profit_atr_mult,
        gold_ticker=config.ensemble.gold_ticker,
        short_scale=float(params.get("short_scale", getattr(config.ensemble, "short_scale", 1.0))),
    )
    risk = replace(
        config.risk,
        halt_on_breach=False,
        max_position_size_pct=float(
            params.get("max_position_size_pct", config.risk.max_position_size_pct)
        ),
        max_leverage=float(params.get("max_leverage", config.risk.max_leverage)),
    )
    return replace(config, ensemble=ens, risk=risk)


def split_bars_by_date(
    bars: List[Bar],
    is_end_date: str,
) -> Tuple[List[Bar], List[Bar]]:
    """Split bars into IS (timestamp.date <= is_end_date) and OOS (after)."""
    end = datetime.strptime(is_end_date, "%Y-%m-%d").date()
    is_bars: List[Bar] = []
    oos_bars: List[Bar] = []
    for b in bars:
        ts = b.timestamp
        d = ts.date() if hasattr(ts, "date") else datetime.strptime(str(ts)[:10], "%Y-%m-%d").date()
        if d <= end:
            is_bars.append(b)
        else:
            oos_bars.append(b)
    return is_bars, oos_bars


def _suggest(trial: Any, name: str, spec: Dict[str, Any]) -> float:
    low = float(spec["low"])
    high = float(spec["high"])
    return float(trial.suggest_float(name, low, high))


def evaluate_config_on_bars(
    config: Config,
    bars: List[Bar],
) -> Tuple[Dict[str, float], CalendarYearStats, BacktestResult]:
    engine = BacktestEngine(config)
    result = engine.run(bars)
    stamps = [b.timestamp for b in bars]
    cal = calendar_year_returns(result.equity_curve, stamps)
    metrics = dict(result.metrics)
    metrics["mean_calendar_year_return"] = cal.mean_return
    metrics["pct_positive_years"] = cal.pct_positive
    return metrics, cal, result


def run_calendar_optimize(
    config: Config,
    bars: List[Bar],
    optuna_path: str | Path | None = None,
    n_trials: Optional[int] = None,
    max_dd_limit: float = 0.30,
) -> OptimizeResult:
    """
    Optuna search maximising mean calendar-year return with Max DD < limit.

    ``optimize_scope`` in optuna.yaml: ``full`` (default) trains on all bars;
    ``is`` trains only through ``is_end_date``. Holdout metrics always reported.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as exc:  # pragma: no cover
        raise ImportError("optuna is required for platinum_ensemble optimize") from exc

    bounds = load_optuna_bounds(optuna_path)
    is_end = str(bounds.get("is_end_date", "2019-12-31"))
    trials = int(n_trials if n_trials is not None else bounds.get("n_trials", 60))
    timeout = int(bounds.get("timeout_seconds", 600))
    seed = int(bounds.get("seed", 42))
    scope = str(bounds.get("optimize_scope", "full")).lower()

    is_bars, oos_bars = split_bars_by_date(bars, is_end)
    train_bars = is_bars if scope == "is" else bars
    if len(train_bars) < 500:
        raise ValueError(f"Train set too short ({len(train_bars)} bars)")

    search_keys = [
        "vol_target_annual",
        "kelly_fraction",
        "agreement_min",
        "fade_z",
        "buffer_forecast",
        "rv_entry_z",
        "fdm_cap",
        "stop_atr_mult",
        "max_position_size_pct",
        "max_leverage",
        "weight_tsmom",
        "weight_carry",
        "weight_pl_gc_rv",
        "weight_inventory",
        "weight_fade",
        "short_scale",
    ]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    min_fills = int(bounds.get("min_fills", 20))

    def objective(trial: "optuna.Trial") -> float:
        params = {k: _suggest(trial, k, bounds[k]) for k in search_keys if k in bounds}
        cfg = apply_trial_params(config, params)
        metrics, cal, _ = evaluate_config_on_bars(cfg, train_bars)
        max_dd = float(metrics.get("max_drawdown", 1.0))
        mean_yr = float(cal.mean_return)
        n_fills = float(metrics.get("n_fills", 0.0))
        if max_dd >= max_dd_limit:
            return -10.0 - 50.0 * (max_dd - max_dd_limit)
        if n_fills < min_fills:
            return -5.0 - 0.1 * (min_fills - n_fills)
        # Primary objective is mean calendar-year return; tiny tie-breakers only.
        upi = float(metrics.get("upi", 0.0))
        return mean_yr + 0.005 * float(cal.pct_positive) + 0.0005 * max(upi, 0.0)

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=False)

    best_params = {k: round(float(v), 6) for k, v in study.best_params.items()}
    best_cfg = apply_trial_params(config, best_params)
    best_weights = {k: round(float(v), 6) for k, v in best_cfg.ensemble.weights.items()}

    is_metrics, is_cal, _ = evaluate_config_on_bars(best_cfg, is_bars)
    if oos_bars:
        oos_metrics, oos_cal, _ = evaluate_config_on_bars(best_cfg, oos_bars)
    else:
        oos_metrics, oos_cal = {}, CalendarYearStats({}, 0.0, 0.0, 0.0, 0)
    full_metrics, full_cal, _ = evaluate_config_on_bars(best_cfg, bars)

    notes: List[str] = [
        f"Optimize scope={scope} on {len(train_bars)} bars; IS through {is_end} "
        f"({len(is_bars)}), OOS={len(oos_bars)}",
        f"Trials completed: {len(study.trials)} (best value={study.best_value:.4f})",
        f"Constraint: max_drawdown < {max_dd_limit:.0%}",
    ]
    if float(full_metrics.get("max_drawdown", 1.0)) >= max_dd_limit:
        notes.append("WARNING: full-sample Max DD breaches constraint")
    else:
        notes.append("Full-sample Max DD within 30% limit")
    if full_cal.mean_return > 0:
        notes.append(f"Full-sample mean calendar-year return {full_cal.mean_return:.2%}")
    else:
        notes.append(
            f"Full-sample mean calendar-year return still non-positive "
            f"({full_cal.mean_return:.2%}) — best feasible under DD cap"
        )
    if oos_bars and float(oos_metrics.get("max_drawdown", 1.0)) >= max_dd_limit:
        notes.append("WARNING: holdout OOS Max DD breaches 30%")

    return OptimizeResult(
        best_params=best_params,
        best_weights=best_weights,
        is_metrics=is_metrics,
        is_calendar=is_cal,
        oos_metrics=oos_metrics,
        oos_calendar=oos_cal,
        full_metrics=full_metrics,
        full_calendar=full_cal,
        n_trials=len(study.trials),
        constraint_max_dd=max_dd_limit,
        notes=notes,
    )


def optimize_result_to_dict(result: OptimizeResult) -> Dict[str, Any]:
    def _cal(c: CalendarYearStats) -> Dict[str, Any]:
        return {
            "mean_return": c.mean_return,
            "median_return": c.median_return,
            "pct_positive": c.pct_positive,
            "n_years": c.n_years,
            "years": {str(k): v for k, v in sorted(c.years.items())},
        }

    return {
        "best_params": result.best_params,
        "best_weights": result.best_weights,
        "constraint_max_dd": result.constraint_max_dd,
        "n_trials": result.n_trials,
        "is_metrics": result.is_metrics,
        "is_calendar": _cal(result.is_calendar),
        "oos_metrics": result.oos_metrics,
        "oos_calendar": _cal(result.oos_calendar),
        "full_metrics": result.full_metrics,
        "full_calendar": _cal(result.full_calendar),
        "notes": result.notes,
    }


def write_optimized_yaml(
    base_config_path: Path,
    result: OptimizeResult,
    out_path: Path,
) -> None:
    """Write a new default.yaml with best ensemble weights/params locked in."""
    with base_config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    ens = dict(raw.get("ensemble", {}))
    risk = dict(raw.get("risk", {}))
    p = result.best_params
    ens["weights"] = result.best_weights
    for key in (
        "vol_target_annual",
        "kelly_fraction",
        "agreement_min",
        "fade_z",
        "buffer_forecast",
        "rv_entry_z",
        "fdm_cap",
        "stop_atr_mult",
        "short_scale",
    ):
        if key in p:
            ens[key] = float(p[key])
    for key in ("max_position_size_pct", "max_leverage"):
        if key in p:
            risk[key] = float(p[key])
    raw["ensemble"] = ens
    raw["risk"] = risk
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        yaml.dump(raw, fh, default_flow_style=False, sort_keys=False)
