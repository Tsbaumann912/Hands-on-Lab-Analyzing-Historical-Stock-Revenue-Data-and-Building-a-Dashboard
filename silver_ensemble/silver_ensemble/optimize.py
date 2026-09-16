"""Calendar-year profitability optimization under a max-drawdown constraint."""

from __future__ import annotations

import copy
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from silver_ensemble.engine import BacktestEngine, BacktestResult
from silver_ensemble.models import Bar, Config, EnsembleConfig, RiskConfig, load_config

logger = logging.getLogger(__name__)

SLEEVE_KEYS = ("tsmom", "carry", "basis_mom", "inventory", "fade")


def apply_trial_params(base: Config, params: Mapping[str, Any]) -> Config:
    """Return a new Config with ensemble/risk fields overridden from ``params``."""
    weights = {k: float(params[f"w_{k}"]) for k in SLEEVE_KEYS if f"w_{k}" in params}
    if weights:
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        weights = {k: v / total for k, v in weights.items()}
    else:
        weights = dict(base.ensemble.weights)

    ens = replace(
        base.ensemble,
        weights=weights,
        vol_target_annual=float(params.get("vol_target_annual", base.ensemble.vol_target_annual)),
        kelly_fraction=float(params.get("kelly_fraction", base.ensemble.kelly_fraction)),
        agreement_min=float(params.get("agreement_min", base.ensemble.agreement_min)),
        buffer_forecast=float(params.get("buffer_forecast", base.ensemble.buffer_forecast)),
        fade_z=float(params.get("fade_z", base.ensemble.fade_z)),
        stop_atr_mult=float(params.get("stop_atr_mult", base.ensemble.stop_atr_mult)),
        take_profit_atr_mult=float(
            params.get("take_profit_atr_mult", base.ensemble.take_profit_atr_mult)
        ),
        fdm_cap=float(params.get("fdm_cap", base.ensemble.fdm_cap)),
    )
    risk = replace(
        base.risk,
        max_position_size_pct=float(
            params.get("max_position_size_pct", base.risk.max_position_size_pct)
        ),
        max_leverage=float(params.get("max_leverage", base.risk.max_leverage)),
        # Halt slightly inside the 30% gate so max DD stays strictly below 0.30
        max_daily_drawdown_pct=float(
            params.get("max_daily_drawdown_pct", min(0.29, base.risk.max_daily_drawdown_pct))
        ),
    )
    return replace(base, ensemble=ens, risk=risk)


def _year_of(ts: Any) -> int:
    if hasattr(ts, "year"):
        return int(ts.year)
    return int(str(ts)[:4])


def calendar_year_returns(
    bars: Sequence[Bar],
    equity: np.ndarray,
) -> Dict[int, float]:
    """Map calendar year → equity return from first to last bar of that year."""
    if len(bars) != equity.size or equity.size < 2:
        return {}
    years = np.array([_year_of(b.timestamp) for b in bars], dtype=np.int32)
    out: Dict[int, float] = {}
    for y in np.unique(years):
        mask = years == y
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            continue
        e0 = float(equity[idx[0]])
        e1 = float(equity[idx[-1]])
        if e0 <= 0:
            continue
        out[int(y)] = e1 / e0 - 1.0
    return out


def mean_calendar_year_return(bars: Sequence[Bar], equity: np.ndarray) -> float:
    yrs = calendar_year_returns(bars, equity)
    if not yrs:
        return -1.0
    return float(np.mean(list(yrs.values())))


def evaluate_config(
    config: Config,
    bars: List[Bar],
) -> Dict[str, Any]:
    """Run one backtest and return profitability / risk diagnostics."""
    result = BacktestEngine(config).run(bars)
    yr = calendar_year_returns(bars, result.equity_curve)
    avg_yr = float(np.mean(list(yr.values()))) if yr else -1.0
    pos_frac = float(np.mean([1.0 if r > 0 else 0.0 for r in yr.values()])) if yr else 0.0
    return {
        "metrics": result.metrics,
        "year_returns": {str(k): float(v) for k, v in sorted(yr.items())},
        "avg_calendar_year_return": avg_yr,
        "pct_years_positive": pos_frac,
        "n_years": len(yr),
        "max_drawdown": float(result.metrics.get("max_drawdown", 1.0)),
        "sharpe": float(result.metrics.get("sharpe", 0.0)),
        "cagr": float(result.metrics.get("cagr", 0.0)),
        "upi": float(result.metrics.get("upi", 0.0)),
    }


def load_search_space(path: str | Path | None = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "optuna.yaml"
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


def _suggest(trial: Any, space: Mapping[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in SLEEVE_KEYS:
        lo, hi = space.get("weights", {}).get(key, [0.05, 0.50])
        params[f"w_{key}"] = trial.suggest_float(f"w_{key}", float(lo), float(hi))
    for name, bounds in space.get("ensemble", {}).items():
        lo, hi = bounds
        if isinstance(lo, int) and isinstance(hi, int) and not isinstance(lo, bool):
            params[name] = trial.suggest_int(name, int(lo), int(hi))
        else:
            params[name] = trial.suggest_float(name, float(lo), float(hi))
    for name, bounds in space.get("risk", {}).items():
        lo, hi = bounds
        params[name] = trial.suggest_float(name, float(lo), float(hi))
    return params


def run_optimization(
    base: Config,
    bars: List[Bar],
    n_trials: int = 60,
    max_dd_gate: float = 0.30,
    seed: int = 42,
    search_space_path: str | Path | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Maximize average calendar-year return subject to max DD < ``max_dd_gate``.

    Returns ``(best_params, best_eval, all_feasible_summaries)``.
    """
    import optuna
    from optuna.samplers import TPESampler

    space = load_search_space(search_space_path)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    feasible: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, space)
        cfg = apply_trial_params(base, params)
        ev = evaluate_config(cfg, bars)
        dd = float(ev["max_drawdown"])
        avg_yr = float(ev["avg_calendar_year_return"])
        trial.set_user_attr("max_drawdown", dd)
        trial.set_user_attr("avg_calendar_year_return", avg_yr)
        trial.set_user_attr("pct_years_positive", ev["pct_years_positive"])
        trial.set_user_attr("sharpe", ev["sharpe"])
        trial.set_user_attr("cagr", ev["cagr"])
        trial.set_user_attr("upi", ev["upi"])
        trial.set_user_attr("year_returns", ev["year_returns"])
        # Soft penalty if DD breaches gate (keeps search guided)
        if dd >= max_dd_gate:
            return avg_yr - 5.0 * (dd - max_dd_gate + 0.01)
        feasible.append(
            {
                "params": params,
                "avg_calendar_year_return": avg_yr,
                "max_drawdown": dd,
                "pct_years_positive": ev["pct_years_positive"],
                "sharpe": ev["sharpe"],
                "cagr": ev["cagr"],
                "upi": ev["upi"],
                "year_returns": ev["year_returns"],
            }
        )
        return avg_yr

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    # Prefer best feasible (DD < gate); else best trial overall
    feasible_sorted = sorted(
        feasible,
        key=lambda x: (x["avg_calendar_year_return"], x["sharpe"]),
        reverse=True,
    )
    if feasible_sorted:
        best = feasible_sorted[0]
        best_params = best["params"]
        best_cfg = apply_trial_params(base, best_params)
        best_eval = evaluate_config(best_cfg, bars)
    else:
        t = study.best_trial
        best_params = dict(t.params)
        # Ensure weight keys present
        for k in SLEEVE_KEYS:
            if f"w_{k}" not in best_params and f"w_{k}" in t.params:
                best_params[f"w_{k}"] = t.params[f"w_{k}"]
        best_cfg = apply_trial_params(base, best_params)
        best_eval = evaluate_config(best_cfg, bars)
        logger.warning("No trial satisfied max_dd < %.2f; returning best penalized trial", max_dd_gate)

    # Normalise weights in returned params for YAML writing
    w = {k: float(best_params[f"w_{k}"]) for k in SLEEVE_KEYS}
    s = sum(w.values())
    best_params_out = {
        **{k: v for k, v in best_params.items() if not k.startswith("w_")},
        "weights": {k: v / s for k, v in w.items()},
    }
    return best_params_out, best_eval, feasible_sorted


def write_optimized_yaml(
    base_yaml_path: Path,
    out_yaml_path: Path,
    best_params: Mapping[str, Any],
) -> None:
    """Merge optimized ensemble/risk knobs into a config YAML copy."""
    with base_yaml_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    ens = dict(raw.get("ensemble", {}))
    risk = dict(raw.get("risk", {}))
    if "weights" in best_params:
        ens["weights"] = {k: float(v) for k, v in best_params["weights"].items()}
    for key in (
        "vol_target_annual",
        "kelly_fraction",
        "agreement_min",
        "buffer_forecast",
        "fade_z",
        "stop_atr_mult",
        "take_profit_atr_mult",
        "fdm_cap",
    ):
        if key in best_params:
            ens[key] = float(best_params[key])
    for key in ("max_position_size_pct", "max_leverage", "max_daily_drawdown_pct"):
        if key in best_params:
            risk[key] = float(best_params[key])
    # Keep halt inside 30% gate
    risk["max_daily_drawdown_pct"] = min(float(risk.get("max_daily_drawdown_pct", 0.29)), 0.29)
    risk["halt_on_breach"] = True
    raw["ensemble"] = ens
    raw["risk"] = risk
    raw["optimization"] = {
        "objective": "maximize_avg_calendar_year_return",
        "constraint": "max_drawdown < 0.30",
        "applied_params": {k: (dict(v) if isinstance(v, dict) else float(v) if isinstance(v, (int, float)) else v) for k, v in best_params.items()},
    }
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml_path.open("w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False, default_flow_style=False)
