"""Calendar-year profitability optimization under drawdown and all-years-positive gates."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from silver_ensemble.engine import BacktestEngine
from silver_ensemble.models import Bar, Config

logger = logging.getLogger(__name__)

SLEEVE_KEYS = ("tsmom", "carry", "basis_mom", "inventory", "fade", "stoch_ma")


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
        ma_fast=int(params.get("ma_fast", base.ensemble.ma_fast)),
        ma_slow=int(params.get("ma_slow", base.ensemble.ma_slow)),
        rsi_period=int(params.get("rsi_period", base.ensemble.rsi_period)),
        stoch_rsi_period=int(params.get("stoch_rsi_period", base.ensemble.stoch_rsi_period)),
        stoch_oversold=float(params.get("stoch_oversold", base.ensemble.stoch_oversold)),
        stoch_overbought=float(params.get("stoch_overbought", base.ensemble.stoch_overbought)),
    )
    risk = replace(
        base.risk,
        max_position_size_pct=float(
            params.get("max_position_size_pct", max(base.risk.max_position_size_pct, 100.0))
        ),
        max_leverage=float(params.get("max_leverage", base.risk.max_leverage)),
        max_daily_drawdown_pct=float(
            params.get("max_daily_drawdown_pct", min(0.29, base.risk.max_daily_drawdown_pct))
        ),
        ytd_loss_halt_pct=float(params.get("ytd_loss_halt_pct", base.risk.ytd_loss_halt_pct)),
    )
    # Always uncap position %; leverage / max_contracts / vol-target bind instead
    risk = replace(risk, max_position_size_pct=100.0)
    port = replace(
        base.portfolio,
        collateral_yield_annual=float(
            params.get("collateral_yield_annual", base.portfolio.collateral_yield_annual)
        ),
    )
    return replace(base, ensemble=ens, risk=risk, portfolio=port)


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
    vals = list(yr.values())
    avg_yr = float(np.mean(vals)) if vals else -1.0
    min_yr = float(np.min(vals)) if vals else -1.0
    pos_frac = float(np.mean([1.0 if r > 0.0 else 0.0 for r in vals])) if vals else 0.0
    all_years_positive = bool(vals) and bool(np.all(np.asarray(vals, dtype=float) > 0.0))
    total_return = float(result.metrics.get("total_return", 0.0))
    return {
        "metrics": result.metrics,
        "year_returns": {str(k): float(v) for k, v in sorted(yr.items())},
        "avg_calendar_year_return": avg_yr,
        "min_year_return": min_yr,
        "pct_years_positive": pos_frac,
        "all_years_positive": all_years_positive,
        "n_years": len(yr),
        "max_drawdown": float(result.metrics.get("max_drawdown", 1.0)),
        "sharpe": float(result.metrics.get("sharpe", 0.0)),
        "cagr": float(result.metrics.get("cagr", 0.0)),
        "upi": float(result.metrics.get("upi", 0.0)),
        "total_return": total_return,
        "end_equity": float(result.metrics.get("end_equity", 0.0)),
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
    for name, bounds in space.get("portfolio", {}).items():
        lo, hi = bounds
        params[name] = trial.suggest_float(name, float(lo), float(hi))
    return params


def _profit_score(avg_yr: float, cagr: float, total_return: float) -> float:
    """Blend annual and total profitability (CAGR bridges multi-year total return)."""
    return float(avg_yr + cagr + 0.25 * total_return)


def run_optimization(
    base: Config,
    bars: List[Bar],
    n_trials: int = 60,
    max_dd_gate: float = 0.30,
    seed: int = 42,
    search_space_path: str | Path | None = None,
    require_all_years_positive: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Maximize annual + total returns subject to:
    - every calendar year return > 0 (when ``require_all_years_positive``)
    - max drawdown < ``max_dd_gate``
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
        min_yr = float(ev["min_year_return"])
        cagr = float(ev["cagr"])
        total_ret = float(ev["total_return"])
        all_pos = bool(ev["all_years_positive"])
        score = _profit_score(avg_yr, cagr, total_ret)

        trial.set_user_attr("max_drawdown", dd)
        trial.set_user_attr("avg_calendar_year_return", avg_yr)
        trial.set_user_attr("min_year_return", min_yr)
        trial.set_user_attr("pct_years_positive", ev["pct_years_positive"])
        trial.set_user_attr("all_years_positive", all_pos)
        trial.set_user_attr("sharpe", ev["sharpe"])
        trial.set_user_attr("cagr", cagr)
        trial.set_user_attr("upi", ev["upi"])
        trial.set_user_attr("total_return", total_ret)
        trial.set_user_attr("year_returns", ev["year_returns"])
        trial.set_user_attr("profit_score", score)

        # Lexicographic soft penalties: first all-years+, then DD, then profit
        if require_all_years_positive and not all_pos:
            # Push min year toward positive; keep below any feasible profit score
            return float(min_yr - 1.0)
        if dd >= max_dd_gate:
            return float(score - 5.0 * (dd - max_dd_gate + 0.01))

        feasible.append(
            {
                "params": params,
                "avg_calendar_year_return": avg_yr,
                "min_year_return": min_yr,
                "max_drawdown": dd,
                "pct_years_positive": ev["pct_years_positive"],
                "all_years_positive": all_pos,
                "sharpe": ev["sharpe"],
                "cagr": cagr,
                "upi": ev["upi"],
                "total_return": total_ret,
                "profit_score": score,
                "year_returns": ev["year_returns"],
            }
        )
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    feasible_sorted = sorted(
        feasible,
        key=lambda x: (
            float(x["profit_score"]),
            float(x["avg_calendar_year_return"]),
            float(x["total_return"]),
            float(x["sharpe"]),
        ),
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
        best_cfg = apply_trial_params(base, best_params)
        best_eval = evaluate_config(best_cfg, bars)
        logger.warning(
            "No trial satisfied all-years-positive=%s and max_dd < %.2f; "
            "returning best penalized trial (min_year=%.4f dd=%.4f)",
            require_all_years_positive,
            max_dd_gate,
            best_eval["min_year_return"],
            best_eval["max_drawdown"],
        )

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
        "stoch_oversold",
        "stoch_overbought",
    ):
        if key in best_params:
            ens[key] = float(best_params[key])
    for key in ("ma_fast", "ma_slow", "rsi_period", "stoch_rsi_period", "stoch_rsi_smooth"):
        if key in best_params:
            ens[key] = int(best_params[key])
    for key in ("max_leverage", "max_daily_drawdown_pct", "ytd_loss_halt_pct"):
        if key in best_params:
            risk[key] = float(best_params[key])
    risk["max_position_size_pct"] = 100.0
    risk["max_daily_drawdown_pct"] = min(float(risk.get("max_daily_drawdown_pct", 0.29)), 0.29)
    risk["halt_on_breach"] = True
    port = dict(raw.get("portfolio", {}))
    if "collateral_yield_annual" in best_params:
        port["collateral_yield_annual"] = float(best_params["collateral_yield_annual"])
    raw["ensemble"] = ens
    raw["risk"] = risk
    raw["portfolio"] = port
    raw["optimization"] = {
        "objective": "maximize_avg_calendar_year_return_and_total_return",
        "constraint": "all_calendar_years_positive and max_drawdown < 0.30",
        "max_position_size_pct": "uncapped (100.0)",
        "applied_params": {
            k: (dict(v) if isinstance(v, dict) else float(v) if isinstance(v, (int, float)) else v)
            for k, v in best_params.items()
        },
    }
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml_path.open("w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False, default_flow_style=False)
