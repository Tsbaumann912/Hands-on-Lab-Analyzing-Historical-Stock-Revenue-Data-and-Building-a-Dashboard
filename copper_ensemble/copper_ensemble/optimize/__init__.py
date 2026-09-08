"""Nested purged walk-forward optimisation (IS-only tuning, OOS evaluation)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import optuna
import yaml
from optuna.samplers import TPESampler

from copper_ensemble.engine import BacktestEngine
from copper_ensemble.models import (
    Bar,
    Config,
    clone_config,
    rebuild_weights,
)

logger = logging.getLogger("copper_ensemble.optimize")

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_OPTUNA_PATH = Path(__file__).resolve().parents[2] / "config" / "optuna.yaml"


@dataclass
class WindowOptResult:
    window: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    best_params: Dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    oos_max_dd: float
    oos_total_return: float
    n_trials: int


@dataclass
class OptimizeReport:
    windows: List[WindowOptResult] = field(default_factory=list)
    robust_params: Dict[str, Any] = field(default_factory=dict)
    mean_oos_sharpe: float = 0.0
    median_oos_sharpe: float = 0.0
    mean_is_sharpe: float = 0.0
    baseline_mean_oos_sharpe: float = 0.0
    robust_full_metrics: Dict[str, float] = field(default_factory=dict)
    robust_wfa: List[Dict[str, float]] = field(default_factory=list)
    holdout: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    search_space_path: str = ""


def load_optuna_space(path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(path) if path else DEFAULT_OPTUNA_PATH
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise TypeError("optuna.yaml must be a mapping")
    return raw


def _suggest(trial: optuna.Trial, space: Mapping[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        if name in {
            "n_trials_per_window",
            "n_windows",
            "in_sample_ratio",
            "purge_bars",
            "max_dd_soft_cap",
            "min_fills",
        }:
            continue
        if not isinstance(spec, dict) or "type" not in spec:
            continue
        t = spec["type"]
        if t == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif t == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif t == "categorical":
            choices = list(spec["choices"])
            # Optuna needs hashable choices; encode list horizons as strings
            if choices and isinstance(choices[0], list):
                labels = [",".join(str(x) for x in c) for c in choices]
                picked = trial.suggest_categorical(name, labels)
                params[name] = [int(x) for x in picked.split(",")]
            else:
                # YAML may load true/false as bools
                params[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"unsupported optuna type {t!r} for {name}")
    return params


def params_to_config(base: Config, params: Mapping[str, Any]) -> Config:
    """Map a trial / robust param dict onto a Config."""
    enable_fade = bool(params.get("enable_fade", True))
    tsmom_w = float(params.get("tsmom_weight", base.ensemble.weights.get("tsmom", 0.55)))
    weights = rebuild_weights(base.ensemble.weights, tsmom_w, enable_fade)
    horizons = params.get("horizons_days", params.get("horizon_set", base.ensemble.horizons_days))
    horizons = [int(h) for h in list(horizons)]
    eo = {
        "weights": weights,
        "horizons_days": horizons,
        "agreement_min": float(params.get("agreement_min", base.ensemble.agreement_min)),
        "buffer_forecast": float(params.get("buffer_forecast", base.ensemble.buffer_forecast)),
        "vol_target_annual": float(
            params.get("vol_target_annual", base.ensemble.vol_target_annual)
        ),
        "kelly_fraction": float(params.get("kelly_fraction", base.ensemble.kelly_fraction)),
        "stop_atr_mult": float(params.get("stop_atr_mult", base.ensemble.stop_atr_mult)),
        "take_profit_atr_mult": float(
            params.get("take_profit_atr_mult", base.ensemble.take_profit_atr_mult)
        ),
        "fdm_cap": float(params.get("fdm_cap", base.ensemble.fdm_cap)),
    }
    return clone_config(base, ensemble_overrides=eo)


def _is_objective_score(metrics: Mapping[str, float], space: Mapping[str, Any]) -> float:
    dd_cap = float(space.get("max_dd_soft_cap", 0.25))
    min_fills = float(space.get("min_fills", 8))
    # Prefer Ulcer Performance Index (return per drawdown pain); fall back to Sharpe.
    upi = float(metrics.get("upi", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    dd = float(metrics.get("max_drawdown", 1.0))
    fills = float(metrics.get("n_fills", 0.0))
    if fills < min_fills:
        return -10.0
    penalty = 0.0
    if dd > dd_cap:
        penalty += 3.0 * (dd - dd_cap)
    if dd > 0.40:
        return -8.0
    # Scale UPI into a similar numeric range as Sharpe for TPE
    return upi - penalty + 0.05 * sharpe


def _window_slices(
    n: int,
    n_windows: int,
    is_ratio: float,
    purge: int,
) -> List[Tuple[int, int, int, int]]:
    """Rolling (non-overlapping block) purged WFA slices."""
    return rolling_window_slices(n, n_windows, is_ratio, purge)


def rolling_window_slices(
    n: int,
    n_windows: int,
    is_ratio: float,
    purge: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Rolling purged walk-forward: consecutive non-overlapping blocks.

    Each block is split into IS then purged OOS. Returns
    ``(is_start, is_end, oos_start, oos_end)``.
    """
    out: List[Tuple[int, int, int, int]] = []
    if n_windows < 1 or n < 100:
        return out
    block = n // n_windows
    for w in range(n_windows):
        start = w * block
        end = n if w == n_windows - 1 else (w + 1) * block
        seg_len = end - start
        split = int(seg_len * is_ratio)
        is_end = start + max(split - purge, 10)
        oos_start = start + min(split + purge, seg_len - 10)
        oos_end = end
        if is_end - start < 80 or oos_end - oos_start < 40:
            continue
        out.append((start, is_end, oos_start, oos_end))
    return out


def anchored_window_slices(
    n: int,
    n_windows: int,
    *,
    min_is_bars: int = 504,
    oos_bars: Optional[int] = None,
    purge: int = 5,
) -> List[Tuple[int, int, int, int]]:
    """
    Anchored (expanding) purged walk-forward.

    IS always starts at bar 0 and expands; OOS is the next fixed-length block
    after a purge gap. Returns ``(is_start, is_end, oos_start, oos_end)``.
    """
    out: List[Tuple[int, int, int, int]] = []
    if n_windows < 1 or n < min_is_bars + 80:
        return out
    # Reserve room for n_windows OOS segments after min IS
    remaining = n - min_is_bars
    oos_len = int(oos_bars) if oos_bars is not None else max(remaining // max(n_windows, 1), 60)
    # Place OOS starts evenly from min_is to near the end
    last_oos_start = n - oos_len
    if last_oos_start <= min_is_bars:
        return out
    starts = np.linspace(min_is_bars, last_oos_start, num=n_windows)
    for w, oos_start_f in enumerate(starts):
        oos_start = int(round(oos_start_f)) + purge
        oos_end = min(oos_start + oos_len, n)
        is_end = max(oos_start - purge, min_is_bars)
        is_start = 0
        if is_end - is_start < 80 or oos_end - oos_start < 40:
            continue
        out.append((is_start, is_end, oos_start, oos_end))
    return out


def _run_metrics(cfg: Config, bars: Sequence[Bar]) -> Dict[str, float]:
    return BacktestEngine(cfg).run(list(bars)).metrics


def _run_sharpe(cfg: Config, bars: Sequence[Bar]) -> Dict[str, float]:
    return _run_metrics(cfg, bars)

def optimize_window(
    base: Config,
    bars: Sequence[Bar],
    space: Mapping[str, Any],
    *,
    n_trials: int,
    seed: int,
) -> Tuple[Dict[str, Any], float]:
    """Tune on ``bars`` only (must be IS). Returns best params and IS Sharpe."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, space)
        cfg = params_to_config(base, params)
        metrics = _run_sharpe(cfg, bars)
        score = _is_objective_score(metrics, space)
        trial.set_user_attr("sharpe", float(metrics.get("sharpe", 0.0)))
        trial.set_user_attr("max_drawdown", float(metrics.get("max_drawdown", 0.0)))
        trial.set_user_attr("n_fills", float(metrics.get("n_fills", 0.0)))
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed, multivariate=True),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    # decode horizon_set string if present
    if "horizon_set" in best and isinstance(best["horizon_set"], str):
        best["horizon_set"] = [int(x) for x in best["horizon_set"].split(",")]
        best["horizons_days"] = best["horizon_set"]
    elif "horizon_set" in best:
        best["horizons_days"] = list(best["horizon_set"])
    is_sharpe = float(study.best_trial.user_attrs.get("sharpe", study.best_value))
    return best, is_sharpe


def _median_params(window_params: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate IS-selected params across windows (median / mode) — no OOS peeking."""
    if not window_params:
        return {}
    keys = sorted({k for p in window_params for k in p.keys()})
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [p[k] for p in window_params if k in p]
        if not vals:
            continue
        v0 = vals[0]
        if isinstance(v0, bool) or (isinstance(v0, (int, float)) and k == "enable_fade"):
            # mode
            truths = sum(1 for v in vals if bool(v))
            out[k] = truths >= (len(vals) - truths)
        elif isinstance(v0, (list, tuple)):
            # mode of tuple
            as_t = [tuple(v) for v in vals]
            # pick most common
            best = max(set(as_t), key=as_t.count)
            out[k] = list(best)
        elif isinstance(v0, (int, float)) and not isinstance(v0, bool):
            out[k] = float(np.median(np.asarray(vals, dtype=np.float64)))
        else:
            # categorical mode as string
            as_s = [str(v) for v in vals]
            out[k] = max(set(as_s), key=as_s.count)
    if "horizon_set" in out and "horizons_days" not in out:
        out["horizons_days"] = list(out["horizon_set"])
    if "horizons_days" in out:
        out["horizons_days"] = [int(x) for x in out["horizons_days"]]
    if "enable_fade" in out:
        out["enable_fade"] = bool(out["enable_fade"])
    return out


def nested_walk_forward_optimize(
    base: Config,
    bars: List[Bar],
    space: Optional[Mapping[str, Any]] = None,
    *,
    space_path: str | Path | None = None,
    n_trials: Optional[int] = None,
    n_windows: Optional[int] = None,
    seed: int = 42,
) -> OptimizeReport:
    """
    Nested purged WFA:

    1. Per window: Optuna maximises **IS-only** score.
    2. Freeze those params; evaluate **once** on purged OOS (never used to pick).
    3. Production ``robust_params`` = median/mode of IS winners (still no OOS peeking).
    4. Report nested mean OOS Sharpe + full-sample metrics under robust params.
    """
    space = dict(space) if space is not None else load_optuna_space(space_path)
    path_str = str(space_path or DEFAULT_OPTUNA_PATH)
    n_trials = int(n_trials if n_trials is not None else space.get("n_trials_per_window", 28))
    n_windows = int(n_windows if n_windows is not None else space.get("n_windows", 6))
    is_ratio = float(space.get("in_sample_ratio", 0.70))
    purge = int(space.get("purge_bars", 5))

    slices = _window_slices(len(bars), n_windows, is_ratio, purge)
    report = OptimizeReport(search_space_path=path_str)
    if not slices:
        report.notes.append("No valid WFA windows — series too short.")
        return report

    # Baseline nested OOS under current config (no tuning)
    base_oos: List[float] = []
    for is0, is1, oos0, oos1 in slices:
        m = _run_sharpe(base, bars[oos0:oos1])
        base_oos.append(float(m.get("sharpe", 0.0)))
    report.baseline_mean_oos_sharpe = float(np.mean(base_oos)) if base_oos else 0.0

    for w, (is0, is1, oos0, oos1) in enumerate(slices):
        logger.info(
            "WFA window %s IS[%s:%s] OOS[%s:%s] trials=%s",
            w,
            is0,
            is1,
            oos0,
            oos1,
            n_trials,
        )
        best_params, is_sharpe = optimize_window(
            base,
            bars[is0:is1],
            space,
            n_trials=n_trials,
            seed=seed + w,
        )
        cfg_w = params_to_config(base, best_params)
        oos_m = _run_sharpe(cfg_w, bars[oos0:oos1])
        report.windows.append(
            WindowOptResult(
                window=w,
                is_start=is0,
                is_end=is1,
                oos_start=oos0,
                oos_end=oos1,
                best_params=best_params,
                is_sharpe=float(is_sharpe),
                oos_sharpe=float(oos_m.get("sharpe", 0.0)),
                oos_max_dd=float(oos_m.get("max_drawdown", 0.0)),
                oos_total_return=float(oos_m.get("total_return", 0.0)),
                n_trials=n_trials,
            )
        )

    oos_s = np.array([w.oos_sharpe for w in report.windows], dtype=np.float64)
    is_s = np.array([w.is_sharpe for w in report.windows], dtype=np.float64)
    report.mean_oos_sharpe = float(np.mean(oos_s)) if oos_s.size else 0.0
    report.median_oos_sharpe = float(np.median(oos_s)) if oos_s.size else 0.0
    report.mean_is_sharpe = float(np.mean(is_s)) if is_s.size else 0.0
    report.robust_params = _median_params([w.best_params for w in report.windows])

    robust_cfg = params_to_config(base, report.robust_params)
    full = _run_sharpe(robust_cfg, bars)
    report.robust_full_metrics = dict(full)

    # Anchored holdout honesty check: tune once on first 70%, score last 30% once
    split = int(len(bars) * 0.70)
    purge_h = purge
    train = bars[: max(split - purge_h, 100)]
    test = bars[min(split + purge_h, len(bars) - 50) :]
    hold_params, hold_is = optimize_window(
        base,
        train,
        space,
        n_trials=n_trials,
        seed=seed + 99,
    )
    hold_cfg = params_to_config(base, hold_params)
    hold_m = _run_sharpe(hold_cfg, test)
    report.holdout = {
        "train_bars": float(len(train)),
        "test_bars": float(len(test)),
        "is_sharpe": float(hold_is),
        "oos_sharpe": float(hold_m.get("sharpe", 0.0)),
        "oos_max_dd": float(hold_m.get("max_drawdown", 0.0)),
        "oos_total_return": float(hold_m.get("total_return", 0.0)),
        "params": hold_params,
    }

    # Secondary WFA under frozen robust params (honest OOS of the production set)
    for is0, is1, oos0, oos1 in slices:
        is_m = _run_sharpe(robust_cfg, bars[is0:is1])
        oos_m = _run_sharpe(robust_cfg, bars[oos0:oos1])
        report.robust_wfa.append(
            {
                "is_sharpe": float(is_m.get("sharpe", 0.0)),
                "oos_sharpe": float(oos_m.get("sharpe", 0.0)),
                "oos_max_dd": float(oos_m.get("max_drawdown", 0.0)),
            }
        )

    report.notes.append(
        "Nested WFA: params chosen on IS only; OOS used solely for evaluation."
    )
    report.notes.append(
        f"Baseline mean OOS Sharpe={report.baseline_mean_oos_sharpe:.3f}; "
        f"nested tuned mean OOS={report.mean_oos_sharpe:.3f}; "
        f"robust-params mean OOS="
        f"{float(np.mean([x['oos_sharpe'] for x in report.robust_wfa])) if report.robust_wfa else 0.0:.3f}."
    )
    report.notes.append(
        f"Anchored holdout OOS Sharpe={report.holdout.get('oos_sharpe', 0.0):.3f} "
        f"(train first 70%, test last 30%, purged)."
    )
    if report.mean_oos_sharpe <= report.baseline_mean_oos_sharpe + 0.05:
        report.notes.append(
            "Tuning did not clearly beat untuned baseline on nested OOS — "
            "prefer simpler/robust median params and do not claim large edge."
        )
    return report


def robust_params_to_yaml_ensemble(base: Config, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Materialise ensemble YAML fragment from robust params."""
    cfg = params_to_config(base, params)
    e = cfg.ensemble
    return {
        "horizons_days": list(e.horizons_days),
        "weights": dict(e.weights),
        "forecast_cap": e.forecast_cap,
        "agreement_min": round(e.agreement_min, 4),
        "fdm_cap": round(e.fdm_cap, 4),
        "vol_target_annual": round(e.vol_target_annual, 4),
        "ewma_vol_com_days": e.ewma_vol_com_days,
        "fade_z": e.fade_z,
        "buffer_forecast": round(e.buffer_forecast, 4),
        "kelly_fraction": round(e.kelly_fraction, 4),
        "inventory_sma": e.inventory_sma,
        "inventory_delta_days": e.inventory_delta_days,
        "basis_mom_lookback": e.basis_mom_lookback,
        "price_z_lookback": e.price_z_lookback,
        "atr_period": e.atr_period,
        "stop_atr_mult": round(e.stop_atr_mult, 4),
        "take_profit_atr_mult": round(e.take_profit_atr_mult, 4),
    }


@dataclass
class CandidateResult:
    name: str
    mean_oos_sharpe: float
    median_oos_sharpe: float
    mean_is_sharpe: float
    full: Dict[str, float]
    holdout_sharpe: float
    holdout_return: float
    oos_sharpes: List[float]
    ensemble: Dict[str, Any]
    # UPI dual-WFA fields
    rolling_mean_oos_upi: float = 0.0
    anchored_mean_oos_upi: float = 0.0
    composite_oos_upi: float = 0.0
    min_scheme_oos_upi: float = 0.0
    rolling_median_oos_upi: float = 0.0
    anchored_median_oos_upi: float = 0.0
    rolling_oos_upis: List[float] = field(default_factory=list)
    anchored_oos_upis: List[float] = field(default_factory=list)
    holdout_upi: float = 0.0
    full_upi: float = 0.0
    rolling_mean_oos_return: float = 0.0
    anchored_mean_oos_return: float = 0.0


def _baseline_ensemble_dict(cfg: Config) -> Dict[str, Any]:
    e = cfg.ensemble
    return {
        "horizons_days": list(e.horizons_days),
        "weights": dict(e.weights),
        "forecast_cap": e.forecast_cap,
        "agreement_min": e.agreement_min,
        "fdm_cap": e.fdm_cap,
        "vol_target_annual": e.vol_target_annual,
        "ewma_vol_com_days": e.ewma_vol_com_days,
        "fade_z": e.fade_z,
        "buffer_forecast": e.buffer_forecast,
        "kelly_fraction": e.kelly_fraction,
        "inventory_sma": e.inventory_sma,
        "inventory_delta_days": e.inventory_delta_days,
        "basis_mom_lookback": e.basis_mom_lookback,
        "price_z_lookback": e.price_z_lookback,
        "atr_period": e.atr_period,
        "stop_atr_mult": e.stop_atr_mult,
        "take_profit_atr_mult": e.take_profit_atr_mult,
    }


def build_default_candidates(base: Config) -> List[Tuple[str, Config]]:
    """
    Tiny pre-specified candidate set (economic priors only).

    Selecting among ≤6 candidates via nested OOS is far safer than free Optuna
    over a continuous space on a single futures market.
    """
    # Anchor to a known trading baseline (fade-on, 21/63/252) regardless of
    # whatever is currently in default.yaml — keeps the menu fixed.
    anchor = {
        "horizons_days": [21, 63, 252],
        "weights": {
            "tsmom": 0.55,
            "carry": 0.15,
            "basis_mom": 0.15,
            "inventory": 0.10,
            "fade": 0.05,
        },
        "forecast_cap": 20.0,
        "agreement_min": 0.20,
        "fdm_cap": 2.0,
        "vol_target_annual": 0.15,
        "ewma_vol_com_days": 60,
        "fade_z": 2.0,
        "buffer_forecast": 1.0,
        "kelly_fraction": 0.40,
        "inventory_sma": 60,
        "inventory_delta_days": 20,
        "basis_mom_lookback": 63,
        "price_z_lookback": 20,
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "take_profit_atr_mult": 4.0,
    }
    root = clone_config(base, ensemble_overrides=anchor)
    w_nf = rebuild_weights(anchor["weights"], 0.55, False)
    w_heavy = rebuild_weights(anchor["weights"], 0.80, False)
    return [
        ("A_baseline", root),
        (
            "B_no_fade",
            clone_config(root, ensemble_overrides={**anchor, "weights": w_nf}),
        ),
        (
            "C_no_fade_long_h",
            clone_config(
                root,
                ensemble_overrides={
                    **anchor,
                    "weights": w_nf,
                    "horizons_days": [63, 126, 252],
                },
            ),
        ),
        (
            "D_no_fade_strict",
            clone_config(
                root,
                ensemble_overrides={
                    **anchor,
                    "weights": w_nf,
                    "agreement_min": 0.30,
                    "buffer_forecast": 2.0,
                    "vol_target_annual": 0.12,
                    "kelly_fraction": 0.35,
                },
            ),
        ),
        (
            "E_tsmom_heavy",
            clone_config(
                root,
                ensemble_overrides={
                    **anchor,
                    "weights": w_heavy,
                    "horizons_days": [63, 126, 252],
                    "agreement_min": 0.25,
                    "buffer_forecast": 1.5,
                    "vol_target_annual": 0.12,
                },
            ),
        ),
        (
            "F_no_fade_med_h",
            clone_config(
                root,
                ensemble_overrides={
                    **anchor,
                    "weights": w_nf,
                    "horizons_days": [21, 63, 126],
                },
            ),
        ),
    ]


def _score_slices_upi(
    cfg: Config,
    bars: List[Bar],
    slices: Sequence[Tuple[int, int, int, int]],
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """Return OOS UPIs, IS UPIs, OOS returns, OOS Sharpes, IS Sharpes."""
    oos_upi: List[float] = []
    is_upi: List[float] = []
    oos_ret: List[float] = []
    oos_sharpe: List[float] = []
    is_sharpe: List[float] = []
    for is0, is1, oos0, oos1 in slices:
        is_m = _run_metrics(cfg, bars[is0:is1])
        oos_m = _run_metrics(cfg, bars[oos0:oos1])
        is_upi.append(float(is_m.get("upi", 0.0)))
        oos_upi.append(float(oos_m.get("upi", 0.0)))
        oos_ret.append(float(oos_m.get("total_return", 0.0)))
        oos_sharpe.append(float(oos_m.get("sharpe", 0.0)))
        is_sharpe.append(float(is_m.get("sharpe", 0.0)))
    return oos_upi, is_upi, oos_ret, oos_sharpe, is_sharpe


def select_pre_specified_candidates(
    base: Config,
    bars: List[Bar],
    *,
    n_windows: Optional[int] = None,
    is_ratio: float = 0.70,
    purge: int = 5,
    metric: str = "upi",
) -> Tuple[CandidateResult, List[CandidateResult]]:
    """
    Score a fixed candidate menu on nested purged OOS.

    Default ``metric='upi'`` ranks by dual-scheme Ulcer Performance Index:
    rolling + anchored mean OOS UPI (composite), with robustness via
    ``min(rolling, anchored)``.

    ``metric='sharpe'`` preserves the prior Sharpe-only rolling ranking.
    """
    if n_windows is None:
        n_windows = (
            base.backtest.walk_forward_windows_long
            if len(bars) >= base.backtest.long_history_bars
            else base.backtest.walk_forward_windows
        )
    n_windows = int(n_windows)
    rolling_slices = rolling_window_slices(len(bars), n_windows, is_ratio, purge)
    anchored_slices = anchored_window_slices(
        len(bars),
        n_windows,
        min_is_bars=max(504, int(len(bars) * 0.25)),
        purge=purge,
    )
    if not rolling_slices:
        raise ValueError("series too short for candidate WFA")

    split = int(len(bars) * 0.70)
    hold_bars = bars[min(split + purge, len(bars) - 50) :]
    results: List[CandidateResult] = []

    for name, cfg in build_default_candidates(base):
        r_oos_upi, r_is_upi, r_oos_ret, r_oos_sh, r_is_sh = _score_slices_upi(
            cfg, bars, rolling_slices
        )
        if anchored_slices:
            a_oos_upi, _a_is_upi, a_oos_ret, _a_oos_sh, _a_is_sh = _score_slices_upi(
                cfg, bars, anchored_slices
            )
        else:
            a_oos_upi, a_oos_ret = [], []

        full = _run_metrics(cfg, bars)
        hold = _run_metrics(cfg, hold_bars)

        roll_mean_upi = float(np.mean(r_oos_upi)) if r_oos_upi else 0.0
        anch_mean_upi = float(np.mean(a_oos_upi)) if a_oos_upi else roll_mean_upi
        composite = 0.5 * roll_mean_upi + 0.5 * anch_mean_upi
        min_scheme = float(min(roll_mean_upi, anch_mean_upi))

        mean_oos_sh = float(np.mean(r_oos_sh)) if r_oos_sh else 0.0
        med_oos_sh = float(np.median(r_oos_sh)) if r_oos_sh else 0.0
        mean_is_sh = float(np.mean(r_is_sh)) if r_is_sh else 0.0

        results.append(
            CandidateResult(
                name=name,
                mean_oos_sharpe=mean_oos_sh,
                median_oos_sharpe=med_oos_sh,
                mean_is_sharpe=mean_is_sh,
                full=dict(full),
                holdout_sharpe=float(hold.get("sharpe", 0.0)),
                holdout_return=float(hold.get("total_return", 0.0)),
                oos_sharpes=r_oos_sh,
                ensemble=_baseline_ensemble_dict(cfg),
                rolling_mean_oos_upi=roll_mean_upi,
                anchored_mean_oos_upi=anch_mean_upi,
                composite_oos_upi=composite,
                min_scheme_oos_upi=min_scheme,
                rolling_median_oos_upi=float(np.median(r_oos_upi)) if r_oos_upi else 0.0,
                anchored_median_oos_upi=float(np.median(a_oos_upi)) if a_oos_upi else 0.0,
                rolling_oos_upis=r_oos_upi,
                anchored_oos_upis=a_oos_upi,
                holdout_upi=float(hold.get("upi", 0.0)),
                full_upi=float(full.get("upi", 0.0)),
                rolling_mean_oos_return=float(np.mean(r_oos_ret)) if r_oos_ret else 0.0,
                anchored_mean_oos_return=float(np.mean(a_oos_ret)) if a_oos_ret else 0.0,
            )
        )

    if metric == "sharpe":
        ranked = sorted(
            results,
            key=lambda r: (r.mean_oos_sharpe, r.median_oos_sharpe, r.holdout_sharpe),
            reverse=True,
        )
    else:
        # Profitable + robust: composite UPI, then worst-scheme UPI, then full UPI
        ranked = sorted(
            results,
            key=lambda r: (
                r.composite_oos_upi,
                r.min_scheme_oos_upi,
                r.full_upi,
                r.holdout_upi,
            ),
            reverse=True,
        )
    return ranked[0], ranked
