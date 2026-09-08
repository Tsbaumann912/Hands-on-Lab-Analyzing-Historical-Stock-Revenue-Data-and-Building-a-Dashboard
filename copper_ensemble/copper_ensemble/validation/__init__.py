"""Ablation, purged walk-forward, and Deflated Sharpe helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm

from copper_ensemble.blend import SLEEVE_ORDER
from copper_ensemble.engine import BacktestEngine, BacktestResult
from copper_ensemble.models import Bar, Config


@dataclass
class WindowResult:
    window: int
    is_sharpe: float
    oos_sharpe: float
    oos_max_dd: float


@dataclass
class ValidationReport:
    full: Dict[str, float]
    ablations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    walk_forward: List[WindowResult] = field(default_factory=list)
    deflated_sharpe: float = 0.0
    oos_retention: float = 0.0
    passed: bool = False
    notes: List[str] = field(default_factory=list)


def probabilistic_sharpe(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """Bailey–López de Prado Probabilistic Sharpe Ratio (PSR)."""
    if n_obs < 3 or not np.isfinite(sharpe):
        return 0.0
    se = np.sqrt(
        (1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe**2) / max(n_obs - 1, 1)
    )
    if se <= 1e-12:
        return 1.0 if sharpe > sr_benchmark else 0.0
    return float(norm.cdf((sharpe - sr_benchmark) / se))


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_var: float = 1.0,
) -> float:
    """
    Approximate DSR: PSR against expected max Sharpe under n_trials nulls.
    """
    n_trials = max(int(n_trials), 1)
    # E[max SR] ≈ sr_var * ((1-γ) Φ^{-1}(1-1/N) + γ Φ^{-1}(1-1/(N e)))
    gamma = 0.5772156649
    e = np.e
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * e))
    sr0 = np.sqrt(max(sr_var, 1e-12)) * ((1.0 - gamma) * z1 + gamma * z2)
    return probabilistic_sharpe(sharpe, n_obs, skew=skew, kurt=kurt, sr_benchmark=float(sr0))


def run_ablation(engine: BacktestEngine, bars: List[Bar]) -> Dict[str, Dict[str, float]]:
    """Full book plus each sleeve alone and each sleeve dropped."""
    out: Dict[str, Dict[str, float]] = {}
    out["all"] = engine.run(bars).metrics

    for name in SLEEVE_ORDER:
        mask = {s: (s == name) for s in SLEEVE_ORDER}
        out[f"only_{name}"] = engine.run(bars, sleeve_mask=mask).metrics
        mask_drop = {s: (s != name) for s in SLEEVE_ORDER}
        out[f"drop_{name}"] = engine.run(bars, sleeve_mask=mask_drop).metrics
    return out


def purged_walk_forward(
    engine: BacktestEngine,
    bars: List[Bar],
    n_windows: int,
    is_ratio: float,
    purge: int,
) -> List[WindowResult]:
    n = len(bars)
    results: List[WindowResult] = []
    if n_windows < 1 or n < 100:
        return results

    # rolling blocks
    block = n // n_windows
    for w in range(n_windows):
        start = w * block
        end = n if w == n_windows - 1 else (w + 1) * block
        segment = bars[start:end]
        split = int(len(segment) * is_ratio)
        is_end = max(split - purge, 10)
        oos_start = min(split + purge, len(segment) - 10)
        if is_end < 50 or oos_start >= len(segment) - 5:
            continue
        is_res = engine.run(segment[:is_end])
        oos_res = engine.run(segment[oos_start:])
        results.append(
            WindowResult(
                window=w,
                is_sharpe=float(is_res.metrics.get("sharpe", 0.0)),
                oos_sharpe=float(oos_res.metrics.get("sharpe", 0.0)),
                oos_max_dd=float(oos_res.metrics.get("max_drawdown", 0.0)),
            )
        )
    return results


def validate_ensemble(config: Config, bars: List[Bar]) -> ValidationReport:
    engine = BacktestEngine(config)
    full = engine.run(bars)
    ablations = run_ablation(engine, bars)
    wf = purged_walk_forward(
        engine,
        bars,
        n_windows=config.backtest.walk_forward_windows,
        is_ratio=config.backtest.in_sample_ratio,
        purge=config.backtest.purge_bars,
    )

    rets = full.returns[np.isfinite(full.returns)]
    skew = float(np.mean(((rets - np.mean(rets)) / (np.std(rets) + 1e-12)) ** 3)) if rets.size > 3 else 0.0
    kurt = float(np.mean(((rets - np.mean(rets)) / (np.std(rets) + 1e-12)) ** 4)) if rets.size > 3 else 3.0
    n_trials = 1 + 2 * len(SLEEVE_ORDER)  # all + only + drop
    dsr = deflated_sharpe_ratio(
        sharpe=float(full.metrics.get("sharpe", 0.0)),
        n_obs=max(int(rets.size), 3),
        n_trials=n_trials,
        skew=skew,
        kurt=kurt,
    )

    if wf:
        mean_is = float(np.mean([w.is_sharpe for w in wf]))
        mean_oos = float(np.mean([w.oos_sharpe for w in wf]))
        if mean_is > 1e-9:
            retention = mean_oos / mean_is
        else:
            retention = 0.0 if mean_oos <= 0 else 1.0
    else:
        retention = 0.0

    notes: List[str] = []
    passed = True
    if full.metrics.get("sharpe", 0.0) <= 0:
        passed = False
        notes.append("Full-sample Sharpe is non-positive")
    if dsr < config.validation.dsr_pass_threshold:
        notes.append(
            f"DSR {dsr:.3f} below threshold {config.validation.dsr_pass_threshold}; "
            "raise trial discipline or improve edge before live."
        )
        if dsr < 0.5:
            passed = False
    if wf and mean_is > 0 and retention < config.validation.oos_retention_min:
        passed = False
        notes.append(
            f"OOS retention {retention:.2%} below min {config.validation.oos_retention_min:.0%}"
        )
    if not wf:
        notes.append("Walk-forward produced no windows (series too short?)")
        passed = False

    # Ensemble should not be worse than every single sleeve on Sharpe
    only_sharpes = [ablations[f"only_{s}"]["sharpe"] for s in SLEEVE_ORDER]
    if full.metrics["sharpe"] + 1e-9 < min(only_sharpes):
        notes.append("Warning: full blend Sharpe below every single-sleeve Sharpe")

    if passed and not notes:
        notes.append("All soft gates reviewed")
    elif passed:
        notes.append("Promotion gates passed with notes above")

    return ValidationReport(
        full=full.metrics,
        ablations=ablations,
        walk_forward=wf,
        deflated_sharpe=dsr,
        oos_retention=retention,
        passed=passed,
        notes=notes,
    )