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
    is_upi: float = 0.0
    oos_upi: float = 0.0
    oos_ulcer_index: float = 0.0
    scheme: str = "rolling"


@dataclass
class ValidationReport:
    full: Dict[str, float]
    ablations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    walk_forward: List[WindowResult] = field(default_factory=list)
    walk_forward_anchored: List[WindowResult] = field(default_factory=list)
    deflated_sharpe: float = 0.0
    oos_retention: float = 0.0
    mean_oos_sharpe: float = 0.0
    mean_is_sharpe: float = 0.0
    mean_oos_upi: float = 0.0
    mean_anchored_oos_upi: float = 0.0
    composite_oos_upi: float = 0.0
    passed: bool = False
    target_oos_met: bool = False
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
    """Rolling (block) purged walk-forward with Sharpe + UPI."""
    from copper_ensemble.optimize import rolling_window_slices

    results: List[WindowResult] = []
    for w, (is0, is1, oos0, oos1) in enumerate(
        rolling_window_slices(len(bars), n_windows, is_ratio, purge)
    ):
        is_res = engine.run(bars[is0:is1])
        oos_res = engine.run(bars[oos0:oos1])
        results.append(
            WindowResult(
                window=w,
                is_sharpe=float(is_res.metrics.get("sharpe", 0.0)),
                oos_sharpe=float(oos_res.metrics.get("sharpe", 0.0)),
                oos_max_dd=float(oos_res.metrics.get("max_drawdown", 0.0)),
                is_upi=float(is_res.metrics.get("upi", 0.0)),
                oos_upi=float(oos_res.metrics.get("upi", 0.0)),
                oos_ulcer_index=float(oos_res.metrics.get("ulcer_index", 0.0)),
                scheme="rolling",
            )
        )
    return results


def anchored_walk_forward(
    engine: BacktestEngine,
    bars: List[Bar],
    n_windows: int,
    purge: int,
) -> List[WindowResult]:
    """Anchored (expanding IS) purged walk-forward with Sharpe + UPI."""
    from copper_ensemble.optimize import anchored_window_slices

    results: List[WindowResult] = []
    for w, (is0, is1, oos0, oos1) in enumerate(
        anchored_window_slices(
            len(bars),
            n_windows,
            min_is_bars=max(504, int(len(bars) * 0.25)),
            purge=purge,
        )
    ):
        is_res = engine.run(bars[is0:is1])
        oos_res = engine.run(bars[oos0:oos1])
        results.append(
            WindowResult(
                window=w,
                is_sharpe=float(is_res.metrics.get("sharpe", 0.0)),
                oos_sharpe=float(oos_res.metrics.get("sharpe", 0.0)),
                oos_max_dd=float(oos_res.metrics.get("max_drawdown", 0.0)),
                is_upi=float(is_res.metrics.get("upi", 0.0)),
                oos_upi=float(oos_res.metrics.get("upi", 0.0)),
                oos_ulcer_index=float(oos_res.metrics.get("ulcer_index", 0.0)),
                scheme="anchored",
            )
        )
    return results


def validate_ensemble(config: Config, bars: List[Bar]) -> ValidationReport:
    engine = BacktestEngine(config)
    full = engine.run(bars)
    ablations = run_ablation(engine, bars)

    n_windows = config.backtest.walk_forward_windows
    if len(bars) >= config.backtest.long_history_bars:
        n_windows = config.backtest.walk_forward_windows_long

    wf = purged_walk_forward(
        engine,
        bars,
        n_windows=n_windows,
        is_ratio=config.backtest.in_sample_ratio,
        purge=config.backtest.purge_bars,
    )
    wf_anch = anchored_walk_forward(
        engine,
        bars,
        n_windows=n_windows,
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
        mean_oos_upi = float(np.mean([w.oos_upi for w in wf]))
        if mean_is > 1e-9:
            retention = mean_oos / mean_is
        else:
            retention = 0.0 if mean_oos <= 0 else 1.0
    else:
        mean_is = 0.0
        mean_oos = 0.0
        mean_oos_upi = 0.0
        retention = 0.0

    mean_anch_upi = (
        float(np.mean([w.oos_upi for w in wf_anch])) if wf_anch else mean_oos_upi
    )
    composite_upi = 0.5 * mean_oos_upi + 0.5 * mean_anch_upi

    target = config.validation.target_mean_oos_sharpe
    target_oos_met = bool(wf) and mean_oos >= target

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

    notes.append(
        f"Mean rolling OOS Sharpe = {mean_oos:.3f} "
        f"(target {target:.1f}; {'MET' if target_oos_met else 'NOT MET'})."
    )
    notes.append(
        f"UPI dual-WFA: rolling mean OOS UPI={mean_oos_upi:.3f}, "
        f"anchored mean OOS UPI={mean_anch_upi:.3f}, "
        f"composite={composite_upi:.3f}; full-sample UPI={float(full.metrics.get('upi', 0.0)):.3f} "
        f"(Ulcer Index={float(full.metrics.get('ulcer_index', 0.0)):.2f}%)."
    )
    if not target_oos_met:
        passed = False
        notes.append(
            "Single-name HG cannot honestly clear ~1.5 mean OOS Sharpe: "
            "AQR/Moskowitz TSMOM Sharpes ~1.5–1.8 require 50+ diversified futures, "
            "not one copper contract. Yahoo curve/inventory/PMI are proxies. "
            "Synthetic data can exceed 1.5 because the edge is planted — that is not "
            "evidence for live copper trading."
        )

    only_sharpes = [ablations[f"only_{s}"]["sharpe"] for s in SLEEVE_ORDER]
    if full.metrics["sharpe"] + 1e-9 < min(only_sharpes):
        notes.append("Warning: full blend Sharpe below every single-sleeve Sharpe")

    if passed and len(notes) <= 1:
        notes.append("All soft gates reviewed")
    elif passed:
        notes.append("Promotion gates passed with notes above")

    return ValidationReport(
        full=full.metrics,
        ablations=ablations,
        walk_forward=wf,
        walk_forward_anchored=wf_anch,
        deflated_sharpe=dsr,
        oos_retention=retention,
        mean_oos_sharpe=mean_oos,
        mean_is_sharpe=mean_is,
        mean_oos_upi=mean_oos_upi,
        mean_anchored_oos_upi=mean_anch_upi,
        composite_oos_upi=composite_upi,
        passed=passed,
        target_oos_met=target_oos_met,
        notes=notes,
    )
