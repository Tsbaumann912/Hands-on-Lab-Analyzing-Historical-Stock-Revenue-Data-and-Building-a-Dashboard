"""Ablation, purged walk-forward, institutional anchored/rolling WFO, Deflated Sharpe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from silver_ensemble.blend import SLEEVE_ORDER
from silver_ensemble.engine import BacktestEngine, BacktestResult, _compute_metrics
from silver_ensemble.models import Bar, Config

BARS_PER_YEAR = 252


@dataclass
class WindowResult:
    """Legacy block-WFO result (used by validate_ensemble)."""

    window: int
    is_sharpe: float
    oos_sharpe: float
    oos_max_dd: float


@dataclass
class InstitutionalWindowResult:
    """One IS/OOS fold for anchored or rolling WFO."""

    mode: str
    window: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    is_metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    oos_equity: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


@dataclass
class ModeWFOReport:
    mode: str
    windows: List[InstitutionalWindowResult]
    stitched_metrics: Dict[str, float]
    stitched_equity: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    passed: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class InstitutionalWFOReport:
    rolling: ModeWFOReport
    anchored: ModeWFOReport
    passed: bool = False
    notes: List[str] = field(default_factory=list)
    account_size: float = 0.0
    n_bars: int = 0


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
    """Approximate DSR: PSR against expected max Sharpe under n_trials nulls."""
    n_trials = max(int(n_trials), 1)
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
    """Legacy disjoint-block WFO (kept for ablation validate_ensemble path)."""
    n = len(bars)
    results: List[WindowResult] = []
    if n_windows < 1 or n < 100:
        return results

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


def _years_to_bars(years: int) -> int:
    return int(years) * BARS_PER_YEAR


def build_rolling_windows(
    n_bars: int,
    is_bars: int,
    oos_bars: int,
    step_bars: int,
    purge: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Return list of (is_start, is_end, oos_start, oos_end) exclusive-end indices.

    IS is ``[is_start, is_end)``; OOS is ``[oos_start, oos_end)`` with
    ``oos_start = is_end + purge``.
    """
    windows: List[Tuple[int, int, int, int]] = []
    if is_bars < 50 or oos_bars < 20 or step_bars < 1 or n_bars < is_bars + purge + oos_bars:
        return windows

    is_start = 0
    w = 0
    while True:
        is_end = is_start + is_bars
        oos_start = is_end + purge
        oos_end = oos_start + oos_bars
        if oos_end > n_bars:
            break
        windows.append((is_start, is_end, oos_start, oos_end))
        w += 1
        is_start += step_bars
    return windows


def build_anchored_windows(
    n_bars: int,
    min_is_bars: int,
    oos_bars: int,
    step_bars: int,
    purge: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Anchored / expanding IS: always starts at 0.

    First OOS begins after ``min_is_bars + purge``; each step advances OOS by
    ``step_bars`` while IS grows to ``oos_start - purge``.
    """
    windows: List[Tuple[int, int, int, int]] = []
    if min_is_bars < 50 or oos_bars < 20 or step_bars < 1:
        return windows

    oos_start = min_is_bars + purge
    w = 0
    while True:
        oos_end = oos_start + oos_bars
        if oos_end > n_bars:
            break
        is_end = oos_start - purge
        if is_end < min_is_bars:
            break
        windows.append((0, is_end, oos_start, oos_end))
        w += 1
        oos_start += step_bars
    return windows


def stitch_oos_equity(
    window_equities: List[np.ndarray],
    initial_cash: float,
) -> np.ndarray:
    """
    Compound independent OOS equity segments into one path.

    Each segment is rebased to continue from the prior stitched level
    (flat restart between folds matches separate ``engine.run`` calls).
    """
    if not window_equities:
        return np.array([], dtype=np.float64)

    pieces: List[np.ndarray] = []
    level = float(initial_cash)
    for eq in window_equities:
        if eq.size < 2 or eq[0] <= 0:
            continue
        scaled = eq / eq[0] * level
        # Drop first point of subsequent segments to avoid duplicate join
        if pieces:
            pieces.append(scaled[1:])
        else:
            pieces.append(scaled)
        level = float(scaled[-1])
    if not pieces:
        return np.array([], dtype=np.float64)
    return np.concatenate(pieces)


def _gate_metrics(metrics: Dict[str, float], config: Config) -> Tuple[bool, List[str]]:
    v = config.validation
    notes: List[str] = []
    ok = True
    sharpe = float(metrics.get("sharpe", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    upi = float(metrics.get("upi", 0.0))
    max_dd = float(metrics.get("max_drawdown", 1.0))

    if sharpe < v.min_oos_sharpe:
        ok = False
        notes.append(f"stitched OOS Sharpe {sharpe:.3f} < min {v.min_oos_sharpe}")
    if cagr < v.min_oos_cagr:
        ok = False
        notes.append(f"stitched OOS CAGR {cagr:.3%} < min {v.min_oos_cagr:.3%}")
    if upi < v.min_oos_upi:
        ok = False
        notes.append(f"stitched OOS UPI {upi:.3f} < min {v.min_oos_upi}")
    if not (max_dd < v.max_drawdown_gate):
        ok = False
        notes.append(
            f"stitched OOS max DD {max_dd:.2%} not strictly below gate {v.max_drawdown_gate:.0%}"
        )
    if ok:
        notes.append(
            f"Gates OK: Sharpe={sharpe:.3f} CAGR={cagr:.3%} UPI={upi:.3f} maxDD={max_dd:.2%}"
        )
    return ok, notes


def _run_windows(
    engine: BacktestEngine,
    bars: List[Bar],
    mode: str,
    index_windows: List[Tuple[int, int, int, int]],
) -> List[InstitutionalWindowResult]:
    out: List[InstitutionalWindowResult] = []
    for w, (is_s, is_e, oos_s, oos_e) in enumerate(index_windows):
        is_res = engine.run(bars[is_s:is_e])
        oos_res = engine.run(bars[oos_s:oos_e])
        out.append(
            InstitutionalWindowResult(
                mode=mode,
                window=w,
                is_start=is_s,
                is_end=is_e,
                oos_start=oos_s,
                oos_end=oos_e,
                is_metrics=dict(is_res.metrics),
                oos_metrics=dict(oos_res.metrics),
                oos_equity=np.asarray(oos_res.equity_curve, dtype=np.float64),
            )
        )
    return out


def _mode_report(
    mode: str,
    windows: List[InstitutionalWindowResult],
    config: Config,
) -> ModeWFOReport:
    eqs = [w.oos_equity for w in windows]
    stitched = stitch_oos_equity(eqs, config.portfolio.initial_cash)
    metrics = _compute_metrics(stitched) if stitched.size else {
        "sharpe": 0.0,
        "cagr": 0.0,
        "upi": 0.0,
        "max_drawdown": 1.0,
        "ulcer_index": 0.0,
        "total_return": 0.0,
        "n_bars": 0.0,
    }
    passed, notes = _gate_metrics(metrics, config)
    if not windows:
        passed = False
        notes = ["No walk-forward windows produced (series too short?)"]
    return ModeWFOReport(
        mode=mode,
        windows=windows,
        stitched_metrics=metrics,
        stitched_equity=stitched,
        passed=passed,
        notes=notes,
    )


def rolling_walk_forward(
    engine: BacktestEngine,
    bars: List[Bar],
    is_bars: int,
    oos_bars: int,
    step_bars: int,
    purge: int,
) -> List[InstitutionalWindowResult]:
    idx = build_rolling_windows(len(bars), is_bars, oos_bars, step_bars, purge)
    return _run_windows(engine, bars, "rolling", idx)


def anchored_walk_forward(
    engine: BacktestEngine,
    bars: List[Bar],
    min_is_bars: int,
    oos_bars: int,
    step_bars: int,
    purge: int,
) -> List[InstitutionalWindowResult]:
    idx = build_anchored_windows(len(bars), min_is_bars, oos_bars, step_bars, purge)
    return _run_windows(engine, bars, "anchored", idx)


def run_institutional_wfo(config: Config, bars: List[Bar]) -> InstitutionalWFOReport:
    """Run both rolling and anchored WFO; gate on stitched OOS Sharpe/UPI/CAGR/maxDD."""
    engine = BacktestEngine(config)
    bt = config.backtest
    is_bars = _years_to_bars(bt.is_years)
    oos_bars = _years_to_bars(bt.oos_years)
    step_bars = _years_to_bars(bt.step_years)
    purge = int(bt.purge_bars)

    rolling_wins = rolling_walk_forward(engine, bars, is_bars, oos_bars, step_bars, purge)
    anchored_wins = anchored_walk_forward(engine, bars, is_bars, oos_bars, step_bars, purge)

    rolling_rep = _mode_report("rolling", rolling_wins, config)
    anchored_rep = _mode_report("anchored", anchored_wins, config)

    notes: List[str] = []
    notes.append(f"account_size={config.portfolio.initial_cash:,.0f} n_bars={len(bars)}")
    notes.append(f"rolling windows={len(rolling_wins)} anchored windows={len(anchored_wins)}")
    notes.extend([f"rolling: {n}" for n in rolling_rep.notes])
    notes.extend([f"anchored: {n}" for n in anchored_rep.notes])

    if config.validation.require_both_modes:
        passed = rolling_rep.passed and anchored_rep.passed
        if not passed:
            notes.append("require_both_modes: overall FAIL unless both modes pass gates")
    else:
        passed = rolling_rep.passed or anchored_rep.passed

    if passed:
        notes.append("Institutional WFO PASSED (Sharpe, UPI, CAGR, maxDD<30%)")
    else:
        notes.append("Institutional WFO FAILED")

    return InstitutionalWFOReport(
        rolling=rolling_rep,
        anchored=anchored_rep,
        passed=passed,
        notes=notes,
        account_size=float(config.portfolio.initial_cash),
        n_bars=len(bars),
    )


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
    n_trials = 1 + 2 * len(SLEEVE_ORDER)
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


def institutional_report_to_dict(report: InstitutionalWFOReport) -> Dict[str, object]:
    """JSON-serialisable institutional WFO payload."""

    def _win(w: InstitutionalWindowResult) -> Dict[str, object]:
        return {
            "mode": w.mode,
            "window": w.window,
            "is_start": w.is_start,
            "is_end": w.is_end,
            "oos_start": w.oos_start,
            "oos_end": w.oos_end,
            "is_metrics": {
                k: w.is_metrics.get(k)
                for k in ("sharpe", "cagr", "upi", "max_drawdown", "ulcer_index", "total_return")
            },
            "oos_metrics": {
                k: w.oos_metrics.get(k)
                for k in ("sharpe", "cagr", "upi", "max_drawdown", "ulcer_index", "total_return")
            },
        }

    def _mode(m: ModeWFOReport) -> Dict[str, object]:
        return {
            "mode": m.mode,
            "passed": m.passed,
            "notes": m.notes,
            "n_windows": len(m.windows),
            "stitched_metrics": m.stitched_metrics,
            "windows": [_win(w) for w in m.windows],
        }

    return {
        "account_size": report.account_size,
        "n_bars": report.n_bars,
        "passed": report.passed,
        "notes": report.notes,
        "passed_rolling": report.rolling.passed,
        "passed_anchored": report.anchored.passed,
        "rolling": _mode(report.rolling),
        "anchored": _mode(report.anchored),
    }
