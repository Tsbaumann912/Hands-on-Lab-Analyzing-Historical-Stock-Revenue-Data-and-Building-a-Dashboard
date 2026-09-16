"""Ablation, purged / anchored / rolling walk-forward, and Deflated Sharpe helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from platinum_ensemble.blend import SLEEVE_ORDER
from platinum_ensemble.engine import BacktestEngine, BacktestResult, _compute_metrics
from platinum_ensemble.models import Bar, Config


@dataclass
class WindowResult:
    window: int
    is_sharpe: float
    oos_sharpe: float
    oos_max_dd: float
    oos_cagr: float = 0.0
    oos_ulcer_index: float = 0.0
    oos_upi: float = 0.0
    is_start: int = 0
    is_end: int = 0
    oos_start: int = 0
    oos_end: int = 0


@dataclass
class ValidationReport:
    full: Dict[str, float]
    ablations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    walk_forward: List[WindowResult] = field(default_factory=list)
    deflated_sharpe: float = 0.0
    oos_retention: float = 0.0
    passed: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class ModeWFOResult:
    mode: str
    windows: List[WindowResult]
    stitched_metrics: Dict[str, float]
    stitched_equity: np.ndarray
    passed: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class InstitutionalWFOReport:
    account_size: float
    n_bars: int
    data_start: Optional[str]
    data_end: Optional[str]
    anchored: ModeWFOResult
    rolling: ModeWFOResult
    passed: bool
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
                oos_cagr=float(oos_res.metrics.get("cagr", 0.0)),
                oos_ulcer_index=float(oos_res.metrics.get("ulcer_index", 0.0)),
                oos_upi=float(oos_res.metrics.get("upi", 0.0)),
                is_start=start,
                is_end=start + is_end,
                oos_start=start + oos_start,
                oos_end=end,
            )
        )
    return results


def _window_from_runs(
    window: int,
    is_res: BacktestResult,
    oos_res: BacktestResult,
    is_start: int,
    is_end: int,
    oos_start: int,
    oos_end: int,
) -> WindowResult:
    return WindowResult(
        window=window,
        is_sharpe=float(is_res.metrics.get("sharpe", 0.0)),
        oos_sharpe=float(oos_res.metrics.get("sharpe", 0.0)),
        oos_max_dd=float(oos_res.metrics.get("max_drawdown", 0.0)),
        oos_cagr=float(oos_res.metrics.get("cagr", 0.0)),
        oos_ulcer_index=float(oos_res.metrics.get("ulcer_index", 0.0)),
        oos_upi=float(oos_res.metrics.get("upi", 0.0)),
        is_start=is_start,
        is_end=is_end,
        oos_start=oos_start,
        oos_end=oos_end,
    )


def anchored_walk_forward(
    engine: BacktestEngine,
    bars: List[Bar],
    initial_is_bars: int,
    oos_bars: int,
    purge: int,
) -> Tuple[List[WindowResult], List[np.ndarray]]:
    """
    Anchored WFO: IS always starts at bar 0; OOS segments advance by ``oos_bars``.

    Window k: IS = [0, T_k), OOS = [T_k + purge, T_k + purge + oos_bars)
    with T_0 = initial_is_bars, T_{k+1} = T_k + oos_bars.
    """
    n = len(bars)
    results: List[WindowResult] = []
    oos_equities: List[np.ndarray] = []
    if n < initial_is_bars + purge + oos_bars:
        return results, oos_equities

    t_k = initial_is_bars
    w = 0
    while t_k + purge + oos_bars <= n:
        is_end = t_k
        oos_start = t_k + purge
        oos_end = min(oos_start + oos_bars, n)
        if oos_end - oos_start < max(20, oos_bars // 4):
            break
        is_res = engine.run(bars[:is_end])
        oos_res = engine.run(bars[oos_start:oos_end])
        results.append(
            _window_from_runs(w, is_res, oos_res, 0, is_end, oos_start, oos_end)
        )
        oos_equities.append(oos_res.equity_curve)
        t_k += oos_bars
        w += 1
    return results, oos_equities


def rolling_walk_forward(
    engine: BacktestEngine,
    bars: List[Bar],
    is_bars: int,
    oos_bars: int,
    step_bars: int,
    purge: int,
) -> Tuple[List[WindowResult], List[np.ndarray]]:
    """
    Rolling WFO: fixed-length IS and OOS windows sliding by ``step_bars``.

    Window k: IS = [k·step, k·step + is_bars),
              OOS = [IS_end + purge, IS_end + purge + oos_bars).
    """
    n = len(bars)
    results: List[WindowResult] = []
    oos_equities: List[np.ndarray] = []
    if n < is_bars + purge + oos_bars:
        return results, oos_equities

    start = 0
    w = 0
    while True:
        is_start = start
        is_end = start + is_bars
        oos_start = is_end + purge
        oos_end = oos_start + oos_bars
        if oos_end > n:
            break
        is_res = engine.run(bars[is_start:is_end])
        oos_res = engine.run(bars[oos_start:oos_end])
        results.append(
            _window_from_runs(w, is_res, oos_res, is_start, is_end, oos_start, oos_end)
        )
        oos_equities.append(oos_res.equity_curve)
        start += step_bars
        w += 1
    return results, oos_equities


def stitch_oos_equity(
    oos_equities: List[np.ndarray],
    initial_cash: float,
) -> np.ndarray:
    """
    Stitch OOS equity curves by compounding period returns onto ``initial_cash``.

    Each segment contributes its relative equity path (normalised to 1 at start).
    """
    if not oos_equities:
        return np.array([initial_cash], dtype=np.float64)

    pieces: List[np.ndarray] = []
    level = float(initial_cash)
    for eq in oos_equities:
        if eq.size < 2:
            continue
        rel = eq / max(float(eq[0]), 1e-12)
        # Drop first point of subsequent segments to avoid duplicate join bar
        seg = level * rel
        if pieces:
            pieces.append(seg[1:])
        else:
            pieces.append(seg)
        level = float(seg[-1])
    if not pieces:
        return np.array([initial_cash], dtype=np.float64)
    return np.concatenate(pieces)


def evaluate_stitched_gates(
    metrics: Dict[str, float],
    config: Config,
) -> Tuple[bool, List[str]]:
    """Apply institutional success gates on stitched OOS metrics."""
    notes: List[str] = []
    passed = True
    max_dd = float(metrics.get("max_drawdown", 1.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    upi = float(metrics.get("upi", 0.0))

    if max_dd >= config.validation.max_oos_drawdown:
        passed = False
        notes.append(
            f"Max DD {max_dd:.2%} ≥ limit {config.validation.max_oos_drawdown:.0%}"
        )
    if sharpe <= config.validation.min_oos_sharpe:
        passed = False
        notes.append(f"Sharpe {sharpe:.3f} ≤ min {config.validation.min_oos_sharpe}")
    if cagr <= config.validation.min_oos_cagr:
        passed = False
        notes.append(f"CAGR {cagr:.2%} ≤ min {config.validation.min_oos_cagr:.0%}")
    if upi <= config.validation.min_oos_upi:
        passed = False
        notes.append(f"UPI {upi:.3f} ≤ min {config.validation.min_oos_upi}")
    if passed:
        notes.append("Stitched OOS gates passed (Sharpe, UPI, CAGR, Max DD)")
    return passed, notes


def _institutional_config(config: Config, cash: Optional[float] = None) -> Config:
    """
    Copy config for institutional WFO.

    Disables the daily DD circuit-breaker halt so Max DD reflects strategy path
    risk rather than a permanent flat after a 3% daily halt.
    """
    risk = replace(config.risk, halt_on_breach=False)
    portfolio = replace(
        config.portfolio,
        initial_cash=float(cash if cash is not None else config.portfolio.initial_cash),
    )
    return replace(config, risk=risk, portfolio=portfolio)


def run_institutional_wfo(
    config: Config,
    bars: List[Bar],
    cash: Optional[float] = None,
) -> InstitutionalWFOReport:
    """Run anchored + rolling WFO and gate on stitched OOS Sharpe/UPI/CAGR/MaxDD."""
    cfg = _institutional_config(config, cash=cash)
    engine = BacktestEngine(cfg)
    initial = cfg.portfolio.initial_cash
    bt = cfg.backtest

    data_start = str(bars[0].timestamp) if bars else None
    data_end = str(bars[-1].timestamp) if bars else None

    a_windows, a_eqs = anchored_walk_forward(
        engine,
        bars,
        initial_is_bars=bt.anchored_initial_is_bars,
        oos_bars=bt.oos_bars,
        purge=bt.purge_bars,
    )
    a_stitched = stitch_oos_equity(a_eqs, initial)
    a_metrics = _compute_metrics(a_stitched)
    a_pass, a_notes = evaluate_stitched_gates(a_metrics, cfg)
    if not a_windows:
        a_pass = False
        a_notes.append("Anchored WFO produced no windows (series too short?)")

    r_windows, r_eqs = rolling_walk_forward(
        engine,
        bars,
        is_bars=bt.rolling_is_bars,
        oos_bars=bt.oos_bars,
        step_bars=bt.rolling_step_bars,
        purge=bt.purge_bars,
    )
    r_stitched = stitch_oos_equity(r_eqs, initial)
    r_metrics = _compute_metrics(r_stitched)
    r_pass, r_notes = evaluate_stitched_gates(r_metrics, cfg)
    if not r_windows:
        r_pass = False
        r_notes.append("Rolling WFO produced no windows (series too short?)")

    overall = a_pass and r_pass
    notes = [
        f"Account size ${initial:,.0f}",
        f"Bars={len(bars)} from {data_start} to {data_end}",
        "Daily DD halt disabled for institutional path evaluation",
        "Evaluation WFO (fixed YAML params — IS is not re-optimised)",
    ]
    # With matching IS length / OOS / step, OOS calendars coincide; IS paths still differ.
    if (
        bt.anchored_initial_is_bars == bt.rolling_is_bars
        and bt.oos_bars == bt.rolling_step_bars
        and a_windows
        and r_windows
        and len(a_windows) == len(r_windows)
    ):
        notes.append(
            "Note: under default IS=756 / OOS=step=252, anchored and rolling share "
            "the same OOS calendar; stitched OOS metrics match when params are fixed."
        )
    if overall:
        notes.append("Overall PASS: both anchored and rolling stitched OOS gates met")
    else:
        notes.append("Overall FAIL: one or both modes failed stitched OOS gates")

    return InstitutionalWFOReport(
        account_size=initial,
        n_bars=len(bars),
        data_start=data_start,
        data_end=data_end,
        anchored=ModeWFOResult(
            mode="anchored",
            windows=a_windows,
            stitched_metrics=a_metrics,
            stitched_equity=a_stitched,
            passed=a_pass,
            notes=a_notes,
        ),
        rolling=ModeWFOResult(
            mode="rolling",
            windows=r_windows,
            stitched_metrics=r_metrics,
            stitched_equity=r_stitched,
            passed=r_pass,
            notes=r_notes,
        ),
        passed=overall,
        notes=notes,
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
    """JSON-serialisable institutional WFO payload (equity curves omitted)."""

    def _mode(m: ModeWFOResult) -> Dict[str, object]:
        return {
            "mode": m.mode,
            "passed": m.passed,
            "notes": m.notes,
            "stitched_metrics": m.stitched_metrics,
            "n_windows": len(m.windows),
            "windows": [
                {
                    "window": w.window,
                    "is_sharpe": w.is_sharpe,
                    "oos_sharpe": w.oos_sharpe,
                    "oos_max_dd": w.oos_max_dd,
                    "oos_cagr": w.oos_cagr,
                    "oos_ulcer_index": w.oos_ulcer_index,
                    "oos_upi": w.oos_upi,
                    "is_start": w.is_start,
                    "is_end": w.is_end,
                    "oos_start": w.oos_start,
                    "oos_end": w.oos_end,
                }
                for w in m.windows
            ],
        }

    return {
        "account_size": report.account_size,
        "n_bars": report.n_bars,
        "data_start": report.data_start,
        "data_end": report.data_end,
        "passed": report.passed,
        "notes": report.notes,
        "anchored": _mode(report.anchored),
        "rolling": _mode(report.rolling),
    }
