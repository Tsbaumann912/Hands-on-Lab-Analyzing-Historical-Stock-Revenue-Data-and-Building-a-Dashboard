"""
Anchored and sliding-rolling walk-forward window builders, OOS equity stitch,
and promotion gates (Sharpe, UPI, CAGR, max DD).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np

from core.config import Config
from core.models import Bar
from engine.backtest import BacktestEngine, BacktestResult
from engine.metrics import compute_metrics
from engine.optimizer import (
    HAS_OPTUNA,
    SearchSpace,
    WFOResult,
    WFOWindow,
    WalkForwardOptimizer,
)
from strategies.base import Strategy

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def years_to_bars(years: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> int:
    """Convert year fractions to an integer bar count."""
    return max(1, int(round(float(years) * periods_per_year)))


def _ref_length(bars: Dict[str, List[Bar]]) -> int:
    ref = next(iter(bars.values()))
    return len(ref)


def _slice_bars(
    bars: Dict[str, List[Bar]], start: int, end: int
) -> Dict[str, List[Bar]]:
    return {sym: b[start:end] for sym, b in bars.items()}


def build_anchored_windows(
    bars: Dict[str, List[Bar]],
    min_is_bars: int,
    oos_bars: int,
    step_bars: Optional[int] = None,
) -> List[WFOWindow]:
    """
    Anchored (expanding) walk-forward windows.

    Fold ``k``: IS = ``[0, min_is + k·step)``, OOS = next ``oos_bars`` bars.
    The series start is fixed (anchored).
    """
    n = _ref_length(bars)
    step = int(step_bars if step_bars is not None else oos_bars)
    min_is = int(min_is_bars)
    oos = int(oos_bars)
    if min_is < 1 or oos < 1 or step < 1:
        raise ValueError("min_is_bars, oos_bars, and step_bars must be >= 1")
    if n < min_is + oos:
        return []

    windows: List[WFOWindow] = []
    wid = 0
    k = 0
    while True:
        is_end = min_is + k * step
        oos_start = is_end
        oos_end = oos_start + oos
        if oos_end > n:
            break
        windows.append(
            WFOWindow(
                window_id=wid,
                is_bars=_slice_bars(bars, 0, is_end),
                oos_bars=_slice_bars(bars, oos_start, oos_end),
            )
        )
        wid += 1
        k += 1
    return windows


def build_rolling_windows(
    bars: Dict[str, List[Bar]],
    is_bars: int,
    oos_bars: int,
    step_bars: Optional[int] = None,
    purge_bars: int = 0,
) -> List[WFOWindow]:
    """
    Sliding rolling walk-forward windows with optional purge gap.

    IS length and OOS length are fixed; the window advances by ``step_bars``.
    When ``purge_bars > 0``, that many bars between IS end and OOS start are
    dropped (embargo against leakage).
    """
    n = _ref_length(bars)
    is_len = int(is_bars)
    oos = int(oos_bars)
    step = int(step_bars if step_bars is not None else oos)
    purge = max(0, int(purge_bars))
    if is_len < 1 or oos < 1 or step < 1:
        raise ValueError("is_bars, oos_bars, and step_bars must be >= 1")
    if n < is_len + purge + oos:
        return []

    windows: List[WFOWindow] = []
    wid = 0
    start = 0
    while True:
        is_end = start + is_len
        oos_start = is_end + purge
        oos_end = oos_start + oos
        if oos_end > n:
            break
        windows.append(
            WFOWindow(
                window_id=wid,
                is_bars=_slice_bars(bars, start, is_end),
                oos_bars=_slice_bars(bars, oos_start, oos_end),
            )
        )
        wid += 1
        start += step
    return windows


def stitch_oos_equity(
    window_results: Sequence[WFOWindow],
    initial_cash: float,
) -> np.ndarray:
    """
    Stitch per-window OOS equity curves into one path starting at ``initial_cash``.

    Uses period returns from each window's OOS equity so capital compounds
    across folds without resetting to the window-local starting equity.
    """
    eq, _ = stitch_oos_equity_timed(window_results, initial_cash)
    return eq


def stitch_oos_equity_timed(
    window_results: Sequence[WFOWindow],
    initial_cash: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like ``stitch_oos_equity`` but also returns timestamps aligned 1:1 with equity.

    The synthetic starting cash point inherits the first available OOS timestamp
    (or is omitted from year gating when no stamps exist).
    """
    equity = float(initial_cash)
    path: List[float] = [equity]
    times: List[Any] = []
    first_ts: Any = None

    for window in window_results:
        result = window.oos_result
        if result is None or len(result.equity_curve) < 2:
            continue
        eq = np.asarray(result.equity_curve, dtype=np.float64)
        ts = result.equity_timestamps
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.diff(eq) / np.where(eq[:-1] != 0.0, eq[:-1], np.nan)
        valid = np.isfinite(rets)
        rets = rets[valid]
        if len(rets) == 0:
            continue

        growth = np.cumprod(1.0 + rets)
        segment = equity * growth
        path.extend(segment.tolist())
        equity = float(segment[-1])

        if ts is not None and len(ts) == len(eq):
            seg_ts = [ts[i + 1] for i, ok in enumerate(valid) if ok]
            if first_ts is None and len(ts) > 0:
                first_ts = ts[0]
            times.extend(seg_ts)
        elif window.oos_bars:
            # Fall back to OOS bar timestamps when engine stamps are missing.
            ref = next(iter(window.oos_bars.values()))
            if len(ref) >= 2:
                n = min(len(rets), max(0, len(ref) - 1))
                seg_ts = [ref[i + 1].timestamp for i in range(n)]
                if first_ts is None and len(ref) > 0:
                    first_ts = ref[0].timestamp
                times.extend(seg_ts)
        else:
            # Synthetic / unit-test windows without stamps — keep equity path only.
            pass

    if first_ts is not None:
        times = [first_ts] + times
    # If lengths diverge (missing stamps), return empty times → year gate fails closed.
    if len(times) != len(path):
        return np.asarray(path, dtype=np.float64), np.asarray([], dtype=object)
    return np.asarray(path, dtype=np.float64), np.asarray(times, dtype=object)


@dataclass
class GateThresholds:
    max_dd_limit: float = 0.30
    min_sharpe: float = 0.0
    min_upi: float = 0.0
    min_cagr: float = 0.0
    require_all_years_profitable: bool = False
    min_year_return: float = 0.0
    min_year_bars: int = 2
    # When True with require_all_years_profitable, enforce each OOS window's
    # total_return >= min_year_return (WFO-native ~annual folds) in addition
    # to stitched calendar-year returns.
    require_oos_windows_profitable: bool = True


@dataclass
class GateResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    calendar_year_returns: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "metrics": dict(self.metrics),
            "calendar_year_returns": {
                str(k): v for k, v in self.calendar_year_returns.items()
            },
        }


def evaluate_gates(
    metrics: Dict[str, float],
    thresholds: Optional[GateThresholds] = None,
    *,
    equity_curve: Optional[np.ndarray] = None,
    equity_timestamps: Optional[np.ndarray] = None,
    oos_window_returns: Optional[Sequence[float]] = None,
) -> GateResult:
    """
    Hard promotion gates on stitched OOS metrics.

    Requires positive Sharpe, UPI, CAGR and ``abs(max_drawdown) < max_dd_limit``.
    Optionally requires every evaluable calendar year and/or every OOS window
    to be non-losing.
    """
    from engine.metrics import all_calendar_years_profitable

    thr = thresholds or GateThresholds()
    failures: List[str] = []
    year_rets: Dict[int, float] = {}
    sharpe = float(metrics.get("sharpe_ratio", float("nan")))
    upi = float(metrics.get("ulcer_performance_index", float("nan")))
    cagr = float(metrics.get("cagr", float("nan")))
    max_dd = float(metrics.get("max_drawdown", float("nan")))

    if not np.isfinite(sharpe) or sharpe <= thr.min_sharpe:
        failures.append(f"sharpe_ratio={sharpe} <= {thr.min_sharpe}")
    if not np.isfinite(upi) or upi <= thr.min_upi:
        failures.append(f"ulcer_performance_index={upi} <= {thr.min_upi}")
    if not np.isfinite(cagr) or cagr <= thr.min_cagr:
        failures.append(f"cagr={cagr} <= {thr.min_cagr}")
    if not np.isfinite(max_dd) or abs(max_dd) >= thr.max_dd_limit:
        failures.append(
            f"abs(max_drawdown)={abs(max_dd) if np.isfinite(max_dd) else max_dd} "
            f">= {thr.max_dd_limit}"
        )

    if thr.require_all_years_profitable:
        if equity_curve is None or equity_timestamps is None or len(equity_timestamps) == 0:
            failures.append("calendar_year_gate_missing_timestamps")
        else:
            ok, year_rets, year_failures = all_calendar_years_profitable(
                equity_curve,
                equity_timestamps,
                min_bars=thr.min_year_bars,
                min_year_return=thr.min_year_return,
            )
            if not ok:
                failures.extend(year_failures)

    if thr.require_oos_windows_profitable and oos_window_returns is not None:
        for i, ret in enumerate(oos_window_returns):
            r = float(ret)
            if not np.isfinite(r) or r < thr.min_year_return - 1e-12:
                failures.append(
                    f"oos_window_{i}_return={r:.6f} < {thr.min_year_return}"
                )

    return GateResult(
        passed=len(failures) == 0,
        failures=failures,
        metrics=dict(metrics),
        calendar_year_returns=year_rets,
    )


@dataclass
class ModeValidationReport:
    mode: str
    wfo: WFOResult
    stitched_equity: np.ndarray
    stitched_metrics: Dict[str, float]
    gates: GateResult
    stitched_timestamps: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "n_windows": len(self.wfo.windows),
            "aggregated_oos_metrics": self.wfo.aggregated_oos_metrics,
            "best_params_per_window": self.wfo.best_params_per_window,
            "stitched_metrics": self.stitched_metrics,
            "gates": self.gates.to_dict(),
            "stitched_equity_final": float(self.stitched_equity[-1])
            if len(self.stitched_equity)
            else None,
            "stitched_equity_n": int(len(self.stitched_equity)),
            "calendar_year_returns": {
                str(k): v for k, v in self.gates.calendar_year_returns.items()
            },
        }


class WalkForwardValidator:
    """
    Run anchored and/or rolling WFO, stitch OOS equity, and evaluate gates.
    """

    def __init__(
        self,
        config: Config,
        strategy_cls: Type[Strategy],
        search_space: SearchSpace,
        objective_metric: str = "sharpe_ratio",
        thresholds: Optional[GateThresholds] = None,
    ) -> None:
        self._config = config
        self._strategy_cls = strategy_cls
        self._search_space = search_space
        self._objective_metric = objective_metric
        self._thresholds = thresholds or GateThresholds()
        self._optimizer = WalkForwardOptimizer(
            config,
            strategy_cls,
            search_space,
            objective_metric=objective_metric,
            oos_warmup_bars=int(
                getattr(config.cl_validation, "oos_warmup_bars", 320)
            ),
            require_all_years_profitable=bool(
                getattr(
                    config.cl_validation,
                    "optuna_require_years_profitable",
                    self._thresholds.require_all_years_profitable,
                )
            ),
            min_year_return=float(self._thresholds.min_year_return),
            min_year_bars=int(self._thresholds.min_year_bars),
        )

    def run_mode(
        self,
        bars: Dict[str, List[Bar]],
        mode: str,
        *,
        n_trials: int = 25,
        timeout: Optional[int] = 120,
        min_is_bars: Optional[int] = None,
        is_bars: Optional[int] = None,
        oos_bars: Optional[int] = None,
        step_bars: Optional[int] = None,
        purge_bars: int = 5,
    ) -> ModeValidationReport:
        mode_l = mode.lower().strip()
        if mode_l == "anchored":
            min_is = min_is_bars if min_is_bars is not None else years_to_bars(3)
            oos = oos_bars if oos_bars is not None else years_to_bars(1)
            step = step_bars if step_bars is not None else oos
            windows = build_anchored_windows(bars, min_is, oos, step)
        elif mode_l == "rolling":
            is_len = is_bars if is_bars is not None else years_to_bars(5)
            oos = oos_bars if oos_bars is not None else years_to_bars(1)
            step = step_bars if step_bars is not None else oos
            windows = build_rolling_windows(
                bars, is_len, oos, step, purge_bars=purge_bars
            )
        elif mode_l == "legacy_blocks":
            # Delegate to existing non-overlapping partition builder.
            wfo = self._optimizer.run(
                bars, n_windows=4, in_sample_ratio=0.70, n_trials=n_trials, timeout=timeout
            )
            return self._finalize(mode_l, wfo)
        else:
            raise ValueError(f"Unknown WFO mode: {mode!r}")

        if not windows:
            empty = WFOResult(windows=[], aggregated_oos_metrics={}, best_params_per_window=[])
            cash = float(self._config.portfolio.initial_cash)
            eq = np.array([cash], dtype=np.float64)
            return ModeValidationReport(
                mode=mode_l,
                wfo=empty,
                stitched_equity=eq,
                stitched_metrics={},
                gates=GateResult(
                    passed=False,
                    failures=["no_windows_built"],
                    metrics={},
                ),
            )

        wfo = self._optimizer.run_on_windows(
            windows, n_trials=n_trials, timeout=timeout
        )
        return self._finalize(mode_l, wfo)

    def _finalize(self, mode: str, wfo: WFOResult) -> ModeValidationReport:
        cash = float(self._config.portfolio.initial_cash)
        stitched, stamps = stitch_oos_equity_timed(wfo.windows, cash)
        rf = float(getattr(self._config.backtest, "risk_free_rate", 0.0))
        metrics = (
            compute_metrics(stitched, risk_free_rate=rf)
            if len(stitched) >= 2
            else {}
        )
        oos_rets: List[float] = []
        for window in wfo.windows:
            if window.oos_result is None:
                oos_rets.append(float("nan"))
                continue
            oos_rets.append(
                float(window.oos_result.metrics.get("total_return", float("nan")))
            )
        gates = evaluate_gates(
            metrics,
            self._thresholds,
            equity_curve=stitched,
            equity_timestamps=stamps if len(stamps) else None,
            oos_window_returns=oos_rets,
        )
        return ModeValidationReport(
            mode=mode,
            wfo=wfo,
            stitched_equity=stitched,
            stitched_metrics=metrics,
            gates=gates,
            stitched_timestamps=stamps if len(stamps) else None,
        )
