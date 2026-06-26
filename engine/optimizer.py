"""
Walk-forward optimisation using Optuna.

Architecture:
  - Splits the full historical dataset into N windows
  - Each window has an in-sample (IS) and out-of-sample (OOS) segment
  - Optuna finds the best hyperparameters on IS; they are evaluated on OOS
  - Reports aggregated OOS metrics to guard against curve-fitting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np

from core.config import Config
from engine.backtest import BacktestEngine, BacktestResult
from core.models import Bar
from strategies.base import Strategy

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    HAS_OPTUNA = False
    optuna = None  # type: ignore[assignment]


@dataclass
class WFOWindow:
    window_id: int
    is_bars: Dict[str, List[Bar]]
    oos_bars: Dict[str, List[Bar]]
    best_params: Dict[str, Any] = field(default_factory=dict)
    is_result: Optional[BacktestResult] = None
    oos_result: Optional[BacktestResult] = None


@dataclass
class WFOResult:
    windows: List[WFOWindow]
    aggregated_oos_metrics: Dict[str, float]
    best_params_per_window: List[Dict[str, Any]]

    def summary(self) -> str:
        lines = ["── Walk-Forward Optimisation Result ────────────"]
        for k, v in self.aggregated_oos_metrics.items():
            lines.append(f"  {k:<22}: {v}")
        return "\n".join(lines)


# Search-space definition type
SearchSpace = Dict[str, Callable[["optuna.Trial"], Any]]


class WalkForwardOptimizer:
    """
    Orchestrates walk-forward optimisation over a strategy's hyperparameters.

    Parameters
    ----------
    config:
        Terminal configuration.
    strategy_cls:
        Strategy class to optimise.
    search_space:
        Dict mapping parameter name to a callable ``(trial) → value``.
        Example::

            search_space = {
                "rsi_period": lambda t: t.suggest_int("rsi_period", 5, 30),
                "rsi_oversold": lambda t: t.suggest_float("rsi_oversold", 20, 40),
            }
    objective_metric:
        Metric from ``compute_metrics`` to maximise (default ``"sharpe_ratio"``).
    """

    def __init__(
        self,
        config: Config,
        strategy_cls: Type[Strategy],
        search_space: SearchSpace,
        objective_metric: str = "sharpe_ratio",
    ) -> None:
        self._config = config
        self._strategy_cls = strategy_cls
        self._search_space = search_space
        self._objective_metric = objective_metric
        self._engine = BacktestEngine(config, strategy_cls)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        bars: Dict[str, List[Bar]],
        n_windows: int = 5,
        in_sample_ratio: float = 0.70,
        n_trials: int = 50,
        timeout: Optional[int] = 120,
    ) -> WFOResult:
        """
        Execute walk-forward optimisation.

        Parameters
        ----------
        bars:
            Full multi-symbol bar dataset.
        n_windows:
            Number of rolling WFO windows.
        in_sample_ratio:
            Fraction of each window used for IS optimisation.
        n_trials:
            Number of Optuna trials per IS window.
        timeout:
            Maximum seconds per Optuna study (``None`` = unlimited).
        """
        if not HAS_OPTUNA:
            raise RuntimeError(
                "optuna not installed. Run: pip install optuna"
            )

        windows = self._build_windows(bars, n_windows, in_sample_ratio)
        oos_metrics_list: List[Dict[str, float]] = []
        best_params_list: List[Dict[str, Any]] = []

        for window in windows:
            logger.info("WFO window %d: optimising IS …", window.window_id)
            best_params, is_result = self._optimise_window(
                window.is_bars, n_trials=n_trials, timeout=timeout
            )
            window.best_params = best_params
            window.is_result = is_result

            logger.info(
                "WFO window %d: evaluating OOS with params %s …",
                window.window_id,
                best_params,
            )
            oos_result = self._engine.run(window.oos_bars, strategy_params=best_params)
            window.oos_result = oos_result
            oos_metrics_list.append(oos_result.metrics)
            best_params_list.append(best_params)

        aggregated = self._aggregate_metrics(oos_metrics_list)

        return WFOResult(
            windows=windows,
            aggregated_oos_metrics=aggregated,
            best_params_per_window=best_params_list,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _optimise_window(
        self,
        is_bars: Dict[str, List[Bar]],
        n_trials: int,
        timeout: Optional[int],
    ) -> tuple[Dict[str, Any], BacktestResult]:
        """Run an Optuna study on the IS window and return the best params."""

        def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
            params = {k: fn(trial) for k, fn in self._search_space.items()}
            result = self._engine.run(is_bars, strategy_params=params)
            return result.metrics.get(self._objective_metric, -999.0)

        study = optuna.create_study(direction="maximize")  # type: ignore[union-attr]
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        best_result = self._engine.run(is_bars, strategy_params=best_params)
        return best_params, best_result

    @staticmethod
    def _build_windows(
        bars: Dict[str, List[Bar]],
        n_windows: int,
        in_sample_ratio: float,
    ) -> List[WFOWindow]:
        """Slice the bar dataset into rolling WFO windows."""
        # Use the first symbol's bar list as the time axis
        ref_symbol = next(iter(bars))
        ref_bars = bars[ref_symbol]
        n = len(ref_bars)
        window_size = n // n_windows

        windows: List[WFOWindow] = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size if i < n_windows - 1 else n
            split = start + int((end - start) * in_sample_ratio)

            is_bars = {sym: b[start:split] for sym, b in bars.items()}
            oos_bars = {sym: b[split:end] for sym, b in bars.items()}
            windows.append(WFOWindow(window_id=i, is_bars=is_bars, oos_bars=oos_bars))

        return windows

    @staticmethod
    def _aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average OOS metrics across all WFO windows."""
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        agg = {}
        for k in keys:
            vals = np.array([m.get(k, np.nan) for m in metrics_list], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            agg[k] = float(vals.mean()) if len(vals) > 0 else np.nan
        return agg
