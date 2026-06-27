"""
Walk-forward optimisation using Optuna.

Architecture:
  - Splits the full historical dataset into N windows
  - Each window has an in-sample (IS) and out-of-sample (OOS) segment
  - Optuna finds the best hyperparameters on IS; they are evaluated on OOS
  - Reports aggregated OOS metrics to guard against curve-fitting
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import yaml

from core.config import Config
from engine.backtest import BacktestEngine, BacktestResult
from core.models import Bar
from strategies.base import Strategy

logger = logging.getLogger(__name__)


# ── Optimisable-parameter registry ───────────────────────────────────────────
# Maps each strategy to the parameters that meaningfully affect its signals.
# Names must match keys in config/optuna.yaml and config.indicators/strategy.
STRATEGY_PARAM_SPACES: Dict[str, List[str]] = {
    "MeanReversionRSI": [
        "rsi_period",
        "rsi_oversold",
        "rsi_overbought",
        "bb_period",
        "bb_std",
    ],
    "MomentumBreakout": ["lookback", "atr_period"],
    "TrendFollowingMACD": ["atr_period", "sma_short", "sma_long"],
}


def load_optuna_bounds(path: str | Path = "config/optuna.yaml") -> Dict[str, Dict[str, Any]]:
    """Load the raw hyperparameter search bounds from ``config/optuna.yaml``."""
    path = Path(path)
    if not path.exists():
        logger.warning("optuna search-space file not found: %s", path)
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _suggest_factory(
    name: str, spec: Dict[str, Any]
) -> Callable[["optuna.Trial"], Any]:  # type: ignore[name-defined]
    """Build a single ``trial → value`` callable, binding *name*/*spec* eagerly."""
    param_type = str(spec.get("type", "float")).lower()
    low = spec["low"]
    high = spec["high"]

    if param_type == "int":
        return lambda t, _n=name, _lo=int(low), _hi=int(high): t.suggest_int(_n, _lo, _hi)
    return lambda t, _n=name, _lo=float(low), _hi=float(high): t.suggest_float(_n, _lo, _hi)


def build_search_space_from_yaml(
    param_names: List[str],
    path: str | Path = "config/optuna.yaml",
) -> "SearchSpace":
    """
    Construct an Optuna ``SearchSpace`` for *param_names* from ``optuna.yaml``.

    Unknown parameter names are skipped with a warning so callers can request a
    superset without crashing.
    """
    bounds = load_optuna_bounds(path)
    space: SearchSpace = {}
    for name in param_names:
        spec = bounds.get(name)
        if spec is None:
            logger.warning("No optuna bounds for parameter %r — skipped.", name)
            continue
        space[name] = _suggest_factory(name, spec)
    return space


def build_param_grid_from_yaml(
    param_names: List[str],
    steps: int,
    path: str | Path = "config/optuna.yaml",
) -> Dict[str, List[Any]]:
    """
    Build a discrete value grid for *param_names* using ``optuna.yaml`` bounds.

    Each parameter is sampled at *steps* evenly-spaced points between its low and
    high bound (integer parameters are rounded and de-duplicated).
    """
    bounds = load_optuna_bounds(path)
    grid: Dict[str, List[Any]] = {}
    steps = max(2, int(steps))
    for name in param_names:
        spec = bounds.get(name)
        if spec is None:
            logger.warning("No optuna bounds for parameter %r — skipped.", name)
            continue
        low = float(spec["low"])
        high = float(spec["high"])
        samples = np.linspace(low, high, steps)
        if str(spec.get("type", "float")).lower() == "int":
            values = sorted({int(round(v)) for v in samples})
        else:
            values = [round(float(v), 4) for v in samples]
        grid[name] = values
    return grid

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
        """
        Aggregate OOS metrics across all WFO windows using a robust mean.

        Values beyond ±1 000 are clipped before averaging so that a single
        degenerate window (e.g. equity collapses to zero producing Sharpe ≈ −1e6)
        does not poison the aggregate — which would otherwise misrepresent
        expected live performance.
        """
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        agg = {}
        for k in keys:
            vals = np.array([m.get(k, np.nan) for m in metrics_list], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                agg[k] = np.nan
                continue
            vals_clipped = np.clip(vals, -1_000.0, 1_000.0)
            agg[k] = float(vals_clipped.mean())
        return agg


# ── Exhaustive grid optimisation (AmiBroker-style "Optimize") ─────────────────

@dataclass
class GridTrial:
    """One parameter combination and the metrics it produced."""

    params: Dict[str, Any]
    metrics: Dict[str, float]
    objective_value: float


@dataclass
class GridOptimizationResult:
    trials: List[GridTrial]
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    objective_metric: str
    n_combinations: int


class GridOptimizer:
    """
    Brute-force grid search over a strategy's parameters on a fixed dataset.

    Mirrors AmiBroker's *Optimize* mode: every combination in the parameter grid
    is backtested and the resulting metrics are tabulated and ranked by the chosen
    objective.  Iteration is over parameter combinations (control flow), never over
    price-series elements.

    Parameters
    ----------
    config, strategy_cls:
        As for :class:`BacktestEngine`.
    param_grid:
        Mapping ``{param_name: [value, …]}`` of values to sweep.
    objective_metric:
        Metric from ``compute_metrics`` to maximise.
    max_combinations:
        Hard cap; raises ``ValueError`` if the grid exceeds it.
    """

    def __init__(
        self,
        config: Config,
        strategy_cls: Type[Strategy],
        param_grid: Dict[str, List[Any]],
        objective_metric: str = "sharpe_ratio",
        max_combinations: int = 96,
    ) -> None:
        self._config = config
        self._strategy_cls = strategy_cls
        self._param_grid = {k: v for k, v in param_grid.items() if v}
        self._objective_metric = objective_metric
        self._max_combinations = max_combinations
        self._engine = BacktestEngine(config, strategy_cls)

    def combinations(self) -> List[Dict[str, Any]]:
        """Materialise the cartesian product of the parameter grid."""
        if not self._param_grid:
            return [{}]
        names = list(self._param_grid.keys())
        value_lists = [self._param_grid[n] for n in names]
        return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]

    def run(self, bars: Dict[str, List[Bar]]) -> GridOptimizationResult:
        """Backtest every combination and return ranked results."""
        combos = self.combinations()
        if len(combos) > self._max_combinations:
            raise ValueError(
                f"Grid has {len(combos)} combinations, exceeding the cap of "
                f"{self._max_combinations}. Reduce parameters or steps."
            )

        trials: List[GridTrial] = []
        for params in combos:
            result = self._engine.run(bars, strategy_params=params)
            objective_value = float(result.metrics.get(self._objective_metric, np.nan))
            trials.append(
                GridTrial(
                    params=params,
                    metrics=result.metrics,
                    objective_value=objective_value,
                )
            )

        trials.sort(
            key=lambda tr: (np.isnan(tr.objective_value), -np.nan_to_num(tr.objective_value, nan=-1e18))
        )
        best = trials[0] if trials else GridTrial({}, {}, float("nan"))

        return GridOptimizationResult(
            trials=trials,
            best_params=best.params,
            best_metrics=best.metrics,
            objective_metric=self._objective_metric,
            n_combinations=len(combos),
        )
