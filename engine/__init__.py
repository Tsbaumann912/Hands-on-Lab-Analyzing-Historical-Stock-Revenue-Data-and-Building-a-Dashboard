"""Execution engine: event-driven live loop, backtester, and walk-forward optimiser."""

from __future__ import annotations

from engine.backtest import BacktestEngine
from engine.live import LiveEngine
from engine.metrics import compute_metrics
from engine.optimizer import WalkForwardOptimizer
from engine.walk_forward import (
    WalkForwardValidator,
    build_anchored_windows,
    build_rolling_windows,
    evaluate_gates,
    stitch_oos_equity,
)

__all__ = [
    "BacktestEngine",
    "LiveEngine",
    "compute_metrics",
    "WalkForwardOptimizer",
    "WalkForwardValidator",
    "build_anchored_windows",
    "build_rolling_windows",
    "evaluate_gates",
    "stitch_oos_equity",
]
