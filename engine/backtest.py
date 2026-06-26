"""
Event-driven backtesting engine.

Architecture:
  1. Feed historical bars into the strategy one bar at a time
  2. Collect Signal output → pass through RiskManager
  3. Convert approved signals to Orders → fill via PaperBroker
  4. Update Portfolio state and mark-to-market each bar
  5. Produce a BacktestResult with metrics, equity curve, and trade log
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import numpy as np

from brokers.paper import PaperBroker
from core.config import Config
from core.enums import Direction, OrderType
from core.models import Bar, Fill, Order, Signal
from engine.metrics import compute_metrics
from portfolio.portfolio import Portfolio
from risk.risk_manager import RiskManager
from strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    symbol: str
    direction: Direction
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    commission: float
    net_pnl: float


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    equity_curve: np.ndarray
    trade_log: List[TradeRecord]
    fills: List[Fill]
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["── Backtest Result ─────────────────────────────"]
        for k, v in self.metrics.items():
            lines.append(f"  {k:<22}: {v}")
        lines.append(f"  {'trades':<22}: {len(self.trade_log)}")
        return "\n".join(lines)


class BacktestEngine:
    """
    Bar-by-bar event-driven backtester.

    Parameters
    ----------
    config:
        Terminal configuration.
    strategy_cls:
        Strategy class (not instance) to instantiate per run.
    """

    def __init__(self, config: Config, strategy_cls: Type[Strategy]) -> None:
        self._config = config
        self._strategy_cls = strategy_cls

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        bars: Dict[str, List[Bar]],
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """
        Execute a backtest over *bars*.

        Parameters
        ----------
        bars:
            ``{symbol: [Bar, …]}`` — aligned by timestamp.
        strategy_params:
            Optional parameter overrides injected into the config before run.
        """
        config = self._apply_params(self._config, strategy_params or {})

        portfolio = Portfolio(config)
        broker = PaperBroker(config)
        strategy = self._strategy_cls(config, symbols=list(bars.keys()))
        risk_mgr = RiskManager(config, portfolio)

        # Build a unified timeline: list of (timestamp, symbol, bar)
        timeline = self._build_timeline(bars)

        for _ts, symbol, bar in timeline:
            signal = strategy.update(bar)

            if signal.direction != Direction.FLAT:
                risk_decision = risk_mgr.evaluate(signal)
                if risk_decision.approved:
                    order = self._signal_to_order(
                        risk_decision.adjusted_signal,
                        risk_decision.suggested_quantity,
                        bar.close,
                    )
                    fill = broker.submit_order(order)
                    if fill:
                        portfolio.process_fill(fill)
                        strategy.set_position(symbol, signal.direction)
            elif signal.direction == Direction.FLAT:
                pos = portfolio.open_positions.get(symbol)
                if pos is not None:
                    close_order = Order(
                        symbol=symbol,
                        direction=Direction.SHORT if pos.direction == Direction.LONG else Direction.LONG,
                        order_type=OrderType.MARKET,
                        quantity=abs(pos.quantity),
                        timestamp=bar.timestamp,
                        limit_price=bar.close,
                    )
                    fill = broker.submit_order(close_order)
                    if fill:
                        portfolio.process_fill(fill)
                        strategy.set_position(symbol, Direction.FLAT)

            portfolio.mark_to_market({symbol: bar.close}, bar.timestamp)

        eq_snapshots = portfolio.equity_curve
        eq_array = np.array([s.total_equity for s in eq_snapshots], dtype=np.float64)
        metrics = compute_metrics(
            eq_array,
            periods_per_year=self._periods_per_year(timeline),
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=eq_array,
            trade_log=[],        # full trade reconstruction omitted for brevity
            fills=portfolio.fills,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_timeline(bars: Dict[str, List[Bar]]) -> List[tuple]:
        timeline = []
        for symbol, bar_list in bars.items():
            for bar in bar_list:
                timeline.append((bar.timestamp, symbol, bar))
        timeline.sort(key=lambda x: (x[0], x[1]))
        return timeline

    @staticmethod
    def _signal_to_order(signal: Signal, qty: float, ref_price: float) -> Order:
        return Order(
            symbol=signal.symbol,
            direction=signal.direction,
            order_type=OrderType.MARKET,
            quantity=max(1.0, qty),
            timestamp=signal.timestamp,
            limit_price=ref_price,
            stop_price=signal.stop_loss,
        )

    @staticmethod
    def _apply_params(config: Config, params: Dict[str, Any]) -> Config:
        """Return a shallow copy of config with overridden indicator/strategy params."""
        import copy
        cfg = copy.deepcopy(config)
        for key, val in params.items():
            if hasattr(cfg.indicators, key):
                setattr(cfg.indicators, key, val)
            elif hasattr(cfg.strategy, key):
                setattr(cfg.strategy, key, val)
        return cfg

    @staticmethod
    def _periods_per_year(timeline: list) -> float:
        if len(timeline) < 2:
            return 252.0
        delta = (timeline[-1][0] - timeline[0][0]).total_seconds()
        n = len(timeline)
        period_seconds = delta / max(n - 1, 1)
        seconds_per_year = 365.25 * 24 * 3600
        return seconds_per_year / period_seconds
