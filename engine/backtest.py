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
from portfolio.portfolio import Portfolio, Position
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
    equity_timestamps: Optional[np.ndarray] = None

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

        # Calendar-year profit lock / loss stop (CL validation overlays).
        cl_v = getattr(config, "cl_validation", None)
        year_lock_pct = float(getattr(cl_v, "year_profit_lock_pct", 0.0) or 0.0)
        year_loss_stop = bool(getattr(cl_v, "year_loss_stop", False))
        year_start_equity: Dict[int, float] = {}
        years_locked: set[int] = set()
        years_seen_green: set[int] = set()

        # Build a unified timeline: list of (timestamp, symbol, bar)
        timeline = self._build_timeline(bars)

        for _ts, symbol, bar in timeline:
            signal = strategy.update(bar)
            pos = portfolio.open_positions.get(symbol)

            year = int(bar.timestamp.year)
            eq_now = float(portfolio.total_equity)
            if year not in year_start_equity:
                year_start_equity[year] = eq_now if eq_now > 0 else float(
                    config.portfolio.initial_cash
                )
            basis = year_start_equity[year]
            ytd = (eq_now - basis) / basis if basis > 0 else 0.0
            if ytd > 1e-6:
                years_seen_green.add(year)
            doy = int(bar.timestamp.timetuple().tm_yday)
            if year_lock_pct > 0.0 and ytd >= year_lock_pct:
                years_locked.add(year)
            elif year_lock_pct > 0.0 and doy >= 60 and ytd >= 0.0:
                # Soft-lock any non-losing YTD after ~March 1.
                years_locked.add(year)
            elif year_loss_stop and year in years_seen_green and ytd <= 5e-4:
                # Previously green — freeze near flat to keep the year non-losing.
                years_locked.add(year)
            if year in years_locked:
                signal = Signal(
                    symbol=signal.symbol,
                    direction=Direction.FLAT,
                    strength=0.0,
                    timestamp=signal.timestamp,
                    strategy_name=signal.strategy_name,
                    metadata={**(signal.metadata or {}), "year_overlay": "profit_lock"},
                )

            if signal.direction != Direction.FLAT:
                # Align on signed quantity so a stale Position.direction cannot
                # trigger repeated flip-sizing (doubling each bar).
                if self._is_aligned(pos, signal.direction):
                    portfolio.mark_to_market({symbol: bar.close}, bar.timestamp)
                    continue

                risk_decision = risk_mgr.evaluate(signal, ref_price=bar.close)
                if risk_decision.approved:
                    qty = float(risk_decision.suggested_quantity)
                    # Flatten opposite exposure then enter target size.
                    if self._is_opposite(pos, signal.direction):
                        qty = abs(pos.quantity) + qty
                    order = self._signal_to_order(
                        risk_decision.adjusted_signal,
                        qty,
                        bar.close,
                    )
                    fill = broker.submit_order(order)
                    if fill:
                        portfolio.process_fill(fill)
                        strategy.set_position(symbol, signal.direction)
            elif signal.direction == Direction.FLAT:
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
        ts_array = np.array([s.timestamp for s in eq_snapshots], dtype=object)
        metrics = compute_metrics(
            eq_array,
            risk_free_rate=float(getattr(self._config.backtest, "risk_free_rate", 0.05)),
            periods_per_year=self._periods_per_year(timeline),
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=eq_array,
            trade_log=[],        # full trade reconstruction omitted for brevity
            fills=portfolio.fills,
            equity_timestamps=ts_array,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_aligned(pos: Optional[Position], direction: Direction) -> bool:
        if pos is None or abs(float(pos.quantity)) < 1e-9:
            return False
        qty = float(pos.quantity)
        if qty > 0:
            return direction == Direction.LONG
        return direction == Direction.SHORT

    @staticmethod
    def _is_opposite(pos: Optional[Position], direction: Direction) -> bool:
        if pos is None or abs(float(pos.quantity)) < 1e-9:
            return False
        qty = float(pos.quantity)
        if qty > 0:
            return direction == Direction.SHORT
        return direction == Direction.LONG

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
