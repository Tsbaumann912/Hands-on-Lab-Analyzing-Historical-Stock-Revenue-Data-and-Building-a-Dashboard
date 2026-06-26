"""
Live execution engine — connects the strategy/risk pipeline to a live broker.

Includes a ``KeyboardInterrupt`` handler that automatically triggers the
broker panic button to cancel orders and flatten all positions.
"""

from __future__ import annotations

import asyncio
import logging
import signal as _signal
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type

from brokers.base import BrokerBase
from core.config import Config
from core.enums import Direction, OrderType
from core.models import Bar, Order
from portfolio.portfolio import Portfolio
from risk.risk_manager import RiskManager
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class LiveEngine:
    """
    Real-time execution loop.

    The engine subscribes to the bar event bus, processes each bar through
    the strategy → risk manager → broker pipeline, and updates the portfolio
    after each fill.

    Parameters
    ----------
    config:
        Terminal configuration.
    strategy:
        Instantiated strategy (already seeded with warm-up data if needed).
    broker:
        Live or paper broker adapter.
    """

    def __init__(
        self,
        config: Config,
        strategy: Strategy,
        broker: BrokerBase,
    ) -> None:
        self._config = config
        self._strategy = strategy
        self._broker = broker
        self._portfolio = Portfolio(config)
        self._risk_mgr = RiskManager(config, self._portfolio)
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, bar_stream: "asyncio.Queue[Bar]") -> None:
        """
        Synchronous entry point — runs the async event loop until interrupted.

        On ``KeyboardInterrupt``, the broker's ``emergency_shutdown`` is invoked
        before the process exits.
        """
        try:
            asyncio.run(self._run(bar_stream))
        except KeyboardInterrupt:
            logger.critical("KeyboardInterrupt received — activating panic button.")
            self._broker.emergency_shutdown()
        finally:
            self._running = False
            self._log_summary()

    async def _run(self, bar_stream: "asyncio.Queue[Bar]") -> None:
        self._running = True
        logger.info("LiveEngine started.")

        while self._running:
            bar: Bar = await asyncio.wait_for(bar_stream.get(), timeout=60.0)
            await self._process_bar(bar)

    async def _process_bar(self, bar: Bar) -> None:
        signal = self._strategy.update(bar)

        # Route to risk manager for direction LONG or SHORT
        if signal.direction != Direction.FLAT:
            decision = self._risk_mgr.evaluate(signal)
            if decision.approved:
                order = Order(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    order_type=OrderType.MARKET,
                    quantity=max(1.0, decision.suggested_quantity),
                    timestamp=bar.timestamp,
                    limit_price=bar.close,
                    stop_price=decision.adjusted_signal.stop_loss,
                )
                fill = self._broker.submit_order(order)
                if fill:
                    self._portfolio.process_fill(fill)
                    self._strategy.set_position(bar.symbol, signal.direction)
            else:
                for v in decision.violations:
                    logger.warning("Risk breach: %s", v.message)

        # Close position on FLAT signal
        elif signal.direction == Direction.FLAT:
            pos = self._portfolio.open_positions.get(bar.symbol)
            if pos is not None:
                close_order = Order(
                    symbol=bar.symbol,
                    direction=Direction.SHORT if pos.direction == Direction.LONG else Direction.LONG,
                    order_type=OrderType.MARKET,
                    quantity=abs(pos.quantity),
                    timestamp=bar.timestamp,
                    limit_price=bar.close,
                )
                fill = self._broker.submit_order(close_order)
                if fill:
                    self._portfolio.process_fill(fill)
                    self._strategy.set_position(bar.symbol, Direction.FLAT)

        self._portfolio.mark_to_market({bar.symbol: bar.close}, bar.timestamp)

    def _log_summary(self) -> None:
        summary = self._portfolio.summary()
        logger.info("── Portfolio Summary ──────────────────────────")
        for k, v in summary.items():
            logger.info("  %-22s: %s", k, v)
