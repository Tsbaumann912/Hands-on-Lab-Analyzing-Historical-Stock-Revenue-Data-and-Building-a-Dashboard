"""
Paper trading broker — simulates order fills locally with configurable
slippage and commission.  Useful for live strategy testing without
real capital at risk.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from brokers.base import BrokerBase
from core.config import Config
from core.enums import Direction, OrderStatus, OrderType
from core.models import Fill, Order

logger = logging.getLogger(__name__)


class PaperBroker(BrokerBase):
    """
    Simulated broker that fills orders at the submitted price ± slippage.

    Parameters
    ----------
    config:
        Terminal config; uses ``config.portfolio`` for commission / tick sizing.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.portfolio
        self._open_orders: Dict[str, Order] = {}
        self._filled_orders: List[Fill] = []

    # ── BrokerBase interface ──────────────────────────────────────────────────

    def submit_order(self, order: Order) -> Optional[Fill]:
        order_id = order.order_id or str(uuid.uuid4())
        order.order_id = order_id
        order.status = OrderStatus.SUBMITTED

        fill_price = self._simulate_fill_price(order)
        slippage_ticks = self._cfg.slippage_ticks
        commission = self._cfg.commission_per_contract * order.quantity

        fill = Fill(
            order_id=order_id,
            symbol=order.symbol,
            direction=order.direction,
            filled_quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=float(slippage_ticks),
            timestamp=datetime.now(tz=timezone.utc),
        )

        order.status = OrderStatus.FILLED
        self._filled_orders.append(fill)

        logger.info(
            "[Paper] FILL %s %s %.0f contracts @ %.4f  commission=%.2f",
            order.direction.value,
            order.symbol,
            order.quantity,
            fill_price,
            commission,
        )
        return fill

    def cancel_all_orders(self) -> List[str]:
        cancelled = list(self._open_orders.keys())
        for order in self._open_orders.values():
            order.status = OrderStatus.CANCELLED
        self._open_orders.clear()
        logger.info("[Paper] Cancelled %d orders.", len(cancelled))
        return cancelled

    def flatten_all_positions(self) -> List[Fill]:
        """
        Paper broker does not track positions directly — returns empty list.
        The portfolio layer is authoritative; the engine handles flattening.
        """
        logger.info("[Paper] flatten_all_positions: no-op (portfolio handles state).")
        return []

    def get_account_info(self) -> dict:
        return {
            "broker": "paper",
            "open_orders": len(self._open_orders),
            "fills": len(self._filled_orders),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _simulate_fill_price(self, order: Order) -> float:
        """
        Estimate fill price with worst-case slippage for market orders.

        Limit orders fill at the limit price (assuming sufficient liquidity).
        """
        slippage_value = self._cfg.slippage_ticks * self._cfg.tick_size

        if order.order_type == OrderType.MARKET:
            base_price = order.limit_price or 0.0
            if order.direction == Direction.LONG:
                return base_price + slippage_value
            return base_price - slippage_value

        if order.order_type == OrderType.LIMIT and order.limit_price:
            return order.limit_price

        if order.order_type == OrderType.STOP and order.stop_price:
            slip = slippage_value
            return (
                order.stop_price + slip
                if order.direction == Direction.LONG
                else order.stop_price - slip
            )

        return order.limit_price or order.stop_price or 0.0
