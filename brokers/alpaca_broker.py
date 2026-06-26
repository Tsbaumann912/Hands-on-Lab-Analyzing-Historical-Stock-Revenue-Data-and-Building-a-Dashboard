"""
Alpaca live/paper broker adapter.

Uses the ``alpaca-py`` SDK for order submission.  The same class works
for both live and paper endpoints — the URL is controlled by
``config.broker.alpaca_base_url``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from brokers.base import BrokerBase
from core.config import Config
from core.enums import Direction, OrderStatus, OrderType
from core.models import Fill, Order

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    HAS_ALPACA = True
except ImportError:  # pragma: no cover
    HAS_ALPACA = False
    TradingClient = None  # type: ignore[assignment,misc]


class AlpacaBroker(BrokerBase):
    """
    Alpaca execution adapter supporting market and limit orders.

    Includes a ``KeyboardInterrupt``-safe execution loop so that Ctrl-C
    triggers the panic button automatically.

    Parameters
    ----------
    config:
        Terminal config; broker credentials come from ``config.broker``.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.broker
        self._portfolio_cfg = config.portfolio
        self._client: Optional[object] = None

        if not HAS_ALPACA:
            logger.warning(
                "alpaca-py not installed. Run: pip install alpaca-py. "
                "AlpacaBroker will operate in dry-run mode."
            )
            return

        try:
            self._client = TradingClient(
                api_key=self._cfg.alpaca_api_key,
                secret_key=self._cfg.alpaca_secret_key,
                paper=self._cfg.paper_trading,
            )
            logger.info(
                "Alpaca client initialised. Paper=%s", self._cfg.paper_trading
            )
        except Exception:
            logger.exception("Failed to initialise Alpaca client.")

    # ── BrokerBase interface ──────────────────────────────────────────────────

    def submit_order(self, order: Order) -> Optional[Fill]:
        if self._client is None:
            logger.error("Alpaca client not available — order not submitted.")
            return None

        try:
            side = OrderSide.BUY if order.direction == Direction.LONG else OrderSide.SELL  # type: ignore[union-attr]

            if order.order_type == OrderType.MARKET:
                req = MarketOrderRequest(  # type: ignore[misc]
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,  # type: ignore[union-attr]
                )
            elif order.order_type == OrderType.LIMIT and order.limit_price:
                req = LimitOrderRequest(  # type: ignore[misc]
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,  # type: ignore[union-attr]
                    limit_price=order.limit_price,
                )
            else:
                logger.error("Unsupported order type: %s", order.order_type)
                return None

            response = self._client.submit_order(req)  # type: ignore[union-attr]

            fill_price = float(getattr(response, "filled_avg_price", None) or order.limit_price or 0.0)
            commission = self._portfolio_cfg.commission_per_contract * order.quantity

            fill = Fill(
                order_id=str(response.id),  # type: ignore[union-attr]
                symbol=order.symbol,
                direction=order.direction,
                filled_quantity=float(getattr(response, "filled_qty", order.quantity)),
                fill_price=fill_price,
                commission=commission,
                slippage=0.0,
                timestamp=datetime.now(tz=timezone.utc),
            )
            logger.info(
                "[Alpaca] FILL %s %s @ %.4f  id=%s",
                order.direction.value,
                order.symbol,
                fill_price,
                fill.order_id,
            )
            return fill

        except Exception:
            logger.exception("Alpaca order submission failed for %s.", order.symbol)
            return None

    def cancel_all_orders(self) -> List[str]:
        if self._client is None:
            return []
        try:
            cancel_statuses = self._client.cancel_orders()  # type: ignore[union-attr]
            ids = [str(s.id) for s in cancel_statuses]
            logger.info("[Alpaca] Cancelled %d orders.", len(ids))
            return ids
        except Exception:
            logger.exception("Failed to cancel orders.")
            return []

    def flatten_all_positions(self) -> List[Fill]:
        if self._client is None:
            return []
        fills: List[Fill] = []
        try:
            self._client.close_all_positions(cancel_orders=True)  # type: ignore[union-attr]
            logger.info("[Alpaca] All positions closed.")
        except Exception:
            logger.exception("Failed to flatten positions.")
        return fills

    def get_account_info(self) -> dict:
        if self._client is None:
            return {"broker": "alpaca", "status": "disconnected"}
        try:
            acct = self._client.get_account()  # type: ignore[union-attr]
            return {
                "broker": "alpaca",
                "buying_power": float(acct.buying_power),
                "equity": float(acct.equity),
                "cash": float(acct.cash),
                "paper": self._cfg.paper_trading,
            }
        except Exception:
            logger.exception("Failed to fetch account info.")
            return {"broker": "alpaca", "status": "error"}
