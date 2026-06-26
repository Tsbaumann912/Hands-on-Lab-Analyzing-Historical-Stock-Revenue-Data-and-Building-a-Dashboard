"""Abstract broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import Fill, Order


class BrokerBase(ABC):
    """
    Contract that every broker adapter must satisfy.

    The execution pipeline is:
        RiskDecision → BrokerBase.submit_order → Fill → Portfolio
    """

    @abstractmethod
    def submit_order(self, order: Order) -> Optional[Fill]:
        """Submit an order and return the resulting Fill (or None on failure)."""

    @abstractmethod
    def cancel_all_orders(self) -> List[str]:
        """Cancel all open limit/stop orders. Returns list of cancelled order IDs."""

    @abstractmethod
    def flatten_all_positions(self) -> List[Fill]:
        """Market-sell/cover all open positions immediately."""

    @abstractmethod
    def get_account_info(self) -> dict:
        """Return broker-specific account metadata."""

    # ── Panic button ──────────────────────────────────────────────────────────

    def emergency_shutdown(self) -> None:
        """
        Gracefully cancel all limit orders then flatten all positions.

        This is the 'panic button' that should be invoked on ``KeyboardInterrupt``
        or unhandled exceptions in the live execution loop.
        """
        import logging
        log = logging.getLogger(self.__class__.__name__)
        log.critical("EMERGENCY SHUTDOWN: cancelling all orders …")
        cancelled = self.cancel_all_orders()
        log.critical("Cancelled %d orders. Flattening all positions …", len(cancelled))
        fills = self.flatten_all_positions()
        log.critical("Flattened %d positions. Shutdown complete.", len(fills))
