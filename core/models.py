"""Shared data model dataclasses used across all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from core.enums import Direction, OrderStatus, OrderType, AssetClass


@dataclass(frozen=True)
class Bar:
    """OHLCV bar for a single instrument."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    asset_class: AssetClass = AssetClass.FUTURES
    contract_multiplier: float = 50.0          # override per symbol in config

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class Tick:
    """Level-1 tick (best bid/ask) for a single instrument."""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last: float
    last_size: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Signal:
    """Trade intent produced by a strategy."""

    symbol: str
    direction: Direction
    strength: float                            # 0.0 – 1.0
    timestamp: datetime
    strategy_name: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    suggested_size: Optional[float] = None    # in contracts
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be in [0, 1], got {self.strength}")


@dataclass
class Order:
    """Order sent to the broker execution layer."""

    symbol: str
    direction: Direction
    order_type: OrderType
    quantity: float                            # in contracts
    timestamp: datetime
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)


@dataclass
class Fill:
    """Confirmed execution of (part of) an order."""

    order_id: str
    symbol: str
    direction: Direction
    filled_quantity: float
    fill_price: float
    commission: float
    timestamp: datetime
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def net_cost(self) -> float:
        """Total cost including commissions (positive = cash outflow for longs)."""
        sign = 1.0 if self.direction == Direction.LONG else -1.0
        return sign * self.fill_price * self.filled_quantity + self.commission
