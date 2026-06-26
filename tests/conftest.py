"""Shared pytest fixtures for the futures terminal test suite."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pytest

from core.config import Config
from core.enums import AssetClass, Direction
from core.models import Bar, Fill


# ── Synthetic price data helpers ──────────────────────────────────────────────

def make_bars(
    n: int = 300,
    symbol: str = "ES.c.0",
    start_price: float = 4500.0,
    volatility: float = 5.0,
    seed: int = 42,
) -> List[Bar]:
    """Generate a list of synthetic OHLCV bars for testing."""
    rng = np.random.default_rng(seed)
    prices = start_price + np.cumsum(rng.normal(0, volatility, n))
    prices = np.clip(prices, 100.0, None)

    base_ts = datetime(2023, 1, 1, 9, 30, tzinfo=timezone.utc)
    bars = []
    for i in range(n):
        close = float(prices[i])
        spread = float(rng.uniform(0.5, 2.0))
        high = close + spread
        low = close - spread
        open_ = float(prices[i - 1]) if i > 0 else close
        volume = float(rng.uniform(500, 5000))
        ts = base_ts + timedelta(minutes=i)
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=ts,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                asset_class=AssetClass.FUTURES,
            )
        )
    return bars


def make_fill(
    symbol: str = "ES.c.0",
    direction: Direction = Direction.LONG,
    qty: float = 1.0,
    price: float = 4500.0,
    commission: float = 2.25,
) -> Fill:
    return Fill(
        order_id="test-fill-001",
        symbol=symbol,
        direction=direction,
        filled_quantity=qty,
        fill_price=price,
        commission=commission,
        timestamp=datetime(2023, 1, 2, 9, 30, tzinfo=timezone.utc),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> Config:
    return Config()


@pytest.fixture
def synthetic_bars() -> List[Bar]:
    return make_bars(n=300)


@pytest.fixture
def short_bars() -> List[Bar]:
    """Only 20 bars — triggers warm-up FLAT signals in most strategies."""
    return make_bars(n=20)
