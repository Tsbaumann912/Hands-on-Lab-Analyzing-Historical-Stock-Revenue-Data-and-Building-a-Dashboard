"""Strategy templates and concrete implementations."""

from __future__ import annotations

from strategies.base import Strategy, BarBuffer
from strategies.mean_reversion import MeanReversionRSI
from strategies.momentum import MomentumBreakout
from strategies.trend_following import TrendFollowingMACD

__all__ = [
    "Strategy",
    "BarBuffer",
    "MeanReversionRSI",
    "MomentumBreakout",
    "TrendFollowingMACD",
]
