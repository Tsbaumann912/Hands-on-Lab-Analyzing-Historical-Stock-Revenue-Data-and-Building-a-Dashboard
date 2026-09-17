"""Strategy templates and concrete implementations."""

from __future__ import annotations

from strategies.base import Strategy, BarBuffer
from strategies.mean_reversion import MeanReversionRSI
from strategies.momentum import MomentumBreakout
from strategies.trend_following import TrendFollowingMACD
from strategies.platinum_tsmom import PlatinumTSMOM
from strategies.platinum_gold_spread import PlatinumGoldSpread

__all__ = [
    "Strategy",
    "BarBuffer",
    "MeanReversionRSI",
    "MomentumBreakout",
    "TrendFollowingMACD",
    "PlatinumTSMOM",
    "PlatinumGoldSpread",
]
