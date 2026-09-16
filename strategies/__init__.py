"""Strategy templates and concrete implementations."""

from __future__ import annotations

from strategies.base import Strategy, BarBuffer
from strategies.mean_reversion import MeanReversionRSI
from strategies.momentum import MomentumBreakout
from strategies.trend_following import TrendFollowingMACD
from strategies.cl_carry_curve import CLCarryCurve
from strategies.cl_carry_momentum import CLCarryMomentum
from strategies.cl_vol_target_tsmom import CLVolTargetTSMOM
from strategies.cl_inventory_confirm import CLInventoryConfirm

__all__ = [
    "Strategy",
    "BarBuffer",
    "MeanReversionRSI",
    "MomentumBreakout",
    "TrendFollowingMACD",
    "CLCarryCurve",
    "CLCarryMomentum",
    "CLVolTargetTSMOM",
    "CLInventoryConfirm",
]
