"""
CL carry-momentum strategy — primary oil risk-premium sleeve.

Applies momentum to the carry series so slow contango/backwardation flips
become nimbler (Bouchouev; Rebellion Research N≈10).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from core.config import Config
from core.enums import Direction
from core.models import Bar, Signal
from data.cl_features import (
    carry_momentum_z,
    clip01,
    resolve_curve,
    spread_carry,
)
from indicators.volatility import atr
from strategies.base import BarBuffer, Strategy

logger = logging.getLogger(__name__)


class CLCarryMomentum(Strategy):
    """Momentum on curve carry: z = (C - MA_N(C)) / σ(C)."""

    def __init__(self, config: Config, symbols: Optional[List[str]] = None) -> None:
        super().__init__(config, symbols=symbols)
        sc = config.strategy
        self._back_month = int(getattr(sc, "carry_back_month", 3))
        self._basis_lookback = int(getattr(sc, "carry_basis_lookback", 63))
        self._lookback = int(getattr(sc, "carry_mom_lookback", 10))
        self._z_max = float(getattr(sc, "carry_mom_z_max", 2.0))
        self._atr_period = int(config.indicators.atr_period)
        self._stop_atr_mult = config.risk.default_stop_loss_atr_mult
        self._tp_atr_mult = config.risk.default_take_profit_atr_mult
        maxlen = max(
            self._basis_lookback + self._lookback + self._atr_period + 20,
            250,
        )
        self._buffers = {sym: BarBuffer(maxlen=maxlen) for sym in self.symbols}

    def on_bar(self, bar: Bar) -> Signal:
        buf = self._buffers[bar.symbol]
        closes = buf.closes()
        highs = buf.highs()
        lows = buf.lows()

        front, back = resolve_curve(
            closes, None, self._back_month, self._basis_lookback
        )
        carry = spread_carry(front, back)
        z_arr = carry_momentum_z(carry, self._lookback)
        atr_vals = atr(highs, lows, closes, self._atr_period)
        if atr_vals is None:
            return self._flat_signal(bar)

        z = z_arr[-1]
        a = atr_vals[-1]
        if not np.isfinite(z) or not np.isfinite(a) or a <= 0.0:
            return self._flat_signal(bar)
        if abs(z) < 1e-12:
            return self._flat_signal(bar)

        direction = Direction.LONG if z > 0.0 else Direction.SHORT
        strength = clip01(abs(z) / max(self._z_max, 1e-9))
        close = closes[-1]

        if direction == Direction.LONG:
            stop = close - self._stop_atr_mult * a
            tp = close + self._tp_atr_mult * a
        else:
            stop = close + self._stop_atr_mult * a
            tp = close - self._tp_atr_mult * a

        return Signal(
            symbol=bar.symbol,
            direction=direction,
            strength=strength,
            timestamp=bar.timestamp,
            strategy_name=self.__class__.__name__,
            stop_loss=stop,
            take_profit=tp,
            metadata={
                "carry_mom_z": round(float(z), 4),
                "carry_spread": round(float(carry[-1]), 6)
                if np.isfinite(carry[-1])
                else None,
                "lookback": self._lookback,
                "atr": round(float(a), 4),
            },
        )
