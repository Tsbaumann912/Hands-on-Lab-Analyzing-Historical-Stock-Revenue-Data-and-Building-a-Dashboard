"""
Mean-Reversion RSI strategy for futures.

Entry logic:
  - LONG when RSI < oversold threshold and close < lower Bollinger Band
  - SHORT when RSI > overbought threshold and close > upper Bollinger Band
Exit logic:
  - FLAT when RSI crosses back through the midline (50)
  - FLAT when close re-enters the Bollinger Band mid-zone
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from core.config import Config
from core.enums import Direction
from core.models import Bar, Signal
from indicators.momentum import rsi
from indicators.volatility import bollinger_bands, atr
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class MeanReversionRSI(Strategy):
    """
    RSI + Bollinger Band mean-reversion strategy.

    Parameters are read from ``config.strategy`` and ``config.indicators``.
    """

    def __init__(self, config: Config, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        self._rsi_period = config.indicators.rsi_period
        self._bb_period = config.indicators.bb_period
        self._bb_std = config.indicators.bb_std
        self._atr_period = config.indicators.atr_period
        self._oversold = config.strategy.rsi_oversold
        self._overbought = config.strategy.rsi_overbought
        self._stop_atr_mult = config.risk.default_stop_loss_atr_mult
        self._tp_atr_mult = config.risk.default_take_profit_atr_mult

    def on_bar(self, bar: Bar) -> Signal:
        buf = self._buffers[bar.symbol]
        closes = buf.closes()
        highs = buf.highs()
        lows = buf.lows()

        rsi_vals = rsi(closes, self._rsi_period)
        bb = bollinger_bands(closes, self._bb_period, self._bb_std)
        atr_vals = atr(highs, lows, closes, self._atr_period)

        if rsi_vals is None or bb is None or atr_vals is None:
            return self._flat_signal(bar)

        current_rsi = rsi_vals[-1]
        current_close = closes[-1]
        bb_upper = bb.upper[-1]
        bb_lower = bb.lower[-1]
        current_atr = atr_vals[-1]

        if np.isnan(current_rsi) or np.isnan(bb_upper) or np.isnan(current_atr):
            return self._flat_signal(bar)

        current_pos = self.current_position(bar.symbol)

        # ── Entry logic ────────────────────────────────────────────────────────
        if current_pos == Direction.FLAT:
            if current_rsi < self._oversold and current_close < bb_lower:
                strength = self._compute_strength(current_rsi, self._oversold, long=True)
                return Signal(
                    symbol=bar.symbol,
                    direction=Direction.LONG,
                    strength=strength,
                    timestamp=bar.timestamp,
                    strategy_name=self.__class__.__name__,
                    stop_loss=current_close - self._stop_atr_mult * current_atr,
                    take_profit=current_close + self._tp_atr_mult * current_atr,
                    metadata={
                        "rsi": round(current_rsi, 2),
                        "bb_lower": round(bb_lower, 4),
                        "atr": round(current_atr, 4),
                    },
                )

            if current_rsi > self._overbought and current_close > bb_upper:
                strength = self._compute_strength(current_rsi, self._overbought, long=False)
                return Signal(
                    symbol=bar.symbol,
                    direction=Direction.SHORT,
                    strength=strength,
                    timestamp=bar.timestamp,
                    strategy_name=self.__class__.__name__,
                    stop_loss=current_close + self._stop_atr_mult * current_atr,
                    take_profit=current_close - self._tp_atr_mult * current_atr,
                    metadata={
                        "rsi": round(current_rsi, 2),
                        "bb_upper": round(bb_upper, 4),
                        "atr": round(current_atr, 4),
                    },
                )

        # ── Exit logic ─────────────────────────────────────────────────────────
        bb_mid = bb.middle[-1]
        if current_pos == Direction.LONG and (current_rsi > 50 or current_close > bb_mid):
            return Signal(
                symbol=bar.symbol,
                direction=Direction.FLAT,
                strength=1.0,
                timestamp=bar.timestamp,
                strategy_name=self.__class__.__name__,
                metadata={"exit_reason": "rsi_cross_50_or_bb_mid"},
            )

        if current_pos == Direction.SHORT and (current_rsi < 50 or current_close < bb_mid):
            return Signal(
                symbol=bar.symbol,
                direction=Direction.FLAT,
                strength=1.0,
                timestamp=bar.timestamp,
                strategy_name=self.__class__.__name__,
                metadata={"exit_reason": "rsi_cross_50_or_bb_mid"},
            )

        return self._flat_signal(bar)

    @staticmethod
    def _compute_strength(rsi_val: float, threshold: float, long: bool) -> float:
        """Scale signal strength by how far RSI is beyond the threshold."""
        if long:
            deviation = max(0.0, threshold - rsi_val)
            return min(1.0, deviation / threshold)
        deviation = max(0.0, rsi_val - threshold)
        return min(1.0, deviation / (100.0 - threshold))
