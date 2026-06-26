"""MACD + SuperTrend trend-following strategy for futures."""

from __future__ import annotations

import logging

import numpy as np

from core.config import Config
from core.enums import Direction
from core.models import Bar, Signal
from indicators.momentum import macd
from indicators.trend import supertrend
from indicators.volatility import atr
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class TrendFollowingMACD(Strategy):
    """
    Dual-confirmation trend strategy: MACD crossover + SuperTrend direction.

    Entry:
      - LONG when MACD line crosses above signal line AND SuperTrend is bullish
      - SHORT when MACD line crosses below signal line AND SuperTrend is bearish
    Exit:
      - Opposite MACD crossover
    """

    def __init__(self, config: Config, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        self._atr_period = config.indicators.atr_period
        self._stop_atr_mult = config.risk.default_stop_loss_atr_mult
        self._tp_atr_mult = config.risk.default_take_profit_atr_mult

    def on_bar(self, bar: Bar) -> Signal:
        buf = self._buffers[bar.symbol]
        closes = buf.closes()
        highs = buf.highs()
        lows = buf.lows()

        macd_result = macd(closes, fast=12, slow=26, signal=9)
        st_result = supertrend(highs, lows, closes, period=10, multiplier=3.0)
        atr_vals = atr(highs, lows, closes, self._atr_period)

        if macd_result is None or st_result is None or atr_vals is None:
            return self._flat_signal(bar)

        macd_line = macd_result.macd_line
        signal_line = macd_result.signal_line
        st_direction = st_result.direction

        current_atr = atr_vals[-1]
        current_close = closes[-1]

        if len(macd_line) < 2:
            return self._flat_signal(bar)

        if np.isnan(macd_line[-1]) or np.isnan(signal_line[-1]) or np.isnan(current_atr):
            return self._flat_signal(bar)

        # MACD crossover detection (current bar vs previous bar)
        macd_cross_up = (
            macd_line[-2] < signal_line[-2] and macd_line[-1] > signal_line[-1]
        )
        macd_cross_down = (
            macd_line[-2] > signal_line[-2] and macd_line[-1] < signal_line[-1]
        )
        st_bullish = st_direction[-1] == 1.0
        st_bearish = st_direction[-1] == -1.0

        current_pos = self.current_position(bar.symbol)

        if current_pos == Direction.FLAT:
            if macd_cross_up and st_bullish:
                macd_diff = abs(macd_line[-1] - signal_line[-1])
                # Normalise against ATR; clamp to [0, 1]
                raw_strength = macd_diff / (current_atr * 2 + 1e-9)
                strength = float(np.clip(raw_strength, 0.0, 1.0))
                return Signal(
                    symbol=bar.symbol,
                    direction=Direction.LONG,
                    strength=strength,
                    timestamp=bar.timestamp,
                    strategy_name=self.__class__.__name__,
                    stop_loss=current_close - self._stop_atr_mult * current_atr,
                    take_profit=current_close + self._tp_atr_mult * current_atr,
                    metadata={
                        "macd": round(float(macd_line[-1]), 4),
                        "signal": round(float(signal_line[-1]), 4),
                        "supertrend_dir": int(st_direction[-1]),
                    },
                )

            if macd_cross_down and st_bearish:
                macd_diff = abs(macd_line[-1] - signal_line[-1])
                raw_strength = macd_diff / (current_atr * 2 + 1e-9)
                strength = float(np.clip(raw_strength, 0.0, 1.0))
                return Signal(
                    symbol=bar.symbol,
                    direction=Direction.SHORT,
                    strength=strength,
                    timestamp=bar.timestamp,
                    strategy_name=self.__class__.__name__,
                    stop_loss=current_close + self._stop_atr_mult * current_atr,
                    take_profit=current_close - self._tp_atr_mult * current_atr,
                    metadata={
                        "macd": round(float(macd_line[-1]), 4),
                        "signal": round(float(signal_line[-1]), 4),
                        "supertrend_dir": int(st_direction[-1]),
                    },
                )

        if current_pos == Direction.LONG and macd_cross_down:
            return Signal(
                symbol=bar.symbol,
                direction=Direction.FLAT,
                strength=1.0,
                timestamp=bar.timestamp,
                strategy_name=self.__class__.__name__,
                metadata={"exit_reason": "macd_cross_down"},
            )

        if current_pos == Direction.SHORT and macd_cross_up:
            return Signal(
                symbol=bar.symbol,
                direction=Direction.FLAT,
                strength=1.0,
                timestamp=bar.timestamp,
                strategy_name=self.__class__.__name__,
                metadata={"exit_reason": "macd_cross_up"},
            )

        return self._flat_signal(bar)
