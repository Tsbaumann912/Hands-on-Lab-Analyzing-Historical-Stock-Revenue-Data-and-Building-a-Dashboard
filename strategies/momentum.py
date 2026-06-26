"""Momentum breakout strategy — trades range expansions above recent highs/lows."""

from __future__ import annotations

import logging

import numpy as np

from core.config import Config
from core.enums import Direction
from core.models import Bar, Signal
from indicators.trend import sma, ema
from indicators.volatility import atr
from indicators.volume import volume_oscillator
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class MomentumBreakout(Strategy):
    """
    Breakout strategy based on Donchian-channel highs/lows with volume confirmation.

    Entry: close breaks above (long) or below (short) a rolling N-period high/low
           AND volume is above average.
    Exit:  price falls back inside the channel or crosses a trailing ATR stop.
    """

    def __init__(self, config: Config, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        self._lookback = config.strategy.lookback
        self._atr_period = config.indicators.atr_period
        self._vol_fast = 5
        self._vol_slow = config.indicators.volume_ma_period
        self._stop_atr_mult = config.risk.default_stop_loss_atr_mult
        self._tp_atr_mult = config.risk.default_take_profit_atr_mult
        self._trailing_stops: dict = {}

    def on_bar(self, bar: Bar) -> Signal:
        buf = self._buffers[bar.symbol]
        closes = buf.closes()
        highs = buf.highs()
        lows = buf.lows()
        volumes = buf.volumes()

        atr_vals = atr(highs, lows, closes, self._atr_period)
        vol_osc = volume_oscillator(volumes, self._vol_fast, self._vol_slow)

        if atr_vals is None or vol_osc is None:
            return self._flat_signal(bar)

        current_atr = atr_vals[-1]
        current_vol_osc = vol_osc[-1]

        if np.isnan(current_atr) or np.isnan(current_vol_osc):
            return self._flat_signal(bar)

        # Donchian channel over lookback window
        lookback_highs = highs[-self._lookback - 1 : -1]
        lookback_lows = lows[-self._lookback - 1 : -1]
        channel_high = lookback_highs.max()
        channel_low = lookback_lows.min()
        current_close = closes[-1]
        volume_positive = current_vol_osc > 0

        current_pos = self.current_position(bar.symbol)

        if current_pos == Direction.FLAT:
            if current_close > channel_high and volume_positive:
                strength = min(1.0, (current_close - channel_high) / current_atr)
                stop = current_close - self._stop_atr_mult * current_atr
                self._trailing_stops[bar.symbol] = stop
                return Signal(
                    symbol=bar.symbol,
                    direction=Direction.LONG,
                    strength=strength,
                    timestamp=bar.timestamp,
                    strategy_name=self.__class__.__name__,
                    stop_loss=stop,
                    take_profit=current_close + self._tp_atr_mult * current_atr,
                    metadata={"channel_high": round(channel_high, 4)},
                )

            if current_close < channel_low and volume_positive:
                strength = min(1.0, (channel_low - current_close) / current_atr)
                stop = current_close + self._stop_atr_mult * current_atr
                self._trailing_stops[bar.symbol] = stop
                return Signal(
                    symbol=bar.symbol,
                    direction=Direction.SHORT,
                    strength=strength,
                    timestamp=bar.timestamp,
                    strategy_name=self.__class__.__name__,
                    stop_loss=stop,
                    take_profit=current_close - self._tp_atr_mult * current_atr,
                    metadata={"channel_low": round(channel_low, 4)},
                )

        # Trailing stop exit
        trailing = self._trailing_stops.get(bar.symbol)
        if trailing is not None:
            if current_pos == Direction.LONG:
                new_stop = current_close - self._stop_atr_mult * current_atr
                if new_stop > trailing:
                    self._trailing_stops[bar.symbol] = new_stop
                if current_close < self._trailing_stops[bar.symbol]:
                    return Signal(
                        symbol=bar.symbol,
                        direction=Direction.FLAT,
                        strength=1.0,
                        timestamp=bar.timestamp,
                        strategy_name=self.__class__.__name__,
                        metadata={"exit_reason": "trailing_stop"},
                    )
            elif current_pos == Direction.SHORT:
                new_stop = current_close + self._stop_atr_mult * current_atr
                if new_stop < trailing:
                    self._trailing_stops[bar.symbol] = new_stop
                if current_close > self._trailing_stops[bar.symbol]:
                    return Signal(
                        symbol=bar.symbol,
                        direction=Direction.FLAT,
                        strength=1.0,
                        timestamp=bar.timestamp,
                        strategy_name=self.__class__.__name__,
                        metadata={"exit_reason": "trailing_stop"},
                    )

        return self._flat_signal(bar)
