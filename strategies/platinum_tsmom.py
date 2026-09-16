"""Multi-horizon time-series momentum strategy (CTA-style TSMOM)."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from core.config import Config
from core.enums import Direction
from core.models import Bar, Signal
from indicators.volatility import atr
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class PlatinumTSMOM(Strategy):
    """
    Multi-horizon time-series momentum on a single futures symbol.

    Goes long when the equal-weight sign of rolling returns over configured
    horizons is positive, short when negative. Strength scales with agreement
    across horizons. Designed for NYMEX Platinum but usable on any contract.
    """

    def __init__(self, config: Config, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        horizons = getattr(config.strategy, "tsmom_horizons", None) or [21, 63, 252]
        self._horizons: List[int] = [int(h) for h in horizons]
        self._atr_period = config.indicators.atr_period
        self._stop_atr_mult = config.risk.default_stop_loss_atr_mult
        self._tp_atr_mult = config.risk.default_take_profit_atr_mult
        self._min_bars = max(self._horizons) + 2
        # Expand buffers so the longest horizon fits (default Strategy buffer is ~200)
        need = max(self._min_bars + 10, 280)
        from strategies.base import BarBuffer

        self._buffers = {sym: BarBuffer(maxlen=need) for sym in self.symbols}

    def on_bar(self, bar: Bar) -> Signal:
        buf = self._buffers[bar.symbol]
        closes = buf.closes()
        highs = buf.highs()
        lows = buf.lows()

        if len(closes) < self._min_bars:
            return self._flat_signal(bar)

        atr_vals = atr(highs, lows, closes, self._atr_period)
        if atr_vals is None:
            return self._flat_signal(bar)
        current_atr = atr_vals[-1]
        if np.isnan(current_atr) or current_atr <= 0:
            return self._flat_signal(bar)

        rets = closes[1:] / closes[:-1] - 1.0
        # Align so rets[-1] is the latest completed return
        signs: list[float] = []
        for h in self._horizons:
            if rets.shape[0] < h:
                continue
            window_sum = float(np.nansum(rets[-h:]))
            if np.isfinite(window_sum) and window_sum != 0.0:
                signs.append(float(np.sign(window_sum)))

        if not signs:
            return self._flat_signal(bar)

        avg_sign = float(np.mean(signs))
        if abs(avg_sign) < 1e-9:
            return self._flat_signal(bar)

        direction = Direction.LONG if avg_sign > 0 else Direction.SHORT
        strength = min(1.0, abs(avg_sign))
        if self.config.strategy.signal_strength_scaling:
            strength = min(1.0, abs(avg_sign) * (len(signs) / max(len(self._horizons), 1)))

        stop = (
            bar.close - self._stop_atr_mult * current_atr
            if direction == Direction.LONG
            else bar.close + self._stop_atr_mult * current_atr
        )
        tp = (
            bar.close + self._tp_atr_mult * current_atr
            if direction == Direction.LONG
            else bar.close - self._tp_atr_mult * current_atr
        )

        return Signal(
            symbol=bar.symbol,
            direction=direction,
            strength=strength,
            timestamp=bar.timestamp,
            strategy_name=self.__class__.__name__,
            stop_loss=stop,
            take_profit=tp,
            metadata={
                "avg_sign": round(avg_sign, 4),
                "horizons": self._horizons,
                "n_agree": len(signs),
            },
        )
