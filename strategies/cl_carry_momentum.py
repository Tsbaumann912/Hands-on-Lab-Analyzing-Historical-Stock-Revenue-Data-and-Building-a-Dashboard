"""
CL carry-momentum strategy — primary oil risk-premium sleeve.

Applies momentum to the carry series so slow contango/backwardation flips
become nimbler (Bouchouev; Rebellion Research N≈10).

Confirmation filters (must agree, else FLAT):
  - Fast MA vs Slow MA trend
  - Stochastic RSI regime (not overbought for longs / not oversold for shorts)
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
from indicators.momentum import stochastic_rsi
from indicators.trend import sma
from indicators.volatility import atr
from strategies.base import BarBuffer, Strategy

logger = logging.getLogger(__name__)


class CLCarryMomentum(Strategy):
    """Carry-momentum with Stochastic RSI + Fast/Slow MA confirmation."""

    def __init__(self, config: Config, symbols: Optional[List[str]] = None) -> None:
        super().__init__(config, symbols=symbols)
        sc = config.strategy
        ind = config.indicators
        self._back_month = int(getattr(sc, "carry_back_month", 3))
        self._basis_lookback = int(getattr(sc, "carry_basis_lookback", 63))
        self._lookback = int(getattr(sc, "carry_mom_lookback", 10))
        self._z_max = float(getattr(sc, "carry_mom_z_max", 2.0))
        self._fast_ma = int(getattr(sc, "fast_ma_period", ind.sma_short))
        self._slow_ma = int(getattr(sc, "slow_ma_period", ind.sma_long))
        self._stoch_rsi_period = int(getattr(sc, "stoch_rsi_period", ind.rsi_period))
        self._stoch_period = int(getattr(sc, "stoch_rsi_stoch_period", 14))
        self._stoch_k = int(getattr(sc, "stoch_rsi_k", 3))
        self._stoch_d = int(getattr(sc, "stoch_rsi_d", 3))
        self._stoch_oversold = float(getattr(sc, "stoch_rsi_oversold", 20.0))
        self._stoch_overbought = float(getattr(sc, "stoch_rsi_overbought", 80.0))
        self._atr_period = int(ind.atr_period)
        self._stop_atr_mult = config.risk.default_stop_loss_atr_mult
        self._tp_atr_mult = config.risk.default_take_profit_atr_mult
        if self._fast_ma >= self._slow_ma:
            # Keep a valid fast/slow ordering if misconfigured.
            self._fast_ma = min(self._fast_ma, self._slow_ma - 1)
            self._fast_ma = max(self._fast_ma, 2)
        maxlen = max(
            self._basis_lookback
            + self._lookback
            + self._slow_ma
            + self._stoch_rsi_period
            + self._stoch_period
            + self._stoch_k
            + self._stoch_d
            + self._atr_period
            + 20,
            300,
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
        fast = sma(closes, self._fast_ma)
        slow = sma(closes, self._slow_ma)
        stoch = stochastic_rsi(
            closes,
            rsi_period=self._stoch_rsi_period,
            stoch_period=self._stoch_period,
            k_period=self._stoch_k,
            d_period=self._stoch_d,
        )
        if atr_vals is None or fast is None or slow is None or stoch is None:
            return self._flat_signal(bar)

        z = z_arr[-1]
        a = atr_vals[-1]
        f_ma = fast[-1]
        s_ma = slow[-1]
        k = stoch.pct_k[-1]
        d = stoch.pct_d[-1]
        if not all(np.isfinite(v) for v in (z, a, f_ma, s_ma, k, d)) or a <= 0.0:
            return self._flat_signal(bar)
        if abs(z) < 1e-12:
            return self._flat_signal(bar)

        ma_bull = f_ma > s_ma
        carry_long = z > 0.0
        # Stoch RSI: avoid chasing extremes against the carry signal.
        stoch_allows_long = k < self._stoch_overbought
        stoch_allows_short = k > self._stoch_oversold

        if carry_long and ma_bull and stoch_allows_long:
            direction = Direction.LONG
        elif (not carry_long) and (not ma_bull) and stoch_allows_short:
            direction = Direction.SHORT
        else:
            return Signal(
                symbol=bar.symbol,
                direction=Direction.FLAT,
                strength=0.0,
                timestamp=bar.timestamp,
                strategy_name=self.__class__.__name__,
                metadata={
                    "exit_reason": "indicator_disagreement",
                    "carry_mom_z": round(float(z), 4),
                    "fast_ma": round(float(f_ma), 4),
                    "slow_ma": round(float(s_ma), 4),
                    "stoch_rsi_k": round(float(k), 4),
                    "stoch_rsi_d": round(float(d), 4),
                },
            )

        strength = clip01(abs(z) / max(self._z_max, 1e-9))
        # Slight boost when Stoch RSI leaves an extreme in the trade direction.
        if direction == Direction.LONG and k <= self._stoch_oversold + 10.0:
            strength = clip01(strength * 1.15)
        elif direction == Direction.SHORT and k >= self._stoch_overbought - 10.0:
            strength = clip01(strength * 1.15)

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
                "fast_ma": round(float(f_ma), 4),
                "slow_ma": round(float(s_ma), 4),
                "stoch_rsi_k": round(float(k), 4),
                "stoch_rsi_d": round(float(d), 4),
                "lookback": self._lookback,
                "atr": round(float(a), 4),
            },
        )
