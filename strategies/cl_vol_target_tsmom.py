"""
CL vol-targeted time-series momentum with nonlinear reaction sizing.

Multi-horizon TSMOM, EWMA vol targeting, and reaction R(z) that shrinks
extreme forecasts. Designed as an overlay sleeve, not a standalone oil hero.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from core.config import Config
from core.enums import Direction
from core.models import Bar, Signal
from data.cl_features import (
    clip01,
    ewma_volatility,
    multi_horizon_tsmom,
    reaction_function,
)
from indicators.volatility import atr
from strategies.base import BarBuffer, Strategy

logger = logging.getLogger(__name__)


class CLVolTargetTSMOM(Strategy):
    """Vol-scaled multi-horizon TSMOM with reaction-function sizing."""

    def __init__(self, config: Config, symbols: Optional[List[str]] = None) -> None:
        super().__init__(config, symbols=symbols)
        sc = config.strategy
        h_short = int(getattr(sc, "tsmom_horizon_short", 20))
        h_med = int(getattr(sc, "tsmom_horizon_med", 60))
        h_long = int(getattr(sc, "tsmom_horizon_long", 120))
        self._horizons: Tuple[int, ...] = (h_short, h_med, h_long)
        self._vol_target = float(getattr(sc, "vol_target_annual", 0.12))
        self._ewma_com = float(getattr(sc, "ewma_vol_com", 60.0))
        self._reaction_b = float(getattr(sc, "reaction_b", 1.0))
        self._z_cap = float(getattr(sc, "tsmom_z_cap", 3.0))
        self._w_max = float(getattr(sc, "vol_target_leverage_cap", 2.0))
        self._atr_period = int(config.indicators.atr_period)
        self._stop_atr_mult = config.risk.default_stop_loss_atr_mult
        self._tp_atr_mult = config.risk.default_take_profit_atr_mult
        self._equity = float(config.portfolio.initial_cash)
        self._multiplier = float(config.portfolio.contract_multiplier)
        maxlen = max(max(self._horizons) + int(self._ewma_com) + 30, 300)
        self._buffers = {sym: BarBuffer(maxlen=maxlen) for sym in self.symbols}

    def on_bar(self, bar: Bar) -> Signal:
        buf = self._buffers[bar.symbol]
        closes = buf.closes()
        highs = buf.highs()
        lows = buf.lows()

        log_c = np.log(np.clip(closes, 1e-12, None))
        rets = np.empty_like(log_c)
        rets[0] = np.nan
        rets[1:] = log_c[1:] - log_c[:-1]

        vol = ewma_volatility(rets, self._ewma_com)
        z_arr = multi_horizon_tsmom(closes, self._horizons, vol)
        atr_vals = atr(highs, lows, closes, self._atr_period)
        if atr_vals is None:
            return self._flat_signal(bar)

        z = z_arr[-1]
        sigma = vol[-1]
        a = atr_vals[-1]
        if not np.isfinite(z) or not np.isfinite(sigma) or sigma <= 0.0:
            return self._flat_signal(bar)
        if not np.isfinite(a) or a <= 0.0:
            return self._flat_signal(bar)

        # Cap z before reaction — R(z) → 0 for |z| ≫ 1 (intended for O(1) scores).
        z_cap = max(self._z_cap, 1e-9)
        z_capped = float(np.clip(z, -z_cap, z_cap))
        r_arr = reaction_function(
            np.array([z_capped], dtype=np.float64), self._reaction_b
        )
        r = float(r_arr[0])
        if not np.isfinite(r) or abs(r) < 1e-12:
            return self._flat_signal(bar)

        # Annualise daily-ish vol; bar spacing unknown — treat sigma as per-bar and
        # scale by sqrt(252) as a conservative CTA default for daily research series.
        sigma_ann = float(sigma) * np.sqrt(252.0)
        w = self._vol_target / max(sigma_ann, 1e-8)
        w = float(np.clip(w, 0.0, self._w_max))

        close = float(closes[-1])
        notional_per = max(close * self._multiplier, 1e-8)
        scaled = w * (r / z_cap) * self._equity / notional_per
        contracts = float(np.round(abs(scaled)))
        if contracts < 1.0 and abs(scaled) >= 0.5:
            contracts = 1.0

        direction = Direction.LONG if r > 0.0 else Direction.SHORT
        strength = clip01(abs(r) / z_cap)

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
            suggested_size=contracts if contracts >= 1.0 else None,
            metadata={
                "tsmom_z": round(float(z), 4),
                "reaction": round(float(r), 4),
                "vol_ann": round(sigma_ann, 6),
                "leverage_w": round(w, 4),
                "suggested_contracts": contracts,
            },
        )
