"""
CL inventory-confirm overlay.

Takes carry direction only when inventory surprise agrees (build surprise → short
bias; draw surprise → long). Uses injected inventory when present on bar metadata
via a parallel buffer; otherwise a price-implied inventory proxy (research only).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from core.config import Config
from core.enums import Direction
from core.models import Bar, Signal
from data.cl_features import (
    clip01,
    inventory_proxy_from_price,
    inventory_surprise,
    resolve_curve,
    spread_carry,
)
from indicators.volatility import atr
from strategies.base import BarBuffer, Strategy

logger = logging.getLogger(__name__)


class CLInventoryConfirm(Strategy):
    """Carry signal gated / scaled by inventory-surprise agreement."""

    def __init__(self, config: Config, symbols: Optional[List[str]] = None) -> None:
        super().__init__(config, symbols=symbols)
        sc = config.strategy
        self._back_month = int(getattr(sc, "carry_back_month", 3))
        self._basis_lookback = int(getattr(sc, "carry_basis_lookback", 63))
        self._inv_sma = int(getattr(sc, "inventory_sma", 60))
        self._expect_window = int(getattr(sc, "inventory_expect_window", 8))
        self._confirm_scale = bool(getattr(sc, "inventory_confirm_scale", True))
        self._enabled = bool(getattr(sc, "inventory_enabled", True))
        self._strength_atr_mult = float(getattr(sc, "carry_strength_atr_mult", 1.0))
        self._atr_period = int(config.indicators.atr_period)
        self._stop_atr_mult = config.risk.default_stop_loss_atr_mult
        self._tp_atr_mult = config.risk.default_take_profit_atr_mult
        # Optional external inventory series keyed by symbol (set in tests / research).
        self._external_inventory: Dict[str, np.ndarray] = {}
        maxlen = max(
            self._basis_lookback + self._inv_sma + self._atr_period + 20,
            250,
        )
        self._buffers = {sym: BarBuffer(maxlen=maxlen) for sym in self.symbols}

    def set_inventory_series(self, symbol: str, inventory: np.ndarray) -> None:
        """Inject an EIA (or other) inventory series aligned to upcoming bars."""
        self._external_inventory[symbol] = inventory.astype(np.float64)

    def on_bar(self, bar: Bar) -> Signal:
        if not self._enabled:
            return self._flat_signal(bar)

        buf = self._buffers[bar.symbol]
        closes = buf.closes()
        highs = buf.highs()
        lows = buf.lows()
        n = len(closes)

        front, back = resolve_curve(
            closes, None, self._back_month, self._basis_lookback
        )
        carry = spread_carry(front, back)
        c = carry[-1]

        ext = self._external_inventory.get(bar.symbol)
        if ext is not None and len(ext) >= n and np.isfinite(ext[:n]).any():
            inv = ext[:n]
            inv_source = "external"
        else:
            inv = inventory_proxy_from_price(closes, self._inv_sma)
            inv_source = "price_proxy"

        surprise = inventory_surprise(inv, self._expect_window)
        s = surprise[-1]
        atr_vals = atr(highs, lows, closes, self._atr_period)
        if atr_vals is None:
            return self._flat_signal(bar)

        a = atr_vals[-1]
        if not np.isfinite(c) or not np.isfinite(s) or not np.isfinite(a) or a <= 0.0:
            return self._flat_signal(bar)
        if abs(c) < 1e-12:
            return self._flat_signal(bar)

        carry_sign = 1.0 if c > 0.0 else -1.0
        # Positive surprise (build) → short bias ⇒ desired sign = -sign(s)
        inv_sign = -1.0 if s > 0.0 else (1.0 if s < 0.0 else 0.0)
        if inv_sign == 0.0:
            return self._flat_signal(bar)

        agrees = carry_sign == inv_sign
        if not agrees and not self._confirm_scale:
            return self._flat_signal(bar)
        if not agrees and self._confirm_scale:
            # Soft disagreement: flatten (confirmation required).
            return Signal(
                symbol=bar.symbol,
                direction=Direction.FLAT,
                strength=0.0,
                timestamp=bar.timestamp,
                strategy_name=self.__class__.__name__,
                metadata={
                    "exit_reason": "inventory_disagreement",
                    "carry_spread": round(float(c), 6),
                    "inventory_surprise": round(float(s), 4),
                    "inventory_source": inv_source,
                },
            )

        direction = Direction.LONG if carry_sign > 0.0 else Direction.SHORT
        kappa = max(self._strength_atr_mult, 1e-9)
        strength = clip01(abs(c) / (kappa * a))
        # Boost strength slightly when |surprise| is large vs its recent scale.
        s_scale = float(np.nanstd(surprise[-max(self._expect_window, 2) :]))
        if np.isfinite(s_scale) and s_scale > 1e-12:
            strength = clip01(strength * min(1.5, abs(s) / s_scale / 2.0 + 0.5))

        close = float(closes[-1])
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
                "carry_spread": round(float(c), 6),
                "inventory_surprise": round(float(s), 4),
                "inventory_source": inv_source,
                "atr": round(float(a), 4),
            },
        )
