"""Gold–Platinum relative-value mean-reversion strategy."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Config
from core.enums import Direction
from core.models import Bar, Signal
from indicators.volatility import atr
from strategies.base import Strategy

logger = logging.getLogger(__name__)


def _aligned_gold_closes(index: pd.DatetimeIndex, gold_ticker: str) -> Optional[np.ndarray]:
    """Fetch gold closes aligned to ``index`` via yfinance; None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance unavailable for gold hedge leg")
        return None
    try:
        raw = yf.download(gold_ticker, period="max", auto_adjust=True, progress=False)
        if raw.empty:
            return None
        col = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        series = col.copy()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        aligned = series.reindex(index).ffill()
        return aligned.to_numpy(dtype=np.float64)
    except Exception as exc:  # pragma: no cover
        logger.warning("gold fetch failed: %s", exc)
        return None


class PlatinumGoldSpread(Strategy):
    """
    Mean-revert the log PL–GC spread.

    Builds ``s = ln(PL) - β ln(GC)`` with rolling OLS β, z-scores the spread,
    and trades the platinum leg: long PL when cheap vs gold (z < -entry),
    short PL when rich (z > entry). Gold prices are loaded once and aligned
    to the PL bar buffer timestamps.
    """

    def __init__(self, config: Config, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        self._lookback = int(getattr(config.strategy, "rv_lookback", None) or config.strategy.lookback)
        self._entry_z = float(config.strategy.entry_z_score)
        self._exit_z = float(config.strategy.exit_z_score)
        self._gold_ticker = str(getattr(config.strategy, "gold_ticker", "GC=F"))
        self._atr_period = config.indicators.atr_period
        self._stop_atr_mult = config.risk.default_stop_loss_atr_mult
        self._tp_atr_mult = config.risk.default_take_profit_atr_mult
        self._gold_cache: Optional[np.ndarray] = None
        self._gold_index_len: int = 0
        self._synthetic_gold_ratio: float = 1.8

    def _gold_series(self, timestamps: list, n: int) -> np.ndarray:
        if self._gold_cache is not None and self._gold_index_len == n:
            return self._gold_cache
        idx = pd.to_datetime(pd.Index(timestamps))
        fetched = _aligned_gold_closes(idx, self._gold_ticker)
        if fetched is None or np.all(~np.isfinite(fetched)):
            # Fallback: synthetic gold path proportional to PL (no live edge; tests/offline)
            closes = self._buffers[list(self._buffers.keys())[0]].closes()
            fetched = closes * self._synthetic_gold_ratio
            logger.info("using synthetic gold proxy for %s", self._gold_ticker)
        self._gold_cache = np.asarray(fetched, dtype=np.float64)
        self._gold_index_len = n
        return self._gold_cache

    def on_bar(self, bar: Bar) -> Signal:
        buf = self._buffers[bar.symbol]
        closes = buf.closes()
        highs = buf.highs()
        lows = buf.lows()
        n = len(closes)
        L = self._lookback

        if n < L + 5:
            return self._flat_signal(bar)

        atr_vals = atr(highs, lows, closes, self._atr_period)
        if atr_vals is None:
            return self._flat_signal(bar)
        current_atr = atr_vals[-1]
        if np.isnan(current_atr) or current_atr <= 0:
            return self._flat_signal(bar)

        gold = self._gold_series(buf.timestamps(), n)
        if gold.shape[0] != n:
            gold = np.resize(gold, n)

        ln_pl = np.log(np.clip(closes, 1e-12, None))
        ln_gc = np.log(np.clip(gold, 1e-12, None))

        # Rolling OLS β on last L bars
        y = ln_pl[-L:]
        x = ln_gc[-L:]
        x_c = x - np.mean(x)
        y_c = y - np.mean(y)
        var_x = float(np.dot(x_c, x_c))
        if var_x < 1e-12:
            return self._flat_signal(bar)
        beta = float(np.dot(x_c, y_c) / var_x)

        spread = ln_pl - beta * ln_gc
        window = spread[-L:]
        mu = float(np.mean(window))
        sd = float(np.std(window, ddof=1))
        if sd < 1e-12:
            return self._flat_signal(bar)
        z = float((spread[-1] - mu) / sd)

        current_pos = self.current_position(bar.symbol)
        direction = Direction.FLAT
        strength = 0.0

        if current_pos == Direction.FLAT:
            if z <= -self._entry_z:
                direction = Direction.LONG
                strength = min(1.0, abs(z) / max(self._entry_z * 2.0, 1e-6))
            elif z >= self._entry_z:
                direction = Direction.SHORT
                strength = min(1.0, abs(z) / max(self._entry_z * 2.0, 1e-6))
        else:
            # Hold until mean reversion toward exit band
            if current_pos == Direction.LONG and z < -self._exit_z:
                direction = Direction.LONG
                strength = min(1.0, abs(z) / max(self._entry_z * 2.0, 1e-6))
            elif current_pos == Direction.SHORT and z > self._exit_z:
                direction = Direction.SHORT
                strength = min(1.0, abs(z) / max(self._entry_z * 2.0, 1e-6))
            else:
                direction = Direction.FLAT
                strength = 1.0

        if direction == Direction.FLAT:
            return Signal(
                symbol=bar.symbol,
                direction=Direction.FLAT,
                strength=strength if current_pos != Direction.FLAT else 0.0,
                timestamp=bar.timestamp,
                strategy_name=self.__class__.__name__,
                metadata={"z": round(z, 4), "beta": round(beta, 4)},
            )

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
            metadata={"z": round(z, 4), "beta": round(beta, 4), "spread": round(float(spread[-1]), 6)},
        )
