"""Abstract base class for all trading strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional

import numpy as np

from core.config import Config
from core.enums import Direction
from core.models import Bar, Signal

logger = logging.getLogger(__name__)


class BarBuffer:
    """
    Rolling fixed-size buffer of ``Bar`` objects.

    Provides O(1) append and vectorised extraction of OHLCV arrays,
    making it efficient to pass to NumPy-based indicator functions.
    """

    def __init__(self, maxlen: int) -> None:
        self._buffer: Deque[Bar] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def push(self, bar: Bar) -> None:
        self._buffer.append(bar)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self._maxlen

    def __len__(self) -> int:
        return len(self._buffer)

    # ── Vectorised extraction ─────────────────────────────────────────────────

    def opens(self) -> np.ndarray:
        return np.array([b.open for b in self._buffer], dtype=np.float64)

    def highs(self) -> np.ndarray:
        return np.array([b.high for b in self._buffer], dtype=np.float64)

    def lows(self) -> np.ndarray:
        return np.array([b.low for b in self._buffer], dtype=np.float64)

    def closes(self) -> np.ndarray:
        return np.array([b.close for b in self._buffer], dtype=np.float64)

    def volumes(self) -> np.ndarray:
        return np.array([b.volume for b in self._buffer], dtype=np.float64)

    def timestamps(self) -> List[datetime]:
        return [b.timestamp for b in self._buffer]

    def latest(self) -> Optional[Bar]:
        return self._buffer[-1] if self._buffer else None


class Strategy(ABC):
    """
    Abstract base class for all strategies.

    Subclasses must implement ``on_bar``, which maps an incoming ``Bar``
    to a ``Signal`` (including ``Direction.FLAT`` for no-trade intent).

    Parameters
    ----------
    config:
        Global terminal config; strategy-specific params live in
        ``config.strategy``.
    symbols:
        List of instrument symbols this strategy should trade.
    """

    def __init__(self, config: Config, symbols: Optional[List[str]] = None) -> None:
        self.config = config
        self.symbols = symbols or config.data.symbols
        self._buffers: Dict[str, BarBuffer] = {
            sym: BarBuffer(maxlen=max(config.indicators.sma_long + 10, 200))
            for sym in self.symbols
        }
        self._position: Dict[str, Direction] = {
            sym: Direction.FLAT for sym in self.symbols
        }
        logger.info(
            "%s initialised for symbols: %s", self.__class__.__name__, self.symbols
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def on_bar(self, bar: Bar) -> Signal:
        """
        Process one OHLCV bar and return a trading signal.

        Must always return a ``Signal`` — use ``Direction.FLAT`` to indicate
        no actionable intent.
        """

    # ── Provided helpers ──────────────────────────────────────────────────────

    def update(self, bar: Bar) -> Signal:
        """
        Public entry point: push bar into buffer then call ``on_bar``.

        Ensures warm-up guard: returns a FLAT signal until the buffer
        for this symbol is full.
        """
        if bar.symbol not in self._buffers:
            logger.warning("Unknown symbol %s — ignored.", bar.symbol)
            return self._flat_signal(bar)

        self._buffers[bar.symbol].push(bar)

        if not self._buffers[bar.symbol].is_full:
            return self._flat_signal(bar)

        return self.on_bar(bar)

    def _flat_signal(self, bar: Bar) -> Signal:
        return Signal(
            symbol=bar.symbol,
            direction=Direction.FLAT,
            strength=0.0,
            timestamp=bar.timestamp,
            strategy_name=self.__class__.__name__,
        )

    def set_position(self, symbol: str, direction: Direction) -> None:
        self._position[symbol] = direction

    def current_position(self, symbol: str) -> Direction:
        return self._position.get(symbol, Direction.FLAT)
