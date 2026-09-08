"""CopperEnsembleStrategy — combines sleeves A–E into one Signal stream."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from copper_ensemble.blend import SLEEVE_ORDER, blend_forecasts
from copper_ensemble.data import build_feature_matrix
from copper_ensemble.forecasts import compute_all_forecasts
from copper_ensemble.models import Bar, Config, Direction, Signal
from copper_ensemble.sizing import apply_forecast_buffer, forecast_to_contracts


class CopperEnsembleStrategy:
    """
    Standalone HG ensemble.

    Call ``prepare(bars)`` once (or when history updates), then ``signal_at(i)``
    / ``generate_all()`` for backtests. Warm-up bars yield FLAT.
    """

    def __init__(self, config: Config, equity: Optional[float] = None) -> None:
        self.config = config
        self.equity = float(equity if equity is not None else config.portfolio.initial_cash)
        self._bars: List[Bar] = []
        self._features: Dict[str, np.ndarray] = {}
        self._forecasts: Dict[str, np.ndarray] = {}
        self._f_star: np.ndarray = np.array([])
        self._agreement: np.ndarray = np.array([])
        self._fdm: np.ndarray = np.array([])
        self._contracts: np.ndarray = np.array([])

    def prepare(self, bars: List[Bar]) -> None:
        self._bars = bars
        self._features = build_feature_matrix(bars, self.config)
        self._forecasts = compute_all_forecasts(self._features, self.config.ensemble)
        f_star, agreement, fdm, _raw = blend_forecasts(self._forecasts, self.config.ensemble)
        buffered = apply_forecast_buffer(f_star, self.config.ensemble.buffer_forecast)
        self._f_star = buffered
        self._agreement = agreement
        self._fdm = fdm
        self._contracts = forecast_to_contracts(
            buffered,
            self._features["close"],
            self._features["vol"],
            self.equity,
            self.config,
        )

    def _flat(self, bar: Bar) -> Signal:
        return Signal(
            symbol=bar.symbol,
            direction=Direction.FLAT,
            strength=0.0,
            timestamp=bar.timestamp,
            suggested_size=0.0,
            metadata={},
        )

    def _metadata_at(self, i: int) -> dict:
        meta: dict = {}
        if i >= len(self._f_star):
            return meta
        f = self._f_star[i]
        meta["forecast"] = None if np.isnan(f) else float(f)
        meta["agreement"] = float(self._agreement[i]) if i < len(self._agreement) else None
        meta["fdm"] = float(self._fdm[i]) if i < len(self._fdm) else None
        if self._features:
            vol = self._features["vol"][i]
            carry = self._features["carry"][i]
            meta["vol"] = None if np.isnan(vol) else float(vol)
            meta["carry"] = None if np.isnan(carry) else float(carry)
        for k in SLEEVE_ORDER:
            val = self._forecasts[k][i]
            meta[f"sleeve_{k}"] = None if np.isnan(val) else float(val)
        return meta

    def signal_at(self, i: int) -> Signal:
        bar = self._bars[i]
        meta = self._metadata_at(i)
        if i >= len(self._f_star) or np.isnan(self._f_star[i]):
            sig = self._flat(bar)
            sig.metadata = meta
            return sig

        n_contracts = float(self._contracts[i])
        f = float(self._f_star[i])
        if n_contracts == 0.0 or abs(f) < 1e-9:
            sig = self._flat(bar)
            sig.metadata = meta
            return sig

        direction = Direction.LONG if n_contracts > 0 else Direction.SHORT
        strength = float(min(1.0, abs(f) / self.config.ensemble.forecast_cap))
        atr = self._features["atr"][i]
        stop = tp = None
        if not np.isnan(atr):
            if direction == Direction.LONG:
                stop = bar.close - self.config.ensemble.stop_atr_mult * atr
                tp = bar.close + self.config.ensemble.take_profit_atr_mult * atr
            else:
                stop = bar.close + self.config.ensemble.stop_atr_mult * atr
                tp = bar.close - self.config.ensemble.take_profit_atr_mult * atr

        return Signal(
            symbol=bar.symbol,
            direction=direction,
            strength=strength,
            timestamp=bar.timestamp,
            stop_loss=stop,
            take_profit=tp,
            suggested_size=abs(n_contracts),
            metadata=meta,
        )

    def generate_all(self) -> List[Signal]:
        return [self.signal_at(i) for i in range(len(self._bars))]
