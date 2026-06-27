"""Unit tests for data/ticks.py — tick → OHLCV bar aggregation."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.enums import AssetClass
from core.models import Tick
from data.ticks import (
    ohlcv_dataframe_to_bars,
    tick_objects_to_dataframe,
    ticks_to_ohlcv,
)


class TestTicksToOHLCV:
    def _ticks(self) -> pd.DataFrame:
        # 6 ticks spread across two 1-minute buckets.
        ts = pd.to_datetime([
            "2023-01-01 09:30:05", "2023-01-01 09:30:20", "2023-01-01 09:30:55",
            "2023-01-01 09:31:10", "2023-01-01 09:31:30", "2023-01-01 09:31:50",
        ])
        prices = [100.0, 101.0, 99.5, 102.0, 103.0, 101.5]
        sizes = [1, 2, 1, 3, 1, 2]
        return pd.DataFrame({"timestamp": ts, "price": prices, "size": sizes})

    def test_returns_correct_number_of_bars(self):
        bars = ticks_to_ohlcv(self._ticks(), "1m")
        assert len(bars) == 2

    def test_ohlc_values(self):
        bars = ticks_to_ohlcv(self._ticks(), "1m").reset_index(drop=True)
        first = bars.iloc[0]
        assert first["Open"] == 100.0
        assert first["High"] == 101.0
        assert first["Low"] == 99.5
        assert first["Close"] == 99.5
        assert first["Volume"] == 4  # 1 + 2 + 1

    def test_volume_defaults_to_tick_count(self):
        df = self._ticks().drop(columns=["size"])
        bars = ticks_to_ohlcv(df, "1m")
        assert bars["Volume"].sum() == 6

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["timestamp", "price"])
        bars = ticks_to_ohlcv(empty, "1m")
        assert bars.empty
        assert list(bars.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]

    def test_coarser_timeframe_merges_buckets(self):
        bars = ticks_to_ohlcv(self._ticks(), "5m")
        assert len(bars) == 1
        assert bars.iloc[0]["Open"] == 100.0
        assert bars.iloc[0]["Close"] == 101.5


class TestTickObjectsToDataFrame:
    def test_uses_last_price_then_mid(self):
        ticks = [
            Tick("ES.c.0", datetime(2023, 1, 1, tzinfo=timezone.utc),
                 bid=99.0, ask=101.0, bid_size=5, ask_size=5, last=100.5, last_size=2),
            Tick("ES.c.0", datetime(2023, 1, 1, 0, 1, tzinfo=timezone.utc),
                 bid=99.0, ask=101.0, bid_size=5, ask_size=5, last=0.0, last_size=0),
        ]
        df = tick_objects_to_dataframe(ticks)
        assert len(df) == 2
        assert df["price"].iloc[0] == 100.5
        assert df["price"].iloc[1] == 100.0  # falls back to mid

    def test_empty(self):
        df = tick_objects_to_dataframe([])
        assert df.empty


class TestOHLCVDataFrameToBars:
    def test_builds_tz_aware_bars_with_multiplier(self):
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        })
        bars = ohlcv_dataframe_to_bars(df, "ES.c.0", contract_multiplier=50.0)
        assert len(bars) == 2
        assert bars[0].symbol == "ES.c.0"
        assert bars[0].contract_multiplier == 50.0
        assert bars[0].asset_class == AssetClass.FUTURES
        assert bars[0].timestamp.tzinfo is not None
        assert bars[1].close == 102.0

    def test_empty_returns_empty_list(self):
        empty = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        assert ohlcv_dataframe_to_bars(empty, "ES.c.0", 50.0) == []
