"""
Tick → OHLCV bar aggregation.

Futures historical tick data (e.g. Databento ``mbp-1`` / ``trades`` schemas, or
the live :class:`~data.live.LiveTickStream`) arrives as a stream of last-trade
prints.  Backtesting and optimisation in this terminal operate on
:class:`~core.models.Bar` objects, so raw ticks must first be aggregated into
fixed-interval OHLCV bars.

All aggregation is fully vectorised (NumPy + pandas ``groupby``); there are no
Python-level loops over individual ticks.
"""

from __future__ import annotations

import logging
from datetime import timezone
from typing import List

import numpy as np
import pandas as pd

from core.enums import AssetClass
from core.models import Bar, Tick

logger = logging.getLogger(__name__)

# Pandas resample rule per supported terminal timeframe.
TIMEFRAME_TO_PANDAS_RULE: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
    "4h": "240min",
    "1d": "1D",
}


def ticks_to_ohlcv(
    ticks_df: pd.DataFrame,
    timeframe: str = "1m",
    *,
    price_col: str = "price",
    size_col: str = "size",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Aggregate a tick DataFrame into an OHLCV bar DataFrame.

    Parameters
    ----------
    ticks_df:
        DataFrame with at least a timestamp column and a trade ``price`` column.
        An optional ``size`` column is summed into bar volume (defaults to 1
        contract per tick when absent).
    timeframe:
        Target bar interval — one of the keys of :data:`TIMEFRAME_TO_PANDAS_RULE`.
    price_col, size_col, timestamp_col:
        Column names within *ticks_df*.

    Returns
    -------
    DataFrame with columns ``Date, Open, High, Low, Close, Volume`` sorted by
    time.  Empty (correctly-typed) DataFrame when *ticks_df* is empty.
    """
    empty = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    if ticks_df is None or ticks_df.empty or price_col not in ticks_df.columns:
        return empty

    rule = TIMEFRAME_TO_PANDAS_RULE.get(timeframe, TIMEFRAME_TO_PANDAS_RULE["1m"])

    frame = ticks_df[[timestamp_col, price_col]].copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    frame["__size"] = (
        ticks_df[size_col].to_numpy(dtype=np.float64)
        if size_col in ticks_df.columns
        else np.ones(len(ticks_df), dtype=np.float64)
    )

    grouped = frame.set_index(timestamp_col).sort_index().resample(rule)
    ohlc = grouped[price_col].ohlc()
    volume = grouped["__size"].sum()

    bars = ohlc.join(volume.rename("Volume")).dropna(subset=["open", "close"])
    bars = bars.reset_index().rename(
        columns={
            timestamp_col: "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
        }
    )
    return bars[["Date", "Open", "High", "Low", "Close", "Volume"]]


def tick_objects_to_dataframe(ticks: List[Tick]) -> pd.DataFrame:
    """
    Convert a list of :class:`~core.models.Tick` objects into a tick DataFrame.

    Uses the last-trade price (falling back to the mid when ``last`` is zero)
    so the output is directly consumable by :func:`ticks_to_ohlcv`.
    """
    if not ticks:
        return pd.DataFrame(columns=["timestamp", "price", "size"])

    timestamps = pd.to_datetime([t.timestamp for t in ticks], utc=True).tz_localize(None)
    last = np.array([t.last for t in ticks], dtype=np.float64)
    mids = np.array([t.mid for t in ticks], dtype=np.float64)
    prices = np.where(last > 0.0, last, mids)
    sizes = np.array([t.last_size for t in ticks], dtype=np.float64)

    return pd.DataFrame({"timestamp": timestamps, "price": prices, "size": sizes})


def ohlcv_dataframe_to_bars(
    df: pd.DataFrame,
    symbol: str,
    contract_multiplier: float,
    asset_class: AssetClass = AssetClass.FUTURES,
) -> List[Bar]:
    """
    Convert an OHLCV DataFrame (``Date, Open, High, Low, Close, Volume``) into a
    list of :class:`~core.models.Bar` objects with tz-aware UTC timestamps.

    NumPy arrays are extracted once; timestamp conversion is vectorised via
    pandas before the final list comprehension assembles Bar dataclasses.
    """
    if df is None or df.empty:
        return []

    dates_utc = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    opens   = df["Open"].to_numpy(dtype=np.float64)
    highs   = df["High"].to_numpy(dtype=np.float64)
    lows    = df["Low"].to_numpy(dtype=np.float64)
    closes  = df["Close"].to_numpy(dtype=np.float64)
    volumes = df["Volume"].to_numpy(dtype=np.float64)

    return [
        Bar(
            symbol=symbol,
            timestamp=ts.to_pydatetime().replace(tzinfo=timezone.utc),
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            volume=float(volumes[i]),
            asset_class=asset_class,
            contract_multiplier=contract_multiplier,
        )
        for i, ts in enumerate(dates_utc)
    ]
