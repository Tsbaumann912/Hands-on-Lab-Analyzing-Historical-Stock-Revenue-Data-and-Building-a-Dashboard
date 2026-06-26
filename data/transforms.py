"""
Pure-function data transformations: continuous contract roll, resampling,
and OHLCV normalisation.

All operations are vectorised (Polars expressions / NumPy); no row-by-row loops.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:  # pragma: no cover
    HAS_POLARS = False
    import pandas as pl  # type: ignore[no-redef]


def continuous_contract_roll(
    df: "pl.DataFrame",
    method: Literal["calendar", "open_interest", "panama"] = "calendar",
    price_col: str = "close",
) -> "pl.DataFrame":
    """
    Stitch a futures contract series into a continuous series.

    Parameters
    ----------
    df:
        DataFrame with at minimum ``timestamp``, ``close``, and ``expiry``
        (or ``contract``) columns.
    method:
        ``"calendar"`` – roll on the last trading day of the expiry month.
        ``"panama"`` – backward-adjust all prior prices on each roll to
        eliminate price gaps (preserves returns but not price levels).
        ``"open_interest"`` – roll when next contract OI exceeds front month.
    price_col:
        Column on which to compute back-adjustment ratios.
    """
    if "expiry" not in df.columns and "contract" not in df.columns:
        logger.debug("No expiry column found; returning df unchanged.")
        return df

    if method == "panama":
        return _panama_roll(df, price_col)

    # For calendar and OI, we just return the data sorted by timestamp —
    # the caller is responsible for fetching already-rolled continuous symbols
    # (e.g. Databento's ".c.0" notation handles this server-side).
    if HAS_POLARS:
        return df.sort("timestamp")
    return df.sort_values("timestamp").reset_index(drop=True)  # type: ignore[union-attr]


def _panama_roll(df: "pl.DataFrame", price_col: str) -> "pl.DataFrame":
    """
    Backward-adjust prices so that returns are continuous across roll dates.

    The most-recent contract's prices are kept as-is; all prior contracts
    are scaled by the cumulative ratio of ``close_next / close_prev`` at each
    roll boundary.
    """
    if HAS_POLARS:
        df = df.sort("timestamp")

        # Detect roll dates: rows where the contract label changes
        roll_mask = df["contract"] != df["contract"].shift(1)
        roll_indices = df.with_row_index().filter(roll_mask)["index"].to_list()

        prices = df[price_col].to_numpy().copy().astype(np.float64)

        # Walk backwards through roll points and apply the back-adjustment ratio
        cumulative_adj = 1.0
        for idx in reversed(roll_indices[1:]):
            ratio = prices[idx] / prices[idx - 1] if prices[idx - 1] != 0 else 1.0
            cumulative_adj *= ratio
            prices[:idx] *= cumulative_adj

        return df.with_columns(pl.Series(name=price_col, values=prices))

    # Pandas fallback
    df = df.sort_values("timestamp").reset_index(drop=True)  # type: ignore[union-attr]
    roll_mask = df["contract"] != df["contract"].shift(1)
    roll_indices = df.index[roll_mask].tolist()
    prices = df[price_col].to_numpy().copy().astype(np.float64)
    cumulative_adj = 1.0
    for idx in reversed(roll_indices[1:]):
        ratio = prices[idx] / prices[idx - 1] if prices[idx - 1] != 0 else 1.0
        cumulative_adj *= ratio
        prices[:idx] *= cumulative_adj
    df = df.copy()
    df[price_col] = prices
    return df


def resample_bars(
    df: "pl.DataFrame",
    freq: str = "1h",
    timestamp_col: str = "timestamp",
) -> "pl.DataFrame":
    """
    Downsample a minute-bar DataFrame to a coarser frequency.

    Parameters
    ----------
    df:
        Sorted minute-bar DataFrame with OHLCV columns.
    freq:
        Target frequency, e.g. ``"1h"``, ``"4h"``, ``"1d"``.
    """
    if HAS_POLARS:
        return (
            df.sort(timestamp_col)
            .group_by_dynamic(timestamp_col, every=freq)
            .agg(
                pl.first("open"),
                pl.max("high"),
                pl.min("low"),
                pl.last("close"),
                pl.sum("volume"),
            )
            .sort(timestamp_col)
        )

    # Pandas fallback
    df = df.set_index(timestamp_col)  # type: ignore[union-attr]
    resampled = df.resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return resampled.dropna().reset_index()  # type: ignore[return-value]


def normalize_ohlcv(
    df: "pl.DataFrame",
    price_cols: Optional[list] = None,
    volume_col: str = "volume",
) -> "pl.DataFrame":
    """
    Ensure OHLCV columns are float64 and drop rows with NaN in price columns.

    Returns a clean, consistently-typed DataFrame.
    """
    price_cols = price_cols or ["open", "high", "low", "close"]
    all_cols = price_cols + [volume_col]

    if HAS_POLARS:
        existing = [c for c in all_cols if c in df.columns]
        df = df.with_columns(
            [pl.col(c).cast(pl.Float64) for c in existing]
        ).drop_nulls(subset=[c for c in price_cols if c in df.columns])
        return df

    for col in all_cols:
        if col in df.columns:  # type: ignore[union-attr]
            df[col] = df[col].astype(np.float64)  # type: ignore[index]
    return df.dropna(subset=[c for c in price_cols if c in df.columns]).reset_index(drop=True)  # type: ignore[union-attr, return-value]
