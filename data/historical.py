"""Historical OHLCV ingestion via Databento and local Parquet cache."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Optional heavy imports — gracefully degraded when not installed
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:  # pragma: no cover
    HAS_POLARS = False
    import pandas as pl  # type: ignore[no-redef]

try:
    import databento as db
    HAS_DATABENTO = True
except ImportError:  # pragma: no cover
    HAS_DATABENTO = False
    db = None  # type: ignore[assignment]


class DatabentoHistoricalClient:
    """
    Wrapper around the Databento Python SDK for CME futures data.

    Caches results as Parquet files to minimise API usage costs.

    Parameters
    ----------
    api_key:
        Databento API key.  Falls back to the ``DATABENTO_API_KEY`` env var.
    cache_dir:
        Local directory for Parquet cache.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str | Path = "data/cache",
    ) -> None:
        self._api_key = api_key or os.getenv("DATABENTO_API_KEY", "")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[object] = None

        if HAS_DATABENTO and self._api_key:
            try:
                self._client = db.Historical(self._api_key)
                logger.info("Databento Historical client initialised.")
            except Exception:
                logger.exception("Failed to create Databento client.")

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_bars(
        self,
        dataset: str,
        symbols: List[str],
        schema: str,
        start: str | date,
        end: str | date,
        *,
        force_refresh: bool = False,
    ) -> "pl.DataFrame":
        """
        Return an OHLCV Polars DataFrame for *symbols* over [start, end].

        Checks the local Parquet cache before hitting the API.
        """
        cache_path = self._cache_path(dataset, symbols, schema, start, end)

        if not force_refresh and cache_path.exists():
            logger.info("Loading from cache: %s", cache_path)
            return self._load_parquet(cache_path)

        if self._client is None:
            raise RuntimeError(
                "Databento client not initialised — check DATABENTO_API_KEY."
            )

        logger.info(
            "Fetching %s %s %s → %s from Databento …", schema, symbols, start, end
        )
        data = self._client.timeseries.get_range(  # type: ignore[union-attr]
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            start=str(start),
            end=str(end),
        )
        df = data.to_df()

        if HAS_POLARS:
            df = pl.from_pandas(df)

        self._save_parquet(df, cache_path)
        return df

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cache_path(
        self,
        dataset: str,
        symbols: List[str],
        schema: str,
        start: object,
        end: object,
    ) -> Path:
        sym_tag = "_".join(s.replace(".", "-") for s in sorted(symbols))
        fname = f"{dataset}__{sym_tag}__{schema}__{start}__{end}.parquet"
        return self._cache_dir / fname

    @staticmethod
    def _load_parquet(path: Path) -> "pl.DataFrame":
        if HAS_POLARS:
            return pl.read_parquet(path)
        import pandas as pd  # type: ignore[import]
        return pd.read_parquet(path)

    @staticmethod
    def _save_parquet(df: "pl.DataFrame", path: Path) -> None:
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            df.write_parquet(path)
        else:
            df.to_parquet(path, index=False)  # type: ignore[union-attr]


class HistoricalDataLoader:
    """
    High-level loader that assembles multi-symbol DataFrames suitable for
    backtesting, applying continuous-contract roll and normalisation.
    """

    def __init__(self, client: DatabentoHistoricalClient) -> None:
        self._client = client

    def load(
        self,
        dataset: str,
        symbols: List[str],
        schema: str,
        start: str,
        end: str,
        *,
        roll_method: str = "calendar",
    ) -> Dict[str, "pl.DataFrame"]:
        """
        Return a ``{symbol: DataFrame}`` mapping for all *symbols*.

        Each DataFrame has columns: timestamp, open, high, low, close, volume.
        Continuous contracts are stitched using the *roll_method*.
        """
        from data.transforms import continuous_contract_roll, normalize_ohlcv

        raw = self._client.fetch_bars(dataset, symbols, schema, start, end)
        results: Dict[str, pl.DataFrame] = {}

        for symbol in symbols:
            if HAS_POLARS:
                sym_df = raw.filter(pl.col("symbol") == symbol)
            else:
                sym_df = raw[raw["symbol"] == symbol].copy()  # type: ignore[index]

            sym_df = continuous_contract_roll(sym_df, method=roll_method)
            sym_df = normalize_ohlcv(sym_df)
            results[symbol] = sym_df
            logger.info("Loaded %s rows for %s", len(sym_df), symbol)

        return results

    def to_numpy(self, df: "pl.DataFrame") -> Dict[str, np.ndarray]:
        """Extract OHLCV columns as a dict of numpy arrays for fast computation."""
        cols = ["open", "high", "low", "close", "volume"]
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            return {c: df[c].to_numpy() for c in cols if c in df.columns}
        return {c: df[c].to_numpy() for c in cols if c in df.columns}  # type: ignore[union-attr]
