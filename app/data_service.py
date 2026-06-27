"""
Centralised data service: wraps yfinance fetches + our quant terminal modules.

All methods are cached for the session to avoid redundant API calls.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Ensure the workspace root is importable for the quant-stack modules below.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


FUTURES_ASSET_CONFIG: Dict[str, Dict[str, str]] = {
    "ES": {"name": "E-mini S&P 500", "yahoo_symbol": "ES=F"},
    "NQ": {"name": "E-mini Nasdaq-100", "yahoo_symbol": "NQ=F"},
    "CL": {"name": "Crude Oil WTI", "yahoo_symbol": "CL=F"},
    "GC": {"name": "Gold", "yahoo_symbol": "GC=F"},
    "ZN": {"name": "10-Year T-Note", "yahoo_symbol": "ZN=F"},
    "ZB": {"name": "30-Year T-Bond", "yahoo_symbol": "ZB=F"},
    "SI": {"name": "Silver", "yahoo_symbol": "SI=F"},
    "NG": {"name": "Natural Gas", "yahoo_symbol": "NG=F"},
}

_POSITIVE_SENTIMENT_TOKENS = {
    "beat", "bullish", "gain", "growth", "improve", "optimistic",
    "rally", "rise", "strong", "surge", "upside", "record",
}
_NEGATIVE_SENTIMENT_TOKENS = {
    "bearish", "cut", "decline", "drop", "fall", "inflation",
    "loss", "miss", "risk", "selloff", "slowdown", "weak",
}


@dataclass(frozen=True)
class HeadlineRecord:
    title: str
    published_at: str
    source: str
    url: str
    sentiment_score: float
    sentiment_label: str


# ── Stock data (yfinance) ─────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def fetch_stock_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Return OHLCV DataFrame for *ticker* over *period*."""
    if not HAS_YF:
        return _synthetic_ohlcv(ticker, 1000)
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        logger.warning("yfinance failed for %s; using synthetic data.", ticker)
        return _synthetic_ohlcv(ticker, 1000)


@lru_cache(maxsize=8)
def fetch_revenue_data(ticker: str) -> pd.DataFrame:
    """Return quarterly revenue DataFrame {Date, Revenue} for *ticker*."""
    if not HAS_YF:
        return _synthetic_revenue(ticker)
    try:
        t = yf.Ticker(ticker)
        fin = t.quarterly_financials
        if fin is None or fin.empty:
            return _synthetic_revenue(ticker)
        rev_row = None
        for label in ("Total Revenue", "Revenue"):
            if label in fin.index:
                rev_row = fin.loc[label]
                break
        if rev_row is None:
            return _synthetic_revenue(ticker)
        df = rev_row.reset_index()
        df.columns = ["Date", "Revenue"]
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df["Revenue"] = (df["Revenue"] / 1e6).round(2)  # → millions
        df = df.dropna().sort_values("Date")
        return df
    except Exception:
        logger.warning("Revenue fetch failed for %s; using synthetic data.", ticker)
        return _synthetic_revenue(ticker)


# ── Futures market intelligence API data ──────────────────────────────────────

def clear_futures_intelligence_cache() -> None:
    """Clear cached futures intelligence payloads."""
    _fetch_futures_asset_intelligence_cached.cache_clear()


def fetch_all_futures_market_intelligence(
    start_year: int = 2000,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Return intelligence payload for all supported futures assets.
    """
    assets: Dict[str, Dict[str, Any]] = {}
    for symbol in FUTURES_ASSET_CONFIG:
        assets[symbol] = fetch_futures_asset_intelligence(
            symbol,
            start_year=start_year,
            force_refresh=force_refresh,
        )

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "start_year": start_year,
        "assets": assets,
    }


def fetch_futures_asset_intelligence(
    symbol: str,
    start_year: int = 2000,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Return futures intelligence payload for one asset symbol.
    """
    key = (symbol or "GC").upper()
    if key not in FUTURES_ASSET_CONFIG:
        key = "GC"

    if force_refresh:
        _fetch_futures_asset_intelligence_cached.cache_clear()

    return _fetch_futures_asset_intelligence_cached(key, start_year)


@lru_cache(maxsize=32)
def _fetch_futures_asset_intelligence_cached(
    symbol: str,
    start_year: int,
) -> Dict[str, Any]:
    config = FUTURES_ASSET_CONFIG[symbol]
    history = _download_futures_history(config["yahoo_symbol"], start_year)
    history = history.sort_values("Date").reset_index(drop=True)
    history = history.dropna(subset=["Date", "Close"])

    close = history["Close"].to_numpy(dtype=np.float64)
    volume = history["Volume"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(history["Date"])
    pct_returns = pd.Series(close).pct_change().fillna(0.0)

    history_start = dates.min().date().isoformat()
    history_end = dates.max().date().isoformat()

    day_count = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
    cagr_pct = (np.power(close[-1] / close[0], 365.25 / day_count) - 1.0) * 100.0
    annual_vol_pct = float(np.nanstd(pct_returns.to_numpy(dtype=np.float64), ddof=0) * np.sqrt(252) * 100.0)

    fundamentals = {
        "price_change_since_2000_pct": float(((close[-1] / close[0]) - 1.0) * 100.0),
        "annualized_return_pct": float(cagr_pct),
        "annualized_volatility_pct": annual_vol_pct,
        "highest_close_since_2000": float(np.nanmax(close)),
        "lowest_close_since_2000": float(np.nanmin(close)),
        "current_close": float(close[-1]),
        "current_volume": float(volume[-1]) if len(volume) else 0.0,
        "history_start": history_start,
        "history_end": history_end,
    }

    headlines = _fetch_yahoo_news_rss(config["yahoo_symbol"], limit=25)
    if not headlines:
        fallback_score = _score_text_sentiment(f"{config['name']} futures market stable")
        headlines = [
            HeadlineRecord(
                title=f"{config['name']} market update unavailable, using synthetic placeholder.",
                published_at=datetime.now(tz=timezone.utc).isoformat(),
                source="Synthetic Feed",
                url="",
                sentiment_score=fallback_score,
                sentiment_label=_score_to_label(fallback_score),
            )
        ]

    headline_scores = np.array([h.sentiment_score for h in headlines], dtype=np.float64)
    headline_sentiment = float(np.nanmean(headline_scores)) if len(headline_scores) else 0.0
    momentum_sentiment = float(np.clip(pct_returns.tail(21).mean() * 45.0, -1.0, 1.0))
    composite_sentiment = float(np.clip((headline_sentiment * 0.70) + (momentum_sentiment * 0.30), -1.0, 1.0))

    monthly_returns = pct_returns.groupby(dates.dt.to_period("M")).mean()
    rolling_mean = monthly_returns.rolling(window=6, min_periods=1).mean()
    rolling_std = monthly_returns.rolling(window=6, min_periods=1).std(ddof=0).replace(0.0, np.nan)
    monthly_score = ((rolling_mean / rolling_std).fillna(0.0) / 2.0).clip(-1.0, 1.0)
    sentiment_series = [
        {
            "month": str(period),
            "score": float(score),
            "label": _score_to_label(float(score)),
        }
        for period, score in monthly_score.items()
    ]

    return {
        "asset_symbol": symbol,
        "asset_name": config["name"],
        "yahoo_symbol": config["yahoo_symbol"],
        "last_updated": datetime.now(tz=timezone.utc).isoformat(),
        "news_headlines": [asdict(item) for item in headlines],
        "fundamentals": fundamentals,
        "sentiment": {
            "current_composite_score": composite_sentiment,
            "current_composite_label": _score_to_label(composite_sentiment),
            "headline_sentiment_score": headline_sentiment,
            "momentum_sentiment_score": momentum_sentiment,
            "series_from_2000": sentiment_series,
        },
    }


def _download_futures_history(yahoo_symbol: str, start_year: int) -> pd.DataFrame:
    if HAS_YF:
        try:
            raw = yf.download(
                yahoo_symbol,
                start=f"{start_year}-01-01",
                auto_adjust=False,
                progress=False,
                interval="1d",
            )
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                frame = raw.reset_index()
                frame.rename(columns={"index": "Date"}, inplace=True)
                frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
                required = ["Date", "Open", "High", "Low", "Close", "Volume"]
                missing = [col for col in required if col not in frame.columns]
                if not missing:
                    return frame[required].dropna(subset=["Date", "Close"])
        except Exception:
            logger.warning("Futures history download failed for %s; using synthetic history.", yahoo_symbol)

    synthetic_n = int((datetime.now(tz=timezone.utc).year - start_year + 1) * 252)
    synthetic = get_synthetic_futures_bars(
        symbol=yahoo_symbol.replace("=F", ""),
        n=max(synthetic_n, 252),
        start_price=1200.0 if yahoo_symbol == "GC=F" else 4000.0,
        volatility=9.0,
        seed=sum(ord(c) for c in yahoo_symbol),
    )
    synthetic = synthetic.copy()
    synthetic["Date"] = pd.date_range(f"{start_year}-01-03", periods=len(synthetic), freq="B")
    return synthetic[["Date", "Open", "High", "Low", "Close", "Volume"]]


def _fetch_yahoo_news_rss(yahoo_symbol: str, limit: int = 25) -> List[HeadlineRecord]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={yahoo_symbol}&region=US&lang=en-US"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        payload = pd.read_xml(response.text, xpath=".//item")
        if payload is None or payload.empty:
            return []
    except Exception:
        logger.warning("Yahoo RSS fetch failed for %s.", yahoo_symbol)
        return []

    clipped = payload.head(limit).fillna("")
    records: List[HeadlineRecord] = []
    for row in clipped.itertuples(index=False):
        title = str(getattr(row, "title", "")).strip()
        link = str(getattr(row, "link", "")).strip()
        source = str(getattr(row, "source", "Yahoo Finance")).strip() or "Yahoo Finance"
        pub_date = str(getattr(row, "pubDate", "")).strip()
        published_at = _normalise_news_timestamp(pub_date)
        score = _score_text_sentiment(title)
        records.append(
            HeadlineRecord(
                title=title,
                published_at=published_at,
                source=source,
                url=link,
                sentiment_score=score,
                sentiment_label=_score_to_label(score),
            )
        )
    return records


def _normalise_news_timestamp(pub_date: str) -> str:
    if not pub_date:
        return datetime.now(tz=timezone.utc).isoformat()
    try:
        parsed = parsedate_to_datetime(pub_date)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        return datetime.now(tz=timezone.utc).isoformat()


def _score_text_sentiment(text: str) -> float:
    tokens = [t.strip(".,:;!?()[]{}'\"").lower() for t in text.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0

    token_array = np.array(tokens, dtype=object)
    positive_count = np.isin(token_array, list(_POSITIVE_SENTIMENT_TOKENS)).sum()
    negative_count = np.isin(token_array, list(_NEGATIVE_SENTIMENT_TOKENS)).sum()
    score = float((positive_count - negative_count) / max(len(tokens), 1))
    return float(np.clip(score * 3.5, -1.0, 1.0))


def _score_to_label(score: float) -> str:
    if score > 0.2:
        return "bullish"
    if score < -0.2:
        return "bearish"
    return "neutral"


# ── Futures / quant terminal data ─────────────────────────────────────────────

def get_synthetic_futures_bars(
    symbol: str = "ES",
    n: int = 500,
    start_price: float = 5200.0,
    volatility: float = 12.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic-looking synthetic ES futures price series.

    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, volatility / start_price, n)
    closes = start_price * np.cumprod(1 + returns)

    highs  = closes * (1 + rng.uniform(0.001, 0.003, n))
    lows   = closes * (1 - rng.uniform(0.001, 0.003, n))
    opens  = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = rng.uniform(80_000, 250_000, n)

    base = pd.Timestamp("2023-01-02 09:30:00")
    dates = pd.date_range(base, periods=n, freq="1min")

    return pd.DataFrame({
        "Date":   dates,
        "Open":   opens.round(2),
        "High":   highs.round(2),
        "Low":    lows.round(2),
        "Close":  closes.round(2),
        "Volume": volumes.astype(int),
    })


def run_backtest_for_ui(
    symbol: str = "ES",
    strategy_name: str = "MeanReversionRSI",
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    bb_period: int = 20,
    initial_cash: float = 100_000.0,
    n_bars: int = 500,
) -> Dict:
    """
    Run a synthetic-data backtest and return a results dict.

    Retained for the Futures/Indicator/Risk pages; the Strategies tab now uses
    :func:`run_strategy_analysis` for full historical-data backtesting.
    """
    from engine.backtest import BacktestEngine
    from strategies.mean_reversion import MeanReversionRSI

    config = _load_analysis_config()
    config.portfolio.initial_cash = initial_cash
    config.indicators.rsi_period = rsi_period
    config.strategy.rsi_oversold = rsi_oversold
    config.strategy.rsi_overbought = rsi_overbought
    config.indicators.bb_period = bb_period
    _apply_contract_spec(config, symbol)

    df = get_synthetic_futures_bars(symbol, n=n_bars)
    sym = f"{symbol}.c.0"
    bars = _dataframe_to_bars(df, sym, config.portfolio.contract_multiplier)

    cls = STRATEGY_REGISTRY.get(strategy_name, MeanReversionRSI)
    engine = BacktestEngine(config, cls)
    result = engine.run({sym: bars})

    n_eq = len(result.equity_curve)
    if n_eq > 0:
        dates_for_eq = df["Date"].iloc[:n_eq].reset_index(drop=True)
        eq_df = pd.DataFrame({
            "Date":   dates_for_eq,
            "Equity": result.equity_curve[:len(dates_for_eq)],
        })
    else:
        eq_df = pd.DataFrame({"Date": df["Date"], "Equity": [initial_cash] * len(df)})

    return {
        "metrics":      result.metrics,
        "equity_curve": eq_df,
        "fills":        result.fills,
        "price_df":     df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ── Strategies tab — historical analysis (backtest / optimize / walk-forward)
# ═══════════════════════════════════════════════════════════════════════════

_ANALYSIS_CONFIG = None       # lazy-loaded Config
_STRATEGY_REGISTRY_CACHE = None


def _load_analysis_config():
    """Load the YAML-backed Config once (falls back to dataclass defaults)."""
    global _ANALYSIS_CONFIG
    from core.config import Config

    root = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(root, "config", "default.yaml")
    try:
        _ANALYSIS_CONFIG = Config.from_yaml(cfg_path)
    except Exception:
        logger.warning("Config.from_yaml failed; using defaults.", exc_info=True)
        _ANALYSIS_CONFIG = Config()
    return _ANALYSIS_CONFIG


def get_analysis_config():
    """Public accessor for the loaded analysis Config."""
    if _ANALYSIS_CONFIG is None:
        return _load_analysis_config()
    return _ANALYSIS_CONFIG


def _strategy_registry() -> Dict[str, Any]:
    global _STRATEGY_REGISTRY_CACHE
    if _STRATEGY_REGISTRY_CACHE is None:
        from strategies.mean_reversion import MeanReversionRSI
        from strategies.momentum import MomentumBreakout
        from strategies.trend_following import TrendFollowingMACD

        _STRATEGY_REGISTRY_CACHE = {
            "MeanReversionRSI":   MeanReversionRSI,
            "MomentumBreakout":   MomentumBreakout,
            "TrendFollowingMACD": TrendFollowingMACD,
        }
    return _STRATEGY_REGISTRY_CACHE


class _LazyRegistry(dict):
    """Dict that populates strategy classes on first access."""

    def get(self, key, default=None):  # type: ignore[override]
        return _strategy_registry().get(key, default)

    def __getitem__(self, key):  # type: ignore[override]
        return _strategy_registry()[key]

    def __contains__(self, key) -> bool:  # type: ignore[override]
        return key in _strategy_registry()


STRATEGY_REGISTRY = _LazyRegistry()


def _contract_yahoo_symbol(symbol: str) -> str:
    cfg = get_analysis_config()
    spec = cfg.contracts.get(symbol)
    if spec is not None:
        return spec.yfinance_ticker
    asset = FUTURES_ASSET_CONFIG.get(symbol)
    if asset is not None:
        return asset["yahoo_symbol"]
    return f"{symbol}=F"


def _apply_contract_spec(config, symbol: str) -> None:
    """Inject per-contract economics into ``config.portfolio`` for notional P&L."""
    spec = config.contracts.get(symbol)
    if spec is None:
        return
    config.portfolio.contract_multiplier = spec.contract_multiplier
    config.portfolio.tick_size = spec.tick_size
    config.portfolio.tick_value = spec.tick_value
    config.portfolio.commission_per_contract = spec.commission_per_contract


def _dataframe_to_bars(df: pd.DataFrame, symbol: str, multiplier: float) -> list:
    from data.ticks import ohlcv_dataframe_to_bars

    return ohlcv_dataframe_to_bars(df, symbol, multiplier)


def load_futures_bars_for_ui(
    symbol: str,
    timeframe: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_source: str = "auto",
    max_bars: int = 60_000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Return an OHLCV DataFrame (``Date, Open, High, Low, Close, Volume``) for a
    futures contract over [start_date, end_date] at *timeframe*, plus a coverage
    metadata dict.

    Data-source resolution
    -----------------------
    ``databento``   — real CME tick/OHLCV via :class:`DatabentoHistoricalClient`
                      (requires ``DATABENTO_API_KEY``); ticks are aggregated to
                      bars when a tick schema is returned.
    ``historical``  — yfinance real daily bars back to ~2000, with synthetic
                      intraday fill for sub-daily timeframes (no API key needed).
    ``synthetic``   — fully synthetic GBM series spanning the requested range.
    ``auto``        — Databento when a key is present, else ``historical``.
    """
    cfg = get_analysis_config()
    start_date = start_date or cfg.analysis.history_start_date
    yahoo_symbol = _contract_yahoo_symbol(symbol)
    resolved = _resolve_data_source(data_source)

    meta: Dict[str, Any]
    if resolved == "databento":
        df, meta = _load_databento_bars(symbol, timeframe, start_date, end_date)
        if df.empty:
            logger.warning("Databento returned no data; falling back to historical.")
            resolved = "historical"

    if resolved == "historical":
        start_year = _year_of(start_date, cfg.analysis.history_start_date)
        df, meta = fetch_futures_timeframe_data(yahoo_symbol, timeframe, start_year)
    elif resolved == "synthetic":
        df, meta = _synthetic_range_bars(symbol, timeframe, start_date, end_date)

    df = _filter_date_range(df, start_date, end_date)

    if len(df) > max_bars:
        df = df.tail(max_bars).reset_index(drop=True)
        meta["bar_cap_applied"] = max_bars

    meta["data_source"] = resolved
    meta["symbol"] = symbol
    meta["yahoo_symbol"] = yahoo_symbol
    meta["bars_used"] = len(df)
    if not df.empty:
        meta["range_from"] = str(pd.to_datetime(df["Date"].iloc[0]).date())
        meta["range_to"] = str(pd.to_datetime(df["Date"].iloc[-1]).date())
    return df, meta


def _resolve_data_source(data_source: str) -> str:
    source = (data_source or "auto").lower()
    if source == "auto":
        return "databento" if os.getenv("DATABENTO_API_KEY") else "historical"
    return source


def _year_of(value: Optional[str], fallback: str) -> int:
    for candidate in (value, fallback, "2000-01-01"):
        if candidate:
            try:
                return pd.to_datetime(candidate).year
            except Exception:
                continue
    return 2000


def _filter_date_range(
    df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    dates = pd.to_datetime(df["Date"])
    mask = np.ones(len(df), dtype=bool)
    if start_date:
        mask &= (dates >= pd.to_datetime(start_date)).to_numpy()
    if end_date:
        mask &= (dates <= pd.to_datetime(end_date)).to_numpy()
    return df.loc[mask].reset_index(drop=True)


def _load_databento_bars(
    symbol: str, timeframe: str, start_date: str, end_date: Optional[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch real CME data from Databento, aggregating ticks to bars if needed."""
    empty = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    try:
        from data.historical import DatabentoHistoricalClient, HAS_DATABENTO
        from data.ticks import ticks_to_ohlcv

        if not HAS_DATABENTO:
            return empty, {"error": "databento sdk not installed"}

        cfg = get_analysis_config()
        client = DatabentoHistoricalClient(cache_dir=cfg.data.cache_dir)
        end = end_date or str(date.today())
        schema = cfg.data.schema
        cont_symbol = f"{symbol}.c.0"
        raw = client.fetch_bars(
            dataset=cfg.data.dataset,
            symbols=[cont_symbol],
            schema=schema,
            start=start_date,
            end=end,
        )
        frame = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        if frame.empty:
            return empty, {"error": "no databento rows"}

        if {"open", "high", "low", "close"}.issubset(frame.columns):
            frame = frame.reset_index().rename(columns={
                "ts_event": "Date", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume",
            })
            cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df = frame[[c for c in cols if c in frame.columns]].dropna()
        else:
            price_col = "price" if "price" in frame.columns else "close"
            frame = frame.reset_index().rename(columns={"ts_event": "timestamp"})
            df = ticks_to_ohlcv(frame, timeframe, price_col=price_col)

        return df, {
            "source": f"Databento {cfg.data.dataset} {schema}",
            "real_bars": len(df),
            "synthetic_bars": 0,
        }
    except Exception:
        logger.warning("Databento fetch failed for %s.", symbol, exc_info=True)
        return empty, {"error": "databento fetch failed"}


def _synthetic_range_bars(
    symbol: str, timeframe: str, start_date: str, end_date: Optional[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate a synthetic OHLCV series spanning the requested date range."""
    cfg = get_analysis_config()
    spec = cfg.contracts.get(symbol)
    start_price = spec.synthetic_start_price if spec else 4500.0

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
    freq_map = {"1d": "B", "4h": "4h", "1h": "h", "30m": "30min",
                "15m": "15min", "5m": "5min", "1m": "min"}
    freq = freq_map.get(timeframe, "B")
    dates = pd.date_range(start_ts, end_ts, freq=freq)
    n = len(dates)
    if n < 2:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"]), {}

    df = get_synthetic_futures_bars(
        symbol=symbol, n=n, start_price=start_price,
        volatility=max(0.5, start_price * 0.012),
        seed=sum(ord(c) for c in symbol),
    )
    df = df.copy()
    df["Date"] = dates
    meta = {"source": "Synthetic GBM", "real_bars": 0, "synthetic_bars": n}
    return df, meta


def run_strategy_analysis(
    mode: str,
    symbol: str,
    strategy_name: str,
    timeframe: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_source: str = "auto",
    initial_cash: float = 100_000.0,
    params: Optional[Dict[str, Any]] = None,
    optimize_params: Optional[List[str]] = None,
    objective_metric: str = "sharpe_ratio",
    grid_steps: Optional[int] = None,
    n_windows: Optional[int] = None,
    n_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Unified entry point for the Strategies tab.

    Loads historical futures bars for the contract/date-range/timeframe and runs
    one of three analyses:

    * ``"backtest"``       — single run with *params*.
    * ``"optimize"``       — exhaustive grid sweep over *optimize_params*.
    * ``"walk_forward"``   — Optuna walk-forward optimisation over *optimize_params*.

    Returns a JSON-serialisable dict consumed by the page callbacks.
    """
    cfg = get_analysis_config()
    _apply_contract_spec(cfg, symbol)
    cfg.portfolio.initial_cash = float(initial_cash)

    is_research = mode in ("optimize", "walk_forward")
    max_bars = cfg.analysis.max_optimize_bars if is_research else cfg.analysis.max_backtest_bars

    df, meta = load_futures_bars_for_ui(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        data_source=data_source,
        max_bars=max_bars,
    )
    if df.empty or len(df) < 60:
        return {"error": f"Insufficient data ({len(df)} bars) for {symbol} {timeframe}.", "coverage": meta}

    sym = f"{symbol}.c.0"
    bars = _dataframe_to_bars(df, sym, cfg.portfolio.contract_multiplier)
    bar_map = {sym: bars}
    strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
    if strategy_cls is None:
        return {"error": f"Unknown strategy {strategy_name}.", "coverage": meta}

    if mode == "backtest":
        return _run_backtest_mode(cfg, strategy_cls, bar_map, df, params or {}, meta)
    if mode == "optimize":
        return _run_optimize_mode(
            cfg, strategy_name, strategy_cls, bar_map, df,
            optimize_params, objective_metric, grid_steps, meta,
        )
    if mode == "walk_forward":
        return _run_walk_forward_mode(
            cfg, strategy_name, strategy_cls, bar_map, df,
            optimize_params, objective_metric, n_windows, n_trials, meta,
        )
    return {"error": f"Unknown analysis mode {mode}.", "coverage": meta}


def _equity_df_from_result(result, df: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    n_eq = len(result.equity_curve)
    if n_eq > 0:
        dates_for_eq = df["Date"].iloc[:n_eq].reset_index(drop=True)
        return pd.DataFrame({
            "Date":   dates_for_eq,
            "Equity": result.equity_curve[:len(dates_for_eq)],
        })
    return pd.DataFrame({"Date": df["Date"], "Equity": [initial_cash] * len(df)})


def _run_backtest_mode(cfg, strategy_cls, bar_map, df, params, meta) -> Dict[str, Any]:
    from engine.backtest import BacktestEngine

    engine = BacktestEngine(cfg, strategy_cls)
    result = engine.run(bar_map, strategy_params=params or None)
    eq_df = _equity_df_from_result(result, df, cfg.portfolio.initial_cash)
    return {
        "mode": "backtest",
        "metrics": result.metrics,
        "equity_curve": eq_df,
        "fills": result.fills,
        "price_df": df,
        "params": params,
        "coverage": meta,
    }


def _run_optimize_mode(
    cfg, strategy_name, strategy_cls, bar_map, df,
    optimize_params, objective_metric, grid_steps, meta,
) -> Dict[str, Any]:
    from engine.optimizer import (
        GridOptimizer,
        STRATEGY_PARAM_SPACES,
        build_param_grid_from_yaml,
    )

    param_names = optimize_params or STRATEGY_PARAM_SPACES.get(strategy_name, [])
    if not param_names:
        return {"error": f"No optimisable parameters for {strategy_name}.", "coverage": meta}

    steps = int(grid_steps or cfg.analysis.grid_steps_per_param)
    root = os.path.dirname(os.path.dirname(__file__))
    optuna_path = os.path.join(root, "config", "optuna.yaml")
    grid = build_param_grid_from_yaml(param_names, steps, path=optuna_path)

    optimizer = GridOptimizer(
        cfg, strategy_cls, grid,
        objective_metric=objective_metric,
        max_combinations=cfg.analysis.max_grid_combinations,
    )
    n_combos = len(optimizer.combinations())
    if n_combos > cfg.analysis.max_grid_combinations:
        return {
            "error": (
                f"Grid has {n_combos} combinations (cap {cfg.analysis.max_grid_combinations}). "
                "Fewer parameters or steps required."
            ),
            "coverage": meta,
        }

    opt_result = optimizer.run(bar_map)
    trials_rows = [
        {"params": tr.params, "objective": tr.objective_value, "metrics": tr.metrics}
        for tr in opt_result.trials
    ]

    from engine.backtest import BacktestEngine
    best_engine = BacktestEngine(cfg, strategy_cls)
    best_result = best_engine.run(bar_map, strategy_params=opt_result.best_params)
    eq_df = _equity_df_from_result(best_result, df, cfg.portfolio.initial_cash)

    return {
        "mode": "optimize",
        "objective_metric": objective_metric,
        "param_names": param_names,
        "grid": grid,
        "n_combinations": opt_result.n_combinations,
        "trials": trials_rows,
        "best_params": opt_result.best_params,
        "best_metrics": opt_result.best_metrics,
        "equity_curve": eq_df,
        "price_df": df,
        "coverage": meta,
    }


def _run_walk_forward_mode(
    cfg, strategy_name, strategy_cls, bar_map, df,
    optimize_params, objective_metric, n_windows, n_trials, meta,
) -> Dict[str, Any]:
    from engine.optimizer import (
        STRATEGY_PARAM_SPACES,
        WalkForwardOptimizer,
        build_search_space_from_yaml,
        HAS_OPTUNA,
    )

    if not HAS_OPTUNA:
        return {"error": "optuna is not installed; walk-forward unavailable.", "coverage": meta}

    param_names = optimize_params or STRATEGY_PARAM_SPACES.get(strategy_name, [])
    root = os.path.dirname(os.path.dirname(__file__))
    optuna_path = os.path.join(root, "config", "optuna.yaml")
    search_space = build_search_space_from_yaml(param_names, path=optuna_path)
    if not search_space:
        return {"error": f"No optimisable parameters for {strategy_name}.", "coverage": meta}

    windows = int(n_windows or cfg.analysis.walk_forward_windows)
    trials = int(n_trials or cfg.analysis.walk_forward_trials)

    optimizer = WalkForwardOptimizer(
        cfg, strategy_cls, search_space, objective_metric=objective_metric
    )
    wfo = optimizer.run(
        bar_map,
        n_windows=windows,
        in_sample_ratio=cfg.analysis.in_sample_ratio,
        n_trials=trials,
        timeout=cfg.analysis.walk_forward_timeout_seconds,
    )

    window_rows = []
    for w in wfo.windows:
        is_obj = (w.is_result.metrics.get(objective_metric) if w.is_result else None)
        oos_obj = (w.oos_result.metrics.get(objective_metric) if w.oos_result else None)
        window_rows.append({
            "window_id": w.window_id,
            "best_params": w.best_params,
            "is_objective": is_obj,
            "oos_objective": oos_obj,
            "oos_metrics": w.oos_result.metrics if w.oos_result else {},
        })

    return {
        "mode": "walk_forward",
        "objective_metric": objective_metric,
        "param_names": param_names,
        "n_windows": windows,
        "n_trials": trials,
        "in_sample_ratio": cfg.analysis.in_sample_ratio,
        "windows": window_rows,
        "aggregated_oos_metrics": wfo.aggregated_oos_metrics,
        "coverage": meta,
    }


def compute_indicators_for_ui(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute all indicators from our library on the given price DataFrame.
    Returns a dict of {indicator_name: array}.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from indicators.momentum import rsi, macd
    from indicators.trend import sma, ema
    from indicators.volatility import atr, bollinger_bands
    from indicators.volume import obv, vwap

    close  = df["Close"].to_numpy(dtype=np.float64)
    high   = df["High"].to_numpy(dtype=np.float64)
    low    = df["Low"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    result = {}

    r = rsi(close, 14)
    if r is not None:
        result["RSI_14"] = r

    m = macd(close, 12, 26, 9)
    if m is not None:
        result["MACD_line"]   = m.macd_line
        result["MACD_signal"] = m.signal_line
        result["MACD_hist"]   = m.histogram

    bb = bollinger_bands(close, 20, 2.0)
    if bb is not None:
        result["BB_upper"]  = bb.upper
        result["BB_middle"] = bb.middle
        result["BB_lower"]  = bb.lower

    atr_v = atr(high, low, close, 14)
    if atr_v is not None:
        result["ATR_14"] = atr_v

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    ema20 = ema(close, 20)
    if sma20 is not None:
        result["SMA_20"] = sma20
    if sma50 is not None:
        result["SMA_50"] = sma50
    if ema20 is not None:
        result["EMA_20"] = ema20

    obv_v = obv(close, volume)
    if obv_v is not None:
        result["OBV"] = obv_v

    vwap_v = vwap(high, low, close, volume)
    if vwap_v is not None:
        result["VWAP"] = vwap_v

    return result


# ── Futures chart catalogue & data fetch ─────────────────────────────────────

FUTURES_CHART_CATALOG: Dict[str, Dict[str, str]] = {
    # Equity Index Futures
    "ES=F":  {"name": "E-mini S&P 500",      "sector": "Index",  "exchange": "CME",   "label": "ES  — E-mini S&P 500"},
    "NQ=F":  {"name": "E-mini Nasdaq-100",    "sector": "Index",  "exchange": "CME",   "label": "NQ  — E-mini Nasdaq-100"},
    "YM=F":  {"name": "E-mini Dow Jones",     "sector": "Index",  "exchange": "CBOT",  "label": "YM  — E-mini Dow Jones"},
    "RTY=F": {"name": "E-mini Russell 2000",  "sector": "Index",  "exchange": "CME",   "label": "RTY — E-mini Russell 2000"},
    "MES=F": {"name": "Micro E-mini S&P 500", "sector": "Index",  "exchange": "CME",   "label": "MES — Micro E-mini S&P 500"},
    "MNQ=F": {"name": "Micro E-mini Nasdaq",  "sector": "Index",  "exchange": "CME",   "label": "MNQ — Micro E-mini Nasdaq-100"},
    # Energy Futures
    "CL=F":  {"name": "Crude Oil WTI",        "sector": "Energy", "exchange": "NYMEX", "label": "CL  — Crude Oil WTI"},
    "BZ=F":  {"name": "Brent Crude Oil",      "sector": "Energy", "exchange": "ICE",   "label": "BZ  — Brent Crude Oil"},
    "NG=F":  {"name": "Natural Gas",          "sector": "Energy", "exchange": "NYMEX", "label": "NG  — Natural Gas"},
    "HO=F":  {"name": "Heating Oil",          "sector": "Energy", "exchange": "NYMEX", "label": "HO  — Heating Oil"},
    "RB=F":  {"name": "Gasoline RBOB",        "sector": "Energy", "exchange": "NYMEX", "label": "RB  — Gasoline RBOB"},
    # Metals Futures
    "GC=F":  {"name": "Gold",                 "sector": "Metals", "exchange": "COMEX", "label": "GC  — Gold"},
    "SI=F":  {"name": "Silver",               "sector": "Metals", "exchange": "COMEX", "label": "SI  — Silver"},
    "HG=F":  {"name": "Copper",               "sector": "Metals", "exchange": "COMEX", "label": "HG  — Copper"},
    "PA=F":  {"name": "Palladium",            "sector": "Metals", "exchange": "NYMEX", "label": "PA  — Palladium"},
    "PL=F":  {"name": "Platinum",             "sector": "Metals", "exchange": "NYMEX", "label": "PL  — Platinum"},
    # Fixed Income Futures
    "ZN=F":  {"name": "10-Year T-Note",       "sector": "Bonds",  "exchange": "CBOT",  "label": "ZN  — 10-Year T-Note"},
    "ZB=F":  {"name": "30-Year T-Bond",       "sector": "Bonds",  "exchange": "CBOT",  "label": "ZB  — 30-Year T-Bond"},
    "ZT=F":  {"name": "2-Year T-Note",        "sector": "Bonds",  "exchange": "CBOT",  "label": "ZT  — 2-Year T-Note"},
    "ZF=F":  {"name": "5-Year T-Note",        "sector": "Bonds",  "exchange": "CBOT",  "label": "ZF  — 5-Year T-Note"},
    # FX Futures
    "6E=F":  {"name": "Euro FX",              "sector": "FX",     "exchange": "CME",   "label": "6E  — Euro FX  (EUR/USD)"},
    "6J=F":  {"name": "Japanese Yen FX",      "sector": "FX",     "exchange": "CME",   "label": "6J  — Japanese Yen FX"},
    "6B=F":  {"name": "British Pound FX",     "sector": "FX",     "exchange": "CME",   "label": "6B  — British Pound FX"},
    "6C=F":  {"name": "Canadian Dollar FX",   "sector": "FX",     "exchange": "CME",   "label": "6C  — Canadian Dollar FX"},
    "6A=F":  {"name": "Australian Dollar FX", "sector": "FX",     "exchange": "CME",   "label": "6A  — Australian Dollar FX"},
    "6S=F":  {"name": "Swiss Franc FX",       "sector": "FX",     "exchange": "CME",   "label": "6S  — Swiss Franc FX"},
    # Agricultural Futures
    "ZC=F":  {"name": "Corn",                 "sector": "Ags",    "exchange": "CBOT",  "label": "ZC  — Corn"},
    "ZS=F":  {"name": "Soybeans",             "sector": "Ags",    "exchange": "CBOT",  "label": "ZS  — Soybeans"},
    "ZW=F":  {"name": "Wheat",                "sector": "Ags",    "exchange": "CBOT",  "label": "ZW  — Wheat"},
    "LE=F":  {"name": "Live Cattle",          "sector": "Ags",    "exchange": "CME",   "label": "LE  — Live Cattle"},
    "HE=F":  {"name": "Lean Hogs",            "sector": "Ags",    "exchange": "CME",   "label": "HE  — Lean Hogs"},
    "KC=F":  {"name": "Coffee",               "sector": "Ags",    "exchange": "ICE",   "label": "KC  — Coffee"},
    "SB=F":  {"name": "Sugar #11",            "sector": "Ags",    "exchange": "ICE",   "label": "SB  — Sugar #11"},
    "CC=F":  {"name": "Cocoa",                "sector": "Ags",    "exchange": "ICE",   "label": "CC  — Cocoa"},
    "CT=F":  {"name": "Cotton",               "sector": "Ags",    "exchange": "ICE",   "label": "CT  — Cotton"},
    # Crypto Futures (CME)
    "BTC=F": {"name": "Bitcoin Futures (CME)", "sector": "Crypto", "exchange": "CME",  "label": "BTC — Bitcoin Futures (CME)"},
    "ETH=F": {"name": "Ethereum Futures (CME)","sector": "Crypto", "exchange": "CME",  "label": "ETH — Ethereum Futures (CME)"},
}

# ── Timeframe configuration ───────────────────────────────────────────────────
#
# max_real_days: how far back yfinance actually provides data at that interval.
# interval_min:  bar size in minutes.
# yf_interval:   the string passed to yf.download(interval=...).

TIMEFRAME_CONFIG: Dict[str, Dict] = {
    "1m":  {"yf_interval": "1m",  "max_real_days": 7,    "interval_min": 1},
    "5m":  {"yf_interval": "5m",  "max_real_days": 60,   "interval_min": 5},
    "15m": {"yf_interval": "15m", "max_real_days": 60,   "interval_min": 15},
    "30m": {"yf_interval": "30m", "max_real_days": 60,   "interval_min": 30},
    "1h":  {"yf_interval": "60m", "max_real_days": 730,  "interval_min": 60},
    "4h":  {"yf_interval": "60m", "max_real_days": 730,  "interval_min": 240},
    "1d":  {"yf_interval": "1d",  "max_real_days": None, "interval_min": 1440},
}

_SESSION_MINUTES = 1380   # 23-hour CME continuous futures session per day


def _fetch_real_intraday(symbol: str, yf_interval: str) -> pd.DataFrame:
    """Download the maximum available real intraday bars from yfinance."""
    period_map = {"1m": "7d", "5m": "60d", "15m": "60d",
                  "30m": "60d", "60m": "730d"}
    period = period_map.get(yf_interval, "60d")

    if not HAS_YF:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    try:
        raw = yf.download(
            symbol, period=period, interval=yf_interval,
            auto_adjust=True, progress=False, multi_level_index=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        frame = raw.reset_index()
        date_col = "Datetime" if "Datetime" in frame.columns else "Date"
        frame = frame.rename(columns={date_col: "Date"})
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
        needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
        frame = frame[[c for c in needed if c in frame.columns]]
        frame = frame.dropna(subset=["Date", "Close"]).reset_index(drop=True)
        if "Volume" not in frame.columns:
            frame["Volume"] = 0
        return frame

    except Exception:
        logger.warning("_fetch_real_intraday failed for %s / %s", symbol, yf_interval)
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])


def _resample_ohlcv_df(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Downsample an OHLCV DataFrame to a coarser bar size using pandas resample."""
    if df.empty:
        return df
    df2 = df.set_index("Date").sort_index()
    rule = f"{interval_minutes}min"
    agg = df2.resample(rule).agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Close"]).reset_index()
    return agg


def _resample_daily_to_intraday(
    daily_df: pd.DataFrame,
    interval_minutes: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Decompose daily OHLCV bars into synthetic intraday bars using vectorised GBM.

    Each daily bar is split into bars_per_day = floor(SESSION_MINUTES / interval_minutes)
    sub-bars.  Price paths satisfy the daily O/H/L/C constraints; volume is
    distributed with a U-shaped profile (heavier at session open and close).

    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume.
    Timestamps start at 23:00 UTC (≈ 18:00 ET) on the trade date's eve,
    matching the CME Globex open for continuous futures.
    """
    if daily_df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    bars_per_day = max(1, _SESSION_MINUTES // interval_minutes)
    n_days       = len(daily_df)
    rng          = np.random.default_rng(seed)

    opens   = daily_df["Open"].to_numpy(dtype=np.float64)
    highs   = daily_df["High"].to_numpy(dtype=np.float64)
    lows    = daily_df["Low"].to_numpy(dtype=np.float64)
    closes  = daily_df["Close"].to_numpy(dtype=np.float64)
    volumes = daily_df["Volume"].to_numpy(dtype=np.float64)

    # ── Price path (n_days × bars_per_day) ────────────────────────────────
    noise = rng.standard_normal((n_days, bars_per_day))
    path  = np.cumsum(noise, axis=1)
    path -= path[:, :1]                                    # anchor left end at 0

    # Normalise to [0, 1]
    p_min = path.min(axis=1, keepdims=True)
    p_max = path.max(axis=1, keepdims=True)
    span  = np.where(p_max > p_min, p_max - p_min, 1.0)
    path  = (path - p_min) / span

    # Steer right endpoint toward (close − low) / (high − low)
    day_range  = np.where(highs > lows, highs - lows, highs * 1e-4)
    target_end = np.clip((closes - lows) / day_range, 0.0, 1.0)
    ramp       = np.linspace(0.0, 1.0, bars_per_day)[np.newaxis, :]
    path       = path + ramp * (target_end[:, np.newaxis] - path[:, -1:])
    path       = np.clip(path, 0.0, 1.0)

    # Map to price space
    prices = lows[:, np.newaxis] + path * day_range[:, np.newaxis]   # (n_days, bars_per_day)

    # ── Bar OHLCV construction ─────────────────────────────────────────────
    bar_c = prices
    bar_o = np.empty_like(bar_c)
    bar_o[:, 0]  = opens
    bar_o[:, 1:] = bar_c[:, :-1]

    # Small intraday wicks
    body_h = np.abs(bar_c - bar_o) * 0.5
    wick   = rng.uniform(0.001, 0.003, (n_days, bars_per_day)) * prices
    bar_h  = np.minimum(np.maximum(bar_o, bar_c) + body_h + wick, highs[:, np.newaxis])
    bar_l  = np.maximum(np.minimum(bar_o, bar_c) - body_h - wick, lows[:, np.newaxis])
    bar_l  = np.minimum(bar_l, bar_h - 1e-8)   # guarantee H ≥ L

    # U-shaped volume profile
    idx      = np.arange(bars_per_day, dtype=np.float64)
    decay    = bars_per_day * 0.15
    u_weight = 1.0 + 2.5 * (np.exp(-idx / decay) + np.exp(-(bars_per_day - 1 - idx) / decay))
    u_weight /= u_weight.sum()
    v_noise  = rng.uniform(0.5, 1.5, (n_days, bars_per_day))
    bar_v    = (volumes[:, np.newaxis] * u_weight[np.newaxis, :] * v_noise).astype(np.int64)

    # ── Timestamps (fully vectorised, no Python loops) ─────────────────────
    # Session open = 23:00 UTC on the calendar day preceding the trade date.
    # Force datetime64[ns] resolution before calling view() so the int64
    # values represent nanoseconds, not days or seconds.
    date_ns = (
        pd.to_datetime(daily_df["Date"])
        .values
        .astype("datetime64[ns]")
        .view(np.int64)
    )
    sess_ns = date_ns - np.int64(1 * 3_600_000_000_000)   # −1 h → 23:00 UTC
    ivl_ns  = np.int64(interval_minutes * 60_000_000_000)
    bar_idx = np.arange(bars_per_day, dtype=np.int64)
    ts_ns   = sess_ns[:, np.newaxis] + bar_idx[np.newaxis, :] * ivl_ns  # (n_days, bars_per_day)

    return pd.DataFrame({
        "Date":   pd.to_datetime(ts_ns.ravel()),
        "Open":   bar_o.ravel().round(6),
        "High":   bar_h.ravel().round(6),
        "Low":    bar_l.ravel().round(6),
        "Close":  bar_c.ravel().round(6),
        "Volume": bar_v.ravel(),
    })


def fetch_futures_timeframe_data(
    symbol: str,
    timeframe: str = "1d",
    start_year: int = 2000,
) -> tuple[pd.DataFrame, dict]:
    """
    Return a full-history OHLCV dataset for a futures contract at any timeframe.

    Strategy
    --------
    1d  — Download real daily bars from ``start_year`` via yfinance (6 000+ bars).
    4h  — Download real 1-hour bars (max 730 days) → resample to 4h;  for years
          before the yfinance cutoff, resample the daily base.
    1h  — Download real 1-hour bars (max 730 days); fill earlier with daily-based.
    30m / 15m / 5m — Real sub-hourly bars (max 60 days); fill earlier with daily-based.
    1m  — Real 1-minute bars (max 7 days); fill earlier with daily-based.

    Parameters
    ----------
    symbol     : yfinance futures symbol, e.g. "ES=F", "CL=F"
    timeframe  : one of "1m" "5m" "15m" "30m" "1h" "4h" "1d"
    start_year : earliest year to include in the synthetic history

    Returns
    -------
    (df, meta) where meta is a dict describing data coverage.
    """
    cfg          = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["1d"])
    interval_min = cfg["interval_min"]

    # 1. Daily base — real data free from yfinance back to ~start_year
    daily_df = _download_futures_history(symbol, start_year)
    daily_df = daily_df.sort_values("Date").reset_index(drop=True)

    _empty = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    if daily_df.empty:
        return _empty, {"error": "no daily data available"}

    if timeframe == "1d":
        meta = {
            "timeframe": "1d",
            "real_bars": len(daily_df),
            "synthetic_bars": 0,
            "total_bars": len(daily_df),
            "real_from": str(pd.to_datetime(daily_df["Date"].min()).date()),
            "real_to":   str(pd.to_datetime(daily_df["Date"].max()).date()),
            "source": "Yahoo Finance · real daily bars",
        }
        return daily_df, meta

    # 2. Fetch maximum real intraday data from yfinance
    real_intraday = _fetch_real_intraday(symbol, cfg["yf_interval"])

    # For 4h: resample the 1h real data → 4h bars
    if timeframe == "4h" and not real_intraday.empty:
        real_intraday = _resample_ohlcv_df(real_intraday, 240)

    # 3. Determine the boundary between real and synthetic data
    if real_intraday.empty:
        cutoff_date = pd.Timestamp.now()
    else:
        cutoff_date = pd.to_datetime(real_intraday["Date"].min())

    # 4. Resample daily bars that pre-date the real-intraday window
    daily_hist = daily_df[pd.to_datetime(daily_df["Date"]) < cutoff_date].copy()

    if not daily_hist.empty:
        synthetic = _resample_daily_to_intraday(daily_hist, interval_min, seed=42)
    else:
        synthetic = _empty.copy()

    # 5. Stitch: synthetic history (old) + real intraday (recent)
    combined = pd.concat([synthetic, real_intraday], ignore_index=True)
    combined = (
        combined
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    synth_from = (str(daily_hist["Date"].min().date()) if not daily_hist.empty else "N/A")
    synth_to   = str(cutoff_date.date())
    real_from  = (str(pd.to_datetime(real_intraday["Date"].min()).date())
                  if not real_intraday.empty else "N/A")
    real_to    = (str(pd.to_datetime(real_intraday["Date"].max()).date())
                  if not real_intraday.empty else "N/A")

    meta = {
        "timeframe":       timeframe,
        "real_bars":       len(real_intraday),
        "synthetic_bars":  len(synthetic),
        "total_bars":      len(combined),
        "real_from":       real_from,
        "real_to":         real_to,
        "synthetic_from":  synth_from,
        "synthetic_to":    synth_to,
        "source":          (
            f"Real: {real_from} → {real_to} (yfinance {cfg['yf_interval']}) | "
            f"Synthetic: {synth_from} → {synth_to} (daily-resampled)"
        ),
    }
    return combined, meta


# ── Synthetic fallbacks ───────────────────────────────────────────────────────

def _synthetic_ohlcv(ticker: str, n: int = 500) -> pd.DataFrame:
    seed = sum(ord(c) for c in ticker)
    rng = np.random.default_rng(seed)
    start = 150.0 if ticker == "TSLA" else 20.0
    returns = rng.normal(0.0003, 0.025, n)
    closes = start * np.cumprod(1 + returns)
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Date":   dates,
        "Open":   (closes * rng.uniform(0.998, 1.002, n)).round(2),
        "High":   (closes * rng.uniform(1.001, 1.015, n)).round(2),
        "Low":    (closes * rng.uniform(0.985, 0.999, n)).round(2),
        "Close":  closes.round(2),
        "Volume": rng.integers(5_000_000, 50_000_000, n),
    })


def _synthetic_revenue(ticker: str) -> pd.DataFrame:
    seed = sum(ord(c) for c in ticker)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=20, freq="QE")
    revenues = rng.uniform(1_500, 25_000, 20)
    return pd.DataFrame({"Date": dates, "Revenue": revenues.round(0)})


# ── Market intelligence (news / fundamentals / sentiment) ─────────────────────

_intelligence_service = None


def _get_intelligence_service():
    global _intelligence_service
    if _intelligence_service is None:
        from data.market_intelligence import MarketIntelligenceService
        _intelligence_service = MarketIntelligenceService()
    return _intelligence_service


def get_futures_intelligence(force_refresh: bool = False) -> Dict:
    """Return intelligence bundles for all configured futures assets."""
    service = _get_intelligence_service()
    bundles = service.get_all(force_refresh=force_refresh)
    from data.market_intelligence import bundle_to_dict
    return {sym: bundle_to_dict(b) for sym, b in bundles.items()}


def get_asset_intelligence(symbol: str, force_refresh: bool = False) -> Dict | None:
    """Return intelligence bundle for a single futures asset."""
    service = _get_intelligence_service()
    bundle = service.get_asset(symbol, force_refresh=force_refresh)
    if bundle is None:
        return None
    from data.market_intelligence import bundle_to_dict
    return bundle_to_dict(bundle)


def refresh_futures_intelligence() -> Dict:
    """Force-refresh all futures intelligence data."""
    service = _get_intelligence_service()
    bundles = service.refresh(force=True)
    return {"refreshed": len(bundles), "assets": list(bundles.keys())}
