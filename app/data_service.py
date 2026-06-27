"""
Centralised data service: wraps yfinance fetches + our quant terminal modules.

All methods are cached for the session to avoid redundant API calls.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

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
    Run a full backtest using the quant terminal engine and return a results dict.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from core.config import Config
    from core.enums import AssetClass
    from core.models import Bar
    from engine.backtest import BacktestEngine
    from strategies.mean_reversion import MeanReversionRSI
    from strategies.momentum import MomentumBreakout
    from strategies.trend_following import TrendFollowingMACD

    strategy_map = {
        "MeanReversionRSI":    MeanReversionRSI,
        "MomentumBreakout":    MomentumBreakout,
        "TrendFollowingMACD":  TrendFollowingMACD,
    }

    config = Config()
    config.portfolio.initial_cash = initial_cash
    config.indicators.rsi_period = rsi_period
    config.strategy.rsi_oversold = rsi_oversold
    config.strategy.rsi_overbought = rsi_overbought
    config.indicators.bb_period = bb_period

    df = get_synthetic_futures_bars(symbol, n=n_bars)
    sym = f"{symbol}.c.0"

    bars = [
        Bar(
            symbol=sym,
            timestamp=row["Date"].to_pydatetime().replace(tzinfo=timezone.utc),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
            asset_class=AssetClass.FUTURES,
        )
        for _, row in df.iterrows()
    ]

    cls = strategy_map.get(strategy_name, MeanReversionRSI)
    engine = BacktestEngine(config, cls)
    result = engine.run({sym: bars})

    # Build equity curve DataFrame — always use price bar dates for alignment
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

_PERIOD_CFG: Dict[str, Dict[str, str]] = {
    "1mo":  {"period": "1mo",  "interval": "1h"},
    "3mo":  {"period": "3mo",  "interval": "1d"},
    "6mo":  {"period": "6mo",  "interval": "1d"},
    "1y":   {"period": "1y",   "interval": "1d"},
    "2y":   {"period": "2y",   "interval": "1d"},
    "5y":   {"period": "5y",   "interval": "1wk"},
    "max":  {"period": "max",  "interval": "1wk"},
}


def fetch_futures_chart_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch OHLCV bars for a futures continuous contract via yfinance.

    symbol: yfinance futures symbol, e.g. "ES=F", "CL=F", "GC=F"
    period: one of "1mo" | "3mo" | "6mo" | "1y" | "2y" | "5y" | "max"

    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume.
    Falls back to synthetic data on any error so the chart never crashes.
    """
    if not HAS_YF:
        return _synthetic_ohlcv(symbol, 252)

    cfg = _PERIOD_CFG.get(period, _PERIOD_CFG["1y"])
    try:
        raw = yf.download(
            symbol,
            period=cfg["period"],
            interval=cfg["interval"],
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if raw is None or raw.empty:
            raise ValueError("empty result from yfinance")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        frame = raw.reset_index()
        date_col = "Datetime" if "Datetime" in frame.columns else "Date"
        frame = frame.rename(columns={date_col: "Date"})
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
        needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
        available = [c for c in needed if c in frame.columns]
        frame = frame[available].dropna(subset=["Date", "Close"]).reset_index(drop=True)
        if "Volume" not in frame.columns:
            frame["Volume"] = 0
        return frame

    except Exception:
        logger.warning("fetch_futures_chart_data failed for %s / %s; using synthetic.", symbol, period)
        return _synthetic_ohlcv(symbol, 252)


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
