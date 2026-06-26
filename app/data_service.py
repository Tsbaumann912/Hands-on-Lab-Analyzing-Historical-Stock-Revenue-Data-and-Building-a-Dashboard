"""
Centralised data service: wraps yfinance fetches + our quant terminal modules.

All methods are cached for the session to avoid redundant API calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


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
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from core.config import Config
    from core.enums import AssetClass
    from core.models import Bar
    from engine.backtest import BacktestEngine
    from engine.metrics import compute_metrics
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
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from indicators.momentum import rsi, macd
    from indicators.trend import sma, ema, supertrend
    from indicators.volatility import atr, bollinger_bands
    from indicators.volume import obv, vwap

    close  = df["Close"].to_numpy(dtype=np.float64)
    high   = df["High"].to_numpy(dtype=np.float64)
    low    = df["Low"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    result = {}

    r = rsi(close, 14)
    if r is not None: result["RSI_14"] = r

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
    if atr_v is not None: result["ATR_14"] = atr_v

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    ema20 = ema(close, 20)
    if sma20 is not None: result["SMA_20"] = sma20
    if sma50 is not None: result["SMA_50"] = sma50
    if ema20 is not None: result["EMA_20"] = ema20

    obv_v = obv(close, volume)
    if obv_v is not None: result["OBV"] = obv_v

    vwap_v = vwap(high, low, close, volume)
    if vwap_v is not None: result["VWAP"] = vwap_v

    return result


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
