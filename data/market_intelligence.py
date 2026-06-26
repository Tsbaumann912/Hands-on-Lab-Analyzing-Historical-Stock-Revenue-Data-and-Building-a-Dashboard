"""
Market intelligence data layer — news, fundamentals, and sentiment for futures assets.

Fetches live news via yfinance and builds historical fundamental / sentiment
time-series from price data back to the configured start date (default 2000).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import Config, MarketIntelligenceAssetConfig
from core.models import (
    AssetIntelligenceBundle,
    FundamentalDataPoint,
    NewsItem,
    SentimentDataPoint,
)

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

_BULLISH_KEYWORDS = frozenset({
    "surge", "rally", "gain", "rise", "bullish", "record", "strong",
    "boost", "jump", "soar", "upbeat", "optimism", "demand", "growth",
})
_BEARISH_KEYWORDS = frozenset({
    "fall", "drop", "decline", "bearish", "crash", "weak", "slump",
    "plunge", "fear", "recession", "cut", "loss", "sell", "concern",
})


class MarketIntelligenceService:
    """Fetches and caches news, fundamentals, and sentiment for futures assets."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config.from_yaml()
        self._mi = self._config.market_intelligence
        self._cache: Dict[str, AssetIntelligenceBundle] = {}
        self._cache_ts: float = 0.0

    @property
    def assets(self) -> List[MarketIntelligenceAssetConfig]:
        return self._mi.assets

    def get_asset_config(self, symbol: str) -> Optional[MarketIntelligenceAssetConfig]:
        sym = symbol.upper()
        for asset in self._mi.assets:
            if asset.symbol.upper() == sym:
                return asset
        return None

    def _is_cache_valid(self) -> bool:
        if not self._cache:
            return False
        elapsed = time.monotonic() - self._cache_ts
        return elapsed < self._mi.cache_ttl_seconds

    def refresh(self, force: bool = False) -> Dict[str, AssetIntelligenceBundle]:
        """Refresh intelligence data for all configured assets."""
        if not force and self._is_cache_valid():
            return self._cache

        updated: Dict[str, AssetIntelligenceBundle] = {}
        for asset in self._mi.assets:
            try:
                updated[asset.symbol] = self._build_bundle(asset)
            except Exception:
                logger.exception("Failed to build intelligence for %s", asset.symbol)
                updated[asset.symbol] = self._synthetic_bundle(asset)

        self._cache = updated
        self._cache_ts = time.monotonic()
        return self._cache

    def get_all(self, force_refresh: bool = False) -> Dict[str, AssetIntelligenceBundle]:
        """Return intelligence bundles for every configured asset."""
        if force_refresh or not self._is_cache_valid():
            return self.refresh(force=force_refresh)
        return self._cache

    def get_asset(self, symbol: str, force_refresh: bool = False) -> Optional[AssetIntelligenceBundle]:
        """Return intelligence bundle for a single asset."""
        all_data = self.get_all(force_refresh=force_refresh)
        return all_data.get(symbol.upper())

    def _build_bundle(self, asset: MarketIntelligenceAssetConfig) -> AssetIntelligenceBundle:
        price_df = self._fetch_price_history(asset.yfinance_ticker)
        last_price, day_chg_pct = self._latest_quote(price_df, asset)
        news = self._fetch_news(asset)
        fundamentals = self._build_fundamentals(asset, price_df)
        sentiment = self._build_sentiment(asset, price_df, news)

        return AssetIntelligenceBundle(
            symbol=asset.symbol,
            label=asset.label,
            last_price=last_price,
            day_change_pct=day_chg_pct,
            news=news,
            fundamentals=fundamentals,
            sentiment=sentiment,
            updated_at=datetime.now(tz=timezone.utc),
        )

    def _fetch_price_history(self, yf_ticker: str) -> pd.DataFrame:
        if not HAS_YF:
            return self._synthetic_price_df(yf_ticker)

        try:
            ticker = yf.Ticker(yf_ticker)
            df = ticker.history(start=self._mi.history_start, auto_adjust=True)
            if df is None or df.empty:
                logger.warning("No price history for %s; using synthetic.", yf_ticker)
                return self._synthetic_price_df(yf_ticker)
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            return df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            logger.warning("Price fetch failed for %s; using synthetic.", yf_ticker)
            return self._synthetic_price_df(yf_ticker)

    def _latest_quote(
        self,
        price_df: pd.DataFrame,
        asset: MarketIntelligenceAssetConfig,
    ) -> tuple[float, float]:
        if price_df.empty or len(price_df) < 2:
            return 0.0, 0.0
        last = float(price_df["Close"].iloc[-1])
        prev = float(price_df["Close"].iloc[-2])
        chg_pct = (last - prev) / prev * 100.0 if prev else 0.0
        return round(last, 2), round(chg_pct, 2)

    def _fetch_news(self, asset: MarketIntelligenceAssetConfig) -> List[NewsItem]:
        if not HAS_YF:
            return self._synthetic_news(asset)

        try:
            ticker = yf.Ticker(asset.yfinance_ticker)
            raw_news = getattr(ticker, "news", None) or []
            items: List[NewsItem] = []
            for entry in raw_news[: self._mi.news_limit]:
                headline = entry.get("title", "")
                if not headline:
                    continue
                pub_ts = entry.get("providerPublishTime", 0)
                published = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
                summary = entry.get("summary", "") or ""
                score = self._headline_sentiment(headline + " " + summary)
                items.append(NewsItem(
                    symbol=asset.symbol,
                    headline=headline,
                    source=entry.get("publisher", "Unknown"),
                    published_at=published,
                    url=entry.get("link", ""),
                    summary=summary[:500],
                    sentiment_score=score,
                ))
            if items:
                return items
        except Exception:
            logger.warning("News fetch failed for %s; using synthetic.", asset.symbol)

        return self._synthetic_news(asset)

    def _build_fundamentals(
        self,
        asset: MarketIntelligenceAssetConfig,
        price_df: pd.DataFrame,
    ) -> List[FundamentalDataPoint]:
        if price_df.empty:
            return []

        df = price_df.copy()
        df["month"] = df["Date"].dt.to_period("M")
        monthly = df.groupby("month", observed=True).agg(
            avg_price=("Close", "mean"),
            avg_volume=("Volume", "mean"),
            high=("High", "max"),
            low=("Low", "min"),
            volatility=("Close", lambda s: float(s.pct_change().std() * np.sqrt(252))),
        ).reset_index()

        monthly["month_start"] = monthly["month"].dt.to_timestamp()
        vol_ma = monthly["avg_volume"].rolling(6, min_periods=1).mean()
        oi_proxy = (monthly["avg_volume"] / vol_ma.replace(0, np.nan)).fillna(1.0)

        points: List[FundamentalDataPoint] = []
        for idx, row in monthly.iterrows():
            points.append(FundamentalDataPoint(
                symbol=asset.symbol,
                date=row["month_start"].to_pydatetime().replace(tzinfo=timezone.utc),
                avg_price=round(float(row["avg_price"]), 2),
                avg_volume=round(float(row["avg_volume"]), 0),
                volatility=round(float(row["volatility"]) if pd.notna(row["volatility"]) else 0.0, 4),
                high=round(float(row["high"]), 2),
                low=round(float(row["low"]), 2),
                open_interest_proxy=round(float(oi_proxy.iloc[idx]), 3),
            ))
        return points

    def _build_sentiment(
        self,
        asset: MarketIntelligenceAssetConfig,
        price_df: pd.DataFrame,
        news: List[NewsItem],
    ) -> List[SentimentDataPoint]:
        if price_df.empty:
            return []

        df = price_df.copy()
        df["month"] = df["Date"].dt.to_period("M")
        monthly_close = df.groupby("month", observed=True)["Close"].last()
        momentum = monthly_close.pct_change(3).clip(-0.3, 0.3) / 0.3

        news_by_month: Dict[str, List[NewsItem]] = {}
        for item in news:
            key = item.published_at.strftime("%Y-%m")
            news_by_month.setdefault(key, []).append(item)

        points: List[SentimentDataPoint] = []
        for period, mom in momentum.items():
            month_start = period.to_timestamp().replace(tzinfo=timezone.utc)
            month_key = period.strftime("%Y-%m")
            month_news = news_by_month.get(month_key, [])
            news_score = (
                float(np.mean([n.sentiment_score for n in month_news]))
                if month_news else 0.0
            )
            mom_val = float(mom) if pd.notna(mom) else 0.0
            combined = float(np.clip(0.6 * mom_val + 0.4 * news_score, -1.0, 1.0))
            points.append(SentimentDataPoint(
                symbol=asset.symbol,
                date=month_start,
                score=round(combined, 3),
                news_count=len(month_news),
                momentum_score=round(mom_val, 3),
            ))
        return points

    @staticmethod
    def _headline_sentiment(text: str) -> float:
        words = set(text.lower().split())
        bull = len(words & _BULLISH_KEYWORDS)
        bear = len(words & _BEARISH_KEYWORDS)
        total = bull + bear
        if total == 0:
            return 0.0
        return float(np.clip((bull - bear) / total, -1.0, 1.0))

    def _synthetic_bundle(self, asset: MarketIntelligenceAssetConfig) -> AssetIntelligenceBundle:
        price_df = self._synthetic_price_df(asset.yfinance_ticker)
        last_price, day_chg_pct = self._latest_quote(price_df, asset)
        news = self._synthetic_news(asset)
        fundamentals = self._build_fundamentals(asset, price_df)
        sentiment = self._build_sentiment(asset, price_df, news)
        return AssetIntelligenceBundle(
            symbol=asset.symbol,
            label=asset.label,
            last_price=last_price,
            day_change_pct=day_chg_pct,
            news=news,
            fundamentals=fundamentals,
            sentiment=sentiment,
            updated_at=datetime.now(tz=timezone.utc),
        )

    def _synthetic_price_df(self, yf_ticker: str) -> pd.DataFrame:
        seed = sum(ord(c) for c in yf_ticker)
        rng = np.random.default_rng(seed)
        start = pd.Timestamp(self._mi.history_start)
        end = pd.Timestamp.now()
        dates = pd.bdate_range(start, end)
        n = len(dates)
        base = 100.0 + seed % 500
        returns = rng.normal(0.0001, 0.015, n)
        closes = base * np.cumprod(1 + returns)
        return pd.DataFrame({
            "Date": dates,
            "Open": (closes * rng.uniform(0.998, 1.002, n)).round(2),
            "High": (closes * rng.uniform(1.001, 1.015, n)).round(2),
            "Low": (closes * rng.uniform(0.985, 0.999, n)).round(2),
            "Close": closes.round(2),
            "Volume": rng.integers(50_000, 500_000, n),
        })

    def _synthetic_news(self, asset: MarketIntelligenceAssetConfig) -> List[NewsItem]:
        seed = sum(ord(c) for c in asset.symbol)
        rng = np.random.default_rng(seed)
        templates = [
            f"{asset.label} futures edge higher on strong demand outlook",
            f"Traders watch {asset.label} as macro data shifts sentiment",
            f"{asset.label} contract volatility rises ahead of key report",
            f"Analysts debate {asset.label} price direction for next quarter",
            f"{asset.label} open interest climbs as institutions reposition",
        ]
        now = datetime.now(tz=timezone.utc)
        items: List[NewsItem] = []
        for i, headline in enumerate(templates[: self._mi.news_limit]):
            published = now.replace(
                hour=max(0, now.hour - i * 3),
                minute=rng.integers(0, 59),
            )
            items.append(NewsItem(
                symbol=asset.symbol,
                headline=headline,
                source="MarketWire",
                published_at=published,
                url=f"https://finance.yahoo.com/quote/{asset.yfinance_ticker}",
                summary=f"Synthetic headline for {asset.label} ({asset.symbol}).",
                sentiment_score=self._headline_sentiment(headline),
            ))
        return items


def bundle_to_dict(bundle: AssetIntelligenceBundle) -> Dict:
    """Serialise an AssetIntelligenceBundle to a JSON-safe dict."""

    def _dt(val: datetime) -> str:
        return val.isoformat()

    return {
        "symbol": bundle.symbol,
        "label": bundle.label,
        "last_price": bundle.last_price,
        "day_change_pct": bundle.day_change_pct,
        "updated_at": _dt(bundle.updated_at) if bundle.updated_at else None,
        "news": [
            {
                "headline": n.headline,
                "source": n.source,
                "published_at": _dt(n.published_at),
                "url": n.url,
                "summary": n.summary,
                "sentiment_score": n.sentiment_score,
            }
            for n in bundle.news
        ],
        "fundamentals": [
            {
                "date": _dt(f.date),
                "avg_price": f.avg_price,
                "avg_volume": f.avg_volume,
                "volatility": f.volatility,
                "high": f.high,
                "low": f.low,
                "open_interest_proxy": f.open_interest_proxy,
            }
            for f in bundle.fundamentals
        ],
        "sentiment": [
            {
                "date": _dt(s.date),
                "score": s.score,
                "news_count": s.news_count,
                "momentum_score": s.momentum_score,
            }
            for s in bundle.sentiment
        ],
    }
