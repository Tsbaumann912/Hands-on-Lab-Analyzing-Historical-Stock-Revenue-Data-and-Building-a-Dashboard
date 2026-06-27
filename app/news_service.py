"""
Futures Market News, Fundamentals, and Sentiment Service.

Fetches live and historical data for futures assets:
  - News headlines (yfinance + RSS feeds)
  - Fundamental data (price history, supply/demand metrics, macro context)
  - Sentiment scores via VADER on headline text

Data is cached in-process with configurable TTL to reduce API pressure.
Historical price/fundamental data is fetched from 2000 to present where available.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Optional dependency guards ────────────────────────────────────────────────

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    logger.warning("yfinance not available; news/fundamentals will use fallback data.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    _VADER = None
    HAS_VADER = False
    logger.warning("vaderSentiment not available; sentiment scoring disabled.")

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    logger.warning("feedparser not available; RSS news feeds disabled.")


# ── Futures asset registry ────────────────────────────────────────────────────

FUTURES_ASSETS: dict[str, dict[str, Any]] = {
    "GC": {
        "name": "Gold",
        "full_name": "Gold Futures (COMEX)",
        "yf_symbol": "GC=F",
        "etf_proxy": "GLD",
        "search_query": "gold+futures+price",
        "sector": "Precious Metals",
        "color": "#ff9f0a",
        "icon": "Au",
        "description": (
            "Gold is traded on the COMEX division of the CME Group. "
            "It is a primary safe-haven asset and inflation hedge, "
            "with prices influenced by USD strength, real interest rates, "
            "central bank reserves, and geopolitical risk."
        ),
        "rss_feeds": [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US",
            "https://www.kitco.com/rss/kitconews.rss",
        ],
        "key_drivers": ["USD Index", "Real Rates", "Central Bank Demand", "Inflation", "Geopolitical Risk"],
        "contract_size": 100,  # troy ounces
        "unit": "$/troy oz",
        "tick_size": 0.10,
        "multiplier": 100,
    },
    "CL": {
        "name": "Crude Oil",
        "full_name": "WTI Crude Oil Futures (NYMEX)",
        "yf_symbol": "CL=F",
        "etf_proxy": "USO",
        "search_query": "WTI+crude+oil+futures",
        "sector": "Energy",
        "color": "#ff3b30",
        "icon": "OIL",
        "description": (
            "WTI Crude Oil is the U.S. benchmark for petroleum pricing, "
            "traded on the NYMEX. Prices are driven by OPEC+ output decisions, "
            "U.S. inventory levels (EIA reports), global demand, and geopolitical "
            "supply disruptions."
        ),
        "rss_feeds": [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CL=F&region=US&lang=en-US",
        ],
        "key_drivers": ["OPEC+ Policy", "EIA Inventories", "Global GDP", "USD Strength", "Refinery Demand"],
        "contract_size": 1000,  # barrels
        "unit": "$/barrel",
        "tick_size": 0.01,
        "multiplier": 1000,
    },
    "ES": {
        "name": "S&P 500",
        "full_name": "E-mini S&P 500 Futures (CME)",
        "yf_symbol": "ES=F",
        "etf_proxy": "SPY",
        "search_query": "S%26P+500+futures+ES",
        "sector": "Equity Index",
        "color": "#0071e3",
        "icon": "ES",
        "description": (
            "E-mini S&P 500 futures represent 1/5 the size of the full S&P 500 "
            "contract and are the most liquid equity index futures in the world. "
            "Price is driven by corporate earnings, Fed policy, economic data, "
            "and macro sentiment."
        ),
        "rss_feeds": [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ES=F&region=US&lang=en-US",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
        ],
        "key_drivers": ["Fed Policy", "Earnings Season", "GDP Growth", "Employment", "Credit Spreads"],
        "contract_size": 50,  # × index value
        "unit": "index points",
        "tick_size": 0.25,
        "multiplier": 50,
    },
    "NQ": {
        "name": "Nasdaq-100",
        "full_name": "E-mini Nasdaq-100 Futures (CME)",
        "yf_symbol": "NQ=F",
        "etf_proxy": "QQQ",
        "search_query": "Nasdaq+100+futures+NQ",
        "sector": "Equity Index",
        "color": "#34c759",
        "icon": "NQ",
        "description": (
            "E-mini Nasdaq-100 futures track the 100 largest non-financial "
            "companies on the Nasdaq. Heavy weighting toward mega-cap tech "
            "means prices are sensitive to interest rate expectations, AI/tech "
            "earnings cycles, and growth-vs-value rotations."
        ),
        "rss_feeds": [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NQ=F&region=US&lang=en-US",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=QQQ&region=US&lang=en-US",
        ],
        "key_drivers": ["Tech Earnings", "Real Yields", "AI Sentiment", "Fed Dots", "Growth Outlook"],
        "contract_size": 20,
        "unit": "index points",
        "tick_size": 0.25,
        "multiplier": 20,
    },
    "ZN": {
        "name": "10-Yr Treasury",
        "full_name": "10-Year T-Note Futures (CBOT)",
        "yf_symbol": "ZN=F",
        "etf_proxy": "IEF",
        "search_query": "10-year+treasury+note+futures",
        "sector": "Fixed Income",
        "color": "#af52de",
        "icon": "ZN",
        "description": (
            "10-Year Treasury Note futures are the benchmark for U.S. "
            "government bond pricing and the global risk-free rate. "
            "Prices move inversely with yields, driven by Fed policy, "
            "inflation expectations, fiscal supply, and flight-to-safety flows."
        ),
        "rss_feeds": [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ZN=F&region=US&lang=en-US",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=IEF&region=US&lang=en-US",
        ],
        "key_drivers": ["Fed Funds Rate", "Inflation (CPI/PCE)", "Treasury Supply", "Safe-Haven Flows", "Break-Evens"],
        "contract_size": 100000,  # face value
        "unit": "% of par",
        "tick_size": 0.015625,
        "multiplier": 1000,
    },
    "SI": {
        "name": "Silver",
        "full_name": "Silver Futures (COMEX)",
        "yf_symbol": "SI=F",
        "etf_proxy": "SLV",
        "search_query": "silver+futures+COMEX",
        "sector": "Precious Metals",
        "color": "#86868b",
        "icon": "Ag",
        "description": (
            "Silver has both monetary and industrial uses, making it more "
            "volatile than gold. It tracks gold sentiment but is also "
            "influenced by solar panel demand, electronics manufacturing, "
            "and emerging-market industrial activity."
        ),
        "rss_feeds": [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SI=F&region=US&lang=en-US",
        ],
        "key_drivers": ["Gold Correlation", "Industrial Demand", "Solar Energy", "USD", "Inflation"],
        "contract_size": 5000,  # troy ounces
        "unit": "$/troy oz",
        "tick_size": 0.005,
        "multiplier": 5000,
    },
    "NG": {
        "name": "Natural Gas",
        "full_name": "Natural Gas Futures (NYMEX)",
        "yf_symbol": "NG=F",
        "etf_proxy": "UNG",
        "search_query": "natural+gas+futures+NYMEX",
        "sector": "Energy",
        "color": "#5ac8fa",
        "icon": "NG",
        "description": (
            "Henry Hub Natural Gas futures are the North American benchmark for "
            "natural gas pricing. Prices are highly seasonal (heating demand), "
            "driven by weather forecasts, LNG export capacity, storage levels "
            "(EIA weekly report), and Permian Basin associated gas output."
        ),
        "rss_feeds": [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NG=F&region=US&lang=en-US",
        ],
        "key_drivers": ["Weather/Heating Demand", "LNG Exports", "Storage Levels", "Permian Output", "Power Generation"],
        "contract_size": 10000,  # mmBtu
        "unit": "$/mmBtu",
        "tick_size": 0.001,
        "multiplier": 10000,
    },
    "HG": {
        "name": "Copper",
        "full_name": "Copper Futures (COMEX)",
        "yf_symbol": "HG=F",
        "etf_proxy": "CPER",
        "search_query": "copper+futures+COMEX",
        "sector": "Industrial Metals",
        "color": "#ff6b2b",
        "icon": "Cu",
        "description": (
            "Copper is often called 'Dr. Copper' due to its predictive power "
            "for global economic health. Demand is driven by China (construction, "
            "EVs), energy transition infrastructure, and global manufacturing "
            "activity. Supply is sensitive to Chilean/Peruvian mine output."
        ),
        "rss_feeds": [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=HG=F&region=US&lang=en-US",
        ],
        "key_drivers": ["China Demand", "EV Transition", "Mining Output", "Global PMIs", "USD"],
        "contract_size": 25000,  # pounds
        "unit": "$/lb",
        "tick_size": 0.0005,
        "multiplier": 25000,
    },
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    publisher: str
    link: str
    published_ts: float
    published_label: str
    summary: str
    sentiment_score: float
    sentiment_label: str
    sentiment_color: str
    source: str = "yfinance"


@dataclass
class FundamentalsSnapshot:
    asset_key: str
    asset_name: str
    current_price: float
    price_change_1d: float
    price_change_pct_1d: float
    price_change_1y: float
    price_change_pct_1y: float
    high_52w: float
    low_52w: float
    avg_volume: float
    market_cap: float
    description: str
    key_drivers: list[str]
    contract_info: dict[str, Any]
    etf_info: dict[str, Any]
    macro_context: list[dict[str, str]]
    last_updated: str


@dataclass
class SentimentPoint:
    date: str
    score: float
    label: str
    article_count: int


# ── TTL in-process cache ──────────────────────────────────────────────────────

_CACHE: dict[str, tuple[float, Any]] = {}
_NEWS_TTL = 600        # 10 min
_FUNDS_TTL = 1800      # 30 min
_HIST_TTL  = 3600      # 1 hour


def _cache_get(key: str) -> Any | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.time() - ts > _NEWS_TTL:
        return None
    return value


def _cache_set(key: str, value: Any, ttl: float = _NEWS_TTL) -> None:
    _CACHE[key] = (time.time(), value)


def _cache_key(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


# ── Sentiment scoring ─────────────────────────────────────────────────────────

def _score_sentiment(text: str) -> tuple[float, str, str]:
    """
    Score *text* with VADER. Returns (compound, label, css_color).
    Falls back to 0 / neutral if VADER unavailable.
    """
    if not HAS_VADER or not text:
        return 0.0, "Neutral", "#86868b"
    scores = _VADER.polarity_scores(text)
    compound = round(scores["compound"], 4)
    if compound >= 0.05:
        return compound, "Bullish", "#34c759"
    if compound <= -0.05:
        return compound, "Bearish", "#ff3b30"
    return compound, "Neutral", "#86868b"


# ── News fetching ─────────────────────────────────────────────────────────────

_EPOCH_MIN = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()


def _parse_ts(ts: Any) -> tuple[float, str]:
    """Convert a unix timestamp or struct_time to (float, label)."""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    dt = now
    try:
        if isinstance(ts, (int, float)) and float(ts) > _EPOCH_MIN:
            dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc)
        elif hasattr(ts, "tm_year"):  # struct_time from feedparser
            candidate = datetime.datetime(*ts[:6], tzinfo=datetime.timezone.utc)
            if candidate.timestamp() > _EPOCH_MIN:
                dt = candidate
    except Exception:
        pass
    label = dt.strftime("%b %d, %Y · %H:%M UTC")
    return dt.timestamp(), label


def _fetch_yf_news(asset_key: str) -> list[NewsItem]:
    """Fetch news from yfinance for the asset and its ETF proxy."""
    if not HAS_YF:
        return []
    meta = FUTURES_ASSETS[asset_key]
    items: list[NewsItem] = []
    for sym in (meta["yf_symbol"], meta["etf_proxy"]):
        try:
            t = yf.Ticker(sym)
            raw_news = t.news or []
            for n in raw_news:
                content = n.get("content", {})
                title = (
                    content.get("title")
                    or n.get("title", "")
                )
                summary_html = (
                    content.get("summary")
                    or content.get("description")
                    or n.get("summary", "")
                )
                # Strip basic HTML tags for display
                summary = summary_html.replace("<p>", "").replace("</p>", " ").strip()

                provider_info = content.get("provider", {})
                publisher = (
                    provider_info.get("displayName")
                    or n.get("publisher", "Unknown")
                )
                link = (
                    content.get("canonicalUrl", {}).get("url")
                    or n.get("link", "#")
                )
                ts_raw = n.get("providerPublishTime") or n.get("published", 0)
                pub_ts, pub_label = _parse_ts(ts_raw)
                score, label, color = _score_sentiment(title + " " + summary)
                items.append(NewsItem(
                    title=title,
                    publisher=publisher,
                    link=link,
                    published_ts=pub_ts,
                    published_label=pub_label,
                    summary=summary[:300] if summary else "",
                    sentiment_score=score,
                    sentiment_label=label,
                    sentiment_color=color,
                    source="yfinance",
                ))
        except Exception as exc:
            logger.warning("yfinance news fetch failed for %s: %s", sym, exc)
    return items


def _fetch_rss_news(asset_key: str) -> list[NewsItem]:
    """Fetch news from RSS feeds registered for the asset."""
    if not HAS_FEEDPARSER:
        return []
    meta = FUTURES_ASSETS[asset_key]
    items: list[NewsItem] = []
    for url in meta.get("rss_feeds", []):
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")[:300]
                link = entry.get("link", "#")
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                pub_ts, pub_label = _parse_ts(pub)
                publisher = feed.feed.get("title", "RSS")
                score, label, color = _score_sentiment(title + " " + summary)
                items.append(NewsItem(
                    title=title,
                    publisher=publisher,
                    link=link,
                    published_ts=pub_ts,
                    published_label=pub_label,
                    summary=summary,
                    sentiment_score=score,
                    sentiment_label=label,
                    sentiment_color=color,
                    source="rss",
                ))
        except Exception as exc:
            logger.warning("RSS fetch failed for %s (%s): %s", asset_key, url, exc)
    return items


def fetch_asset_news(asset_key: str, limit: int = 50) -> list[NewsItem]:
    """
    Return merged, deduplicated, time-sorted news items for *asset_key*.
    Combines yfinance and RSS sources. Cached for 10 minutes.
    """
    cache_key = _cache_key("news", asset_key, str(limit))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    items = _fetch_yf_news(asset_key) + _fetch_rss_news(asset_key)
    # Deduplicate by title
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in items:
        slug = item.title.lower().strip()[:60]
        if slug and slug not in seen:
            seen.add(slug)
            unique.append(item)

    unique.sort(key=lambda x: x.published_ts, reverse=True)
    result = unique[:limit]
    _cache_set(cache_key, result, ttl=_NEWS_TTL)
    return result


# ── Fundamentals fetching ─────────────────────────────────────────────────────

def _fetch_etf_info(etf_sym: str) -> dict[str, Any]:
    """Pull key stats from the ETF proxy via yfinance."""
    if not HAS_YF:
        return {}
    try:
        t = yf.Ticker(etf_sym)
        info = t.info or {}
        return {
            "etf_symbol":   etf_sym,
            "etf_name":     info.get("longName", etf_sym),
            "aum":          info.get("totalAssets"),
            "expense_ratio": info.get("annualReportExpenseRatio"),
            "ytd_return":   info.get("ytdReturn"),
            "beta_3y":      info.get("beta3Year"),
            "nav":          info.get("navPrice"),
        }
    except Exception as exc:
        logger.warning("ETF info fetch failed for %s: %s", etf_sym, exc)
        return {}


def _build_macro_context(asset_key: str) -> list[dict[str, str]]:
    """Build a list of macro context bullets for the asset."""
    contexts: dict[str, list[dict[str, str]]] = {
        "GC": [
            {"label": "Fed Funds Rate", "desc": "Inverse relationship — higher real rates pressure gold"},
            {"label": "DXY (USD Index)", "desc": "Gold priced in USD; a strong dollar typically suppresses gold"},
            {"label": "Inflation (CPI)", "desc": "Gold is a classic inflation hedge; rising CPI supports price"},
            {"label": "Central Bank Buying", "desc": "EM central banks (China, India, Turkey) are structural buyers"},
            {"label": "COT Positioning", "desc": "Monitor managed-money net longs in CFTC COT report"},
        ],
        "CL": [
            {"label": "OPEC+ Quota Decisions", "desc": "Monthly production adjustments are the dominant price lever"},
            {"label": "EIA Weekly Inventory", "desc": "Crude draws (bullish) / builds (bearish) published every Wednesday"},
            {"label": "Baker Hughes Rig Count", "desc": "U.S. active rigs proxy for domestic supply trajectory"},
            {"label": "China PMI", "desc": "China is the largest marginal consumer; PMI swings move price"},
            {"label": "Refinery Utilization", "desc": "High crack spreads signal strong refinery demand"},
        ],
        "ES": [
            {"label": "Fed Policy / FOMC", "desc": "Rate decisions and forward guidance dominate index direction"},
            {"label": "S&P 500 EPS Growth", "desc": "Quarterly earnings seasons drive multiple expansion/compression"},
            {"label": "VIX (Volatility Index)", "desc": "Fear gauge; spikes correlate with ES selloffs"},
            {"label": "Credit Spreads (HY)", "desc": "Widening HY spreads often precede equity weakness"},
            {"label": "10-Year Real Yield", "desc": "Rising real yields compress equity multiples (P/E)"},
        ],
        "NQ": [
            {"label": "Mega-Cap Earnings (FAANG+)", "desc": "Top 10 holdings represent ~50% of the index weighting"},
            {"label": "Real Rate Sensitivity", "desc": "Duration-heavy growth stocks price in future earnings"},
            {"label": "AI / Semiconductor Cycle", "desc": "AI capex waves have outsized impact on NQ constituents"},
            {"label": "Tech Regulation Risk", "desc": "Antitrust and data privacy rulings can reprice multiple"},
            {"label": "Growth vs Value Rotation", "desc": "Rising rates tend to push capital from growth to value"},
        ],
        "ZN": [
            {"label": "Fed Funds Target Rate", "desc": "The benchmark that anchors the entire yield curve"},
            {"label": "CPI / PCE Inflation", "desc": "Hot prints raise real yield expectations → price falls"},
            {"label": "Treasury Auction Demand", "desc": "Weak bid-to-cover ratios signal supply indigestion"},
            {"label": "Break-Even Inflation Rate", "desc": "10Y TIPS spread reflects market inflation expectations"},
            {"label": "Flight-to-Safety Flows", "desc": "Risk-off events (crises) drive bond rallies"},
        ],
        "SI": [
            {"label": "Gold/Silver Ratio", "desc": "Historical mean ~65×; extremes signal reversion opportunity"},
            {"label": "Solar Panel Demand", "desc": "Silver is a critical PV cell component; energy transition driven"},
            {"label": "Electronics Manufacturing", "desc": "Industrial silver demand tracks semiconductor output"},
            {"label": "CFTC COT — Managed Money", "desc": "Speculative positioning is amplified vs gold"},
            {"label": "Mine Supply (Mexico/Peru)", "desc": "Political/environmental risks to primary silver producers"},
        ],
        "NG": [
            {"label": "Weather Forecasts (HDD/CDD)", "desc": "Heating/cooling degree days are the dominant short-term driver"},
            {"label": "EIA Storage Report", "desc": "Weekly natural gas storage injection/draw vs seasonal norm"},
            {"label": "LNG Export Capacity", "desc": "Sabine Pass/Corpus Christi utilization rate tightens domestic supply"},
            {"label": "Associated Gas from Permian", "desc": "Oil production growth adds to associated gas supply"},
            {"label": "Power Burn Demand", "desc": "Gas-fired generation competing with renewables for dispatch"},
        ],
        "HG": [
            {"label": "China PMI (Manufacturing)", "desc": "China consumes ~55% of global refined copper output"},
            {"label": "EV & Grid Investment", "desc": "Energy transition is a multi-decade structural demand driver"},
            {"label": "LME / COMEX Inventories", "desc": "Low visible inventories tighten the nearby delivery market"},
            {"label": "Chilean/Peruvian Mine Output", "desc": "Top two producers; labour strikes = immediate supply shock"},
            {"label": "Scrap Copper Availability", "desc": "High prices incentivize scrap recycling, moderating tightness"},
        ],
    }
    return contexts.get(asset_key, [])


# In-process price history cache: key → (timestamp, DataFrame)
_PRICE_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_PRICE_TTL = 3600  # 1 hour


def _fetch_price_history_raw(yf_symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for *yf_symbol* from *start* to *end*."""
    if not HAS_YF:
        return pd.DataFrame()
    try:
        df = yf.download(yf_symbol, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # Flatten MultiIndex columns produced by yfinance ≥ 0.2.x
        flat_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                flat_cols.append("_".join(x for x in c if x).strip("_"))
            else:
                flat_cols.append(str(c))
        df.columns = flat_cols
        # Normalise to canonical names: Date, Open, High, Low, Close, Volume
        rename: dict[str, str] = {}
        for col in df.columns:
            low = col.lower()
            for target in ("date", "open", "high", "low", "close", "volume"):
                if target in low and target.capitalize() not in rename.values():
                    rename[col] = target.capitalize()
                    break
        df = df.rename(columns=rename)
        return df
    except Exception as exc:
        logger.warning("Price history fetch failed for %s: %s", yf_symbol, exc)
        return pd.DataFrame()


def fetch_price_history(asset_key: str, start: str = "2000-01-01") -> pd.DataFrame:
    """
    Return daily OHLCV DataFrame from *start* to today.
    Uses yfinance for futures symbol + ETF proxy as fallback.
    Results are cached for 1 hour in-process.
    """
    end = datetime.date.today().isoformat()
    meta = FUTURES_ASSETS[asset_key]

    for sym in (meta["yf_symbol"], meta["etf_proxy"]):
        cache_key = _cache_key("price", sym, start, end)
        entry = _PRICE_CACHE.get(cache_key)
        if entry is not None:
            ts, df_cached = entry
            if time.time() - ts < _PRICE_TTL and not df_cached.empty:
                return df_cached

        df = _fetch_price_history_raw(sym, start, end)
        if not df.empty:
            _PRICE_CACHE[cache_key] = (time.time(), df)
            return df

    return pd.DataFrame()


def fetch_asset_fundamentals(asset_key: str) -> FundamentalsSnapshot:
    """
    Build a FundamentalsSnapshot for *asset_key*.
    Pulls current price, 52-week range, volume, ETF info. Cached 30 min.
    """
    cache_key = _cache_key("fundamentals", asset_key)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    meta = FUTURES_ASSETS[asset_key]
    now_label = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Defaults
    current_price = 0.0
    pct_1d = 0.0
    chg_1d = 0.0
    pct_1y = 0.0
    chg_1y = 0.0
    high_52w = 0.0
    low_52w = 0.0
    avg_vol = 0.0
    mkt_cap = 0.0

    if HAS_YF:
        for sym in (meta["yf_symbol"], meta["etf_proxy"]):
            try:
                t = yf.Ticker(sym)
                info = t.info or {}
                hist_1y = t.history(period="1y")
                hist_5d = t.history(period="5d")

                if not hist_5d.empty and len(hist_5d) >= 2:
                    current_price = float(hist_5d["Close"].iloc[-1])
                    prev = float(hist_5d["Close"].iloc[-2])
                    chg_1d = current_price - prev
                    pct_1d = chg_1d / prev * 100 if prev else 0.0

                if not hist_1y.empty and len(hist_1y) >= 2:
                    base = float(hist_1y["Close"].iloc[0])
                    end_p = float(hist_1y["Close"].iloc[-1])
                    chg_1y = end_p - base
                    pct_1y = chg_1y / base * 100 if base else 0.0
                    high_52w = float(hist_1y["High"].max())
                    low_52w  = float(hist_1y["Low"].min())
                    avg_vol  = float(hist_1y["Volume"].mean())

                mkt_cap = info.get("marketCap") or info.get("totalAssets") or 0.0
                if current_price > 0:
                    break
            except Exception as exc:
                logger.warning("Fundamentals fetch failed for %s: %s", sym, exc)

    etf_info = _fetch_etf_info(meta["etf_proxy"])
    macro = _build_macro_context(asset_key)

    snapshot = FundamentalsSnapshot(
        asset_key=asset_key,
        asset_name=meta["full_name"],
        current_price=round(current_price, 4),
        price_change_1d=round(chg_1d, 4),
        price_change_pct_1d=round(pct_1d, 4),
        price_change_1y=round(chg_1y, 4),
        price_change_pct_1y=round(pct_1y, 4),
        high_52w=round(high_52w, 4),
        low_52w=round(low_52w, 4),
        avg_volume=round(avg_vol, 0),
        market_cap=mkt_cap,
        description=meta["description"],
        key_drivers=meta["key_drivers"],
        contract_info={
            "contract_size": meta["contract_size"],
            "unit": meta["unit"],
            "tick_size": meta["tick_size"],
            "multiplier": meta["multiplier"],
        },
        etf_info=etf_info,
        macro_context=macro,
        last_updated=now_label,
    )
    _cache_set(cache_key, snapshot, ttl=_FUNDS_TTL)
    return snapshot


# ── Sentiment history ─────────────────────────────────────────────────────────

def fetch_sentiment_history(
    asset_key: str,
    start: str = "2000-01-01",
) -> pd.DataFrame:
    """
    Build a monthly sentiment series from 2000 to today.

    For the recent period (last 90 days): uses live news headlines scored by VADER.
    For the historical period: derives a proxy sentiment from price momentum + vol
    (a common market-implied sentiment approach when headline archives are unavailable).

    Returns DataFrame with columns: Date, Sentiment, Label, Source.
    """
    cache_key = _cache_key("sentiment_hist", asset_key, start)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Build historical proxy sentiment from price returns
    df_price = fetch_price_history(asset_key, start=start)
    rows = []

    if not df_price.empty and "Close" in df_price.columns:
        date_col = "Date" if "Date" in df_price.columns else df_price.columns[0]
        df_price[date_col] = pd.to_datetime(df_price[date_col])
        df_price = df_price.set_index(date_col).sort_index()

        # Monthly resampled OHLCV
        monthly = df_price["Close"].resample("ME").last().dropna()
        if len(monthly) >= 2:
            monthly_returns = monthly.pct_change().dropna()
            # Normalize to [-1, 1] range using rolling z-score
            roll_mean = monthly_returns.rolling(12, min_periods=3).mean()
            roll_std  = monthly_returns.rolling(12, min_periods=3).std().replace(0, np.nan)
            z_scores  = ((monthly_returns - roll_mean) / roll_std).clip(-3, 3)
            norm_scores = (z_scores / 3).fillna(0)  # scale to [-1, 1]

            for dt, score in norm_scores.items():
                s = float(score)
                if s >= 0.1:
                    label = "Bullish"
                elif s <= -0.1:
                    label = "Bearish"
                else:
                    label = "Neutral"
                rows.append({
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Sentiment": round(s, 4),
                    "Label": label,
                    "Source": "price-implied",
                })

    # Overlay recent VADER-scored news sentiment
    recent_news = fetch_asset_news(asset_key, limit=200)
    if recent_news:
        news_df = pd.DataFrame([{
            "Date": datetime.datetime.fromtimestamp(n.published_ts),
            "Sentiment": n.sentiment_score,
        } for n in recent_news])
        news_df["Date"] = pd.to_datetime(news_df["Date"])
        news_df = news_df.set_index("Date").sort_index()
        monthly_news = news_df["Sentiment"].resample("ME").mean().dropna()
        for dt, score in monthly_news.items():
            s = float(score)
            label = "Bullish" if s >= 0.05 else ("Bearish" if s <= -0.05 else "Neutral")
            rows.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Sentiment": round(s, 4),
                "Label": label,
                "Source": "news-vader",
            })

    if not rows:
        # Synthetic fallback — never fail the UI
        seed = sum(ord(c) for c in asset_key)
        rng = np.random.default_rng(seed)
        dates = pd.date_range(start, datetime.date.today(), freq="ME")
        scores = rng.normal(0.0, 0.3, len(dates)).clip(-1, 1)
        for dt, s in zip(dates, scores):
            rows.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Sentiment": round(float(s), 4),
                "Label": "Bullish" if s >= 0.1 else ("Bearish" if s <= -0.1 else "Neutral"),
                "Source": "synthetic",
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["Date"] = pd.to_datetime(result["Date"])
        # Filter to requested range
        result = result[result["Date"] >= pd.to_datetime(start)]
        result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
        result = result.drop_duplicates(subset=["Date"]).sort_values("Date")
    _cache_set(cache_key, result, ttl=_HIST_TTL)
    return result


# ── Cache invalidation helper ─────────────────────────────────────────────────

def invalidate_cache(asset_key: str | None = None) -> int:
    """
    Clear cached entries for *asset_key* (or all entries if None).
    Returns count of entries cleared.
    """
    cleared = 0
    if asset_key is None:
        cleared = len(_CACHE) + len(_PRICE_CACHE)
        _CACHE.clear()
        _PRICE_CACHE.clear()
        return cleared

    prefix_keys = [
        _cache_key("news", asset_key, "50"),
        _cache_key("fundamentals", asset_key),
        _cache_key("sentiment_hist", asset_key, "2000-01-01"),
    ]
    for k in prefix_keys:
        if k in _CACHE:
            del _CACHE[k]
            cleared += 1
    # Clear price cache entries for this asset
    meta = FUTURES_ASSETS.get(asset_key, {})
    for sym in (meta.get("yf_symbol", ""), meta.get("etf_proxy", "")):
        to_del = [k for k in _PRICE_CACHE if sym in k]
        for k in to_del:
            del _PRICE_CACHE[k]
            cleared += 1
    return cleared


# ── JSON serialisation helpers ────────────────────────────────────────────────

def news_items_to_dicts(items: list[NewsItem]) -> list[dict[str, Any]]:
    """Convert NewsItem list to JSON-safe dicts."""
    return [
        {
            "title":           i.title,
            "publisher":       i.publisher,
            "link":            i.link,
            "published_ts":    i.published_ts,
            "published_label": i.published_label,
            "summary":         i.summary,
            "sentiment_score": i.sentiment_score,
            "sentiment_label": i.sentiment_label,
            "sentiment_color": i.sentiment_color,
            "source":          i.source,
        }
        for i in items
    ]


def fundamentals_to_dict(snap: FundamentalsSnapshot) -> dict[str, Any]:
    """Convert FundamentalsSnapshot to JSON-safe dict."""
    return {
        "asset_key":             snap.asset_key,
        "asset_name":            snap.asset_name,
        "current_price":         snap.current_price,
        "price_change_1d":       snap.price_change_1d,
        "price_change_pct_1d":   snap.price_change_pct_1d,
        "price_change_1y":       snap.price_change_1y,
        "price_change_pct_1y":   snap.price_change_pct_1y,
        "high_52w":              snap.high_52w,
        "low_52w":               snap.low_52w,
        "avg_volume":            snap.avg_volume,
        "market_cap":            snap.market_cap,
        "description":           snap.description,
        "key_drivers":           snap.key_drivers,
        "contract_info":         snap.contract_info,
        "etf_info":              snap.etf_info,
        "macro_context":         snap.macro_context,
        "last_updated":          snap.last_updated,
    }
