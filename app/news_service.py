"""
Futures Market Intelligence service — the data-retrieval API behind the
"Overview" tab.

For every tradable contract across the futures market (equity index, metals,
energy, agriculture, rates, FX, crypto) this module retrieves, categorised by
asset name:

* **News headlines** — live Yahoo Finance news for the underlying contract.
* **Fundamental data** — quantitative fundamentals derived from the full price
  history from ``config.news.history_start`` (default 2000) to the current day.
* **Sentiment data** — a lexicon-based compound sentiment score computed over
  the live headlines and aggregated per asset.

Results are cached with a configurable TTL so the terminal can transparently
*update the data points upon loading* (or on an explicit refresh) without
hammering the upstream provider on every callback.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import Config

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:  # pragma: no cover - yfinance is a hard runtime dep here
    HAS_YF = False

_CONFIG = Config.from_yaml()


# ── Asset registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FuturesAsset:
    """Static reference / contract metadata for a single futures market."""

    key: str            # url-safe identifier, e.g. "gold"
    symbol: str         # CME-style root, e.g. "GC"
    name: str           # display name, e.g. "Gold"
    category: str       # grouping, e.g. "Metals"
    yf_ticker: str      # yfinance continuous-contract ticker, e.g. "GC=F"
    exchange: str       # listing venue
    unit: str           # quoted unit / contract size description

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# The investable futures universe, categorised by asset name.
_ASSET_LIST: List[FuturesAsset] = [
    # Equity Index
    FuturesAsset("sp500", "ES", "E-mini S&P 500", "Equity Index", "ES=F", "CME", "$50 × index"),
    FuturesAsset("nasdaq", "NQ", "E-mini Nasdaq-100", "Equity Index", "NQ=F", "CME", "$20 × index"),
    FuturesAsset("dow", "YM", "E-mini Dow", "Equity Index", "YM=F", "CBOT", "$5 × index"),
    FuturesAsset("russell", "RTY", "E-mini Russell 2000", "Equity Index", "RTY=F", "CME", "$50 × index"),
    # Metals
    FuturesAsset("gold", "GC", "Gold", "Metals", "GC=F", "COMEX", "100 troy oz"),
    FuturesAsset("silver", "SI", "Silver", "Metals", "SI=F", "COMEX", "5,000 troy oz"),
    FuturesAsset("copper", "HG", "Copper", "Metals", "HG=F", "COMEX", "25,000 lbs"),
    FuturesAsset("platinum", "PL", "Platinum", "Metals", "PL=F", "NYMEX", "50 troy oz"),
    # Energy
    FuturesAsset("crude", "CL", "WTI Crude Oil", "Energy", "CL=F", "NYMEX", "1,000 barrels"),
    FuturesAsset("brent", "BZ", "Brent Crude Oil", "Energy", "BZ=F", "NYMEX", "1,000 barrels"),
    FuturesAsset("natgas", "NG", "Natural Gas", "Energy", "NG=F", "NYMEX", "10,000 MMBtu"),
    FuturesAsset("gasoline", "RB", "RBOB Gasoline", "Energy", "RB=F", "NYMEX", "42,000 gallons"),
    FuturesAsset("heatoil", "HO", "Heating Oil", "Energy", "HO=F", "NYMEX", "42,000 gallons"),
    # Agriculture
    FuturesAsset("corn", "ZC", "Corn", "Agriculture", "ZC=F", "CBOT", "5,000 bushels"),
    FuturesAsset("wheat", "ZW", "Wheat", "Agriculture", "ZW=F", "CBOT", "5,000 bushels"),
    FuturesAsset("soybeans", "ZS", "Soybeans", "Agriculture", "ZS=F", "CBOT", "5,000 bushels"),
    FuturesAsset("coffee", "KC", "Coffee", "Agriculture", "KC=F", "ICE", "37,500 lbs"),
    FuturesAsset("sugar", "SB", "Sugar No.11", "Agriculture", "SB=F", "ICE", "112,000 lbs"),
    FuturesAsset("cotton", "CT", "Cotton", "Agriculture", "CT=F", "ICE", "50,000 lbs"),
    # Rates
    FuturesAsset("tnote10", "ZN", "10-Year T-Note", "Rates", "ZN=F", "CBOT", "$100,000 face"),
    FuturesAsset("tbond30", "ZB", "30-Year T-Bond", "Rates", "ZB=F", "CBOT", "$100,000 face"),
    FuturesAsset("tnote5", "ZF", "5-Year T-Note", "Rates", "ZF=F", "CBOT", "$100,000 face"),
    # Currencies
    FuturesAsset("euro", "6E", "Euro FX", "Currencies", "6E=F", "CME", "€125,000"),
    FuturesAsset("yen", "6J", "Japanese Yen", "Currencies", "6J=F", "CME", "¥12,500,000"),
    FuturesAsset("pound", "6B", "British Pound", "Currencies", "6B=F", "CME", "£62,500"),
    # Crypto
    FuturesAsset("bitcoin", "BTC", "Bitcoin", "Crypto", "BTC=F", "CME", "5 BTC"),
]

FUTURES_ASSETS: Dict[str, FuturesAsset] = {a.key: a for a in _ASSET_LIST}

# Preserve the declaration order of categories for stable UI grouping.
CATEGORY_ORDER: List[str] = list(dict.fromkeys(a.category for a in _ASSET_LIST))


def list_assets() -> List[Dict[str, str]]:
    """Return metadata for every futures asset in the universe."""
    return [a.to_dict() for a in _ASSET_LIST]


def assets_by_category() -> Dict[str, List[FuturesAsset]]:
    """Group the asset universe by category in declaration order."""
    grouped: Dict[str, List[FuturesAsset]] = {cat: [] for cat in CATEGORY_ORDER}
    for asset in _ASSET_LIST:
        grouped[asset.category].append(asset)
    return grouped


def get_asset(key: str) -> Optional[FuturesAsset]:
    return FUTURES_ASSETS.get(key)


# ── Sentiment engine (lexicon-based, dependency-free) ─────────────────────────

# Finance/commodity polarity lexicon. Weight magnitude encodes intensity.
_POSITIVE_LEXICON: Dict[str, float] = {
    "rise": 1.0, "rises": 1.0, "rising": 1.0, "rose": 1.0, "gain": 1.0,
    "gains": 1.0, "gained": 1.0, "surge": 2.0, "surges": 2.0, "surged": 2.0,
    "soar": 2.0, "soared": 2.0, "soaring": 2.0, "rally": 1.5, "rallies": 1.5,
    "rallied": 1.5, "jump": 1.5, "jumps": 1.5, "jumped": 1.5, "climb": 1.0,
    "climbs": 1.0, "climbed": 1.0, "higher": 1.0, "boom": 1.5, "bullish": 2.0,
    "strong": 1.0, "strength": 1.0, "beat": 1.0, "beats": 1.0, "outperform": 1.5,
    "record": 1.0, "support": 0.5, "demand": 0.5, "growth": 1.0, "grow": 1.0,
    "optimism": 1.5, "optimistic": 1.5, "recover": 1.0, "recovery": 1.0,
    "rebound": 1.5, "rebounds": 1.5, "rebounded": 1.5, "upgrade": 1.5,
    "upgraded": 1.5, "profit": 1.0, "profits": 1.0, "boost": 1.0, "boosted": 1.0,
    "advance": 1.0, "advances": 1.0, "win": 1.0, "wins": 1.0, "positive": 1.0,
    "tops": 1.0, "accelerate": 1.0, "upbeat": 1.5,
}

_NEGATIVE_LEXICON: Dict[str, float] = {
    "fall": 1.0, "falls": 1.0, "falling": 1.0, "fell": 1.0, "drop": 1.0,
    "drops": 1.0, "dropped": 1.0, "plunge": 2.0, "plunges": 2.0, "plunged": 2.0,
    "slump": 1.5, "slumps": 1.5, "slumped": 1.5, "tumble": 1.5, "tumbles": 1.5,
    "tumbled": 1.5, "decline": 1.0, "declines": 1.0, "declined": 1.0,
    "sink": 1.5, "sinks": 1.5, "sank": 1.5, "lower": 1.0, "weak": 1.0,
    "weakness": 1.0, "bearish": 2.0, "loss": 1.0, "losses": 1.0, "miss": 1.0,
    "misses": 1.0, "missed": 1.0, "underperform": 1.5, "fear": 1.5, "fears": 1.5,
    "recession": 2.0, "slowdown": 1.5, "downgrade": 1.5, "downgraded": 1.5,
    "selloff": 2.0, "crash": 2.0, "crashed": 2.0, "slip": 1.0, "slips": 1.0,
    "slipped": 1.0, "pressure": 0.5, "concern": 1.0, "concerns": 1.0,
    "worry": 1.0, "worries": 1.0, "risk": 0.5, "risks": 0.5, "warn": 1.0,
    "warns": 1.0, "warning": 1.0, "negative": 1.0, "glut": 1.5,
    "oversupply": 1.5, "default": 1.5, "crisis": 2.0, "slide": 1.0,
    "slides": 1.0, "slid": 1.0, "cut": 0.5, "cuts": 0.5, "sell": 0.5,
}

_NEGATION_WORDS = frozenset({"not", "no", "never", "without", "fails", "fail", "failed", "lack"})

# VADER-style normalisation constant: compound = score / sqrt(score^2 + alpha).
_NORM_ALPHA = 15.0
_TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")


def _normalise(raw: float) -> float:
    """Squash an unbounded polarity sum into the [-1, 1] interval."""
    return float(raw / math.sqrt(raw * raw + _NORM_ALPHA))


def score_text(text: str) -> float:
    """Return a compound sentiment score in [-1, 1] for an arbitrary string."""
    if not text:
        return 0.0
    tokens = _TOKEN_RE.findall(text.lower())
    raw = 0.0
    for idx, tok in enumerate(tokens):
        weight = _POSITIVE_LEXICON.get(tok, 0.0) - _NEGATIVE_LEXICON.get(tok, 0.0)
        if weight == 0.0:
            continue
        window = tokens[max(0, idx - 3):idx]
        if any(w in _NEGATION_WORDS for w in window):
            weight = -weight
        raw += weight
    return _normalise(raw)


def _label_for(compound: float) -> str:
    if compound >= _CONFIG.news.sentiment_positive_threshold:
        return "Bullish"
    if compound <= _CONFIG.news.sentiment_negative_threshold:
        return "Bearish"
    return "Neutral"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    publisher: str
    link: str
    published: str          # ISO-8601 UTC
    summary: str
    sentiment_score: float
    sentiment_label: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SentimentSummary:
    compound: float
    label: str
    bullish: int
    neutral: int
    bearish: int
    headline_count: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class YearStat:
    year: int
    avg_close: float
    year_end_close: float
    annual_return_pct: float
    volatility_pct: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Fundamentals:
    data_start: Optional[str]
    data_end: Optional[str]
    current_price: Optional[float]
    previous_close: Optional[float]
    day_change: Optional[float]
    day_change_pct: Optional[float]
    high_52w: Optional[float]
    low_52w: Optional[float]
    all_time_high: Optional[float]
    all_time_low: Optional[float]
    cagr_pct: Optional[float]
    annual_volatility_pct: Optional[float]
    max_drawdown_pct: Optional[float]
    return_ytd_pct: Optional[float]
    return_1y_pct: Optional[float]
    return_5y_pct: Optional[float]
    avg_annual_volume: Optional[float]
    yearly: List[YearStat] = field(default_factory=list)
    price_history: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


@dataclass
class AssetIntel:
    asset: Dict[str, str]
    fundamentals: Dict
    sentiment: Dict
    news: List[Dict]
    fetched_at: str

    def to_dict(self) -> Dict:
        return asdict(self)


# ── TTL cache ─────────────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    value: object
    expires_at: float


_CACHE: Dict[str, _CacheEntry] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(cache_key: str):
    with _CACHE_LOCK:
        entry = _CACHE.get(cache_key)
        if entry is not None and entry.expires_at > time.monotonic():
            return entry.value
    return None


def _cache_set(cache_key: str, value: object, ttl_seconds: int) -> None:
    with _CACHE_LOCK:
        _CACHE[cache_key] = _CacheEntry(value=value, expires_at=time.monotonic() + ttl_seconds)


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


# ── Live news retrieval ───────────────────────────────────────────────────────

def _parse_published(content: Dict) -> str:
    for field_name in ("pubDate", "displayTime"):
        raw = content.get(field_name)
        if raw:
            try:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                continue
    return ""


def _extract_news_items(raw_news: List[Dict], limit: int) -> List[NewsItem]:
    items: List[NewsItem] = []
    for entry in raw_news[:limit]:
        # yfinance >= 0.2.40 nests fields under "content"; older versions are flat.
        content = entry.get("content", entry) if isinstance(entry, dict) else {}
        title = (content.get("title") or "").strip()
        if not title:
            continue
        summary = (content.get("summary") or content.get("description") or "").strip()
        provider = content.get("provider") or {}
        publisher = (
            provider.get("displayName")
            if isinstance(provider, dict)
            else content.get("publisher", "")
        ) or "Unknown"
        link = ""
        for url_field in ("clickThroughUrl", "canonicalUrl"):
            url_obj = content.get(url_field)
            if isinstance(url_obj, dict) and url_obj.get("url"):
                link = url_obj["url"]
                break
        if not link:
            link = content.get("link", "")
        compound = score_text(f"{title}. {summary}")
        items.append(NewsItem(
            title=title,
            publisher=publisher,
            link=link,
            published=_parse_published(content),
            summary=summary,
            sentiment_score=round(compound, 4),
            sentiment_label=_label_for(compound),
        ))
    return items


def get_news(key: str, *, force: bool = False) -> List[NewsItem]:
    """Return live, sentiment-scored news headlines for *key*."""
    asset = get_asset(key)
    if asset is None:
        return []
    cache_key = f"news::{key}"
    if not force:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

    items: List[NewsItem] = []
    if HAS_YF:
        try:
            raw = yf.Ticker(asset.yf_ticker).news or []
            items = _extract_news_items(raw, _CONFIG.news.max_headlines)
        except Exception as exc:  # noqa: BLE001 - upstream provider is best-effort
            logger.warning("News fetch failed for %s (%s): %s", key, asset.yf_ticker, exc)
            items = []

    _cache_set(cache_key, items, _CONFIG.news.news_ttl_seconds)
    return items


def get_sentiment(key: str, *, force: bool = False) -> SentimentSummary:
    """Aggregate headline sentiment for *key* into a single summary."""
    news = get_news(key, force=force)
    if not news:
        return SentimentSummary(0.0, "Neutral", 0, 0, 0, 0)

    scores = np.array([n.sentiment_score for n in news], dtype=np.float64)
    labels = np.array([n.sentiment_label for n in news])
    compound = float(scores.mean())
    return SentimentSummary(
        compound=round(compound, 4),
        label=_label_for(compound),
        bullish=int(np.count_nonzero(labels == "Bullish")),
        neutral=int(np.count_nonzero(labels == "Neutral")),
        bearish=int(np.count_nonzero(labels == "Bearish")),
        headline_count=len(news),
    )


# ── Fundamental data retrieval (history since config.news.history_start) ──────

def _empty_fundamentals() -> Fundamentals:
    return Fundamentals(
        data_start=None, data_end=None, current_price=None, previous_close=None,
        day_change=None, day_change_pct=None, high_52w=None, low_52w=None,
        all_time_high=None, all_time_low=None, cagr_pct=None,
        annual_volatility_pct=None, max_drawdown_pct=None, return_ytd_pct=None,
        return_1y_pct=None, return_5y_pct=None, avg_annual_volume=None,
    )


def _fetch_history(ticker: str, start: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in df.columns]
    return df[keep].dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)


def _pct_return_over(close: pd.Series, dates: pd.Series, days: int) -> Optional[float]:
    """Return percentage change from ~``days`` ago to the latest close."""
    if close.empty:
        return None
    last_date = dates.iloc[-1]
    cutoff = last_date - pd.Timedelta(days=days)
    prior = close[dates <= cutoff]
    if prior.empty:
        return None
    base = float(prior.iloc[-1])
    if base == 0.0:
        return None
    return round((float(close.iloc[-1]) / base - 1.0) * 100.0, 2)


def _compute_fundamentals(df: pd.DataFrame) -> Fundamentals:
    f = _empty_fundamentals()
    if df.empty:
        return f

    close = df["Close"].astype(np.float64)
    dates = df["Date"]
    tdays = _CONFIG.news.trading_days_per_year

    f.data_start = dates.iloc[0].date().isoformat()
    f.data_end = dates.iloc[-1].date().isoformat()
    f.current_price = round(float(close.iloc[-1]), 4)
    if len(close) >= 2:
        f.previous_close = round(float(close.iloc[-2]), 4)
        f.day_change = round(f.current_price - f.previous_close, 4)
        if f.previous_close:
            f.day_change_pct = round(f.day_change / f.previous_close * 100.0, 2)

    last_year = dates >= (dates.iloc[-1] - pd.Timedelta(days=365))
    window_52w = close[last_year]
    if not window_52w.empty:
        f.high_52w = round(float(window_52w.max()), 4)
        f.low_52w = round(float(window_52w.min()), 4)

    f.all_time_high = round(float(close.max()), 4)
    f.all_time_low = round(float(close.min()), 4)

    # CAGR over the full sample.
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1e-9)
    start_px = float(close.iloc[0])
    if start_px > 0 and f.current_price and f.current_price > 0:
        f.cagr_pct = round(((f.current_price / start_px) ** (1.0 / years) - 1.0) * 100.0, 2)

    # Annualised volatility from daily log returns (vectorised).
    log_ret = np.diff(np.log(close.to_numpy()))
    if log_ret.size > 1:
        f.annual_volatility_pct = round(float(np.std(log_ret, ddof=1) * math.sqrt(tdays) * 100.0), 2)

    # Maximum drawdown over the full sample (vectorised). Guard against the
    # ratio formula breaking on the April-2020 negative WTI print: a long-only
    # price drawdown is floored at -100%.
    px = close.to_numpy()
    running_max = np.maximum.accumulate(px)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(running_max > 0, px / running_max - 1.0, 0.0)
    drawdown = np.clip(drawdown, -1.0, 0.0)
    f.max_drawdown_pct = round(float(drawdown.min()) * 100.0, 2)

    # Period returns.
    ytd_mask = dates.dt.year == dates.iloc[-1].year
    ytd_close = close[ytd_mask]
    if not ytd_close.empty and float(ytd_close.iloc[0]) != 0.0:
        f.return_ytd_pct = round((f.current_price / float(ytd_close.iloc[0]) - 1.0) * 100.0, 2)
    f.return_1y_pct = _pct_return_over(close, dates, 365)
    f.return_5y_pct = _pct_return_over(close, dates, 365 * 5)

    if "Volume" in df.columns:
        vol_by_year = df.assign(Year=dates.dt.year).groupby("Year")["Volume"].sum()
        if not vol_by_year.empty:
            f.avg_annual_volume = round(float(vol_by_year.mean()), 0)

    # Per-year summary table (vectorised groupby — no row iteration).
    grp = df.assign(Year=dates.dt.year).groupby("Year")["Close"]
    yearly_df = pd.DataFrame({
        "avg_close": grp.mean(),
        "year_end_close": grp.last(),
        "year_start_close": grp.first(),
    })
    yearly_df["annual_return_pct"] = (
        yearly_df["year_end_close"] / yearly_df["year_start_close"] - 1.0
    ) * 100.0
    vol_df = (
        df.assign(Year=dates.dt.year, LogRet=np.log(close).diff())
        .groupby("Year")["LogRet"]
        .std()
        * math.sqrt(tdays)
        * 100.0
    )
    yearly_df["volatility_pct"] = vol_df
    f.yearly = [
        YearStat(
            year=int(year),
            avg_close=round(float(row.avg_close), 2),
            year_end_close=round(float(row.year_end_close), 2),
            annual_return_pct=round(float(row.annual_return_pct), 2),
            volatility_pct=(round(float(row.volatility_pct), 2)
                            if not math.isnan(row.volatility_pct) else 0.0),
        )
        for year, row in yearly_df.iterrows()
    ]

    # Monthly-sampled price history for charting (keeps payload light).
    monthly = (
        df.set_index("Date")["Close"].resample("ME").last().dropna().round(2)
    )
    f.price_history = [
        {"date": idx.date().isoformat(), "close": float(val)}
        for idx, val in monthly.items()
    ]
    return f


def get_fundamentals(key: str, *, force: bool = False) -> Fundamentals:
    """Return fundamentals computed from history since ``config.news.history_start``."""
    asset = get_asset(key)
    if asset is None:
        return _empty_fundamentals()
    cache_key = f"fund::{key}"
    if not force:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

    fundamentals = _empty_fundamentals()
    if HAS_YF:
        try:
            df = _fetch_history(asset.yf_ticker, _CONFIG.news.history_start)
            fundamentals = _compute_fundamentals(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fundamentals fetch failed for %s (%s): %s", key, asset.yf_ticker, exc)
            fundamentals = _empty_fundamentals()

    _cache_set(cache_key, fundamentals, _CONFIG.news.fundamentals_ttl_seconds)
    return fundamentals


# ── Combined intelligence ─────────────────────────────────────────────────────

def get_asset_intel(key: str, *, force: bool = False) -> Optional[AssetIntel]:
    """Return news + fundamentals + sentiment for a single asset."""
    asset = get_asset(key)
    if asset is None:
        return None
    news = get_news(key, force=force)
    return AssetIntel(
        asset=asset.to_dict(),
        fundamentals=get_fundamentals(key, force=force).to_dict(),
        sentiment=get_sentiment(key, force=force).to_dict(),
        news=[n.to_dict() for n in news],
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
