"""Data loaders and feature construction for PL ensemble."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from platinum_ensemble.models import Bar, Config

logger = logging.getLogger(__name__)


def _ewma_vol(returns: np.ndarray, com_days: int) -> np.ndarray:
    """Annualised EWMA volatility with given center-of-mass in days."""
    r = pd.Series(np.asarray(returns, dtype=np.float64))
    var = r.pow(2).ewm(com=float(com_days), min_periods=max(com_days, 5)).mean()
    return np.sqrt(var.to_numpy(dtype=np.float64) * 252.0)


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-style ATR; leading values are NaN until warm-up."""
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    out[period] = float(np.mean(tr[1 : period + 1]))
    alpha = 1.0 / float(period)
    # Vectorised Wilder via recursive expansion is awkward; use cumulative filter
    # Equivalent: out[i] = alpha*tr[i] + (1-alpha)*out[i-1]
    tr_tail = tr[period + 1 :]
    if tr_tail.size:
        # Iterative Wilder — control flow over bars is warm-up sized; keep short path
        prev = out[period]
        for i, t in enumerate(tr_tail, start=period + 1):
            prev = (prev * (period - 1) + t) / period
            out[i] = prev
    return out


def make_synthetic_pl(
    n_days: int = 1500,
    seed: int = 42,
    start_price: float = 1000.0,
    gold_start: float = 1800.0,
) -> pd.DataFrame:
    """
    Synthetic PL panel with price, curve, inventory, gold, and USD.

    Regime switches create TSMOM opportunities; inventory drives carry;
    gold co-moves with PL plus mean-reverting relative-value noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days, freq="C")

    regime = np.ones(n_days)
    cuts = np.linspace(0, n_days, 8, dtype=int)
    for i in range(len(cuts) - 1):
        regime[cuts[i] : cuts[i + 1]] = 1.0 if (i % 2 == 0) else -1.0

    shocks = rng.normal(0.0, 0.012, size=n_days)
    log_rets = 0.0020 * regime + shocks
    close = start_price * np.exp(np.cumsum(log_rets))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.005, n_days))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.005, n_days))
    volume = rng.integers(2_000, 12_000, size=n_days).astype(float)

    # Gold: shared regime + idiosyncratic noise (cointegration-friendly)
    gold_shocks = rng.normal(0.0, 0.008, size=n_days)
    gold_rets = 0.0015 * regime + gold_shocks
    gold_close = gold_start * np.exp(np.cumsum(gold_rets))
    # Inject temporary PL–GC wedges that mean-revert
    wedge = rng.normal(0.0, 0.004, size=n_days)
    wedge = pd.Series(wedge).ewm(span=40).mean().to_numpy()
    close = close * np.exp(wedge)

    # Inventory AR(1); lower inventory → more backwardation
    phi = 0.98
    noise = rng.normal(0.0, 400.0, size=n_days)
    innov = 0.02 * 80_000.0 - 600.0 * regime + noise
    try:
        from scipy.signal import lfilter

        inv = lfilter([1.0], [1.0, -phi], innov, zi=np.array([80_000.0 - innov[0]]))[0]
    except Exception:
        inv = np.empty(n_days, dtype=np.float64)
        inv[0] = 80_000.0
        for i in range(1, n_days):
            inv[i] = phi * inv[i - 1] + innov[i]

    inv_z = (inv - np.mean(inv)) / (np.std(inv) + 1e-9)
    carry_spread = -0.012 * inv_z + rng.normal(0, 0.002, n_days)
    near = close.copy()
    next_px = near * (1.0 - carry_spread)

    usd = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n_days)))
    usd_ret_20 = pd.Series(usd).pct_change(20).to_numpy()
    gold_ret_20 = pd.Series(gold_close).pct_change(20).to_numpy()

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "near_price": near,
            "next_price": next_px,
            "inventory": inv,
            "gold_close": gold_close,
            "gold_ret_20d": gold_ret_20,
            "usd_ret_20d": usd_ret_20,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def load_yfinance_pl(
    ticker: str = "PL=F",
    gold_ticker: str = "GC=F",
    period: str = "5y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load PL continuous proxy from Yahoo; join gold and synthesise curve/inventory.

    Prefer ``start``/``end`` (e.g. ``2008-01-01``) for institutional WFO.
    If ``start`` is set, ``period`` is ignored.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for live PL download") from exc

    if start is not None:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    else:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no data returned for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]

    df = raw[["open", "high", "low", "close", "volume"]].dropna().copy()
    close = df["close"].to_numpy(dtype=np.float64)
    rets = pd.Series(close).pct_change().fillna(0.0).to_numpy()
    basis = pd.Series(rets).rolling(63).mean().fillna(0.0).to_numpy()
    df["near_price"] = close
    df["next_price"] = close * (1.0 - basis)

    z = (pd.Series(close) - pd.Series(close).rolling(60).mean()) / (
        pd.Series(close).rolling(60).std() + 1e-9
    )
    df["inventory"] = (80_000.0 - 8_000.0 * z.fillna(0.0)).to_numpy()

    if start is not None:
        gold = yf.download(gold_ticker, start=start, end=end, auto_adjust=True, progress=False)
        usd = yf.download("DX-Y.NYB", start=start, end=end, auto_adjust=True, progress=False)
    else:
        gold = yf.download(gold_ticker, period=period, auto_adjust=True, progress=False)
        usd = yf.download("DX-Y.NYB", period=period, auto_adjust=True, progress=False)

    if not gold.empty:
        g = gold["Close"] if "Close" in gold.columns else gold.iloc[:, 0]
        if isinstance(g, pd.DataFrame):
            g = g.iloc[:, 0]
        g = g.reindex(df.index).ffill()
        df["gold_close"] = g.to_numpy(dtype=np.float64)
        df["gold_ret_20d"] = g.pct_change(20)
    else:
        df["gold_close"] = close * 1.8
        df["gold_ret_20d"] = 0.0
    df["gold_ret_20d"] = df["gold_ret_20d"].fillna(0.0)

    if not usd.empty:
        u = usd["Close"] if "Close" in usd.columns else usd.iloc[:, 0]
        if isinstance(u, pd.DataFrame):
            u = u.iloc[:, 0]
        u = u.reindex(df.index).ffill()
        df["usd_ret_20d"] = u.pct_change(20)
    else:
        df["usd_ret_20d"] = 0.0
    df["usd_ret_20d"] = df["usd_ret_20d"].fillna(0.0)
    df.index.name = "timestamp"
    return df


def dataframe_to_bars(df: pd.DataFrame, symbol: str = "PL") -> list[Bar]:
    """Convert a feature DataFrame into a list of ``Bar`` objects."""
    bars: list[Bar] = []
    for ts, row in df.iterrows():
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                near_price=float(row["near_price"]) if "near_price" in row else float(row["close"]),
                next_price=float(row["next_price"]) if "next_price" in row else float(row["close"]),
                inventory=float(row["inventory"]) if "inventory" in row and pd.notna(row["inventory"]) else None,
                gold_close=float(row["gold_close"]) if "gold_close" in row and pd.notna(row["gold_close"]) else None,
                gold_ret_20d=(
                    float(row["gold_ret_20d"]) if "gold_ret_20d" in row and pd.notna(row["gold_ret_20d"]) else None
                ),
                usd_ret_20d=float(row["usd_ret_20d"]) if "usd_ret_20d" in row and pd.notna(row["usd_ret_20d"]) else None,
            )
        )
    return bars


def build_feature_matrix(bars: list[Bar], config: Config) -> dict[str, np.ndarray]:
    """Vectorised feature arrays aligned to ``bars``."""
    n = len(bars)
    close = np.array([b.close for b in bars], dtype=np.float64)
    high = np.array([b.high for b in bars], dtype=np.float64)
    low = np.array([b.low for b in bars], dtype=np.float64)
    near = np.array(
        [b.near_price if b.near_price is not None else b.close for b in bars],
        dtype=np.float64,
    )
    nxt = np.array(
        [b.next_price if b.next_price is not None else b.close for b in bars],
        dtype=np.float64,
    )
    inv = np.array(
        [b.inventory if b.inventory is not None else np.nan for b in bars],
        dtype=np.float64,
    )
    gold = np.array(
        [b.gold_close if b.gold_close is not None else np.nan for b in bars],
        dtype=np.float64,
    )
    gold_r = np.array(
        [b.gold_ret_20d if b.gold_ret_20d is not None else np.nan for b in bars],
        dtype=np.float64,
    )
    usd = np.array(
        [b.usd_ret_20d if b.usd_ret_20d is not None else np.nan for b in bars],
        dtype=np.float64,
    )
    rets = np.empty(n, dtype=np.float64)
    rets[0] = np.nan
    rets[1:] = close[1:] / close[:-1] - 1.0
    vol = _ewma_vol(np.nan_to_num(rets, nan=0.0), config.ensemble.ewma_vol_com_days)
    atr_vals = atr(high, low, close, config.ensemble.atr_period)

    carry = np.where(near != 0.0, (near - nxt) / near, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        basis = np.log(np.clip(near, 1e-12, None)) - np.log(np.clip(nxt, 1e-12, None))

    return {
        "close": close,
        "high": high,
        "low": low,
        "near": near,
        "next": nxt,
        "inventory": inv,
        "gold_close": gold,
        "gold_ret_20d": gold_r,
        "usd_ret_20d": usd,
        "returns": rets,
        "vol": vol,
        "atr": atr_vals,
        "carry": carry,
        "basis": basis,
    }
