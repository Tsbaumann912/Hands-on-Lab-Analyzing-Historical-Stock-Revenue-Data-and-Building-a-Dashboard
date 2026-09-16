"""Data loaders and feature construction for SI ensemble."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from silver_ensemble.models import Bar, Config

logger = logging.getLogger(__name__)


def _ewma_vol(returns: np.ndarray, com_days: int) -> np.ndarray:
    """Annualised EWMA volatility with given center-of-mass in days."""
    r = pd.Series(np.asarray(returns, dtype=np.float64))
    var = r.pow(2).ewm(com=float(com_days), min_periods=max(com_days, 5)).mean()
    out = np.sqrt(var.to_numpy(dtype=np.float64) * 252.0)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-style ATR; leading values are NaN until warm-up."""
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    seed = float(np.mean(tr[1 : period + 1]))
    alpha = 1.0 / float(period)
    try:
        from scipy.signal import lfilter

        rest = tr[period + 1 :]
        out[period] = seed
        if rest.size:
            filt, _ = lfilter(
                [alpha],
                [1.0, -(1.0 - alpha)],
                rest,
                zi=np.array([seed * (1.0 - alpha)]),
            )
            out[period + 1 :] = filt
    except Exception:
        out[period] = seed
        for i in range(period + 1, n):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def make_synthetic_si(
    n_days: int = 1500,
    seed: int = 42,
    start_price: float = 22.0,
) -> pd.DataFrame:
    """
    Build a synthetic SI-like panel with price, curve, inventory, gold, and macro.

    Designed so carry, basis changes, inventory, gold/silver ratio, and real-yield
    fades are identifiable for unit tests and offline demos (no network required).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days, freq="C")

    # Regime-switching drift to create TSMOM opportunities
    regime = np.ones(n_days)
    cuts = np.linspace(0, n_days, 8, dtype=int)
    for i in range(len(cuts) - 1):
        regime[cuts[i] : cuts[i + 1]] = 1.0 if (i % 2 == 0) else -1.0

    # Silver is typically more volatile than copper
    shocks = rng.normal(0.0, 0.015, size=n_days)
    log_rets = 0.0030 * regime + shocks
    close = start_price * np.exp(np.cumsum(log_rets))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.006, n_days))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.006, n_days))
    volume = rng.integers(8_000, 40_000, size=n_days).astype(float)

    # Inventory mean-reverting AR(1); lower inventory → more backwardation
    noise = rng.normal(0.0, 400.0, size=n_days)
    phi = 0.98
    innov = 0.02 * 80_000.0 - 600.0 * regime + noise
    inv = np.empty(n_days, dtype=np.float64)
    inv[0] = 80_000.0
    try:
        from scipy.signal import lfilter

        inv = lfilter([1.0], [1.0, -phi], innov, zi=np.array([80_000.0 - innov[0]]))[0]
    except Exception:
        for i in range(1, n_days):
            inv[i] = phi * inv[i - 1] + innov[i]

    inv_z = (inv - np.mean(inv)) / (np.std(inv) + 1e-9)
    # Contango when inventories high: next > near
    carry_spread = -0.012 * inv_z + rng.normal(0, 0.0025, n_days)
    near = close.copy()
    next_px = near * (1.0 - carry_spread)

    # Gold co-moves with silver but with lower vol + monetary premium in bull regimes
    gold_shocks = rng.normal(0.0, 0.008, size=n_days)
    gold_log = 0.0020 * regime + gold_shocks
    gold_close = 1800.0 * np.exp(np.cumsum(gold_log))

    # Real yield changes: rise in bear metal regimes
    real_yield_chg = -0.0008 * regime + rng.normal(0, 0.0005, n_days)

    usd = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n_days) - 0.0005 * regime))
    usd_ret_20 = pd.Series(usd).pct_change(20).to_numpy()

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
            "real_yield_chg": real_yield_chg,
            "usd_ret_20d": usd_ret_20,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def _yf_close_series(ticker: str, period: str, index: pd.DatetimeIndex) -> Optional[pd.Series]:
    """Download a Yahoo close series and align to ``index``; return None on failure."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for live SI download") from exc

    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as exc:
        logger.warning("yfinance download failed for %s: %s", ticker, exc)
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        col = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
    else:
        col = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 3]
    col = col.reindex(index).ffill()
    return col.astype(np.float64)


def load_yfinance_si(
    ticker: str = "SI=F",
    period: str = "5y",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load SI continuous proxy from Yahoo Finance; synthesise curve/inventory/macro proxies.

    Prefer ``start``/``end`` (e.g. ``start='2008-01-01'``) for institutional WFO;
    fall back to ``period`` when ``start`` is unset.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for live SI download") from exc

    try:
        if start is not None:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
        else:
            raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as exc:
        logger.error("SI download failed: %s", exc)
        raise RuntimeError(f"no data returned for {ticker}") from exc
    if raw.empty:
        raise RuntimeError(f"no data returned for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]

    df = raw[["open", "high", "low", "close", "volume"]].dropna().copy()
    close = df["close"].to_numpy(dtype=np.float64)
    # Proxy curve: roll estimate from rolling mean return — mild synthetic basis
    rets = pd.Series(close).pct_change().fillna(0.0).to_numpy()
    basis = pd.Series(rets).rolling(63).mean().fillna(0.0).to_numpy()
    df["near_price"] = close
    df["next_price"] = close * (1.0 - basis)
    # Inventory proxy: inverse of rolling z-score of price (demo only)
    z = (pd.Series(close) - pd.Series(close).rolling(60).mean()) / (
        pd.Series(close).rolling(60).std() + 1e-9
    )
    df["inventory"] = (80_000.0 - 8_000.0 * z.fillna(0.0)).to_numpy()

    # Align auxiliary series to the same calendar as SI
    period_for_aux = period if start is None else "max"
    gold = _yf_close_series("GC=F", period_for_aux, df.index)
    if gold is None and start is not None:
        gold = _yf_close_series_range("GC=F", start, end, df.index)
    if gold is not None:
        df["gold_close"] = gold.ffill().to_numpy()
    else:
        df["gold_close"] = close * 80.0  # rough GS ratio proxy

    tips = _yf_close_series("^TNX", period_for_aux, df.index)
    if tips is None and start is not None:
        tips = _yf_close_series_range("^TNX", start, end, df.index)
    if tips is not None:
        df["real_yield_chg"] = tips.diff().fillna(0.0).to_numpy()
    else:
        df["real_yield_chg"] = 0.0

    usd = _yf_close_series("DX-Y.NYB", period_for_aux, df.index)
    if usd is None and start is not None:
        usd = _yf_close_series_range("DX-Y.NYB", start, end, df.index)
    if usd is not None:
        df["usd_ret_20d"] = usd.pct_change(20).fillna(0.0).to_numpy()
    else:
        df["usd_ret_20d"] = 0.0

    df.index.name = "timestamp"
    return df


def _yf_close_series_range(
    ticker: str,
    start: str | None,
    end: str | None,
    index: pd.DatetimeIndex,
) -> Optional[pd.Series]:
    """Download a Yahoo close series by start/end and align to ``index``."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for live SI download") from exc

    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:
        logger.warning("yfinance download failed for %s: %s", ticker, exc)
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        col = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
    else:
        col = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 3]
    col = col.reindex(index).ffill()
    return col.astype(np.float64)


def dataframe_to_bars(df: pd.DataFrame, symbol: str = "SI") -> list[Bar]:
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
                inventory=(
                    float(row["inventory"])
                    if "inventory" in row and pd.notna(row["inventory"])
                    else None
                ),
                gold_close=(
                    float(row["gold_close"])
                    if "gold_close" in row and pd.notna(row["gold_close"])
                    else None
                ),
                real_yield_chg=(
                    float(row["real_yield_chg"])
                    if "real_yield_chg" in row and pd.notna(row["real_yield_chg"])
                    else None
                ),
                usd_ret_20d=(
                    float(row["usd_ret_20d"])
                    if "usd_ret_20d" in row and pd.notna(row["usd_ret_20d"])
                    else None
                ),
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
    ry = np.array(
        [b.real_yield_chg if b.real_yield_chg is not None else np.nan for b in bars],
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

    # Annualised front-next carry proxy (Koijen-style slope when ΔT≈1/12)
    with np.errstate(divide="ignore", invalid="ignore"):
        carry = np.where(nxt != 0.0, (near - nxt) / nxt, np.nan)
        basis = np.log(np.clip(near, 1e-12, None)) - np.log(np.clip(nxt, 1e-12, None))
        gs_ratio = np.log(np.clip(gold, 1e-12, None)) - np.log(np.clip(close, 1e-12, None))

    from silver_ensemble.indicators import sma, stochastic_rsi, wilder_rsi

    ens = config.ensemble
    ma_fast = sma(close, ens.ma_fast)
    ma_slow = sma(close, ens.ma_slow)
    rsi = wilder_rsi(close, ens.rsi_period)
    stoch_rsi = stochastic_rsi(
        close,
        rsi_period=ens.rsi_period,
        stoch_period=ens.stoch_rsi_period,
        smooth_k=ens.stoch_rsi_smooth,
    )

    return {
        "close": close,
        "high": high,
        "low": low,
        "near": near,
        "next": nxt,
        "inventory": inv,
        "gold_close": gold,
        "real_yield_chg": ry,
        "usd_ret_20d": usd,
        "returns": rets,
        "vol": vol,
        "atr": atr_vals,
        "carry": carry,
        "basis": basis,
        "gs_ratio": gs_ratio,
        "ma_fast": ma_fast,
        "ma_slow": ma_slow,
        "rsi": rsi,
        "stoch_rsi": stoch_rsi,
    }
