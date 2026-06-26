"""Unit tests for the indicators/ module."""

from __future__ import annotations

import numpy as np
import pytest

from indicators.momentum import rsi, macd, stochastic_oscillator
from indicators.trend import sma, ema, wma, supertrend
from indicators.volatility import atr, bollinger_bands, historical_volatility, keltner_channels
from indicators.volume import obv, vwap, volume_oscillator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def price_series() -> np.ndarray:
    rng = np.random.default_rng(0)
    return 4500.0 + np.cumsum(rng.normal(0, 5, 200))


@pytest.fixture
def ohlcv(price_series: np.ndarray):
    rng = np.random.default_rng(1)
    close = price_series
    high = close + rng.uniform(0.5, 3.0, len(close))
    low = close - rng.uniform(0.5, 3.0, len(close))
    volume = rng.uniform(100, 5000, len(close))
    return high, low, close, volume


# ── RSI ───────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_returns_none_on_insufficient_data(self):
        assert rsi(np.ones(5), period=14) is None

    def test_returns_array_of_correct_length(self, price_series):
        result = rsi(price_series, period=14)
        assert result is not None
        assert len(result) == len(price_series)

    def test_values_in_valid_range(self, price_series):
        result = rsi(price_series, period=14)
        assert result is not None
        valid = result[~np.isnan(result)]
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_warmup_period_is_nan(self, price_series):
        result = rsi(price_series, period=14)
        assert result is not None
        assert np.isnan(result[:14]).all()

    def test_constant_series_returns_neutral(self):
        constant = np.full(50, 100.0)
        result = rsi(constant, period=14)
        assert result is not None
        valid = result[~np.isnan(result)]
        # All gains = all losses = 0, RS undefined → implementation returns 50 or 100
        assert np.all((valid == 50.0) | (valid == 100.0) | np.isnan(valid))


# ── MACD ──────────────────────────────────────────────────────────────────────

class TestMACD:
    def test_returns_none_on_short_series(self):
        assert macd(np.ones(30)) is None

    def test_named_tuple_fields(self, price_series):
        result = macd(price_series, fast=12, slow=26, signal=9)
        assert result is not None
        assert hasattr(result, "macd_line")
        assert hasattr(result, "signal_line")
        assert hasattr(result, "histogram")

    def test_lengths_match(self, price_series):
        result = macd(price_series)
        assert result is not None
        n = len(price_series)
        assert len(result.macd_line) == n
        assert len(result.signal_line) == n
        assert len(result.histogram) == n

    def test_histogram_equals_macd_minus_signal(self, price_series):
        result = macd(price_series)
        assert result is not None
        valid_mask = ~(np.isnan(result.macd_line) | np.isnan(result.signal_line))
        diff = result.macd_line[valid_mask] - result.signal_line[valid_mask]
        np.testing.assert_allclose(
            result.histogram[valid_mask], diff, atol=1e-9
        )


# ── Bollinger Bands ───────────────────────────────────────────────────────────

class TestBollingerBands:
    def test_returns_none_on_short_series(self):
        assert bollinger_bands(np.ones(10), period=20) is None

    def test_upper_above_middle_above_lower(self, price_series):
        result = bollinger_bands(price_series, period=20, num_std=2.0)
        assert result is not None
        valid = ~(np.isnan(result.upper) | np.isnan(result.lower))
        assert (result.upper[valid] >= result.middle[valid]).all()
        assert (result.middle[valid] >= result.lower[valid]).all()

    def test_bandwidth_positive(self, price_series):
        result = bollinger_bands(price_series, period=20)
        assert result is not None
        bw = result.bandwidth[~np.isnan(result.bandwidth)]
        assert (bw >= 0).all()


# ── ATR ───────────────────────────────────────────────────────────────────────

class TestATR:
    def test_returns_none_on_short_series(self, ohlcv):
        high, low, close, _ = ohlcv
        assert atr(high[:5], low[:5], close[:5], period=14) is None

    def test_atr_positive(self, ohlcv):
        high, low, close, _ = ohlcv
        result = atr(high, low, close, period=14)
        assert result is not None
        valid = result[~np.isnan(result)]
        assert (valid > 0).all()


# ── SMA / EMA ─────────────────────────────────────────────────────────────────

class TestMovingAverages:
    def test_sma_returns_none_on_short(self):
        assert sma(np.ones(5), period=10) is None

    def test_sma_correct_value(self):
        arr = np.arange(1.0, 11.0)  # 1..10
        result = sma(arr, period=5)
        assert result is not None
        # Last 5-period SMA = mean(6,7,8,9,10) = 8.0
        assert abs(result[-1] - 8.0) < 1e-9

    def test_ema_returns_none_on_short(self):
        assert ema(np.ones(5), period=10) is None

    def test_ema_last_value_reasonable(self, price_series):
        result = ema(price_series, period=20)
        assert result is not None
        # EMA should be close to the price series (within 5%)
        last_price = price_series[-1]
        last_ema = result[-1]
        assert abs(last_ema - last_price) / last_price < 0.05


# ── OBV ───────────────────────────────────────────────────────────────────────

class TestOBV:
    def test_returns_none_on_empty(self):
        assert obv(np.array([]), np.array([])) is None

    def test_correct_length(self, ohlcv):
        _, _, close, volume = ohlcv
        result = obv(close, volume)
        assert result is not None
        assert len(result) == len(close)

    def test_monotone_uptrend_increasing_obv(self):
        close = np.arange(1.0, 11.0)
        volume = np.ones(10) * 1000.0
        result = obv(close, volume)
        assert result is not None
        # All price increases → OBV should be strictly increasing after first bar
        assert (np.diff(result[1:]) > 0).all()


# ── VWAP ──────────────────────────────────────────────────────────────────────

class TestVWAP:
    def test_returns_none_on_empty(self):
        assert vwap(np.array([]), np.array([]), np.array([]), np.array([])) is None

    def test_single_bar_equals_price(self):
        h = np.array([100.0])
        l = np.array([98.0])
        c = np.array([99.0])
        v = np.array([1000.0])
        result = vwap(h, l, c, v)
        assert result is not None
        expected = (100 + 98 + 99) / 3
        assert abs(result[0] - expected) < 1e-9
