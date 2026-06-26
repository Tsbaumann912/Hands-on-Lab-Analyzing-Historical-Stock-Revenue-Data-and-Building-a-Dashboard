"""Unit tests for dashboard intelligence data service payloads."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import data_service


def _mock_history(start: str = "2000-01-03", periods: int = 400) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="B")
    close = np.linspace(1200.0, 2400.0, periods)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": close * 0.998,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.linspace(100_000, 200_000, periods),
        }
    )
    return frame


def test_fetch_futures_asset_intelligence_contains_expected_sections(monkeypatch):
    data_service.clear_futures_intelligence_cache()

    monkeypatch.setattr(data_service, "_download_futures_history", lambda *_args, **_kwargs: _mock_history())
    monkeypatch.setattr(
        data_service,
        "_fetch_yahoo_news_rss",
        lambda *_args, **_kwargs: [
            data_service.HeadlineRecord(
                title="Gold futures rally on strong demand",
                published_at=datetime.now(tz=timezone.utc).isoformat(),
                source="UnitTest Feed",
                url="https://example.com/gold",
                sentiment_score=0.75,
                sentiment_label="bullish",
            )
        ],
    )

    payload = data_service.fetch_futures_asset_intelligence("GC", start_year=2000, force_refresh=True)

    assert payload["asset_symbol"] == "GC"
    assert payload["asset_name"] == "Gold"
    assert "fundamentals" in payload
    assert "sentiment" in payload
    assert len(payload["news_headlines"]) == 1
    assert payload["fundamentals"]["history_start"].startswith("2000-")
    assert payload["sentiment"]["series_from_2000"][0]["month"].startswith("2000-")


def test_fetch_all_futures_market_intelligence_returns_all_assets(monkeypatch):
    data_service.clear_futures_intelligence_cache()
    monkeypatch.setattr(data_service, "_download_futures_history", lambda *_args, **_kwargs: _mock_history())
    monkeypatch.setattr(data_service, "_fetch_yahoo_news_rss", lambda *_args, **_kwargs: [])

    payload = data_service.fetch_all_futures_market_intelligence(start_year=2000, force_refresh=True)

    assert set(data_service.FUTURES_ASSET_CONFIG.keys()).issubset(set(payload["assets"].keys()))
    assert payload["start_year"] == 2000

