"""Tests for the market intelligence data layer and API."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from core.config import Config, MarketIntelligenceAssetConfig
from core.models import AssetIntelligenceBundle, NewsItem
from data.market_intelligence import MarketIntelligenceService, bundle_to_dict


@pytest.fixture
def mi_config(tmp_path) -> Config:
    cfg = Config()
    cfg.market_intelligence = type(cfg.market_intelligence)(
        history_start="2020-01-01",
        cache_ttl_seconds=60,
        news_limit=5,
        assets=[
            MarketIntelligenceAssetConfig(
                symbol="GC",
                label="Gold",
                yfinance_ticker="GC=F",
                color="#ff9f0a",
            ),
        ],
    )
    return cfg


@pytest.fixture
def service(mi_config: Config) -> MarketIntelligenceService:
    return MarketIntelligenceService(config=mi_config)


class TestMarketIntelligenceService:
    def test_get_all_returns_bundles(self, service: MarketIntelligenceService) -> None:
        bundles = service.get_all(force_refresh=True)
        assert "GC" in bundles
        bundle = bundles["GC"]
        assert bundle.symbol == "GC"
        assert bundle.label == "Gold"
        assert len(bundle.news) > 0
        assert len(bundle.fundamentals) > 0
        assert len(bundle.sentiment) > 0

    def test_get_asset_unknown_returns_none(self, service: MarketIntelligenceService) -> None:
        service.get_all(force_refresh=True)
        assert service.get_asset("ZZZ") is None

    def test_cache_valid_without_force(self, service: MarketIntelligenceService) -> None:
        first = service.get_all(force_refresh=True)
        second = service.get_all(force_refresh=False)
        assert first is second

    def test_refresh_force_rebuilds(self, service: MarketIntelligenceService) -> None:
        service.get_all(force_refresh=True)
        refreshed = service.refresh(force=True)
        assert "GC" in refreshed

    def test_fundamentals_start_from_history_start(
        self, service: MarketIntelligenceService
    ) -> None:
        bundles = service.get_all(force_refresh=True)
        fund = bundles["GC"].fundamentals
        assert fund[0].date.year >= 2020

    def test_sentiment_scores_bounded(self, service: MarketIntelligenceService) -> None:
        bundles = service.get_all(force_refresh=True)
        for point in bundles["GC"].sentiment:
            assert -1.0 <= point.score <= 1.0


class TestBundleToDict:
    def test_serialises_bundle(self) -> None:
        bundle = AssetIntelligenceBundle(
            symbol="GC",
            label="Gold",
            last_price=2341.80,
            day_change_pct=0.21,
            news=[
                NewsItem(
                    symbol="GC",
                    headline="Gold rises on safe-haven demand",
                    source="Reuters",
                    published_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
                    url="https://example.com",
                    sentiment_score=0.5,
                ),
            ],
            updated_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        result = bundle_to_dict(bundle)
        assert result["symbol"] == "GC"
        assert len(result["news"]) == 1
        assert result["news"][0]["headline"] == "Gold rises on safe-haven demand"


class TestMarketIntelligenceAPI:
    def test_list_endpoint(self) -> None:
        from wsgi import server
        client = server.test_client()
        resp = client.get("/api/market-intelligence?refresh=true")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "assets" in data
        assert data["count"] >= 1

    def test_asset_endpoint(self) -> None:
        from wsgi import server
        client = server.test_client()
        resp = client.get("/api/market-intelligence/GC?refresh=true")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["symbol"] == "GC"
        assert "news" in data
        assert "fundamentals" in data
        assert "sentiment" in data

    def test_unknown_asset_404(self) -> None:
        from wsgi import server
        client = server.test_client()
        resp = client.get("/api/market-intelligence/INVALID")
        assert resp.status_code == 404

    def test_refresh_endpoint(self) -> None:
        from wsgi import server
        client = server.test_client()
        resp = client.post("/api/market-intelligence/refresh")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"
        assert data["refreshed"] >= 1
