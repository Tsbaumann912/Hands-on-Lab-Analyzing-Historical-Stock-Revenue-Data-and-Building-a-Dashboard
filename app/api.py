"""
REST API for the Futures Market Intelligence service.

Mounted on the Dash Flask server (see ``wsgi.py``). Every endpoint accepts an
optional ``?refresh=1`` flag to bypass the TTL cache and pull live data from the
upstream provider — this is what powers "update the data points upon loading the
terminal".

Routes
------
GET  /api/futures/assets                      → universe metadata (grouped by category)
GET  /api/futures/<key>                        → combined news + fundamentals + sentiment
GET  /api/futures/<key>/news                   → live, sentiment-scored headlines
GET  /api/futures/<key>/fundamentals           → fundamentals since config.news.history_start
GET  /api/futures/<key>/sentiment              → aggregated sentiment summary
POST /api/futures/refresh                      → clear the cache (force re-fetch on next read)
"""

from __future__ import annotations

from typing import Tuple

from flask import Flask, jsonify, request

from app import news_service as ns


def _wants_refresh() -> bool:
    return request.args.get("refresh", "0").lower() in ("1", "true", "yes")


def register_api(server: Flask) -> Flask:
    """Attach all ``/api/futures`` routes to *server* and return it."""

    @server.route("/api/futures/assets", methods=["GET"])
    def _assets():
        grouped = {
            category: [a.to_dict() for a in assets]
            for category, assets in ns.assets_by_category().items()
        }
        return jsonify({
            "count": len(ns.FUTURES_ASSETS),
            "categories": ns.CATEGORY_ORDER,
            "assets": ns.list_assets(),
            "by_category": grouped,
        })

    @server.route("/api/futures/refresh", methods=["POST"])
    def _refresh():
        ns.clear_cache()
        return jsonify({"status": "ok", "message": "intelligence cache cleared"})

    @server.route("/api/futures/<key>", methods=["GET"])
    def _intel(key: str):
        intel = ns.get_asset_intel(key, force=_wants_refresh())
        if intel is None:
            return _not_found(key)
        return jsonify(intel.to_dict())

    @server.route("/api/futures/<key>/news", methods=["GET"])
    def _news(key: str):
        if ns.get_asset(key) is None:
            return _not_found(key)
        news = ns.get_news(key, force=_wants_refresh())
        return jsonify({"key": key, "count": len(news), "news": [n.to_dict() for n in news]})

    @server.route("/api/futures/<key>/fundamentals", methods=["GET"])
    def _fundamentals(key: str):
        if ns.get_asset(key) is None:
            return _not_found(key)
        fundamentals = ns.get_fundamentals(key, force=_wants_refresh())
        return jsonify({"key": key, "fundamentals": fundamentals.to_dict()})

    @server.route("/api/futures/<key>/sentiment", methods=["GET"])
    def _sentiment(key: str):
        if ns.get_asset(key) is None:
            return _not_found(key)
        sentiment = ns.get_sentiment(key, force=_wants_refresh())
        return jsonify({"key": key, "sentiment": sentiment.to_dict()})

    return server


def _not_found(key: str) -> Tuple:
    return (
        jsonify({
            "error": "unknown_asset",
            "key": key,
            "available": sorted(ns.FUTURES_ASSETS.keys()),
        }),
        404,
    )
