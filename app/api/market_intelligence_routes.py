"""
REST API routes for futures market intelligence (news, fundamentals, sentiment).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from data.market_intelligence import MarketIntelligenceService, bundle_to_dict

logger = logging.getLogger(__name__)

_service = MarketIntelligenceService()

market_intel_bp = Blueprint("market_intelligence", __name__, url_prefix="/api/market-intelligence")


@market_intel_bp.route("", methods=["GET"])
def list_assets() -> Any:
    """Return summary intelligence for all configured futures assets."""
    force = request.args.get("refresh", "false").lower() == "true"
    bundles = _service.get_all(force_refresh=force)
    payload: Dict[str, Any] = {
        "assets": [
            {
                "symbol": b.symbol,
                "label": b.label,
                "last_price": b.last_price,
                "day_change_pct": b.day_change_pct,
                "news_count": len(b.news),
                "fundamentals_count": len(b.fundamentals),
                "sentiment_count": len(b.sentiment),
                "updated_at": b.updated_at.isoformat() if b.updated_at else None,
            }
            for b in bundles.values()
        ],
        "count": len(bundles),
    }
    return jsonify(payload)


@market_intel_bp.route("/<symbol>", methods=["GET"])
def get_asset(symbol: str) -> Any:
    """Return full intelligence bundle for a single futures asset."""
    force = request.args.get("refresh", "false").lower() == "true"
    bundle = _service.get_asset(symbol.upper(), force_refresh=force)
    if bundle is None:
        return jsonify({"error": f"Unknown asset symbol: {symbol}"}), 404
    return jsonify(bundle_to_dict(bundle))


@market_intel_bp.route("/refresh", methods=["POST"])
def refresh_all() -> Any:
    """Force-refresh intelligence data for all assets."""
    try:
        bundles = _service.refresh(force=True)
        return jsonify({
            "status": "ok",
            "refreshed": len(bundles),
            "assets": list(bundles.keys()),
        })
    except Exception:
        logger.exception("Market intelligence refresh failed")
        return jsonify({"error": "Refresh failed"}), 500
