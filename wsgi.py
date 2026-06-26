"""
Quantitative Futures Trading Terminal — Desktop Web Application

Run with:
    python3 wsgi.py

Then open your browser at: http://127.0.0.1:8050
"""

from __future__ import annotations

import sys
import os

# Ensure workspace root is on the path so our modules resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import json

import dash
from dash import Input, Output, callback
import dash_bootstrap_components as dbc

# Register custom Plotly theme before anything else
import app.theme  # noqa: F401

from app.layout import root_layout, NAV_ITEMS

# Import page modules (they register their own callbacks via @callback)
from app.pages import (
    dashboard,
    stock_research,
    futures_terminal,
    strategy_lab,
    risk_console,
    indicator_explorer,
)

# ── Initialise Dash app ───────────────────────────────────────────────────────

_assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "assets")

application = dash.Dash(
    __name__,
    assets_folder=_assets_dir,
    external_stylesheets=[
        dbc.themes.FLATLY,
    ],
    suppress_callback_exceptions=True,
    title="QuantTerminal",
    update_title=None,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

server = application.server   # expose Flask server for production deployment


@server.route("/health")
def _health():
    return "ok", 200


# ── News & Fundamentals REST API ──────────────────────────────────────────────

from app.news_service import (  # noqa: E402
    FUTURES_ASSETS,
    fetch_asset_news,
    fetch_asset_fundamentals,
    fetch_sentiment_history,
    invalidate_cache,
    news_items_to_dicts,
    fundamentals_to_dict,
)
from flask import jsonify, request as flask_request  # noqa: E402


def _json_response(data, status: int = 200):
    resp = server.response_class(
        json.dumps(data, default=str),
        status=status,
        mimetype="application/json",
    )
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@server.route("/api/assets")
def api_list_assets():
    """List all registered futures assets with metadata."""
    payload = {
        k: {
            "name":     v["name"],
            "full_name": v["full_name"],
            "sector":   v["sector"],
            "color":    v["color"],
            "icon":     v["icon"],
        }
        for k, v in FUTURES_ASSETS.items()
    }
    return _json_response(payload)


@server.route("/api/news/<asset_key>")
def api_news(asset_key: str):
    """
    GET /api/news/<asset_key>
    Returns latest news headlines with sentiment scores.
    Query params:
      limit  (int, default 50) – max articles to return
    """
    if asset_key not in FUTURES_ASSETS:
        return _json_response({"error": f"Unknown asset: {asset_key}"}, 404)
    limit = min(int(flask_request.args.get("limit", 50)), 200)
    items = fetch_asset_news(asset_key, limit=limit)
    return _json_response({
        "asset":  asset_key,
        "count":  len(items),
        "items":  news_items_to_dicts(items),
    })


@server.route("/api/fundamentals/<asset_key>")
def api_fundamentals(asset_key: str):
    """
    GET /api/fundamentals/<asset_key>
    Returns fundamental snapshot: price, 52-week range, ETF info, macro context.
    """
    if asset_key not in FUTURES_ASSETS:
        return _json_response({"error": f"Unknown asset: {asset_key}"}, 404)
    snap = fetch_asset_fundamentals(asset_key)
    return _json_response(fundamentals_to_dict(snap))


@server.route("/api/sentiment/<asset_key>")
def api_sentiment(asset_key: str):
    """
    GET /api/sentiment/<asset_key>
    Returns monthly sentiment series from 2000 to present.
    Query params:
      start  (str YYYY-MM-DD, default "2000-01-01")
    """
    if asset_key not in FUTURES_ASSETS:
        return _json_response({"error": f"Unknown asset: {asset_key}"}, 404)
    start = flask_request.args.get("start", "2000-01-01")
    df = fetch_sentiment_history(asset_key, start=start)
    return _json_response({
        "asset": asset_key,
        "start": start,
        "rows":  df.to_dict(orient="records"),
    })


@server.route("/api/refresh/<asset_key>", methods=["POST"])
def api_refresh(asset_key: str):
    """
    POST /api/refresh/<asset_key>
    Invalidates the in-process cache for the asset and triggers a fresh fetch.
    """
    if asset_key not in FUTURES_ASSETS and asset_key != "all":
        return _json_response({"error": f"Unknown asset: {asset_key}"}, 404)
    cleared = invalidate_cache(None if asset_key == "all" else asset_key)
    return _json_response({"cleared": cleared, "asset": asset_key})


application.layout = root_layout()


# ── URL routing ───────────────────────────────────────────────────────────────

PAGE_MAP = {
    "/":            dashboard.layout,
    "/stock":       stock_research.layout,
    "/futures":     futures_terminal.layout,
    "/indicators":  indicator_explorer.layout,
    "/strategy-lab":strategy_lab.layout,
    "/risk":        risk_console.layout,
}


@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname: str):
    layout_fn = PAGE_MAP.get(pathname, PAGE_MAP["/"])
    return layout_fn()


# ── Sidebar active-state highlighting ────────────────────────────────────────

nav_ids   = [item["id"]   for item in NAV_ITEMS]
nav_pages = [item["page"] for item in NAV_ITEMS]


@callback(
    *[Output(nid, "className") for nid in nav_ids],
    Input("url", "pathname"),
)
def update_nav_active(pathname: str):
    classes = []
    for page in nav_pages:
        is_active = (pathname == page) or (page != "/" and pathname.startswith(page))
        classes.append("nav-link-custom active" if is_active else "nav-link-custom")
    return classes


# ── Sidebar click navigation ─────────────────────────────────────────────────

for nav_item in NAV_ITEMS:
    @callback(
        Output("url", "pathname", allow_duplicate=True),
        Input(nav_item["id"], "n_clicks"),
        prevent_initial_call=True,
    )
    def navigate(n_clicks, page=nav_item["page"]):
        if n_clicks:
            return page
        return dash.no_update


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    import threading

    def _open_browser():
        import time
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:8050")

    print("\n" + "=" * 60)
    print("  QuantTerminal — Quantitative Futures Trading Terminal")
    print("=" * 60)
    print("  Opening browser at http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    threading.Thread(target=_open_browser, daemon=True).start()

    application.run(
        host="0.0.0.0",
        port=8050,
        debug=False,
        use_reloader=False,
    )
