"""
Quantitative Futures Trading Terminal — Desktop Web Application

Run with:
    python3 app.py

Then open your browser at: http://127.0.0.1:8050
"""

from __future__ import annotations

import sys
import os

# Ensure workspace root is on the path so our modules resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

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

application = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="QuantTerminal — Futures & Research",
    update_title=None,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

server = application.server   # expose Flask server for production deployment
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
