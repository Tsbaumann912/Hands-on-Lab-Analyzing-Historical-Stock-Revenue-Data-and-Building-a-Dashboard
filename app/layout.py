"""
Root application layout — Apple-inspired top navigation and content shell.
"""

from __future__ import annotations

from dash import dcc, html

from app.components import status_pill


NAV_ITEMS = [
    {"id": "nav-dashboard",    "label": "Overview",    "page": "/"},
    {"id": "nav-charts",       "label": "Charts",      "page": "/charts"},
    {"id": "nav-futures",      "label": "Futures",     "page": "/futures"},
    {"id": "nav-indicators",   "label": "Indicators",  "page": "/indicators"},
    {"id": "nav-strategy-lab", "label": "Strategies",  "page": "/strategy-lab"},
    {"id": "nav-risk",         "label": "Risk",        "page": "/risk"},
]


def top_nav() -> html.Nav:
    nav_links = [
        html.Div(item["label"], id=item["id"], className="nav-link-custom", n_clicks=0)
        for item in NAV_ITEMS
    ]

    return html.Nav([
        html.Div([
            html.Div([
                html.Span("Q", className="brand-icon"),
                html.Span("QuantTerminal"),
            ], className="brand-link"),

            html.Div(nav_links, className="nav-links"),

            html.Div([
                status_pill("Paper Trading", "paper"),
            ], className="nav-status"),
        ], className="top-nav-inner"),
    ], className="top-nav", id="top-nav")


def root_layout() -> html.Div:
    return html.Div([
        dcc.Location(id="url", refresh=False),
        html.Div([
            top_nav(),
            html.Main(id="page-content", children=[]),
            html.Footer(
                "QuantTerminal · Futures & Research · Simulation mode",
                className="app-footer",
            ),
        ], className="app-shell"),
    ])
