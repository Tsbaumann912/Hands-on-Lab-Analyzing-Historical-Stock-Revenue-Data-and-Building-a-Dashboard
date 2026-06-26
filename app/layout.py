"""
Root application layout: sidebar navigation + dynamic page content.
"""

from __future__ import annotations

from dash import dcc, html


NAV_ITEMS = [
    {"id": "nav-dashboard",    "label": "Dashboard",          "icon": "📊", "page": "/"},
    {"id": "nav-stock",        "label": "Stock Research",     "icon": "📈", "page": "/stock"},
    {"id": "nav-futures",      "label": "Futures Terminal",   "icon": "⚡", "page": "/futures"},
    {"id": "nav-indicators",   "label": "Indicator Explorer", "icon": "🔬", "page": "/indicators"},
    {"id": "nav-strategy-lab", "label": "Strategy Lab",       "icon": "🧪", "page": "/strategy-lab"},
    {"id": "nav-risk",         "label": "Risk Console",       "icon": "🛡️", "page": "/risk"},
]


def sidebar() -> html.Div:
    nav_links = []
    for item in NAV_ITEMS:
        nav_links.append(
            html.Div(
                [
                    html.Span(item["icon"], style={"fontSize": "0.95rem"}),
                    html.Span(item["label"]),
                ],
                id=item["id"],
                className="nav-link-custom",
                n_clicks=0,
            )
        )

    return html.Div([
        # Brand
        html.Div([
            html.H4("⚙ QuantTerminal"),
            html.Small("FUTURES · RESEARCH · RISK"),
        ], className="sidebar-brand"),

        # Navigation
        html.Div("RESEARCH", className="nav-section-label"),
        nav_links[0],  # Dashboard
        nav_links[1],  # Stock Research

        html.Div("TRADING", className="nav-section-label"),
        nav_links[2],  # Futures Terminal
        nav_links[3],  # Indicator Explorer
        nav_links[4],  # Strategy Lab

        html.Div("MANAGEMENT", className="nav-section-label"),
        nav_links[5],  # Risk Console

        # Status footer
        html.Div([
            html.Div([
                html.Span(className="status-dot paper"),
                html.Span("Paper Trading", style={"color": "#94a3b8", "fontSize": "0.75rem"}),
            ]),
            html.Div("v1.0.0", style={"color": "#334155", "fontSize": "0.65rem", "marginTop": "4px"}),
        ], className="sidebar-status"),
    ], id="sidebar")


def root_layout() -> html.Div:
    return html.Div([
        dcc.Location(id="url", refresh=False),
        sidebar(),
        html.Div(id="page-content", className="", children=[], style={"marginLeft": "220px", "padding": "24px", "minHeight": "100vh", "backgroundColor": "#0a0e1a"}),
    ])
