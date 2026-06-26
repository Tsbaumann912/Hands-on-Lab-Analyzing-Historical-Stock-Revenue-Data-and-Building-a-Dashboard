"""
Market Overview Dashboard — home page with live-style market summary,
watchlist, and quick-access metrics.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html
import dash_bootstrap_components as dbc

from app.theme import COLORS
from app.data_service import fetch_stock_history


# ── Watchlist configuration ───────────────────────────────────────────────────

WATCHLIST = [
    {"ticker": "TSLA",  "label": "Tesla",      "color": COLORS["blue"]},
    {"ticker": "GME",   "label": "GameStop",   "color": COLORS["green"]},
    {"ticker": "AAPL",  "label": "Apple",      "color": COLORS["gold"]},
    {"ticker": "SPY",   "label": "S&P 500 ETF","color": COLORS["purple"]},
]

FUTURES_WATCHLIST = [
    {"sym": "ES", "label": "E-mini S&P 500", "price": 5_247.25, "chg": +0.38},
    {"sym": "NQ", "label": "Nasdaq-100",     "price": 18_432.50,"chg": +0.52},
    {"sym": "CL", "label": "Crude Oil (WTI)","price": 78.42,    "chg": -0.17},
    {"sym": "GC", "label": "Gold",           "price": 2_341.80, "chg": +0.21},
    {"sym": "ZN", "label": "10-Yr T-Note",   "price": 109.28,   "chg": -0.04},
]


def _mini_sparkline(ticker: str, color: str) -> go.Figure:
    df = fetch_stock_history(ticker, period="3mo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba"),
        hoverinfo="skip",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=60,
        showlegend=False,
    )
    return fig


def _market_overview_chart() -> go.Figure:
    """Multi-line normalised performance of watchlist tickers (YTD)."""
    fig = go.Figure()
    for item in WATCHLIST:
        df = fetch_stock_history(item["ticker"], period="1y")
        if df.empty:
            continue
        base = df["Close"].iloc[0]
        norm = (df["Close"] / base - 1) * 100
        fig.add_trace(go.Scatter(
            x=df["Date"], y=norm,
            name=item["label"],
            mode="lines",
            line=dict(color=item["color"], width=1.8),
            hovertemplate=f"<b>{item['label']}</b><br>%{{y:+.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        title="1-Year Normalised Performance (%)",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_ticksuffix="%",
        hovermode="x unified",
    )
    return fig


def _futures_heatmap() -> go.Figure:
    """Simulated intraday futures heatmap (% change)."""
    labels = [f["label"] for f in FUTURES_WATCHLIST]
    changes = [f["chg"] for f in FUTURES_WATCHLIST]
    colors = [COLORS["green"] if c >= 0 else COLORS["red"] for c in changes]

    fig = go.Figure(go.Bar(
        x=labels, y=changes,
        marker_color=colors,
        text=[f"{c:+.2f}%" for c in changes],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Change: %{y:+.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Futures Daily Change (%)",
        height=240,
        yaxis_ticksuffix="%",
        showlegend=False,
        bargap=0.3,
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    # Compute live-style stats for watchlist cards
    cards = []
    for item in WATCHLIST:
        df = fetch_stock_history(item["ticker"], period="5d")
        if df.empty or len(df) < 2:
            price, chg, pct = 0, 0, 0
        else:
            price = df["Close"].iloc[-1]
            prev  = df["Close"].iloc[-2]
            chg   = price - prev
            pct   = chg / prev * 100

        sign_cls = "positive" if chg >= 0 else "negative"
        arrow    = "▲" if chg >= 0 else "▼"

        cards.append(
            dbc.Col(
                html.Div([
                    html.Div(item["label"], className="metric-label"),
                    html.Div(item["ticker"], style={"color": item["color"], "fontSize": "0.7rem", "marginBottom": "4px"}),
                    html.Div(f"${price:,.2f}", className=f"metric-value {sign_cls}"),
                    html.Div(f"{arrow} ${abs(chg):.2f}  ({pct:+.2f}%)", className=f"metric-delta {sign_cls}"),
                    html.Div(style={"height": "8px"}),
                    dcc.Graph(
                        figure=_mini_sparkline(item["ticker"], item["color"]),
                        config={"displayModeBar": False},
                        style={"height": "60px"},
                    ),
                ], className="metric-card"),
                md=3, sm=6, className="mb-3",
            )
        )

    # Futures table rows
    fut_rows = []
    for f in FUTURES_WATCHLIST:
        color = COLORS["green"] if f["chg"] >= 0 else COLORS["red"]
        arrow = "▲" if f["chg"] >= 0 else "▼"
        fut_rows.append(html.Tr([
            html.Td(html.Span(f["sym"], className="tag tag-blue")),
            html.Td(f["label"], style={"color": "#94a3b8"}),
            html.Td(f"${f['price']:,.2f}", style={"fontWeight": "600"}),
            html.Td(f"{arrow} {abs(f['chg']):.2f}%", style={"color": color, "fontWeight": "600"}),
        ]))

    return html.Div([
        # Header
        html.Div([
            html.H2("Market Overview"),
            html.P("Real-time watchlist, futures snapshot, and portfolio summary"),
        ], className="page-header"),

        # KPI cards row
        dbc.Row(cards, className="mb-2"),

        # Charts row
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Portfolio Performance"),
                html.P("1-year normalised return comparison", className="chart-subtitle"),
                dcc.Graph(
                    id="dashboard-perf-chart",
                    figure=_market_overview_chart(),
                    config={"displayModeBar": False},
                ),
            ], className="chart-card"), md=8),

            dbc.Col(html.Div([
                html.H5("Futures Snapshot"),
                html.P("Daily % change across key contracts", className="chart-subtitle"),
                dcc.Graph(
                    id="dashboard-futures-bar",
                    figure=_futures_heatmap(),
                    config={"displayModeBar": False},
                ),
            ], className="chart-card"), md=4),
        ]),

        # Futures table
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("CME Futures Watchlist"),
                html.P("Key contract prices — paper / simulation mode", className="chart-subtitle"),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Symbol"), html.Th("Contract"),
                        html.Th("Last Price"), html.Th("Day Chg"),
                    ], style={"color": "#64748b", "fontSize": "0.7rem", "textTransform": "uppercase"})),
                    html.Tbody(fut_rows),
                ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85rem"}),
            ], className="chart-card"), md=12),
        ]),

        # Auto-refresh interval (every 60s for demo)
        dcc.Interval(id="dashboard-interval", interval=60_000, n_intervals=0),
    ])
