"""
Market Overview Dashboard — home page with live-style market summary,
watchlist, and quick-access metrics.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc

from app.components import data_table, metric_tile, page_header, section_card
from app.theme import COLORS
from app.data_service import (
    FUTURES_ASSET_CONFIG,
    fetch_all_futures_market_intelligence,
    fetch_stock_history,
)


WATCHLIST = [
    {"ticker": "TSLA",  "label": "Tesla",       "color": COLORS["blue"]},
    {"ticker": "GME",   "label": "GameStop",    "color": COLORS["green"]},
    {"ticker": "AAPL",  "label": "Apple",       "color": COLORS["gold"]},
    {"ticker": "SPY",   "label": "S&P 500 ETF", "color": COLORS["purple"]},
]

FUTURES_WATCHLIST = [
    {"sym": "ES", "label": "E-mini S&P 500", "price": 5_247.25, "chg": +0.38},
    {"sym": "NQ", "label": "Nasdaq-100",     "price": 18_432.50, "chg": +0.52},
    {"sym": "CL", "label": "Crude Oil (WTI)", "price": 78.42,    "chg": -0.17},
    {"sym": "GC", "label": "Gold",           "price": 2_341.80, "chg": +0.21},
    {"sym": "ZN", "label": "10-Yr T-Note",   "price": 109.28,   "chg": -0.04},
]


SPARKLINE_FILL = {
    COLORS["blue"]:   "rgba(0, 113, 227, 0.08)",
    COLORS["green"]:  "rgba(52, 199, 89, 0.08)",
    COLORS["gold"]:   "rgba(255, 159, 10, 0.08)",
    COLORS["purple"]: "rgba(175, 82, 222, 0.08)",
}


def _mini_sparkline(ticker: str, color: str) -> go.Figure:
    df = fetch_stock_history(ticker, period="3mo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=SPARKLINE_FILL.get(color, "rgba(0,0,0,0.05)"),
        hoverinfo="skip",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=56,
        showlegend=False,
    )
    return fig


def _market_overview_chart() -> go.Figure:
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
            line=dict(color=item["color"], width=2),
            hovertemplate=f"<b>{item['label']}</b><br>%{{y:+.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        title="",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_ticksuffix="%",
        hovermode="x unified",
    )
    return fig


def _futures_heatmap() -> go.Figure:
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
        title="",
        height=260,
        yaxis_ticksuffix="%",
        showlegend=False,
        bargap=0.35,
    )
    return fig


def _sentiment_timeseries_chart(series: list[dict]) -> go.Figure:
    data = series[-24:] if series else []
    x_vals = [point["month"] for point in data]
    y_vals = [point["score"] for point in data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines+markers",
        line=dict(color=COLORS["blue"], width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(0, 113, 227, 0.06)",
        hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="",
        height=200,
        margin=dict(l=10, r=10, t=8, b=24),
        yaxis=dict(range=[-1.05, 1.05], zeroline=True, zerolinecolor="#d2d2d7"),
        xaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def layout() -> html.Div:
    cards = []
    for item in WATCHLIST:
        df = fetch_stock_history(item["ticker"], period="5d")
        if df.empty or len(df) < 2:
            price, chg, pct = 0, 0, 0
        else:
            price = df["Close"].iloc[-1]
            prev = df["Close"].iloc[-2]
            chg = price - prev
            pct = chg / prev * 100

        sign_cls = "positive" if chg >= 0 else "negative"
        arrow = "▲" if chg >= 0 else "▼"

        cards.append(
            html.Div([
                metric_tile(
                    item["label"],
                    f"${price:,.2f}",
                    delta=f"{arrow} ${abs(chg):.2f} ({pct:+.2f}%)",
                    delta_cls=sign_cls,
                    accent=item["ticker"],
                ),
                dcc.Graph(
                    figure=_mini_sparkline(item["ticker"], item["color"]),
                    config={"displayModeBar": False},
                    style={"height": "56px", "marginTop": "8px"},
                ),
            ], className="metric-tile", style={"borderTop": f"3px solid {item['color']}"})
        )

    fut_rows = []
    for f in FUTURES_WATCHLIST:
        color = COLORS["green"] if f["chg"] >= 0 else COLORS["red"]
        arrow = "▲" if f["chg"] >= 0 else "▼"
        fut_rows.append(html.Tr([
            html.Td(html.Span(f["sym"], className="tag tag-blue")),
            html.Td(f["label"], style={"color": "#6e6e73"}),
            html.Td(f"${f['price']:,.2f}", style={"fontWeight": "600"}),
            html.Td(f"{arrow} {abs(f['chg']):.2f}%", style={"color": color, "fontWeight": "600"}),
        ]))

    return html.Div([
        page_header(
            "Market Overview",
            "Watch live-style quotes, compare performance, and monitor key futures contracts.",
            badge="Today",
        ),

        html.Div(cards, className="metrics-grid"),

        html.Div([
            section_card(
                "Portfolio Performance",
                "One-year normalised return comparison across your watchlist",
                dcc.Graph(
                    id="dashboard-perf-chart",
                    figure=_market_overview_chart(),
                    config={"displayModeBar": False},
                ),
            ),
            section_card(
                "Futures Snapshot",
                "Daily percentage change across major contracts",
                dcc.Graph(
                    id="dashboard-futures-bar",
                    figure=_futures_heatmap(),
                    config={"displayModeBar": False},
                ),
            ),
        ], className="charts-grid-2-1"),

        section_card(
            "CME Futures Watchlist",
            "Key contract prices — paper trading simulation",
            data_table(
                ["Symbol", "Contract", "Last Price", "Day Change"],
                fut_rows,
            ),
        ),

        section_card(
            "Futures Intelligence Feed",
            "Live headlines, fundamentals, and sentiment history (2000 → present) by contract.",
            html.Div([
                dcc.Tabs(
                    id="overview-futures-intel-tabs",
                    value="GC",
                    children=[
                        dcc.Tab(
                            label=f"{sym} — {FUTURES_ASSET_CONFIG[sym]['name']}",
                            value=sym,
                        )
                        for sym in FUTURES_ASSET_CONFIG
                    ],
                ),
                dcc.Loading(
                    html.Div(id="overview-futures-intel-body", style={"marginTop": "16px"}),
                    type="dot",
                ),
            ]),
        ),

        dcc.Interval(id="dashboard-interval", interval=60_000, n_intervals=0),
        dcc.Interval(id="overview-futures-intel-refresh", interval=180_000, n_intervals=0),
    ])


@callback(
    Output("overview-futures-intel-body", "children"),
    Input("overview-futures-intel-tabs", "value"),
    Input("overview-futures-intel-refresh", "n_intervals"),
)
def render_futures_intelligence(symbol: str, n_intervals: int):
    payload = fetch_all_futures_market_intelligence(
        start_year=2000,
        force_refresh=(n_intervals == 0),
    )
    selected = payload["assets"].get(symbol, payload["assets"]["GC"])

    fundamentals = selected.get("fundamentals", {})
    sentiment = selected.get("sentiment", {})
    headlines = selected.get("news_headlines", [])

    fundamental_tiles = [
        ("Current Close", f"{fundamentals.get('current_close', 0.0):,.2f}", "neutral"),
        ("Since 2000", f"{fundamentals.get('price_change_since_2000_pct', 0.0):+.2f}%", "neutral"),
        ("Annualized Return", f"{fundamentals.get('annualized_return_pct', 0.0):+.2f}%", "positive"),
        ("Annualized Volatility", f"{fundamentals.get('annualized_volatility_pct', 0.0):.2f}%", "negative"),
    ]

    headline_rows = [
        html.Li([
            html.A(
                headline.get("title", "Untitled headline"),
                href=headline.get("url") or "#",
                target="_blank",
                rel="noopener noreferrer",
                style={"fontWeight": "600"},
            ),
            html.Div(
                f"{headline.get('source', 'Feed')} · "
                f"{headline.get('published_at', '')[:10]} · "
                f"{headline.get('sentiment_label', 'neutral').title()}",
                style={"fontSize": "12px", "color": "#8e8e93", "marginTop": "4px"},
            ),
        ], style={"marginBottom": "12px"})
        for headline in headlines[:12]
    ]

    composite = float(sentiment.get("current_composite_score", 0.0))
    composite_cls = "positive" if composite > 0.2 else ("negative" if composite < -0.2 else "neutral")

    return html.Div([
        html.Div([
            html.Div(metric_tile(label, value, value_cls=value_cls), className="metric-tile")
            for label, value, value_cls in fundamental_tiles
        ], className="metrics-grid"),
        html.Div([
            html.Div([
                html.H4("Latest Headlines", style={"marginBottom": "10px"}),
                html.Ul(headline_rows, style={"paddingLeft": "18px", "margin": 0}),
            ], style={"padding": "6px 0"}),
            html.Div([
                html.H4("Sentiment (2000 → Current)", style={"marginBottom": "10px"}),
                html.Div(metric_tile(
                    "Composite Sentiment",
                    sentiment.get("current_composite_label", "neutral").title(),
                    delta=f"{composite:+.2f}",
                    delta_cls=composite_cls,
                ), className="metric-tile", style={"marginBottom": "10px"}),
                dcc.Graph(
                    figure=_sentiment_timeseries_chart(sentiment.get("series_from_2000", [])),
                    config={"displayModeBar": False},
                ),
            ]),
        ], className="charts-grid-2-1"),
    ])
