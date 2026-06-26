"""
Market Overview Dashboard — home page with live-style market summary,
futures intelligence (news, fundamentals, sentiment), and watchlist metrics.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update, ALL, ctx
import dash_bootstrap_components as dbc

from app.components import data_table, metric_tile, page_header, section_card
from app.theme import COLORS
from app.data_service import fetch_stock_history, get_asset_intelligence, get_futures_intelligence


WATCHLIST = [
    {"ticker": "TSLA",  "label": "Tesla",       "color": COLORS["blue"]},
    {"ticker": "GME",   "label": "GameStop",    "color": COLORS["green"]},
    {"ticker": "AAPL",  "label": "Apple",       "color": COLORS["gold"]},
    {"ticker": "SPY",   "label": "S&P 500 ETF", "color": COLORS["purple"]},
]

ASSET_COLORS = {
    "ES": COLORS["blue"],
    "NQ": COLORS["purple"],
    "CL": COLORS["red"],
    "GC": COLORS["gold"],
    "ZN": COLORS["green"],
    "ZB": COLORS["cyan"],
    "SI": COLORS["muted"],
    "NG": COLORS["orange"],
}

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


def _futures_heatmap(intel: dict) -> go.Figure:
    if not intel:
        return go.Figure()

    labels = [intel[s]["label"] for s in intel]
    changes = [intel[s]["day_change_pct"] for s in intel]
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


def _sentiment_chart(sentiment: list) -> go.Figure:
    if not sentiment:
        return go.Figure()
    dates = [s["date"][:10] for s in sentiment]
    scores = [s["score"] for s in sentiment]
    colors = [COLORS["green"] if s >= 0 else COLORS["red"] for s in scores]
    fig = go.Figure(go.Bar(
        x=dates, y=scores,
        marker_color=colors,
        hovertemplate="Date: %{x}<br>Score: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="",
        height=220,
        yaxis=dict(range=[-1, 1], title="Sentiment"),
        showlegend=False,
    )
    return fig


def _fundamentals_chart(fundamentals: list) -> go.Figure:
    if not fundamentals:
        return go.Figure()
    dates = [f["date"][:10] for f in fundamentals]
    prices = [f["avg_price"] for f in fundamentals]
    fig = go.Figure(go.Scatter(
        x=dates, y=prices,
        mode="lines",
        line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 113, 227, 0.08)",
        hovertemplate="Date: %{x}<br>Avg Price: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(title="", height=220, showlegend=False)
    return fig


def _asset_card(symbol: str, data: dict) -> html.Div:
    color = ASSET_COLORS.get(symbol, COLORS["blue"])
    chg = data.get("day_change_pct", 0)
    sign_cls = "positive" if chg >= 0 else "negative"
    arrow = "▲" if chg >= 0 else "▼"
    price = data.get("last_price", 0)
    news_count = data.get("news_count", len(data.get("news", [])))

    return html.Div(
        [
            html.Div(symbol, className="asset-card-symbol", style={"color": color}),
            html.Div(data.get("label", symbol), className="asset-card-label"),
            html.Div(f"${price:,.2f}", className="asset-card-price"),
            html.Div(
                f"{arrow} {abs(chg):.2f}%",
                className=f"asset-card-change {sign_cls}",
            ),
            html.Div(f"{news_count} headlines", className="asset-card-meta"),
        ],
        id={"type": "asset-card", "symbol": symbol},
        className="asset-card",
        style={"borderTop": f"3px solid {color}"},
        n_clicks=0,
    )


def _news_panel(news: list) -> html.Div:
    if not news:
        return html.Div("No news headlines available.", className="empty-state")

    rows = []
    for item in news:
        pub = item.get("published_at", "")[:16].replace("T", " ")
        score = item.get("sentiment_score", 0)
        tag_cls = "tag-green" if score > 0.1 else ("tag-red" if score < -0.1 else "tag-blue")
        tag_text = "Bullish" if score > 0.1 else ("Bearish" if score < -0.1 else "Neutral")
        rows.append(html.Div([
            html.Div([
                html.Span(tag_text, className=f"tag {tag_cls}"),
                html.Span(item.get("source", ""), className="news-source"),
                html.Span(pub, className="news-date"),
            ], className="news-meta"),
            html.A(
                item.get("headline", ""),
                href=item.get("url", "#"),
                target="_blank",
                className="news-headline",
            ),
            html.P(item.get("summary", ""), className="news-summary") if item.get("summary") else None,
        ], className="news-item"))

    return html.Div(rows, className="news-list")


def _fundamentals_panel(fundamentals: list) -> html.Div:
    if not fundamentals:
        return html.Div("No fundamental data available.", className="empty-state")

    recent = fundamentals[-12:]
    rows = []
    for f in reversed(recent):
        rows.append(html.Tr([
            html.Td(f["date"][:7]),
            html.Td(f"${f['avg_price']:,.2f}"),
            html.Td(f"{f['avg_volume']:,.0f}"),
            html.Td(f"{f['volatility']:.4f}"),
            html.Td(f"${f['high']:,.2f}"),
            html.Td(f"${f['low']:,.2f}"),
        ]))

    return html.Div([
        dcc.Graph(
            figure=_fundamentals_chart(fundamentals),
            config={"displayModeBar": False},
        ),
        data_table(
            ["Month", "Avg Price", "Avg Volume", "Volatility", "High", "Low"],
            rows,
        ),
    ])


def _sentiment_panel(sentiment: list) -> html.Div:
    if not sentiment:
        return html.Div("No sentiment data available.", className="empty-state")

    recent = sentiment[-12:]
    rows = []
    for s in reversed(recent):
        score = s["score"]
        color = COLORS["green"] if score >= 0 else COLORS["red"]
        rows.append(html.Tr([
            html.Td(s["date"][:7]),
            html.Td(f"{score:+.3f}", style={"color": color, "fontWeight": "600"}),
            html.Td(str(s.get("news_count", 0))),
            html.Td(f"{s.get('momentum_score', 0):+.3f}"),
        ]))

    return html.Div([
        dcc.Graph(
            figure=_sentiment_chart(sentiment),
            config={"displayModeBar": False},
        ),
        data_table(
            ["Month", "Score", "News Count", "Momentum"],
            rows,
        ),
    ])


def _asset_detail_panel(symbol: str, data: dict) -> html.Div:
    color = ASSET_COLORS.get(symbol, COLORS["blue"])
    chg = data.get("day_change_pct", 0)
    sign_cls = "positive" if chg >= 0 else "negative"

    return html.Div([
        html.Div([
            html.Div([
                html.H3(data.get("label", symbol), className="detail-title"),
                html.Span(symbol, className="tag tag-blue"),
            ], className="detail-header-left"),
            html.Div([
                html.Span(f"${data.get('last_price', 0):,.2f}", className="detail-price"),
                html.Span(
                    f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%",
                    className=f"detail-change {sign_cls}",
                ),
            ], className="detail-header-right"),
        ], className="detail-header", style={"borderLeft": f"4px solid {color}"}),

        dbc.Tabs([
            dbc.Tab(_news_panel(data.get("news", [])), label="News", tab_id="tab-news"),
            dbc.Tab(
                _fundamentals_panel(data.get("fundamentals", [])),
                label="Fundamentals",
                tab_id="tab-fundamentals",
            ),
            dbc.Tab(
                _sentiment_panel(data.get("sentiment", [])),
                label="Sentiment",
                tab_id="tab-sentiment",
            ),
        ], id="asset-detail-tabs", active_tab="tab-news"),

        html.Div(
            f"Last updated: {data.get('updated_at', 'N/A')[:19].replace('T', ' ')} UTC",
            className="detail-updated",
        ),
    ], className="asset-detail-panel")


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

    return html.Div([
        page_header(
            "Market Overview",
            "Watch live quotes, explore futures intelligence, and monitor news, fundamentals, and sentiment.",
            badge="Live",
        ),

        html.Div(id="intel-status", className="intel-status"),

        html.Div(cards, className="metrics-grid"),

        section_card(
            "Futures Market Intelligence",
            "Click any contract to view news headlines, fundamentals, and sentiment (2000–present). Data refreshes on load.",
            html.Div([
                html.Div(id="futures-asset-grid", className="asset-grid"),
                html.Div(id="asset-detail-container"),
            ]),
        ),

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
                    figure=go.Figure(),
                    config={"displayModeBar": False},
                ),
            ),
        ], className="charts-grid-2-1"),

        section_card(
            "CME Futures Watchlist",
            "Live contract prices with intelligence data",
            html.Div(id="futures-watchlist-table"),
        ),

        dcc.Store(id="futures-intel-store"),
        dcc.Store(id="selected-asset-store", data="GC"),
        dcc.Interval(id="dashboard-interval", interval=300_000, n_intervals=0),
    ])


@callback(
    Output("futures-intel-store", "data"),
    Output("intel-status", "children"),
    Output("futures-asset-grid", "children"),
    Output("dashboard-futures-bar", "figure"),
    Output("futures-watchlist-table", "children"),
    Input("dashboard-interval", "n_intervals"),
    prevent_initial_call=False,
)
def refresh_intelligence(n_intervals: int):
    force = n_intervals == 0
    intel = get_futures_intelligence(force_refresh=force)

    status = html.Span([
        html.Span(className="status-dot live"),
        f" Intelligence data loaded for {len(intel)} assets",
        " — refreshed on load" if force else " — auto-refresh",
    ], className="status-pill status-pill-live")

    asset_cards = [_asset_card(sym, data) for sym, data in intel.items()]

    fut_rows = []
    for sym, data in intel.items():
        chg = data.get("day_change_pct", 0)
        color = COLORS["green"] if chg >= 0 else COLORS["red"]
        arrow = "▲" if chg >= 0 else "▼"
        fut_rows.append(html.Tr([
            html.Td(html.Span(sym, className="tag tag-blue")),
            html.Td(data.get("label", sym), style={"color": "#6e6e73"}),
            html.Td(f"${data.get('last_price', 0):,.2f}", style={"fontWeight": "600"}),
            html.Td(f"{arrow} {abs(chg):.2f}%", style={"color": color, "fontWeight": "600"}),
            html.Td(str(data.get("news_count", len(data.get("news", []))))),
        ]))

    table = data_table(
        ["Symbol", "Contract", "Last Price", "Day Change", "Headlines"],
        fut_rows,
    )

    return intel, status, asset_cards, _futures_heatmap(intel), table


@callback(
    Output("selected-asset-store", "data"),
    Output("asset-detail-container", "children"),
    Input({"type": "asset-card", "symbol": ALL}, "n_clicks"),
    Input("futures-intel-store", "data"),
    State("selected-asset-store", "data"),
    prevent_initial_call=False,
)
def show_asset_detail(n_clicks_list, intel, current_symbol):
    if not intel:
        return no_update, no_update

    symbol = current_symbol or "GC"

    if ctx.triggered:
        triggered = ctx.triggered[0]
        prop_id = triggered["prop_id"]
        if prop_id != "futures-intel-store.data" and triggered["value"]:
            import json
            id_str = prop_id.rsplit(".", 1)[0]
            card_id = json.loads(id_str)
            symbol = card_id["symbol"]

    data = intel.get(symbol)
    if not data:
        return no_update, no_update

    return symbol, _asset_detail_panel(symbol, data)
