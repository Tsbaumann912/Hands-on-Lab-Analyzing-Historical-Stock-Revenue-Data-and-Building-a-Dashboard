"""
Market Overview Dashboard — home page with live-style market summary,
watchlist, and quick-access metrics.
"""

from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
import dash
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
import dash_bootstrap_components as dbc

from app.components import data_table, metric_tile, page_header, section_card
from app.theme import COLORS
from app.data_service import fetch_stock_history
from app import news_service as ns


CATEGORY_ACCENT = {
    "Equity Index": COLORS["blue"],
    "Metals":       COLORS["gold"],
    "Energy":       COLORS["orange"],
    "Agriculture":  COLORS["green"],
    "Rates":        COLORS["purple"],
    "Currencies":   COLORS["cyan"],
    "Crypto":       COLORS["pink"],
}

SENTIMENT_COLOR = {
    "Bullish": COLORS["green"],
    "Bearish": COLORS["red"],
    "Neutral": COLORS["muted"],
}

DEFAULT_ASSET = "gold"


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


# ── Futures Market Intelligence (news / fundamentals / sentiment) ─────────────

def _asset_selector() -> html.Div:
    """Clickable chips for every futures asset, grouped by category."""
    groups = []
    for category, assets in ns.assets_by_category().items():
        accent = CATEGORY_ACCENT.get(category, COLORS["blue"])
        chips = [
            html.Button(
                [
                    html.Span(a.symbol, className="intel-chip-sym"),
                    html.Span(a.name, className="intel-chip-name"),
                ],
                id={"type": "intel-asset", "key": a.key},
                className="intel-chip",
                n_clicks=0,
                style={"--chip-accent": accent},
            )
            for a in assets
        ]
        groups.append(html.Div([
            html.Div([
                html.Span(className="intel-cat-dot", style={"background": accent}),
                html.Span(category, className="intel-cat-label"),
            ], className="intel-cat-head"),
            html.Div(chips, className="intel-chip-row"),
        ], className="intel-cat-group"))
    return html.Div(groups, className="intel-selector")


def _sentiment_gauge(sentiment: dict) -> go.Figure:
    compound = sentiment.get("compound", 0.0)
    color = SENTIMENT_COLOR.get(sentiment.get("label", "Neutral"), COLORS["muted"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=compound,
        number={"valueformat": "+.2f", "font": {"size": 30, "color": color}},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1, "tickvals": [-1, -0.5, 0, 0.5, 1]},
            "bar": {"color": color, "thickness": 0.28},
            "borderwidth": 0,
            "steps": [
                {"range": [-1, -0.05], "color": "rgba(255,59,48,0.10)"},
                {"range": [-0.05, 0.05], "color": "rgba(134,134,139,0.12)"},
                {"range": [0.05, 1], "color": "rgba(52,199,89,0.10)"},
            ],
        },
    ))
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=10, b=0))
    return fig


def _price_history_fig(fundamentals: dict, accent: str) -> go.Figure:
    history = fundamentals.get("price_history", [])
    fig = go.Figure()
    if history:
        dates = [pt["date"] for pt in history]
        closes = [pt["close"] for pt in history]
        fig.add_trace(go.Scatter(
            x=dates, y=closes, mode="lines",
            line=dict(color=accent, width=2),
            fill="tozeroy", fillcolor="rgba(0,113,227,0.05)",
            hovertemplate="%{x|%b %Y}<br><b>%{y:,.2f}</b><extra></extra>",
        ))
    fig.update_layout(height=260, margin=dict(l=48, r=16, t=10, b=30), showlegend=False)
    return fig


def _yearly_return_fig(fundamentals: dict) -> go.Figure:
    yearly = fundamentals.get("yearly", [])
    fig = go.Figure()
    if yearly:
        years = [y["year"] for y in yearly]
        rets = [y["annual_return_pct"] for y in yearly]
        bar_colors = [COLORS["green"] if r >= 0 else COLORS["red"] for r in rets]
        fig.add_trace(go.Bar(
            x=years, y=rets, marker_color=bar_colors,
            hovertemplate="<b>%{x}</b><br>Return: %{y:+.1f}%<extra></extra>",
        ))
    fig.update_layout(height=260, margin=dict(l=48, r=16, t=10, b=30),
                      yaxis_ticksuffix="%", showlegend=False, bargap=0.25)
    return fig


def _fmt(value, suffix: str = "", prefix: str = "", nd: int = 2) -> str:
    if value is None:
        return "—"
    return f"{prefix}{value:,.{nd}f}{suffix}"


def _fundamental_tiles(fundamentals: dict) -> list:
    cur = fundamentals.get("current_price")
    chg = fundamentals.get("day_change_pct")
    chg_cls = "neutral" if chg is None else ("positive" if chg >= 0 else "negative")
    cagr = fundamentals.get("cagr_pct")
    dd = fundamentals.get("max_drawdown_pct")
    specs = [
        ("Last Price", _fmt(cur, prefix="$"), "neutral"),
        ("Day Change", _fmt(chg, suffix="%"), chg_cls),
        ("52-Wk High", _fmt(fundamentals.get("high_52w"), prefix="$"), "positive"),
        ("52-Wk Low", _fmt(fundamentals.get("low_52w"), prefix="$"), "negative"),
        ("All-Time High", _fmt(fundamentals.get("all_time_high"), prefix="$"), "positive"),
        ("All-Time Low", _fmt(fundamentals.get("all_time_low"), prefix="$"), "negative"),
        ("CAGR (since 2000)", _fmt(cagr, suffix="%"), "positive" if (cagr or 0) >= 0 else "negative"),
        ("Annual Volatility", _fmt(fundamentals.get("annual_volatility_pct"), suffix="%"), "neutral"),
        ("Max Drawdown", _fmt(dd, suffix="%"), "negative"),
        ("YTD Return", _fmt(fundamentals.get("return_ytd_pct"), suffix="%"),
         "positive" if (fundamentals.get("return_ytd_pct") or 0) >= 0 else "negative"),
        ("1-Year Return", _fmt(fundamentals.get("return_1y_pct"), suffix="%"),
         "positive" if (fundamentals.get("return_1y_pct") or 0) >= 0 else "negative"),
        ("5-Year Return", _fmt(fundamentals.get("return_5y_pct"), suffix="%"),
         "positive" if (fundamentals.get("return_5y_pct") or 0) >= 0 else "negative"),
    ]
    return [
        html.Div(metric_tile(label, value, value_cls=cls), className="metric-tile")
        for label, value, cls in specs
    ]


def _yearly_table(fundamentals: dict) -> html.Table:
    rows = []
    for y in reversed(fundamentals.get("yearly", [])):
        ret = y["annual_return_pct"]
        color = COLORS["green"] if ret >= 0 else COLORS["red"]
        rows.append(html.Tr([
            html.Td(str(y["year"]), style={"fontWeight": "600"}),
            html.Td(f"${y['avg_close']:,.2f}", style={"color": "#6e6e73"}),
            html.Td(f"${y['year_end_close']:,.2f}"),
            html.Td(f"{ret:+.1f}%", style={"color": color, "fontWeight": "600"}),
            html.Td(f"{y['volatility_pct']:.1f}%", style={"color": "#6e6e73"}),
        ]))
    return data_table(["Year", "Avg Close", "Year-End", "Return", "Volatility"], rows)


def _news_list(news: list) -> html.Div:
    if not news:
        return html.Div("No live headlines available right now.", className="intel-empty")
    items = []
    for n in news:
        label = n.get("sentiment_label", "Neutral")
        color = SENTIMENT_COLOR.get(label, COLORS["muted"])
        published = n.get("published", "")
        when = ""
        if published:
            try:
                when = datetime.fromisoformat(published).strftime("%b %d, %Y %H:%M UTC")
            except ValueError:
                when = published
        title_el = (
            html.A(n["title"], href=n["link"], target="_blank", className="intel-news-title")
            if n.get("link") else html.Div(n["title"], className="intel-news-title")
        )
        meta = " · ".join(filter(None, [n.get("publisher", ""), when]))
        body = [
            html.Div([
                title_el,
                html.Span(f"{label} {n.get('sentiment_score', 0):+.2f}",
                          className="intel-news-tag",
                          style={"color": color, "borderColor": color}),
            ], className="intel-news-head"),
            html.Div(meta, className="intel-news-meta"),
        ]
        if n.get("summary"):
            body.append(html.Div(n["summary"], className="intel-news-summary"))
        items.append(html.Div(body, className="intel-news-item"))
    return html.Div(items, className="intel-news-list")


def render_intel_detail(key: str, force: bool = False) -> html.Div:
    intel = ns.get_asset_intel(key, force=force)
    if intel is None:
        return html.Div("Select an asset to view its intelligence.", className="intel-empty")

    asset = intel.asset
    fundamentals = intel.fundamentals
    sentiment = intel.sentiment
    accent = CATEGORY_ACCENT.get(asset["category"], COLORS["blue"])

    data_range = "—"
    if fundamentals.get("data_start") and fundamentals.get("data_end"):
        data_range = f"{fundamentals['data_start']} → {fundamentals['data_end']}"

    fetched = intel.fetched_at
    try:
        fetched = datetime.fromisoformat(fetched).strftime("%b %d, %Y %H:%M UTC")
    except ValueError:
        pass

    header = html.Div([
        html.Div([
            html.Span(asset["symbol"], className="intel-detail-sym",
                      style={"background": accent}),
            html.Div([
                html.H3(asset["name"], className="intel-detail-name"),
                html.Div(
                    f"{asset['category']} · {asset['exchange']} · {asset['unit']} · "
                    f"{asset['yf_ticker']}",
                    className="intel-detail-sub",
                ),
            ]),
        ], className="intel-detail-id"),
        html.Div([
            html.Span(sentiment["label"], className="intel-detail-badge",
                      style={"background": SENTIMENT_COLOR.get(sentiment["label"], COLORS["muted"])}),
            html.Div(f"History {data_range}", className="intel-detail-range"),
            html.Div(f"Updated {fetched}", className="intel-detail-range"),
        ], className="intel-detail-meta"),
    ], className="intel-detail-header")

    sentiment_card = section_card(
        "Market Sentiment",
        f"Aggregated from {sentiment['headline_count']} live headlines",
        html.Div([
            dcc.Graph(figure=_sentiment_gauge(sentiment),
                      config={"displayModeBar": False}),
            html.Div([
                html.Div([html.Span("Bullish", className="intel-sent-k"),
                          html.Span(str(sentiment["bullish"]),
                                    style={"color": COLORS["green"], "fontWeight": "700"})],
                         className="intel-sent-row"),
                html.Div([html.Span("Neutral", className="intel-sent-k"),
                          html.Span(str(sentiment["neutral"]),
                                    style={"color": COLORS["muted"], "fontWeight": "700"})],
                         className="intel-sent-row"),
                html.Div([html.Span("Bearish", className="intel-sent-k"),
                          html.Span(str(sentiment["bearish"]),
                                    style={"color": COLORS["red"], "fontWeight": "700"})],
                         className="intel-sent-row"),
            ], className="intel-sent-breakdown"),
        ]),
    )

    fundamentals_card = section_card(
        "Fundamental Snapshot",
        f"Quantitative fundamentals derived from {data_range}",
        html.Div(_fundamental_tiles(fundamentals), className="metrics-grid"),
    )

    price_card = section_card(
        "Price History (since 2000)",
        "Monthly closing price across the full data set",
        dcc.Graph(figure=_price_history_fig(fundamentals, accent),
                  config={"displayModeBar": False}),
    )

    yearly_chart_card = section_card(
        "Annual Returns",
        "Calendar-year performance since 2000",
        dcc.Graph(figure=_yearly_return_fig(fundamentals),
                  config={"displayModeBar": False}),
    )

    news_card = section_card(
        f"Latest {asset['name']} Headlines",
        "Live news with per-headline sentiment scoring",
        _news_list(intel.news),
    )

    yearly_table_card = section_card(
        "Year-by-Year Detail",
        "Average close, year-end close, annual return and realised volatility",
        html.Div(_yearly_table(fundamentals), className="intel-yearly-scroll"),
    )

    return html.Div([
        header,
        html.Div([sentiment_card, fundamentals_card], className="charts-grid-1-2"),
        html.Div([price_card, yearly_chart_card], className="charts-grid-2"),
        news_card,
        yearly_table_card,
    ], className="intel-detail")


def _intelligence_section() -> html.Div:
    return section_card(
        "Futures Market Intelligence",
        "Select any contract to read live news, fundamental data and sentiment — "
        "categorised by asset and computed from 2000 to today.",
        html.Div([
            html.Div([
                _asset_selector(),
                html.Button(
                    [html.Span("\u21bb ", className="intel-refresh-icon"), "Refresh live data"],
                    id="intel-refresh", className="intel-refresh-btn", n_clicks=0,
                ),
            ], className="intel-controls"),
            dcc.Store(id="intel-selected", data=DEFAULT_ASSET),
            dcc.Loading(html.Div(id="intel-detail-container")),
        ]),
        className="intel-section",
    )


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

        _intelligence_section(),

        dcc.Interval(id="dashboard-interval", interval=60_000, n_intervals=0),
    ])


# ── Intelligence callbacks ────────────────────────────────────────────────────

@callback(
    Output("intel-selected", "data"),
    Input({"type": "intel-asset", "key": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _select_asset(_clicks):
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "intel-asset":
        return triggered["key"]
    return dash.no_update


@callback(
    Output({"type": "intel-asset", "key": ALL}, "className"),
    Input("intel-selected", "data"),
    State({"type": "intel-asset", "key": ALL}, "id"),
)
def _highlight_selected(selected, ids):
    return [
        "intel-chip active" if (i and i.get("key") == selected) else "intel-chip"
        for i in ids
    ]


@callback(
    Output("intel-detail-container", "children"),
    Input("intel-selected", "data"),
    Input("intel-refresh", "n_clicks"),
)
def _render_detail(selected, refresh_clicks):
    key = selected or DEFAULT_ASSET
    force = ctx.triggered_id == "intel-refresh"
    return render_intel_detail(key, force=force)
