"""
Market Overview Dashboard — home page with live-style market summary,
watchlist, quick-access metrics, and the Futures News Intelligence panel.

The news intelligence panel lets users click any futures asset and instantly
see categorised news headlines, fundamental data, and a historical sentiment
chart dating back to 2000. Data refreshes on demand via the Refresh button.
"""

from __future__ import annotations

import json
import logging

import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

from app.components import data_table, metric_tile, page_header, section_card
from app.theme import COLORS
from app.data_service import fetch_stock_history
from app.news_service import (
    FUTURES_ASSETS,
    fetch_asset_news,
    fetch_asset_fundamentals,
    fetch_sentiment_history,
    invalidate_cache,
    news_items_to_dicts,
    fundamentals_to_dict,
)

logger = logging.getLogger(__name__)


# ── Watchlist & static data ───────────────────────────────────────────────────

WATCHLIST = [
    {"ticker": "TSLA",  "label": "Tesla",       "color": COLORS["blue"]},
    {"ticker": "GME",   "label": "GameStop",    "color": COLORS["green"]},
    {"ticker": "AAPL",  "label": "Apple",       "color": COLORS["gold"]},
    {"ticker": "SPY",   "label": "S&P 500 ETF", "color": COLORS["purple"]},
]

FUTURES_WATCHLIST = [
    {"sym": "ES", "label": "E-mini S&P 500",  "price": 5_247.25,  "chg": +0.38},
    {"sym": "NQ", "label": "Nasdaq-100",      "price": 18_432.50, "chg": +0.52},
    {"sym": "CL", "label": "Crude Oil (WTI)", "price": 78.42,     "chg": -0.17},
    {"sym": "GC", "label": "Gold",            "price": 2_341.80,  "chg": +0.21},
    {"sym": "ZN", "label": "10-Yr T-Note",    "price": 109.28,    "chg": -0.04},
]

SPARKLINE_FILL = {
    COLORS["blue"]:   "rgba(0, 113, 227, 0.08)",
    COLORS["green"]:  "rgba(52, 199, 89, 0.08)",
    COLORS["gold"]:   "rgba(255, 159, 10, 0.08)",
    COLORS["purple"]: "rgba(175, 82, 222, 0.08)",
}

# Ordered asset list for the selector pill-bar
_ASSET_ORDER = ["ES", "NQ", "GC", "CL", "ZN", "SI", "NG", "HG"]


# ── Chart helpers ─────────────────────────────────────────────────────────────

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
    labels  = [f["label"] for f in FUTURES_WATCHLIST]
    changes = [f["chg"]   for f in FUTURES_WATCHLIST]
    colors  = [COLORS["green"] if c >= 0 else COLORS["red"] for c in changes]
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


# ── News panel component builders ─────────────────────────────────────────────

def _asset_selector() -> html.Div:
    """Pill-bar of clickable asset buttons."""
    buttons = []
    for key in _ASSET_ORDER:
        meta = FUTURES_ASSETS.get(key)
        if not meta:
            continue
        buttons.append(
            html.Button(
                [
                    html.Span(meta["icon"], className="asset-btn-icon"),
                    html.Span(meta["name"], className="asset-btn-label"),
                ],
                id={"type": "asset-btn", "index": key},
                className="asset-selector-btn",
                n_clicks=0,
            )
        )
    return html.Div(buttons, className="asset-selector-bar")


def _empty_news_panel() -> html.Div:
    return html.Div(
        [
            html.Div("📡", className="empty-panel-icon"),
            html.P(
                "Select a futures asset above to load live news, "
                "fundamentals, and sentiment data.",
                className="empty-panel-text",
            ),
        ],
        className="empty-panel",
    )


def _render_headlines(news_dicts: list[dict]) -> html.Div:
    if not news_dicts:
        return html.Div(
            html.P("No recent headlines found for this asset.", className="no-data-msg"),
            className="news-list",
        )
    cards = []
    for item in news_dicts:
        sentiment_color = item.get("sentiment_color", "#86868b")
        sentiment_label = item.get("sentiment_label", "Neutral")
        score           = item.get("sentiment_score", 0.0)
        cards.append(
            html.A(
                html.Div([
                    html.Div([
                        html.Span(
                            sentiment_label,
                            className="sentiment-badge",
                            style={"backgroundColor": sentiment_color + "20",
                                   "color": sentiment_color,
                                   "borderColor": sentiment_color + "40"},
                        ),
                        html.Span(f"{score:+.3f}", className="sentiment-score",
                                  style={"color": sentiment_color}),
                        html.Span(item.get("publisher", ""), className="news-publisher"),
                        html.Span(item.get("published_label", ""), className="news-date"),
                    ], className="news-meta"),
                    html.H4(item.get("title", ""), className="news-title"),
                    html.P(item.get("summary", ""), className="news-summary"),
                ], className="news-card"),
                href=item.get("link", "#"),
                target="_blank",
                style={"textDecoration": "none"},
            )
        )
    return html.Div(cards, className="news-list")


def _render_fundamentals(fund_dict: dict) -> html.Div:
    price     = fund_dict.get("current_price", 0)
    chg_1d    = fund_dict.get("price_change_1d", 0)
    pct_1d    = fund_dict.get("price_change_pct_1d", 0)
    chg_1y    = fund_dict.get("price_change_1y", 0)
    pct_1y    = fund_dict.get("price_change_pct_1y", 0)
    high_52w  = fund_dict.get("high_52w", 0)
    low_52w   = fund_dict.get("low_52w", 0)
    avg_vol   = fund_dict.get("avg_volume", 0)
    desc      = fund_dict.get("description", "")
    drivers   = fund_dict.get("key_drivers", [])
    contract  = fund_dict.get("contract_info", {})
    etf       = fund_dict.get("etf_info", {})
    macro     = fund_dict.get("macro_context", [])
    updated   = fund_dict.get("last_updated", "")

    pos_1d = chg_1d >= 0
    pos_1y = chg_1y >= 0

    def _fmt_num(v: float, decimals: int = 2) -> str:
        if v == 0:
            return "—"
        if abs(v) >= 1e9:
            return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v/1e6:.2f}M"
        if abs(v) >= 1000:
            return f"${v:,.{decimals}f}"
        return f"{v:.{decimals}f}"

    price_section = html.Div([
        html.Div([
            html.Span("Current Price", className="fund-label"),
            html.Span(_fmt_num(price), className="fund-value fund-value-lg"),
        ], className="fund-row"),
        html.Div([
            html.Div([
                html.Span("Day Change", className="fund-label"),
                html.Span(
                    f"{'▲' if pos_1d else '▼'} {abs(chg_1d):.4f} ({pct_1d:+.2f}%)",
                    className="fund-value",
                    style={"color": COLORS["green"] if pos_1d else COLORS["red"]},
                ),
            ], className="fund-cell"),
            html.Div([
                html.Span("1-Year Change", className="fund-label"),
                html.Span(
                    f"{'▲' if pos_1y else '▼'} {abs(chg_1y):.2f} ({pct_1y:+.2f}%)",
                    className="fund-value",
                    style={"color": COLORS["green"] if pos_1y else COLORS["red"]},
                ),
            ], className="fund-cell"),
        ], className="fund-row-grid"),
        html.Div([
            html.Div([
                html.Span("52-Week High", className="fund-label"),
                html.Span(_fmt_num(high_52w), className="fund-value positive"),
            ], className="fund-cell"),
            html.Div([
                html.Span("52-Week Low", className="fund-label"),
                html.Span(_fmt_num(low_52w), className="fund-value negative"),
            ], className="fund-cell"),
            html.Div([
                html.Span("Avg. Daily Volume", className="fund-label"),
                html.Span(f"{avg_vol:,.0f}" if avg_vol else "—", className="fund-value"),
            ], className="fund-cell"),
        ], className="fund-row-grid"),
    ], className="fund-price-block")

    contract_section = html.Div([
        html.H4("Contract Specifications", className="fund-section-title"),
        html.Div([
            html.Div([
                html.Span("Contract Size", className="fund-label"),
                html.Span(f"{contract.get('contract_size', '—'):,}", className="fund-value"),
            ], className="fund-cell"),
            html.Div([
                html.Span("Unit", className="fund-label"),
                html.Span(contract.get("unit", "—"), className="fund-value"),
            ], className="fund-cell"),
            html.Div([
                html.Span("Tick Size", className="fund-label"),
                html.Span(str(contract.get("tick_size", "—")), className="fund-value"),
            ], className="fund-cell"),
            html.Div([
                html.Span("Dollar Multiplier", className="fund-label"),
                html.Span(f"${contract.get('multiplier', 0):,}", className="fund-value"),
            ], className="fund-cell"),
        ], className="fund-row-grid"),
    ], className="fund-section-block")

    etf_aum    = etf.get("aum")
    etf_er     = etf.get("expense_ratio")
    etf_name   = etf.get("etf_name", "—")
    etf_nav    = etf.get("nav")
    etf_beta   = etf.get("beta_3y")
    etf_section = html.Div([
        html.H4("ETF Proxy", className="fund-section-title"),
        html.P(etf_name, className="fund-etf-name"),
        html.Div([
            html.Div([
                html.Span("AUM", className="fund-label"),
                html.Span(
                    f"${etf_aum/1e9:.2f}B" if etf_aum else "—",
                    className="fund-value",
                ),
            ], className="fund-cell"),
            html.Div([
                html.Span("Expense Ratio", className="fund-label"),
                html.Span(
                    f"{etf_er*100:.2f}%" if etf_er else "—",
                    className="fund-value",
                ),
            ], className="fund-cell"),
            html.Div([
                html.Span("NAV", className="fund-label"),
                html.Span(f"${etf_nav:.2f}" if etf_nav else "—", className="fund-value"),
            ], className="fund-cell"),
            html.Div([
                html.Span("3Y Beta", className="fund-label"),
                html.Span(f"{etf_beta:.2f}" if etf_beta else "—", className="fund-value"),
            ], className="fund-cell"),
        ], className="fund-row-grid"),
    ], className="fund-section-block")

    desc_section = html.Div([
        html.H4("Asset Overview", className="fund-section-title"),
        html.P(desc, className="fund-desc"),
    ], className="fund-section-block")

    drivers_section = html.Div([
        html.H4("Key Price Drivers", className="fund-section-title"),
        html.Div([
            html.Span(d, className="driver-tag") for d in drivers
        ], className="driver-tags"),
    ], className="fund-section-block")

    macro_section = html.Div([
        html.H4("Macro Context", className="fund-section-title"),
        html.Div([
            html.Div([
                html.Span(m["label"], className="macro-label"),
                html.Span(m["desc"],  className="macro-desc"),
            ], className="macro-row")
            for m in macro
        ], className="macro-list"),
    ], className="fund-section-block") if macro else html.Div()

    return html.Div([
        price_section,
        contract_section,
        etf_section,
        desc_section,
        drivers_section,
        macro_section,
        html.P(f"Last updated: {updated}", className="fund-updated"),
    ], className="fund-panel")


def _render_sentiment_chart(df_rows: list[dict]) -> go.Figure:
    """Build the historical sentiment chart from monthly rows."""
    import pandas as pd

    if not df_rows:
        fig = go.Figure()
        fig.update_layout(
            height=380,
            annotations=[dict(text="No sentiment data available", showarrow=False,
                              font=dict(size=14, color="#86868b"),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
        )
        return fig

    df = pd.DataFrame(df_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    scores = df["Sentiment"].to_numpy()
    dates  = df["Date"].tolist()
    sources = df["Source"].tolist() if "Source" in df.columns else ["" for _ in dates]

    # Color each bar by label
    bar_colors = []
    for row in df_rows:
        lbl = row.get("Label", "Neutral")
        if lbl == "Bullish":
            bar_colors.append("rgba(52, 199, 89, 0.7)")
        elif lbl == "Bearish":
            bar_colors.append("rgba(255, 59, 48, 0.7)")
        else:
            bar_colors.append("rgba(134, 134, 139, 0.5)")

    fig = go.Figure()

    # Area under curve
    fig.add_trace(go.Scatter(
        x=dates, y=scores,
        mode="lines",
        line=dict(color="#0071e3", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0, 113, 227, 0.08)",
        name="Sentiment (MA)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Score: %{y:.3f}<extra></extra>",
    ))

    # Colour-coded bar overlay
    fig.add_trace(go.Bar(
        x=dates, y=scores,
        marker_color=bar_colors,
        name="Monthly Score",
        opacity=0.5,
        hovertemplate="<b>%{x|%b %Y}</b><br>Score: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.3)", line_width=1)

    fig.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(
            title="Date",
            showgrid=False,
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        yaxis=dict(
            title="Sentiment Score",
            range=[-1.1, 1.1],
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        bargap=0,
        barmode="overlay",
        showlegend=True,
    )
    return fig


# ── Page layout ───────────────────────────────────────────────────────────────

def layout() -> html.Div:
    # ── Stock watchlist KPI cards
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
            ], className="metric-tile", style={"borderTop": f"3px solid {item['color']}"}),
        )

    # ── CME futures table rows
    fut_rows = []
    for f in FUTURES_WATCHLIST:
        color = COLORS["green"] if f["chg"] >= 0 else COLORS["red"]
        arrow = "▲" if f["chg"] >= 0 else "▼"
        fut_rows.append(html.Tr([
            html.Td(html.Span(f["sym"], className="tag tag-blue")),
            html.Td(f["label"], style={"color": "#6e6e73"}),
            html.Td(f"${f['price']:,.2f}", style={"fontWeight": "600"}),
            html.Td(f"{arrow} {abs(f['chg']):.2f}%",
                    style={"color": color, "fontWeight": "600"}),
        ]))

    return html.Div([
        page_header(
            "Market Overview",
            "Watch live-style quotes, compare performance, monitor key futures, "
            "and explore news intelligence across all asset classes.",
            badge="Today",
        ),

        # Stock KPI grid
        html.Div(cards, className="metrics-grid"),

        # Performance + futures snapshot charts
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

        # CME futures table
        section_card(
            "CME Futures Watchlist",
            "Key contract prices — paper trading simulation",
            data_table(
                ["Symbol", "Contract", "Last Price", "Day Change"],
                fut_rows,
            ),
        ),

        # ── News Intelligence Panel ───────────────────────────────────────────
        html.Div([
            # Section header row
            html.Div([
                html.Div([
                    html.H3("Futures News Intelligence", className="section-title"),
                    html.P(
                        "Live headlines, fundamental data, and historical sentiment "
                        "(2000 – present) for any futures market. Click an asset to load.",
                        className="section-subtitle",
                    ),
                ], className="news-header-text"),
                html.Div([
                    html.Button(
                        [html.Span("↺", style={"marginRight": "6px"}), "Refresh"],
                        id="news-refresh-btn",
                        className="news-refresh-btn",
                        n_clicks=0,
                    ),
                    html.Span("", id="news-refresh-status", className="news-refresh-status"),
                ], className="news-header-actions"),
            ], className="news-section-header"),

            # Asset selector pill-bar
            _asset_selector(),

            # Selected asset title bar
            html.Div(id="news-asset-title", className="news-asset-title-bar"),

            # Tabs: Headlines / Fundamentals / Sentiment
            dcc.Tabs(
                id="news-tabs",
                value="headlines",
                className="news-tabs",
                children=[
                    dcc.Tab(label="📰 Headlines",    value="headlines",    className="news-tab", selected_className="news-tab-selected"),
                    dcc.Tab(label="📊 Fundamentals", value="fundamentals", className="news-tab", selected_className="news-tab-selected"),
                    dcc.Tab(label="🧠 Sentiment",    value="sentiment",    className="news-tab", selected_className="news-tab-selected"),
                ],
            ),

            # Tab content area
            html.Div(id="news-tab-content", className="news-tab-body",
                     children=[_empty_news_panel()]),

        ], className="section-card news-intelligence-panel"),

        # State stores
        dcc.Store(id="selected-asset-store", data=None),
        dcc.Store(id="news-data-store",      data=None),
        dcc.Store(id="fund-data-store",      data=None),
        dcc.Store(id="sentiment-data-store", data=None),

        # Refresh interval (60 s)
        dcc.Interval(id="dashboard-interval", interval=60_000, n_intervals=0),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("selected-asset-store", "data"),
    [Input({"type": "asset-btn", "index": key}, "n_clicks") for key in _ASSET_ORDER],
    prevent_initial_call=True,
)
def _select_asset(*n_clicks_tuple):
    """Detect which asset button was clicked and store the asset key."""
    from dash import ctx
    if not ctx.triggered_id:
        return no_update
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "asset-btn":
        return triggered["index"]
    return no_update


@callback(
    Output("news-data-store",      "data"),
    Output("fund-data-store",      "data"),
    Output("sentiment-data-store", "data"),
    Output("news-asset-title",     "children"),
    Output("news-refresh-status",  "children"),
    Input("selected-asset-store",  "data"),
    Input("news-refresh-btn",      "n_clicks"),
    State("selected-asset-store",  "data"),
    prevent_initial_call=True,
)
def _load_asset_data(asset_from_store, _refresh_clicks, asset_from_state):
    """
    Fetch news, fundamentals, and sentiment for the selected asset.
    Triggered either by asset selection or the Refresh button.
    Invalidates cache on refresh.
    """
    from dash import ctx

    # Determine which input triggered
    triggered = ctx.triggered_id if ctx.triggered_id else ""
    asset_key = asset_from_store

    if triggered == "news-refresh-btn":
        asset_key = asset_from_state
        if asset_key:
            invalidate_cache(asset_key)

    if not asset_key or asset_key not in FUTURES_ASSETS:
        return no_update, no_update, no_update, no_update, no_update

    meta = FUTURES_ASSETS[asset_key]

    try:
        news_items = fetch_asset_news(asset_key, limit=50)
        news_dicts = news_items_to_dicts(news_items)
    except Exception as exc:
        logger.error("News fetch failed for %s: %s", asset_key, exc)
        news_dicts = []

    try:
        snap = fetch_asset_fundamentals(asset_key)
        fund_dict = fundamentals_to_dict(snap)
    except Exception as exc:
        logger.error("Fundamentals fetch failed for %s: %s", asset_key, exc)
        fund_dict = {}

    try:
        df_sent = fetch_sentiment_history(asset_key, start="2000-01-01")
        sent_rows = df_sent.to_dict(orient="records")
    except Exception as exc:
        logger.error("Sentiment fetch failed for %s: %s", asset_key, exc)
        sent_rows = []

    # Title bar
    color = meta["color"]
    title_bar = html.Div([
        html.Span(meta["icon"], className="news-asset-icon",
                  style={"backgroundColor": color + "20", "color": color}),
        html.Div([
            html.Span(meta["name"], className="news-asset-name"),
            html.Span(meta["full_name"], className="news-asset-full-name"),
        ], className="news-asset-text"),
        html.Div([
            html.Span(f"{len(news_dicts)} articles", className="news-article-count"),
            html.Span(meta["sector"], className="news-sector-tag",
                      style={"backgroundColor": color + "15", "color": color}),
        ], className="news-asset-badges"),
    ], className="news-asset-title-inner")

    status = f"Updated · {len(news_dicts)} headlines loaded"
    return news_dicts, fund_dict, sent_rows, title_bar, status


@callback(
    Output("news-tab-content", "children"),
    Input("news-tabs",          "value"),
    Input("news-data-store",    "data"),
    Input("fund-data-store",    "data"),
    Input("sentiment-data-store", "data"),
    State("selected-asset-store", "data"),
    prevent_initial_call=True,
)
def _render_tab_content(active_tab, news_dicts, fund_dict, sent_rows, asset_key):
    """Render the correct content panel when tab or data changes."""
    if not asset_key:
        return _empty_news_panel()

    if active_tab == "headlines":
        if not news_dicts:
            return html.Div(
                html.P("Loading headlines… click Refresh if data does not appear.",
                       className="no-data-msg"),
                className="news-list",
            )
        return _render_headlines(news_dicts or [])

    if active_tab == "fundamentals":
        if not fund_dict:
            return html.Div(
                html.P("Loading fundamentals…", className="no-data-msg"),
            )
        return _render_fundamentals(fund_dict)

    if active_tab == "sentiment":
        fig = _render_sentiment_chart(sent_rows or [])
        count_bull = sum(1 for r in (sent_rows or []) if r.get("Label") == "Bullish")
        count_bear = sum(1 for r in (sent_rows or []) if r.get("Label") == "Bearish")
        count_neut = sum(1 for r in (sent_rows or []) if r.get("Label") == "Neutral")
        total      = len(sent_rows or [])
        meta = FUTURES_ASSETS.get(asset_key, {})
        color = meta.get("color", "#0071e3")
        return html.Div([
            html.Div([
                html.Div([
                    html.Span("Bullish Months", className="sent-stat-label"),
                    html.Span(
                        f"{count_bull} / {total}",
                        className="sent-stat-value",
                        style={"color": "#34c759"},
                    ),
                ], className="sent-stat-cell"),
                html.Div([
                    html.Span("Bearish Months", className="sent-stat-label"),
                    html.Span(
                        f"{count_bear} / {total}",
                        className="sent-stat-value",
                        style={"color": "#ff3b30"},
                    ),
                ], className="sent-stat-cell"),
                html.Div([
                    html.Span("Neutral Months", className="sent-stat-label"),
                    html.Span(
                        f"{count_neut} / {total}",
                        className="sent-stat-value",
                        style={"color": "#86868b"},
                    ),
                ], className="sent-stat-cell"),
                html.Div([
                    html.Span("Data From", className="sent-stat-label"),
                    html.Span("2000 – Present", className="sent-stat-value"),
                ], className="sent-stat-cell"),
            ], className="sent-stats-row"),
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": True, "displaylogo": False},
                style={"marginTop": "16px"},
            ),
            html.P(
                "Historical sentiment is price-momentum-implied (monthly z-score of returns) "
                "for the full archive. Recent months overlay VADER-scored news headline scores.",
                className="sent-footnote",
            ),
        ])

    return _empty_news_panel()


@callback(
    [Output({"type": "asset-btn", "index": key}, "className") for key in _ASSET_ORDER],
    Input("selected-asset-store", "data"),
    prevent_initial_call=False,
)
def _highlight_active_btn(asset_key):
    """Add 'active' class to the selected asset button."""
    return [
        "asset-selector-btn active" if key == asset_key else "asset-selector-btn"
        for key in _ASSET_ORDER
    ]
