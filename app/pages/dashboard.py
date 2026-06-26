"""
Market Overview Dashboard — home page with live-style market summary,
futures intelligence (news, fundamentals, sentiment), and watchlist metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from app.components import data_table, metric_tile, page_header, section_card
from app.data_service import fetch_stock_history, get_futures_intelligence
from app.theme import COLORS


WATCHLIST = [
    {"ticker": "TSLA", "label": "Tesla", "color": COLORS["blue"]},
    {"ticker": "GME", "label": "GameStop", "color": COLORS["green"]},
    {"ticker": "AAPL", "label": "Apple", "color": COLORS["gold"]},
    {"ticker": "SPY", "label": "S&P 500 ETF", "color": COLORS["purple"]},
]

SPARKLINE_FILL = {
    COLORS["blue"]: "rgba(0, 113, 227, 0.08)",
    COLORS["green"]: "rgba(52, 199, 89, 0.08)",
    COLORS["gold"]: "rgba(255, 159, 10, 0.08)",
    COLORS["purple"]: "rgba(175, 82, 222, 0.08)",
}


def _mini_sparkline(ticker: str, color: str) -> go.Figure:
    df = fetch_stock_history(ticker, period="3mo")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=SPARKLINE_FILL.get(color, "rgba(0,0,0,0.05)"),
            hoverinfo="skip",
        )
    )
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
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=norm,
                name=item["label"],
                mode="lines",
                line=dict(color=item["color"], width=2),
                hovertemplate=f"<b>{item['label']}</b><br>%{{y:+.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        title="",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_ticksuffix="%",
        hovermode="x unified",
    )
    return fig


def _futures_heatmap(intel: Dict[str, Dict[str, Any]]) -> go.Figure:
    if not intel:
        return go.Figure()

    labels = [intel[s]["label"] for s in intel]
    changes = [intel[s]["day_change_pct"] for s in intel]
    colors = [COLORS["green"] if c >= 0 else COLORS["red"] for c in changes]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=changes,
            marker_color=colors,
            text=[f"{c:+.2f}%" for c in changes],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Change: %{y:+.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="",
        height=260,
        yaxis_ticksuffix="%",
        showlegend=False,
        bargap=0.35,
    )
    return fig


def _sentiment_timeseries_chart(sentiment: List[Dict[str, Any]]) -> go.Figure:
    series = sentiment[-24:] if sentiment else []
    x_vals = [point.get("date", "")[:7] for point in series]
    y_vals = [float(point.get("score", 0.0)) for point in series]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            line=dict(color=COLORS["blue"], width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(0, 113, 227, 0.06)",
            hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="",
        height=220,
        margin=dict(l=10, r=10, t=8, b=24),
        yaxis=dict(range=[-1.05, 1.05], zeroline=True, zerolinecolor="#d2d2d7"),
        xaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def _render_futures_intelligence_panel(asset: Dict[str, Any]) -> html.Div:
    news = asset.get("news", [])
    fundamentals = asset.get("fundamentals", [])
    sentiment = asset.get("sentiment", [])

    first_fund = fundamentals[0] if fundamentals else {}
    latest_fund = fundamentals[-1] if fundamentals else {}
    first_price = float(first_fund.get("avg_price", 0.0) or 0.0)
    latest_price = float(asset.get("last_price", latest_fund.get("avg_price", 0.0)) or 0.0)
    since_start_pct = ((latest_price / first_price) - 1.0) * 100.0 if first_price else 0.0
    latest_sent = sentiment[-1] if sentiment else {}
    latest_sent_score = float(latest_sent.get("score", 0.0) or 0.0)
    sent_cls = "positive" if latest_sent_score > 0.1 else ("negative" if latest_sent_score < -0.1 else "neutral")

    metric_specs = [
        ("Current Close", f"{latest_price:,.2f}", "neutral"),
        ("Since Start", f"{since_start_pct:+.2f}%", "neutral"),
        ("Headlines", str(len(news)), "neutral"),
        ("Sentiment", f"{latest_sent_score:+.3f}", sent_cls),
    ]

    headline_rows = []
    for item in news[:12]:
        headline_rows.append(
            html.Li(
                [
                    html.A(
                        item.get("headline", "Untitled headline"),
                        href=item.get("url") or "#",
                        target="_blank",
                        rel="noopener noreferrer",
                        style={"fontWeight": "600"},
                    ),
                    html.Div(
                        f"{item.get('source', 'Feed')} · "
                        f"{item.get('published_at', '')[:10]} · "
                        f"{float(item.get('sentiment_score', 0.0)):+.2f}",
                        style={"fontSize": "12px", "color": "#8e8e93", "marginTop": "4px"},
                    ),
                ],
                style={"marginBottom": "12px"},
            )
        )
    if not headline_rows:
        headline_rows = [html.Li("No headlines available.")]

    fundamental_rows = []
    for point in reversed(fundamentals[-12:]):
        fundamental_rows.append(
            html.Tr(
                [
                    html.Td(point.get("date", "")[:7]),
                    html.Td(f"${float(point.get('avg_price', 0.0)):,.2f}"),
                    html.Td(f"{float(point.get('avg_volume', 0.0)):,.0f}"),
                    html.Td(f"{float(point.get('volatility', 0.0)):.4f}"),
                    html.Td(f"${float(point.get('high', 0.0)):,.2f}"),
                    html.Td(f"${float(point.get('low', 0.0)):,.2f}"),
                ]
            )
        )

    return html.Div(
        [
            html.Div(
                [html.Div(metric_tile(label, value, value_cls=value_cls), className="metric-tile") for label, value, value_cls in metric_specs],
                className="metrics-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Latest Headlines", style={"marginBottom": "10px"}),
                            html.Ul(headline_rows, style={"paddingLeft": "18px", "margin": 0}),
                        ],
                        style={"padding": "6px 0"},
                    ),
                    html.Div(
                        [
                            html.H4("Sentiment (2000 → Current)", style={"marginBottom": "10px"}),
                            dcc.Graph(
                                figure=_sentiment_timeseries_chart(sentiment),
                                config={"displayModeBar": False},
                            ),
                        ]
                    ),
                ],
                className="charts-grid-2-1",
            ),
            section_card(
                "Fundamentals",
                "Monthly history from configured start date",
                data_table(
                    ["Month", "Avg Price", "Avg Volume", "Volatility", "High", "Low"],
                    fundamental_rows,
                ),
            ),
        ]
    )


def layout() -> html.Div:
    cards = []
    for item in WATCHLIST:
        df = fetch_stock_history(item["ticker"], period="5d")
        if df.empty or len(df) < 2:
            price, chg, pct = 0.0, 0.0, 0.0
        else:
            price = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            chg = price - prev
            pct = (chg / prev) * 100 if prev else 0.0

        sign_cls = "positive" if chg >= 0 else "negative"
        arrow = "▲" if chg >= 0 else "▼"
        cards.append(
            html.Div(
                [
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
                ],
                className="metric-tile",
                style={"borderTop": f"3px solid {item['color']}"},
            )
        )

    return html.Div(
        [
            page_header(
                "Market Overview",
                "Watch live quotes, explore futures intelligence, and monitor news, fundamentals, and sentiment.",
                badge="Live",
            ),
            html.Div(id="intel-status", className="intel-status"),
            html.Div(cards, className="metrics-grid"),
            section_card(
                "Futures Market Intelligence",
                "Select a contract tab to view headlines, fundamentals, and sentiment (2000–present). Data refreshes on load.",
                html.Div(
                    [
                        dcc.Tabs(id="overview-futures-intel-tabs", value="GC", children=[]),
                        dcc.Loading(
                            html.Div(id="overview-futures-intel-body", style={"marginTop": "16px"}),
                            type="dot",
                        ),
                    ]
                ),
            ),
            html.Div(
                [
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
                ],
                className="charts-grid-2-1",
            ),
            section_card(
                "CME Futures Watchlist",
                "Live contract prices with intelligence data",
                html.Div(id="futures-watchlist-table"),
            ),
            dcc.Store(id="futures-intel-store"),
            dcc.Interval(id="dashboard-interval", interval=300_000, n_intervals=0),
        ]
    )


@callback(
    Output("futures-intel-store", "data"),
    Output("intel-status", "children"),
    Output("dashboard-futures-bar", "figure"),
    Output("futures-watchlist-table", "children"),
    Output("overview-futures-intel-tabs", "children"),
    Output("overview-futures-intel-tabs", "value"),
    Input("dashboard-interval", "n_intervals"),
    State("overview-futures-intel-tabs", "value"),
    prevent_initial_call=False,
)
def refresh_intelligence(
    n_intervals: int,
    selected_tab: str | None,
):
    force = n_intervals == 0
    intel = get_futures_intelligence(force_refresh=force)

    if not intel:
        status = html.Span(
            [html.Span(className="status-dot"), " Intelligence data unavailable"],
            className="status-pill",
        )
        return {}, status, go.Figure(), html.Div("No futures data available."), [], selected_tab

    status = html.Span(
        [
            html.Span(className="status-dot live"),
            f" Intelligence data loaded for {len(intel)} assets",
            " — refreshed on load" if force else " — auto-refresh",
        ],
        className="status-pill status-pill-live",
    )

    rows = []
    for sym, data in intel.items():
        chg = float(data.get("day_change_pct", 0.0))
        color = COLORS["green"] if chg >= 0 else COLORS["red"]
        arrow = "▲" if chg >= 0 else "▼"
        rows.append(
            html.Tr(
                [
                    html.Td(html.Span(sym, className="tag tag-blue")),
                    html.Td(data.get("label", sym), style={"color": "#6e6e73"}),
                    html.Td(f"${float(data.get('last_price', 0.0)):,.2f}", style={"fontWeight": "600"}),
                    html.Td(f"{arrow} {abs(chg):.2f}%", style={"color": color, "fontWeight": "600"}),
                    html.Td(str(len(data.get("news", [])))),
                ]
            )
        )

    table = data_table(
        ["Symbol", "Contract", "Last Price", "Day Change", "Headlines"],
        rows,
    )
    tabs = [dcc.Tab(label=f"{sym} — {data.get('label', sym)}", value=sym) for sym, data in intel.items()]
    next_selected = selected_tab if selected_tab in intel else next(iter(intel))

    return intel, status, _futures_heatmap(intel), table, tabs, next_selected


@callback(
    Output("overview-futures-intel-body", "children"),
    Input("overview-futures-intel-tabs", "value"),
    Input("futures-intel-store", "data"),
)
def render_futures_intelligence(
    symbol: str | None,
    intel: Dict[str, Dict[str, Any]] | None,
):
    if not intel:
        return html.Div("No intelligence data loaded yet.")

    selected_symbol = symbol if symbol in intel else next(iter(intel), None)
    if selected_symbol is None:
        return html.Div("No assets available.")
    return _render_futures_intelligence_panel(intel[selected_symbol])
