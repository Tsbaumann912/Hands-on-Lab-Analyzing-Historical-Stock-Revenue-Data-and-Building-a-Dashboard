"""
Stock Research page — integrates the Final Assignment notebook analysis.

Shows Tesla and GameStop (and any ticker):
  • Interactive OHLCV candlestick chart
  • Revenue vs Share Price dual-axis chart (replicating make_graph)
  • Key financial metrics table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from app.theme import COLORS
from app.data_service import fetch_stock_history, fetch_revenue_data


PRESET_TICKERS = ["TSLA", "GME", "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL"]


def _candlestick_chart(ticker: str, period: str = "2y") -> go.Figure:
    df = fetch_stock_history(ticker, period=period)
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=ticker,
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        increasing_fillcolor=COLORS["green"],
        decreasing_fillcolor=COLORS["red"],
    ), row=1, col=1)

    # 20-day and 50-day EMA overlays
    close = df["Close"].to_numpy(dtype=np.float64)
    for period_ma, color_ma, label_ma in [(20, COLORS["blue"], "EMA 20"), (50, COLORS["gold"], "EMA 50")]:
        if len(close) >= period_ma:
            alpha = 2.0 / (period_ma + 1)
            ema = np.full(len(close), np.nan)
            ema[period_ma - 1] = close[:period_ma].mean()
            for i in range(period_ma, len(close)):
                ema[i] = ema[i - 1] + alpha * (close[i] - ema[i - 1])
            fig.add_trace(go.Scatter(
                x=df["Date"], y=ema,
                name=label_ma, mode="lines",
                line=dict(color=color_ma, width=1.2, dash="dot"),
            ), row=1, col=1)

    # Volume bars
    colors_vol = [COLORS["green"] if df["Close"].iloc[i] >= df["Open"].iloc[i] else COLORS["red"]
                  for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"],
        name="Volume", marker_color=colors_vol,
        opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Price & Volume",
        height=480,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.04),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def _revenue_vs_price_chart(ticker: str) -> go.Figure:
    """Dual-panel share price + quarterly revenue — replicates notebook make_graph."""
    stock_df = fetch_stock_history(ticker, period="5y")
    rev_df   = fetch_revenue_data(ticker)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Historical Share Price", "Quarterly Revenue ($M)"),
        vertical_spacing=0.25,
    )

    if not stock_df.empty:
        fig.add_trace(go.Scatter(
            x=stock_df["Date"], y=stock_df["Close"],
            name="Share Price",
            mode="lines",
            line=dict(color=COLORS["blue"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.05)",
        ), row=1, col=1)

    if not rev_df.empty:
        fig.add_trace(go.Bar(
            x=rev_df["Date"], y=rev_df["Revenue"],
            name="Revenue ($M)",
            marker_color=COLORS["green"],
            opacity=0.75,
        ), row=2, col=1)

    fig.update_layout(
        height=520,
        title=f"{ticker} — Share Price & Quarterly Revenue",
        showlegend=True,
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($M)", row=2, col=1)
    return fig


def _metrics_table(ticker: str) -> list:
    """Return a list of dbc.Row metric cards for the selected ticker."""
    df = fetch_stock_history(ticker, period="1y")
    if df.empty or len(df) < 2:
        return []

    last  = df["Close"].iloc[-1]
    prev  = df["Close"].iloc[-2]
    high  = df["High"].max()
    low   = df["Low"].min()
    avg_vol = df["Volume"].mean()
    ytd_ret = (last / df["Close"].iloc[0] - 1) * 100

    def card(label: str, value: str, color_cls: str = "neutral") -> dbc.Col:
        return dbc.Col(html.Div([
            html.Div(label, className="metric-label"),
            html.Div(value, className=f"metric-value {color_cls}"),
        ], className="metric-card"), md=2, sm=4, xs=6, className="mb-3")

    chg = last - prev
    chg_cls = "positive" if chg >= 0 else "negative"
    ytd_cls = "positive" if ytd_ret >= 0 else "negative"

    return [
        card("Last Price",    f"${last:,.2f}",  "neutral"),
        card("Day Change",    f"{'+' if chg >= 0 else ''}{chg:.2f}", chg_cls),
        card("52-Wk High",   f"${high:,.2f}",  "positive"),
        card("52-Wk Low",    f"${low:,.2f}",   "negative"),
        card("Avg Volume",   f"{avg_vol/1e6:.1f}M", "neutral"),
        card("YTD Return",   f"{ytd_ret:+.1f}%", ytd_cls),
    ]


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H2("Stock Research"),
            html.P("Equity analysis, revenue tracking, and price charting — integrated from the Final Assignment notebook"),
        ], className="page-header"),

        # Ticker selector
        dbc.Row([
            dbc.Col([
                html.Div("Select Ticker", className="form-label-dark"),
                dcc.Dropdown(
                    id="stock-ticker-dropdown",
                    options=[{"label": t, "value": t} for t in PRESET_TICKERS],
                    value="TSLA",
                    clearable=False,
                    style={"backgroundColor": "#111827", "color": "#f1f5f9", "border": "1px solid #1e293b"},
                ),
            ], md=3),
            dbc.Col([
                html.Div("Period", className="form-label-dark"),
                dcc.Dropdown(
                    id="stock-period-dropdown",
                    options=[
                        {"label": "3 Months", "value": "3mo"},
                        {"label": "6 Months", "value": "6mo"},
                        {"label": "1 Year",   "value": "1y"},
                        {"label": "2 Years",  "value": "2y"},
                        {"label": "5 Years",  "value": "5y"},
                        {"label": "Max",      "value": "max"},
                    ],
                    value="2y",
                    clearable=False,
                    style={"backgroundColor": "#111827", "color": "#f1f5f9", "border": "1px solid #1e293b"},
                ),
            ], md=2),
        ], className="mb-4"),

        # Metrics row
        dbc.Row(id="stock-metrics-row", className="mb-2"),

        # Candlestick chart
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Price Chart"),
                html.P("Candlestick with EMA overlays and volume", className="chart-subtitle"),
                dcc.Loading(dcc.Graph(
                    id="stock-candlestick",
                    config={"displayModeBar": True, "displaylogo": False},
                )),
            ], className="chart-card"), md=12),
        ]),

        # Revenue vs price
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Revenue vs Share Price"),
                html.P(
                    "Quarterly revenue (from earnings reports) overlaid with share price — "
                    "replicates the IBM Skills Network Final Assignment dashboard",
                    className="chart-subtitle",
                ),
                dcc.Loading(dcc.Graph(
                    id="stock-revenue-chart",
                    config={"displayModeBar": True, "displaylogo": False},
                )),
            ], className="chart-card"), md=12),
        ]),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("stock-candlestick",   "figure"),
    Output("stock-revenue-chart", "figure"),
    Output("stock-metrics-row",   "children"),
    Input("stock-ticker-dropdown", "value"),
    Input("stock-period-dropdown", "value"),
)
def update_stock_charts(ticker: str, period: str):
    if not ticker:
        return go.Figure(), go.Figure(), []
    return (
        _candlestick_chart(ticker, period),
        _revenue_vs_price_chart(ticker),
        _metrics_table(ticker),
    )
