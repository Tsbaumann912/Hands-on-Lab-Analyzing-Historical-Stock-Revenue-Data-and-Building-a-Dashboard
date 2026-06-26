"""
Stock Research page — equity analysis integrated from the Final Assignment notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from app.components import control_group, metric_tile, page_header, section_card
from app.styles import DROPDOWN_STYLE
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
                line=dict(color=color_ma, width=1.5, dash="dot"),
            ), row=1, col=1)

    colors_vol = [
        COLORS["green"] if df["Close"].iloc[i] >= df["Open"].iloc[i] else COLORS["red"]
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"],
        name="Volume", marker_color=colors_vol,
        opacity=0.5,
    ), row=2, col=1)

    fig.update_layout(
        title="",
        height=480,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.04),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def _revenue_vs_price_chart(ticker: str) -> go.Figure:
    stock_df = fetch_stock_history(ticker, period="5y")
    rev_df = fetch_revenue_data(ticker)

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
            line=dict(color=COLORS["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 113, 227, 0.06)",
        ), row=1, col=1)

    if not rev_df.empty:
        fig.add_trace(go.Bar(
            x=rev_df["Date"], y=rev_df["Revenue"],
            name="Revenue ($M)",
            marker_color=COLORS["green"],
            opacity=0.8,
        ), row=2, col=1)

    fig.update_layout(
        height=520,
        title="",
        showlegend=True,
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($M)", row=2, col=1)
    return fig


def _metrics_row(ticker: str) -> list:
    df = fetch_stock_history(ticker, period="1y")
    if df.empty or len(df) < 2:
        return []

    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]
    high = df["High"].max()
    low = df["Low"].min()
    avg_vol = df["Volume"].mean()
    ytd_ret = (last / df["Close"].iloc[0] - 1) * 100
    chg = last - prev
    chg_cls = "positive" if chg >= 0 else "negative"
    ytd_cls = "positive" if ytd_ret >= 0 else "negative"

    metrics = [
        ("Last Price", f"${last:,.2f}", "neutral", None),
        ("Day Change", f"{'+' if chg >= 0 else ''}{chg:.2f}", chg_cls, None),
        ("52-Wk High", f"${high:,.2f}", "positive", None),
        ("52-Wk Low", f"${low:,.2f}", "negative", None),
        ("Avg Volume", f"{avg_vol/1e6:.1f}M", "neutral", None),
        ("YTD Return", f"{ytd_ret:+.1f}%", ytd_cls, None),
    ]

    return [
        html.Div(
            metric_tile(label, value, value_cls=cls),
            className="metric-tile",
        )
        for label, value, cls, _ in metrics
    ]


def layout() -> html.Div:
    return html.Div([
        page_header(
            "Stock Research",
            "Deep-dive equity analysis with price charts and quarterly revenue — built from your assignment notebook.",
            badge="Research",
        ),

        html.Div([
            control_group(
                "Ticker",
                dcc.Dropdown(
                    id="stock-ticker-dropdown",
                    options=[{"label": t, "value": t} for t in PRESET_TICKERS],
                    value="TSLA",
                    clearable=False,
                    className="dash-dropdown",
                    style=DROPDOWN_STYLE,
                ),
            ),
            control_group(
                "Period",
                dcc.Dropdown(
                    id="stock-period-dropdown",
                    options=[
                        {"label": "3 Months", "value": "3mo"},
                        {"label": "6 Months", "value": "6mo"},
                        {"label": "1 Year", "value": "1y"},
                        {"label": "2 Years", "value": "2y"},
                        {"label": "5 Years", "value": "5y"},
                        {"label": "Max", "value": "max"},
                    ],
                    value="2y",
                    clearable=False,
                    className="dash-dropdown",
                    style=DROPDOWN_STYLE,
                ),
            ),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "maxWidth": "480px", "marginBottom": "24px"}),

        html.Div(id="stock-metrics-row", className="metrics-grid"),

        section_card(
            "Price Chart",
            "Candlestick with EMA overlays and volume",
            dcc.Loading(dcc.Graph(
                id="stock-candlestick",
                config={"displayModeBar": True, "displaylogo": False},
            )),
        ),

        section_card(
            "Revenue vs Share Price",
            "Quarterly revenue from earnings reports overlaid with share price history",
            dcc.Loading(dcc.Graph(
                id="stock-revenue-chart",
                config={"displayModeBar": True, "displaylogo": False},
            )),
        ),
    ])


@callback(
    Output("stock-candlestick", "figure"),
    Output("stock-revenue-chart", "figure"),
    Output("stock-metrics-row", "children"),
    Input("stock-ticker-dropdown", "value"),
    Input("stock-period-dropdown", "value"),
)
def update_stock_charts(ticker: str, period: str):
    if not ticker:
        return go.Figure(), go.Figure(), []
    return (
        _candlestick_chart(ticker, period),
        _revenue_vs_price_chart(ticker),
        _metrics_row(ticker),
    )
