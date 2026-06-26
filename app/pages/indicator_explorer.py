"""
Indicator Explorer — interactive panel showing all computed indicators
for any futures contract with full drill-down charts.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from app.theme import COLORS
from app.data_service import get_synthetic_futures_bars, compute_indicators_for_ui


CONTRACTS = ["ES", "NQ", "CL", "GC", "ZN", "SI", "NG", "ZB"]


def _full_indicator_dashboard(contract: str, n_bars: int) -> go.Figure:
    df   = get_synthetic_futures_bars(contract, n=n_bars)
    inds = compute_indicators_for_ui(df)
    dates = df["Date"]
    close = df["Close"]

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.35, 0.14, 0.14, 0.14, 0.14],
        subplot_titles=[
            "Price + Bollinger Bands + EMA 20/50",
            "RSI (14)",
            "MACD",
            "ATR (14)",
            "OBV",
        ],
    )

    # ── Row 1: Price + BB + EMAs ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=close, name="Close",
        mode="lines", line=dict(color=COLORS["text"], width=1.5),
    ), row=1, col=1)

    for key, color, name in [("BB_upper", "#64748b", "BB Upper"), ("BB_lower", "#64748b", "BB Lower")]:
        if key in inds:
            fig.add_trace(go.Scatter(
                x=dates, y=inds[key], name=name, mode="lines",
                line=dict(color=color, width=0.7, dash="dot"),
                showlegend=(name == "BB Upper"),
            ), row=1, col=1)
    if "BB_upper" in inds and "BB_lower" in inds:
        fig.add_trace(go.Scatter(
            x=dates, y=inds["BB_lower"], name="BB Band",
            mode="lines", line=dict(color=COLORS["muted"], width=0),
            fill="tonexty", fillcolor="rgba(100,116,139,0.07)",
            showlegend=False,
        ), row=1, col=1)
    if "BB_middle" in inds:
        fig.add_trace(go.Scatter(
            x=dates, y=inds["BB_middle"], name="BB Mid",
            mode="lines", line=dict(color=COLORS["muted"], width=0.8),
        ), row=1, col=1)

    for key, color, name in [("EMA_20", COLORS["blue"], "EMA 20"), ("SMA_50", COLORS["gold"], "SMA 50")]:
        if key in inds:
            fig.add_trace(go.Scatter(
                x=dates, y=inds[key], name=name,
                mode="lines", line=dict(color=color, width=1.2, dash="dot"),
            ), row=1, col=1)

    # ── Row 2: RSI ────────────────────────────────────────────────────────
    if "RSI_14" in inds:
        rsi = inds["RSI_14"]
        fig.add_trace(go.Scatter(
            x=dates, y=rsi, name="RSI (14)",
            mode="lines", line=dict(color=COLORS["blue"], width=1.3),
        ), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.05)",
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(16,185,129,0.05)",
                      line_width=0, row=2, col=1)
        fig.add_hline(y=70, line=dict(color=COLORS["red"],   width=0.7, dash="dot"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color=COLORS["green"], width=0.7, dash="dot"), row=2, col=1)
        fig.add_hline(y=50, line=dict(color=COLORS["muted"], width=0.5, dash="dot"), row=2, col=1)

    # ── Row 3: MACD ───────────────────────────────────────────────────────
    if "MACD_line" in inds:
        fig.add_trace(go.Scatter(
            x=dates, y=inds["MACD_line"], name="MACD",
            mode="lines", line=dict(color=COLORS["blue"], width=1.3),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=inds["MACD_signal"], name="Signal",
            mode="lines", line=dict(color=COLORS["gold"], width=1.1),
        ), row=3, col=1)
        hist = inds.get("MACD_hist", np.zeros(len(dates)))
        bar_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in hist]
        fig.add_trace(go.Bar(
            x=dates, y=hist, name="Histogram",
            marker_color=bar_colors, opacity=0.55,
        ), row=3, col=1)

    # ── Row 4: ATR ────────────────────────────────────────────────────────
    if "ATR_14" in inds:
        fig.add_trace(go.Scatter(
            x=dates, y=inds["ATR_14"], name="ATR (14)",
            mode="lines", line=dict(color=COLORS["orange"], width=1.3),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.06)",
        ), row=4, col=1)

    # ── Row 5: OBV ────────────────────────────────────────────────────────
    if "OBV" in inds:
        fig.add_trace(go.Scatter(
            x=dates, y=inds["OBV"], name="OBV",
            mode="lines", line=dict(color=COLORS["cyan"], width=1.3),
        ), row=5, col=1)

    fig.update_layout(
        height=780,
        title=f"{contract} — Full Indicator Suite",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.03, x=0, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H2("Indicator Explorer"),
            html.P("Visualise all computed indicators from the quant terminal library on a single synchronised chart"),
        ], className="page-header"),

        dbc.Row([
            dbc.Col([
                html.Div("Contract", className="form-label-dark"),
                dcc.Dropdown(
                    id="ie-contract",
                    options=[{"value": c, "label": c} for c in CONTRACTS],
                    value="ES",
                    clearable=False,
                    style={"backgroundColor": "#111827", "border": "1px solid #1e293b"},
                ),
            ], md=2),

            dbc.Col([
                html.Div("Bars", className="form-label-dark"),
                dcc.Slider(
                    id="ie-nbars",
                    min=100, max=800, step=100, value=300,
                    marks={100: "100", 300: "300", 500: "500", 800: "800"},
                    tooltip={"placement": "bottom"},
                ),
            ], md=4, style={"paddingTop": "8px"}),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Multi-Panel Indicator Dashboard"),
                html.P("Price · RSI · MACD · ATR · OBV — all synchronised on the same time axis",
                       className="chart-subtitle"),
                dcc.Loading(dcc.Graph(
                    id="ie-chart",
                    config={"displayModeBar": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                )),
            ], className="chart-card"), md=12),
        ]),
    ])


@callback(
    Output("ie-chart", "figure"),
    Input("ie-contract", "value"),
    Input("ie-nbars",    "value"),
)
def update_ie(contract, n_bars):
    return _full_indicator_dashboard(contract or "ES", n_bars or 300)
