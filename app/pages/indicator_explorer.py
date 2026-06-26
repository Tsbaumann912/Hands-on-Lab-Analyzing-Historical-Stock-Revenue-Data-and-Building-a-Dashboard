"""
Indicator Explorer — interactive panel showing all computed indicators
for any futures contract with full drill-down charts.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, callback

from app.components import control_group, page_header, section_card
from app.styles import DROPDOWN_STYLE
from app.theme import COLORS
from app.data_service import get_synthetic_futures_bars, compute_indicators_for_ui


CONTRACTS = ["ES", "NQ", "CL", "GC", "ZN", "SI", "NG", "ZB"]


def _full_indicator_dashboard(contract: str, n_bars: int) -> go.Figure:
    df = get_synthetic_futures_bars(contract, n=n_bars)
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

    fig.add_trace(go.Scatter(
        x=dates, y=close, name="Close",
        mode="lines", line=dict(color=COLORS["text"], width=1.5),
    ), row=1, col=1)

    for key, color, name in [("BB_upper", "#86868b", "BB Upper"), ("BB_lower", "#86868b", "BB Lower")]:
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
            fill="tonexty", fillcolor="rgba(134,134,139,0.06)",
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

    if "RSI_14" in inds:
        rsi = inds["RSI_14"]
        fig.add_trace(go.Scatter(
            x=dates, y=rsi, name="RSI (14)",
            mode="lines", line=dict(color=COLORS["blue"], width=1.3),
        ), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,59,48,0.05)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(52,199,89,0.05)", line_width=0, row=2, col=1)
        fig.add_hline(y=70, line=dict(color=COLORS["red"], width=0.7, dash="dot"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color=COLORS["green"], width=0.7, dash="dot"), row=2, col=1)
        fig.add_hline(y=50, line=dict(color=COLORS["muted"], width=0.5, dash="dot"), row=2, col=1)

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

    if "ATR_14" in inds:
        fig.add_trace(go.Scatter(
            x=dates, y=inds["ATR_14"], name="ATR (14)",
            mode="lines", line=dict(color=COLORS["orange"], width=1.3),
            fill="tozeroy", fillcolor="rgba(255,149,0,0.06)",
        ), row=4, col=1)

    if "OBV" in inds:
        fig.add_trace(go.Scatter(
            x=dates, y=inds["OBV"], name="OBV",
            mode="lines", line=dict(color=COLORS["cyan"], width=1.3),
        ), row=5, col=1)

    fig.update_layout(
        height=780,
        title="",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.03, x=0, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
    )
    return fig


def layout() -> html.Div:
    return html.Div([
        page_header(
            "Indicator Explorer",
            "Visualise every indicator from the quant library on one synchronised chart.",
            badge="Analysis",
        ),

        html.Div([
            control_group(
                "Contract",
                dcc.Dropdown(
                    id="ie-contract",
                    options=[{"value": c, "label": c} for c in CONTRACTS],
                    value="ES",
                    clearable=False,
                    className="dash-dropdown",
                    style=DROPDOWN_STYLE,
                ),
            ),
            control_group(
                "Bars",
                dcc.Slider(
                    id="ie-nbars",
                    min=100, max=800, step=100, value=300,
                    marks={100: "100", 300: "300", 500: "500", 800: "800"},
                    tooltip={"placement": "bottom"},
                ),
            ),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "240px 1fr",
            "gap": "24px",
            "maxWidth": "640px",
            "marginBottom": "24px",
        }),

        section_card(
            "Multi-Panel Dashboard",
            "Price · RSI · MACD · ATR · OBV — all aligned on the same time axis",
            dcc.Loading(dcc.Graph(
                id="ie-chart",
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )),
        ),
    ])


@callback(
    Output("ie-chart", "figure"),
    Input("ie-contract", "value"),
    Input("ie-nbars", "value"),
)
def update_ie(contract, n_bars):
    return _full_indicator_dashboard(contract or "ES", n_bars or 300)
