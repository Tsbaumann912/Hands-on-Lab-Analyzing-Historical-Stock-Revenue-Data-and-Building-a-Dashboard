"""
Futures Terminal — live-style price chart with full indicator suite.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from app.components import control_group, metric_tile, page_header, section_card
from app.styles import DROPDOWN_STYLE
from app.theme import COLORS
from app.data_service import get_synthetic_futures_bars, compute_indicators_for_ui


FUTURES_CONTRACTS = [
    {"value": "ES", "label": "ES — E-mini S&P 500"},
    {"value": "NQ", "label": "NQ — E-mini Nasdaq-100"},
    {"value": "CL", "label": "CL — Crude Oil WTI"},
    {"value": "GC", "label": "GC — Gold"},
    {"value": "ZN", "label": "ZN — 10-Year T-Note"},
    {"value": "ZB", "label": "ZB — 30-Year T-Bond"},
    {"value": "SI", "label": "SI — Silver"},
    {"value": "NG", "label": "NG — Natural Gas"},
]

OVERLAY_OPTIONS = [
    {"value": "SMA_20", "label": "SMA 20"},
    {"value": "SMA_50", "label": "SMA 50"},
    {"value": "EMA_20", "label": "EMA 20"},
    {"value": "BB_upper", "label": "Bollinger Bands"},
    {"value": "VWAP", "label": "VWAP"},
]

PANEL_OPTIONS = [
    {"value": "RSI_14", "label": "RSI (14)"},
    {"value": "MACD_line", "label": "MACD"},
    {"value": "ATR_14", "label": "ATR (14)"},
    {"value": "OBV", "label": "OBV"},
]


def _build_chart(contract: str, n_bars: int, overlays: list, panel: str | None) -> go.Figure:
    df = get_synthetic_futures_bars(contract, n=n_bars)
    inds = compute_indicators_for_ui(df)

    rows = 3 if panel else 2
    heights = [0.6, 0.2, 0.2] if panel else [0.75, 0.25]

    row_titles = ["Price", "Volume"]
    if panel:
        label_map = {
            "RSI_14": "RSI (14)",
            "MACD_line": "MACD",
            "ATR_14": "ATR (14)",
            "OBV": "OBV",
        }
        row_titles.append(label_map.get(panel, panel))

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.04,
        subplot_titles=row_titles,
    )

    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=contract,
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        increasing_fillcolor=COLORS["green"],
        decreasing_fillcolor=COLORS["red"],
    ), row=1, col=1)

    overlay_color_map = {
        "SMA_20": COLORS["blue"],
        "SMA_50": COLORS["gold"],
        "EMA_20": COLORS["cyan"],
        "VWAP": COLORS["purple"],
    }

    for ov in (overlays or []):
        if ov == "BB_upper" and "BB_upper" in inds:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=inds["BB_upper"],
                name="BB Upper", mode="lines",
                line=dict(color=COLORS["muted"], width=0.8, dash="dot"),
                showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df["Date"], y=inds["BB_lower"],
                name="BB Lower", mode="lines",
                line=dict(color=COLORS["muted"], width=0.8, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(134, 134, 139, 0.06)",
                showlegend=False,
            ), row=1, col=1)
            if "BB_middle" in inds:
                fig.add_trace(go.Scatter(
                    x=df["Date"], y=inds["BB_middle"],
                    name="BB Mid", mode="lines",
                    line=dict(color=COLORS["muted"], width=1),
                    showlegend=False,
                ), row=1, col=1)
        elif ov in inds:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=inds[ov],
                name=ov, mode="lines",
                line=dict(color=overlay_color_map.get(ov, COLORS["cyan"]), width=1.5),
            ), row=1, col=1)

    vol_colors = [
        COLORS["green"] if df["Close"].iloc[i] >= df["Open"].iloc[i] else COLORS["red"]
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"],
        name="Volume", marker_color=vol_colors, opacity=0.5,
    ), row=2, col=1)

    if panel and rows == 3:
        if panel == "RSI_14" and "RSI_14" in inds:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=inds["RSI_14"],
                name="RSI", mode="lines",
                line=dict(color=COLORS["blue"], width=1.5),
            ), row=3, col=1)
            fig.add_hline(y=70, line=dict(color=COLORS["red"], width=0.8, dash="dot"), row=3, col=1)
            fig.add_hline(y=30, line=dict(color=COLORS["green"], width=0.8, dash="dot"), row=3, col=1)

        elif panel == "MACD_line" and "MACD_line" in inds:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=inds["MACD_line"],
                name="MACD", mode="lines",
                line=dict(color=COLORS["blue"], width=1.5),
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df["Date"], y=inds["MACD_signal"],
                name="Signal", mode="lines",
                line=dict(color=COLORS["gold"], width=1.2),
            ), row=3, col=1)
            hist = inds.get("MACD_hist", np.zeros(len(df)))
            bar_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in hist]
            fig.add_trace(go.Bar(
                x=df["Date"], y=hist, name="Histogram",
                marker_color=bar_colors, opacity=0.5,
            ), row=3, col=1)

        elif panel == "ATR_14" and "ATR_14" in inds:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=inds["ATR_14"],
                name="ATR (14)", mode="lines",
                line=dict(color=COLORS["orange"], width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255, 149, 0, 0.06)",
            ), row=3, col=1)

        elif panel == "OBV" and "OBV" in inds:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=inds["OBV"],
                name="OBV", mode="lines",
                line=dict(color=COLORS["cyan"], width=1.5),
            ), row=3, col=1)

    fig.update_layout(
        title="",
        height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.04, x=0, xanchor="left"),
        hovermode="x unified",
    )
    return fig


def layout() -> html.Div:
    return html.Div([
        page_header(
            "Futures Terminal",
            "Analyse CME continuous contracts with overlays, volume, and technical indicators.",
            badge="Trading",
        ),

        html.Div([
            control_group(
                "Contract",
                dcc.Dropdown(
                    id="ft-contract",
                    options=FUTURES_CONTRACTS,
                    value="ES",
                    clearable=False,
                    className="dash-dropdown",
                    style=DROPDOWN_STYLE,
                ),
            ),
            control_group(
                "Bars",
                dcc.Slider(
                    id="ft-nbars",
                    min=100, max=1000, step=100, value=500,
                    marks={100: "100", 300: "300", 500: "500", 750: "750", 1000: "1K"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ),
            control_group(
                "Price Overlays",
                dcc.Dropdown(
                    id="ft-overlays",
                    options=OVERLAY_OPTIONS,
                    value=["SMA_20", "BB_upper"],
                    multi=True,
                    className="dash-dropdown",
                    style=DROPDOWN_STYLE,
                ),
            ),
            control_group(
                "Sub-Panel",
                dcc.Dropdown(
                    id="ft-panel",
                    options=PANEL_OPTIONS,
                    value="RSI_14",
                    clearable=True,
                    className="dash-dropdown",
                    style=DROPDOWN_STYLE,
                ),
            ),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
            "gap": "16px",
            "marginBottom": "24px",
        }),

        section_card(
            "Price Chart",
            "Synthetic minute bars with selected overlays and indicator panel",
            dcc.Loading(dcc.Graph(
                id="ft-chart",
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )),
        ),

        html.Div(id="ft-stats-row", className="metrics-grid"),
    ])


@callback(
    Output("ft-chart", "figure"),
    Output("ft-stats-row", "children"),
    Input("ft-contract", "value"),
    Input("ft-nbars", "value"),
    Input("ft-overlays", "value"),
    Input("ft-panel", "value"),
)
def update_ft_chart(contract, n_bars, overlays, panel):
    if not contract:
        return go.Figure(), []

    n_bars = n_bars or 500
    df = get_synthetic_futures_bars(contract, n=n_bars)
    inds = compute_indicators_for_ui(df)
    fig = _build_chart(contract, n_bars, overlays or [], panel)

    last = df["Close"].iloc[-1]
    high = df["High"].max()
    low = df["Low"].min()
    vol = df["Volume"].sum()
    atr_v = inds.get("ATR_14", [0])
    last_atr = float(atr_v[~np.isnan(atr_v)][-1]) if len(atr_v) > 0 else 0.0
    rsi_v = inds.get("RSI_14", [50])
    last_rsi = float(rsi_v[~np.isnan(rsi_v)][-1]) if len(rsi_v) > 0 else 50.0
    rsi_cls = "negative" if last_rsi > 70 else ("positive" if last_rsi < 30 else "neutral")

    stats = [
        ("Last Price", f"{last:,.2f}", "neutral"),
        ("Session High", f"{high:,.2f}", "positive"),
        ("Session Low", f"{low:,.2f}", "negative"),
        ("Total Volume", f"{vol/1e6:.1f}M", "neutral"),
        ("ATR (14)", f"{last_atr:.2f}", "neutral"),
        ("RSI (14)", f"{last_rsi:.1f}", rsi_cls),
    ]

    return fig, [
        html.Div(metric_tile(label, value, value_cls=cls), className="metric-tile")
        for label, value, cls in stats
    ]
