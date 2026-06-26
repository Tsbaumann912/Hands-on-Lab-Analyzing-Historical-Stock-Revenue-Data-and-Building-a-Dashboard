"""
Risk Console — portfolio health dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html

from app.components import data_table, metric_tile, page_header, section_card
from app.theme import COLORS
from app.data_service import run_backtest_for_ui


def _drawdown_gauge(drawdown_pct: float, limit_pct: float = 3.0) -> go.Figure:
    color = COLORS["green"] if drawdown_pct < limit_pct * 0.5 else (
        COLORS["gold"] if drawdown_pct < limit_pct * 0.8 else COLORS["red"]
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=drawdown_pct,
        delta={"reference": 0, "suffix": "%"},
        title={"text": "Current Drawdown", "font": {"size": 14, "color": "#6e6e73"}},
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        gauge={
            "axis": {
                "range": [0, limit_pct * 2],
                "tickwidth": 1,
                "tickcolor": "rgba(0,0,0,0.08)",
                "ticksuffix": "%",
                "tickfont": {"color": "#86868b"},
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#f5f5f7",
            "bordercolor": "rgba(0,0,0,0.08)",
            "steps": [
                {"range": [0, limit_pct * 0.5], "color": "rgba(52,199,89,0.12)"},
                {"range": [limit_pct * 0.5, limit_pct * 0.8], "color": "rgba(255,159,10,0.12)"},
                {"range": [limit_pct * 0.8, limit_pct * 2], "color": "rgba(255,59,48,0.12)"},
            ],
            "threshold": {
                "line": {"color": COLORS["red"], "width": 2},
                "thickness": 0.75,
                "value": limit_pct,
            },
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _risk_limits_chart(drawdown_pct, position_pct, open_pos, leverage) -> go.Figure:
    categories = ["Drawdown Limit", "Position Size", "Open Positions (max 5)", "Leverage (max 10x)"]
    values = [
        min(100, drawdown_pct / 3.0 * 100),
        min(100, position_pct / 10.0 * 100),
        min(100, open_pos / 5.0 * 100),
        min(100, leverage / 10.0 * 100),
    ]
    colors = [
        COLORS["green"] if v < 50 else COLORS["gold"] if v < 80 else COLORS["red"]
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=categories,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% utilised<extra></extra>",
    ))
    fig.update_layout(
        title="",
        height=220,
        xaxis=dict(range=[0, 115], ticksuffix="%"),
        bargap=0.35,
        showlegend=False,
    )
    return fig


def _equity_history_chart(eq_df: pd.DataFrame) -> go.Figure:
    eq = eq_df["Equity"].to_numpy(dtype=np.float64)
    cummax = np.maximum.accumulate(eq)
    dd = (eq - cummax) / np.where(cummax > 0, cummax, 1.0) * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.08,
        subplot_titles=["Portfolio Equity ($)", "Drawdown (%)"],
    )

    fig.add_trace(go.Scatter(
        x=eq_df["Date"], y=eq_df["Equity"],
        mode="lines", name="Equity",
        line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(0,113,227,0.06)",
    ), row=1, col=1)

    peak = float(cummax[-1])
    fig.add_hline(
        y=peak, line=dict(color=COLORS["gold"], width=0.8, dash="dot"),
        annotation_text=f"Peak ${peak:,.0f}", row=1, col=1,
    )

    fig.add_trace(go.Scatter(
        x=eq_df["Date"], y=dd,
        mode="lines", name="Drawdown",
        line=dict(color=COLORS["red"], width=1.2),
        fill="tozeroy", fillcolor="rgba(255,59,48,0.08)",
    ), row=2, col=1)

    fig.update_layout(height=380, hovermode="x unified", legend=dict(orientation="h", y=1.04))
    return fig


def layout() -> html.Div:
    result = run_backtest_for_ui(
        symbol="ES", strategy_name="MeanReversionRSI",
        rsi_period=14, initial_cash=100_000, n_bars=400,
    )
    eq = result["equity_curve"]["Equity"].to_numpy(dtype=np.float64)
    initial = 100_000.0
    current = float(eq[-1]) if len(eq) > 0 else initial
    peak = float(np.maximum.accumulate(eq)[-1]) if len(eq) > 0 else initial
    dd_pct = max(0.0, (peak - current) / peak * 100)
    total_ret = (current / initial - 1) * 100
    n_fills = len(result["fills"])

    positions = [
        {"sym": "ES.c.0", "dir": "LONG", "qty": 2, "entry": 5_210.25, "last": 5_247.25, "pnl": +3_700.0},
        {"sym": "NQ.c.0", "dir": "LONG", "qty": 1, "entry": 18_120.00, "last": 18_432.50, "pnl": +3_125.0},
        {"sym": "CL.c.0", "dir": "SHORT", "qty": 3, "entry": 79.10, "last": 78.42, "pnl": +2_040.0},
    ]

    pos_rows = []
    for p in positions:
        color = COLORS["green"] if p["pnl"] >= 0 else COLORS["red"]
        dir_cls = "tag-green" if p["dir"] == "LONG" else "tag-red"
        pos_rows.append(html.Tr([
            html.Td(p["sym"], style={"fontWeight": "600"}),
            html.Td(html.Span(p["dir"], className=f"tag {dir_cls}")),
            html.Td(str(p["qty"])),
            html.Td(f"${p['entry']:,.2f}"),
            html.Td(f"${p['last']:,.2f}"),
            html.Td(f"${p['pnl']:+,.2f}", style={"color": color, "fontWeight": "600"}),
        ]))

    return html.Div([
        page_header(
            "Risk Console",
            "Monitor portfolio health, drawdown limits, and open position exposure.",
            badge="Risk",
        ),

        html.Div([
            html.Div(metric_tile("Portfolio Equity", f"${current:,.0f}", delta=f"Initial ${initial:,.0f}"), className="metric-tile"),
            html.Div(metric_tile("Total Return", f"{total_ret:+.2f}%", value_cls="positive" if total_ret >= 0 else "negative"), className="metric-tile"),
            html.Div(metric_tile("Peak Equity", f"${peak:,.0f}", value_cls="positive"), className="metric-tile"),
            html.Div(metric_tile("Current DD", f"{dd_pct:.2f}%", value_cls="negative" if dd_pct > 1 else "positive"), className="metric-tile"),
            html.Div(metric_tile("Open Positions", "3", delta="Max 5"), className="metric-tile"),
            html.Div(metric_tile("Total Fills", f"{n_fills}"), className="metric-tile"),
        ], className="metrics-grid"),

        html.Div([
            section_card(
                "Drawdown Gauge",
                "Circuit breaker triggers at 3%",
                dcc.Graph(figure=_drawdown_gauge(dd_pct, limit_pct=3.0), config={"displayModeBar": False}),
            ),
            section_card(
                "Risk Limits",
                "Current utilisation vs configured hard limits",
                dcc.Graph(
                    figure=_risk_limits_chart(dd_pct, 5.5, 3, 2.1),
                    config={"displayModeBar": False},
                ),
            ),
        ], className="charts-grid-2-1", style={"gridTemplateColumns": "1fr 2fr"}),

        section_card(
            "Equity History",
            "Simulated portfolio equity and drawdown timeline",
            dcc.Graph(
                figure=_equity_history_chart(result["equity_curve"]),
                config={"displayModeBar": False},
            ),
        ),

        section_card(
            "Open Positions",
            "Current futures positions — paper trading mode",
            data_table(
                ["Symbol", "Direction", "Qty", "Entry", "Last", "Unrealised P&L"],
                pos_rows,
            ),
        ),
    ])
