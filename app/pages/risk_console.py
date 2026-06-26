"""
Risk Console — portfolio health dashboard.

Shows: drawdown gauge, position heatmap, risk limit status,
margin utilisation, and the live equity curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc

from app.theme import COLORS
from app.data_service import run_backtest_for_ui, get_synthetic_futures_bars


def _drawdown_gauge(drawdown_pct: float, limit_pct: float = 3.0) -> go.Figure:
    """Speedometer gauge showing current drawdown vs the circuit-breaker limit."""
    color = COLORS["green"] if drawdown_pct < limit_pct * 0.5 else (
        COLORS["gold"] if drawdown_pct < limit_pct * 0.8 else COLORS["red"]
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=drawdown_pct,
        delta={"reference": 0, "suffix": "%"},
        title={"text": "Current Drawdown (%)", "font": {"size": 13, "color": "#94a3b8"}},
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis":      {"range": [0, limit_pct * 2], "tickwidth": 1, "tickcolor": "#1e293b",
                          "ticksuffix": "%", "tickfont": {"color": "#64748b"}},
            "bar":       {"color": color, "thickness": 0.25},
            "bgcolor":   "#111827",
            "bordercolor": "#1e293b",
            "steps": [
                {"range": [0, limit_pct * 0.5],  "color": "rgba(16,185,129,0.1)"},
                {"range": [limit_pct * 0.5, limit_pct * 0.8], "color": "rgba(245,158,11,0.1)"},
                {"range": [limit_pct * 0.8, limit_pct * 2],   "color": "rgba(239,68,68,0.1)"},
            ],
            "threshold": {
                "line":  {"color": COLORS["red"], "width": 2},
                "thickness": 0.75,
                "value": limit_pct,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _risk_limits_chart(
    drawdown_pct: float,
    position_pct: float,
    open_pos: int,
    leverage: float,
) -> go.Figure:
    """Horizontal bar chart showing four risk limit utilisations."""
    categories = ["Drawdown Limit", "Position Size", "Open Positions (max 5)", "Leverage (max 10x)"]
    values     = [
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
        title="Risk Limit Utilisation",
        height=200,
        xaxis=dict(range=[0, 115], ticksuffix="%"),
        bargap=0.35,
        showlegend=False,
    )
    return fig


def _simulated_portfolio_equity() -> dict:
    """Generate a synthetic portfolio for display purposes."""
    result = run_backtest_for_ui(
        symbol="ES", strategy_name="MeanReversionRSI",
        rsi_period=14, initial_cash=100_000, n_bars=400
    )
    return result


def _equity_history_chart(eq_df: pd.DataFrame) -> go.Figure:
    eq = eq_df["Equity"].to_numpy(dtype=np.float64)
    cummax = np.maximum.accumulate(eq)
    dd = (eq - cummax) / np.where(cummax > 0, cummax, 1.0) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.08,
                        subplot_titles=["Portfolio Equity ($)", "Drawdown (%)"])

    fig.add_trace(go.Scatter(
        x=eq_df["Date"], y=eq_df["Equity"],
        mode="lines", name="Equity",
        line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
    ), row=1, col=1)

    # Mark peak equity line
    peak = float(cummax[-1])
    fig.add_hline(y=peak, line=dict(color=COLORS["gold"], width=0.8, dash="dot"),
                  annotation_text=f"Peak ${peak:,.0f}", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=eq_df["Date"], y=dd,
        mode="lines", name="Drawdown",
        line=dict(color=COLORS["red"], width=1.2),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
    ), row=2, col=1)

    fig.update_layout(height=380, hovermode="x unified",
                      legend=dict(orientation="h", y=1.04))
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    # Build simulated portfolio state
    result   = _simulated_portfolio_equity()
    eq       = result["equity_curve"]["Equity"].to_numpy(dtype=np.float64)
    initial  = 100_000.0
    current  = float(eq[-1]) if len(eq) > 0 else initial
    peak     = float(np.maximum.accumulate(eq)[-1]) if len(eq) > 0 else initial
    dd_pct   = max(0.0, (peak - current) / peak * 100)
    total_ret= (current / initial - 1) * 100
    n_fills  = len(result["fills"])

    # Simulated open positions
    positions = [
        {"sym": "ES.c.0", "dir": "LONG",  "qty": 2, "entry": 5_210.25, "last": 5_247.25, "pnl": +3_700.0},
        {"sym": "NQ.c.0", "dir": "LONG",  "qty": 1, "entry": 18_120.00,"last": 18_432.50,"pnl": +3_125.0},
        {"sym": "CL.c.0", "dir": "SHORT", "qty": 3, "entry": 79.10,    "last": 78.42,    "pnl": +2_040.0},
    ]

    pos_rows = []
    for p in positions:
        color = COLORS["green"] if p["pnl"] >= 0 else COLORS["red"]
        dir_cls = "tag-green" if p["dir"] == "LONG" else "tag-red"
        pos_rows.append(html.Tr([
            html.Td(p["sym"],  style={"fontWeight": "600"}),
            html.Td(html.Span(p["dir"], className=f"tag {dir_cls}")),
            html.Td(str(p["qty"])),
            html.Td(f"${p['entry']:,.2f}"),
            html.Td(f"${p['last']:,.2f}"),
            html.Td(f"${p['pnl']:+,.2f}", style={"color": color, "fontWeight": "600"}),
        ]))

    return html.Div([
        html.Div([
            html.H2("Risk Console"),
            html.P("Portfolio health — drawdown status, position inventory, and risk limit monitoring"),
        ], className="page-header"),

        # Top KPI row
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("Portfolio Equity",  className="metric-label"),
                html.Div(f"${current:,.0f}", className="metric-value neutral"),
                html.Div(f"Initial: ${initial:,.0f}", style={"color": "#64748b", "fontSize": "0.75rem"}),
            ], className="metric-card"), md=2, sm=4, xs=6, className="mb-3"),

            dbc.Col(html.Div([
                html.Div("Total Return", className="metric-label"),
                html.Div(f"{total_ret:+.2f}%",
                         className=f"metric-value {'positive' if total_ret >= 0 else 'negative'}"),
            ], className="metric-card"), md=2, sm=4, xs=6, className="mb-3"),

            dbc.Col(html.Div([
                html.Div("Peak Equity", className="metric-label"),
                html.Div(f"${peak:,.0f}", className="metric-value positive"),
            ], className="metric-card"), md=2, sm=4, xs=6, className="mb-3"),

            dbc.Col(html.Div([
                html.Div("Current DD", className="metric-label"),
                html.Div(f"{dd_pct:.2f}%",
                         className=f"metric-value {'negative' if dd_pct > 1 else 'positive'}"),
            ], className="metric-card"), md=2, sm=4, xs=6, className="mb-3"),

            dbc.Col(html.Div([
                html.Div("Open Positions", className="metric-label"),
                html.Div("3", className="metric-value neutral"),
                html.Div("Max: 5", style={"color": "#64748b", "fontSize": "0.75rem"}),
            ], className="metric-card"), md=2, sm=4, xs=6, className="mb-3"),

            dbc.Col(html.Div([
                html.Div("Total Fills", className="metric-label"),
                html.Div(f"{n_fills}", className="metric-value neutral"),
            ], className="metric-card"), md=2, sm=4, xs=6, className="mb-3"),
        ]),

        # Gauge + limits row
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Drawdown Gauge"),
                html.P("Circuit breaker triggers at 3%", className="chart-subtitle"),
                dcc.Graph(
                    figure=_drawdown_gauge(dd_pct, limit_pct=3.0),
                    config={"displayModeBar": False},
                ),
            ], className="chart-card"), md=4),

            dbc.Col(html.Div([
                html.H5("Risk Limits"),
                html.P("Current utilisation vs configured hard limits", className="chart-subtitle"),
                dcc.Graph(
                    figure=_risk_limits_chart(
                        drawdown_pct=dd_pct,
                        position_pct=5.5,    # simulated
                        open_pos=3,
                        leverage=2.1,
                    ),
                    config={"displayModeBar": False},
                ),
            ], className="chart-card"), md=8),
        ]),

        # Equity history
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Equity History"),
                html.P("Simulated portfolio equity and drawdown timeline", className="chart-subtitle"),
                dcc.Graph(
                    figure=_equity_history_chart(result["equity_curve"]),
                    config={"displayModeBar": False},
                ),
            ], className="chart-card"), md=12),
        ]),

        # Open positions table
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Open Positions"),
                html.P("Current futures positions — paper mode", className="chart-subtitle"),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th(h) for h in
                        ["Symbol", "Direction", "Qty", "Entry", "Last", "Unrealised P&L"]
                    ], style={"fontSize": "0.7rem", "color": "#64748b", "textTransform": "uppercase"})),
                    html.Tbody(pos_rows),
                ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85rem"}),
            ], className="chart-card"), md=12),
        ]),
    ])
