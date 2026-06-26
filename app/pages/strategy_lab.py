"""
Strategy Lab — configure, backtest, and analyse trading strategies.

Users can select from MeanReversionRSI, MomentumBreakout, and TrendFollowingMACD,
tune parameters with sliders, then run a full backtest via our engine
and see equity curves, metrics, and the fills log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

from app.theme import COLORS
from app.data_service import run_backtest_for_ui, get_synthetic_futures_bars


STRATEGIES = [
    {"value": "MeanReversionRSI",   "label": "Mean Reversion RSI"},
    {"value": "MomentumBreakout",   "label": "Momentum Breakout (Donchian)"},
    {"value": "TrendFollowingMACD", "label": "Trend Following MACD"},
]

CONTRACTS = [
    {"value": "ES", "label": "ES — E-mini S&P 500"},
    {"value": "NQ", "label": "NQ — E-mini Nasdaq-100"},
    {"value": "CL", "label": "CL — Crude Oil"},
    {"value": "GC", "label": "GC — Gold"},
]


def _equity_chart(result: dict) -> go.Figure:
    eq_df = result["equity_curve"]
    price_df = result["price_df"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["Portfolio Equity Curve", "Underlying Price"],
        vertical_spacing=0.12,
    )

    # Equity curve
    fig.add_trace(go.Scatter(
        x=eq_df["Date"], y=eq_df["Equity"],
        name="Equity", mode="lines",
        line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.06)",
    ), row=1, col=1)

    # Drawdown shading
    eq = eq_df["Equity"].to_numpy(dtype=np.float64)
    if len(eq) > 1:
        cummax = np.maximum.accumulate(eq)
        drawdown = (eq - cummax) / np.where(cummax > 0, cummax, 1.0)
        fig.add_trace(go.Scatter(
            x=eq_df["Date"], y=drawdown * 100,
            name="Drawdown %", mode="lines",
            line=dict(color=COLORS["red"], width=1),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.1)",
            yaxis="y3",
            visible=True,
        ), row=1, col=1)

    # Underlying price
    fig.add_trace(go.Scatter(
        x=price_df["Date"], y=price_df["Close"],
        name="Price", mode="lines",
        line=dict(color=COLORS["muted"], width=1),
    ), row=2, col=1)

    # Mark fill events
    fills = result.get("fills", [])
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    for f in fills:
        ts = pd.Timestamp(f.timestamp)
        if hasattr(f, "direction"):
            from core.enums import Direction
            if f.direction == Direction.LONG:
                buy_dates.append(ts)
                buy_prices.append(f.fill_price)
            else:
                sell_dates.append(ts)
                sell_prices.append(f.fill_price)

    if buy_dates:
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices, mode="markers",
            marker=dict(symbol="triangle-up", size=8, color=COLORS["green"]),
            name="Buy",
        ), row=2, col=1)
    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices, mode="markers",
            marker=dict(symbol="triangle-down", size=8, color=COLORS["red"]),
            name="Sell",
        ), row=2, col=1)

    fig.update_layout(
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    return fig


def _metrics_cards(metrics: dict) -> list:
    if not metrics:
        return [html.P("No metrics — run a backtest first.", style={"color": "#64748b"})]

    order = [
        ("total_return",  "Total Return",  lambda v: f"{v*100:+.2f}%"),
        ("cagr",          "CAGR",          lambda v: f"{v*100:+.2f}%"),
        ("sharpe_ratio",  "Sharpe Ratio",  lambda v: f"{v:.3f}"),
        ("sortino_ratio", "Sortino Ratio", lambda v: f"{v:.3f}"),
        ("max_drawdown",  "Max Drawdown",  lambda v: f"{v*100:.2f}%"),
        ("calmar_ratio",  "Calmar Ratio",  lambda v: f"{v:.3f}"),
        ("win_rate",      "Win Rate",      lambda v: f"{v*100:.1f}%"),
        ("profit_factor", "Profit Factor", lambda v: f"{v:.2f}"),
        ("var_95",        "VaR (95%)",     lambda v: f"{v*100:.2f}%"),
        ("n_periods",     "Bars Tested",   lambda v: f"{int(v):,}"),
    ]

    cards = []
    for key, label, fmt in order:
        val = metrics.get(key)
        if val is None:
            continue
        # Determine sign colouring
        pos_keys = {"total_return", "cagr", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "win_rate", "profit_factor"}
        neg_keys = {"max_drawdown", "var_95"}
        if key in pos_keys:
            cls = "positive" if val > 0 else "negative"
        elif key in neg_keys:
            cls = "negative" if val < 0 else "positive"
        else:
            cls = "neutral"

        cards.append(dbc.Col(html.Div([
            html.Div(label, className="metric-label"),
            html.Div(fmt(val), className=f"metric-value {cls}", style={"fontSize": "1.2rem"}),
        ], className="metric-card"), md=2, sm=4, xs=6, className="mb-3"))

    return cards


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H2("Strategy Lab"),
            html.P("Configure parameters, run backtests, and analyse performance metrics"),
        ], className="page-header"),

        dbc.Row([
            # ── Left panel: controls ──────────────────────────────────────
            dbc.Col([
                html.Div([
                    html.H5("Configuration", style={"fontSize": "0.85rem", "fontWeight": "700", "color": "#f1f5f9", "marginBottom": "16px"}),

                    html.Div("Strategy", className="form-label-dark"),
                    dcc.Dropdown(
                        id="sl-strategy",
                        options=STRATEGIES,
                        value="MeanReversionRSI",
                        clearable=False,
                        style={"backgroundColor": "#111827", "border": "1px solid #1e293b", "marginBottom": "12px"},
                    ),

                    html.Div("Contract", className="form-label-dark"),
                    dcc.Dropdown(
                        id="sl-contract",
                        options=CONTRACTS,
                        value="ES",
                        clearable=False,
                        style={"backgroundColor": "#111827", "border": "1px solid #1e293b", "marginBottom": "12px"},
                    ),

                    html.Hr(className="divider"),
                    html.Div("RSI Period", className="form-label-dark"),
                    dcc.Slider(id="sl-rsi-period", min=5, max=30, step=1, value=14,
                               marks={5: "5", 14: "14", 21: "21", 30: "30"},
                               tooltip={"placement": "bottom"}),

                    html.Div("RSI Oversold", className="form-label-dark", style={"marginTop": "12px"}),
                    dcc.Slider(id="sl-rsi-oversold", min=15, max=45, step=1, value=30,
                               marks={15: "15", 30: "30", 45: "45"},
                               tooltip={"placement": "bottom"}),

                    html.Div("RSI Overbought", className="form-label-dark", style={"marginTop": "12px"}),
                    dcc.Slider(id="sl-rsi-overbought", min=55, max=85, step=1, value=70,
                               marks={55: "55", 70: "70", 85: "85"},
                               tooltip={"placement": "bottom"}),

                    html.Div("BB Period", className="form-label-dark", style={"marginTop": "12px"}),
                    dcc.Slider(id="sl-bb-period", min=10, max=50, step=5, value=20,
                               marks={10: "10", 20: "20", 30: "30", 50: "50"},
                               tooltip={"placement": "bottom"}),

                    html.Hr(className="divider"),
                    html.Div("Bars in Test", className="form-label-dark"),
                    dcc.Slider(id="sl-nbars", min=200, max=1000, step=100, value=500,
                               marks={200: "200", 500: "500", 1000: "1K"},
                               tooltip={"placement": "bottom"}),

                    html.Div("Initial Capital ($)", className="form-label-dark", style={"marginTop": "12px"}),
                    dcc.Input(
                        id="sl-capital",
                        type="number",
                        value=100_000,
                        min=10_000,
                        step=10_000,
                        style={"backgroundColor": "#111827", "color": "#f1f5f9",
                               "border": "1px solid #1e293b", "borderRadius": "6px",
                               "padding": "6px 10px", "width": "100%", "marginBottom": "20px"},
                    ),

                    dbc.Button(
                        "▶  Run Backtest",
                        id="sl-run-btn",
                        color="primary",
                        className="w-100",
                        style={"fontWeight": "700", "letterSpacing": "0.5px"},
                    ),

                    html.Div(id="sl-run-status", style={"marginTop": "10px", "fontSize": "0.75rem", "color": "#64748b"}),
                ], className="chart-card"),
            ], md=3),

            # ── Right panel: results ──────────────────────────────────────
            dbc.Col([
                dbc.Row(id="sl-metrics-row", className="mb-3"),
                html.Div([
                    html.H5("Equity Curve & Trade Signals"),
                    html.P("Portfolio value over time with buy/sell markers on price chart", className="chart-subtitle"),
                    dcc.Loading(dcc.Graph(
                        id="sl-equity-chart",
                        config={"displayModeBar": True, "displaylogo": False},
                    )),
                ], className="chart-card"),
            ], md=9),
        ]),

        # Fills table
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Fill Log"),
                html.P("All executed fills during the backtest", className="chart-subtitle"),
                html.Div(id="sl-fills-table"),
            ], className="chart-card"), md=12),
        ]),

        # Hidden store for results
        dcc.Store(id="sl-results-store"),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("sl-results-store", "data"),
    Output("sl-run-status", "children"),
    Input("sl-run-btn", "n_clicks"),
    State("sl-strategy",     "value"),
    State("sl-contract",     "value"),
    State("sl-rsi-period",   "value"),
    State("sl-rsi-oversold", "value"),
    State("sl-rsi-overbought","value"),
    State("sl-bb-period",    "value"),
    State("sl-nbars",        "value"),
    State("sl-capital",      "value"),
    prevent_initial_call=True,
)
def run_backtest(n_clicks, strategy, contract, rsi_period, rsi_oversold,
                 rsi_overbought, bb_period, n_bars, capital):
    if not n_clicks:
        return no_update, ""

    try:
        result = run_backtest_for_ui(
            symbol=contract or "ES",
            strategy_name=strategy or "MeanReversionRSI",
            rsi_period=rsi_period or 14,
            rsi_oversold=float(rsi_oversold or 30),
            rsi_overbought=float(rsi_overbought or 70),
            bb_period=bb_period or 20,
            initial_cash=float(capital or 100_000),
            n_bars=n_bars or 500,
        )
        # Serialise for store (fills are dataclass objects — convert to counts)
        store = {
            "metrics":  result["metrics"],
            "equity_x": result["equity_curve"]["Date"].astype(str).tolist(),
            "equity_y": result["equity_curve"]["Equity"].tolist(),
            "price_x":  result["price_df"]["Date"].astype(str).tolist(),
            "price_y":  result["price_df"]["Close"].tolist(),
            "n_fills":  len(result["fills"]),
            "fills_data": [
                {
                    "direction": f.direction.value,
                    "qty": f.filled_quantity,
                    "price": f.fill_price,
                    "commission": f.commission,
                    "time": str(f.timestamp)[:16],
                }
                for f in result["fills"][:50]  # cap at 50 for display
            ],
        }
        n_fills = len(result["fills"])
        n_bars_run = len(result["equity_curve"])
        return store, f"✓ Complete — {n_bars_run} bars, {n_fills} fills"
    except Exception as exc:
        return None, f"✗ Error: {str(exc)[:80]}"


@callback(
    Output("sl-equity-chart", "figure"),
    Output("sl-metrics-row",  "children"),
    Output("sl-fills-table",  "children"),
    Input("sl-results-store", "data"),
)
def update_results(store):
    if not store:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            height=300,
            annotations=[dict(text="Run a backtest to see results", x=0.5, y=0.5,
                              showarrow=False, font=dict(color="#64748b", size=14),
                              xref="paper", yref="paper")],
        )
        return empty_fig, [], html.P("No fills yet.", style={"color": "#64748b"})

    metrics = store.get("metrics", {})

    # Equity figure
    eq_df = pd.DataFrame({"Date": pd.to_datetime(store["equity_x"]), "Equity": store["equity_y"]})
    pr_df = pd.DataFrame({"Date": pd.to_datetime(store["price_x"]),  "Close": store["price_y"]})

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        subplot_titles=["Equity Curve", "Underlying Price"],
                        vertical_spacing=0.12)

    fig.add_trace(go.Scatter(x=eq_df["Date"], y=eq_df["Equity"], name="Equity",
                             mode="lines", line=dict(color=COLORS["blue"], width=2),
                             fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"), row=1, col=1)

    eq = eq_df["Equity"].to_numpy(dtype=np.float64)
    if len(eq) > 1:
        cummax = np.maximum.accumulate(eq)
        dd = (eq - cummax) / np.where(cummax > 0, cummax, 1.0) * 100
        fig.add_trace(go.Scatter(x=eq_df["Date"], y=dd, name="Drawdown %",
                                 mode="lines", line=dict(color=COLORS["red"], width=1),
                                 fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=pr_df["Date"], y=pr_df["Close"], name="Price",
                             mode="lines", line=dict(color=COLORS["muted"], width=1)), row=2, col=1)

    fig.update_layout(height=500, hovermode="x unified",
                      legend=dict(orientation="h", y=1.05))

    # Fills table
    fills = store.get("fills_data", [])
    if fills:
        header_style = {"color": "#64748b", "fontSize": "0.7rem", "textTransform": "uppercase",
                        "borderBottom": "1px solid #1e293b", "padding": "8px 10px"}
        cell_style   = {"padding": "7px 10px", "fontSize": "0.8rem",
                        "borderBottom": "1px solid #111827"}
        rows = []
        for f in fills:
            color = COLORS["green"] if f["direction"] == "LONG" else COLORS["red"]
            rows.append(html.Tr([
                html.Td(html.Span(f["direction"], className=f"tag {'tag-green' if f['direction']=='LONG' else 'tag-red'}"), style=cell_style),
                html.Td(f"{f['qty']:.0f}", style=cell_style),
                html.Td(f"${f['price']:,.2f}", style=cell_style),
                html.Td(f"${f['commission']:.2f}", style=cell_style),
                html.Td(f["time"], style={**cell_style, "color": "#64748b"}),
            ]))
        table = html.Table([
            html.Thead(html.Tr([
                html.Th("Direction", style=header_style),
                html.Th("Qty",       style=header_style),
                html.Th("Price",     style=header_style),
                html.Th("Commission",style=header_style),
                html.Th("Time",      style=header_style),
            ])),
            html.Tbody(rows),
        ], style={"width": "100%", "borderCollapse": "collapse"})
    else:
        table = html.P("No fills during this backtest.", style={"color": "#64748b"})

    return fig, _metrics_cards(metrics), table
