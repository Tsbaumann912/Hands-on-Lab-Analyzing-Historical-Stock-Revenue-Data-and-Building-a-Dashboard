"""
Strategy Lab — configure, backtest, and analyse trading strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

from app.components import control_group, metric_tile, page_header, primary_button, section_card
from app.styles import DROPDOWN_STYLE
from app.theme import COLORS
from app.data_service import run_backtest_for_ui


STRATEGIES = [
    {"value": "MeanReversionRSI", "label": "Mean Reversion RSI"},
    {"value": "MomentumBreakout", "label": "Momentum Breakout (Donchian)"},
    {"value": "TrendFollowingMACD", "label": "Trend Following MACD"},
]

CONTRACTS = [
    {"value": "ES", "label": "ES — E-mini S&P 500"},
    {"value": "NQ", "label": "NQ — E-mini Nasdaq-100"},
    {"value": "CL", "label": "CL — Crude Oil"},
    {"value": "GC", "label": "GC — Gold"},
]


def _metrics_cards(metrics: dict) -> list:
    if not metrics:
        return [html.P("Run a backtest to see performance metrics.", className="empty-state")]

    order = [
        ("total_return", "Total Return", lambda v: f"{v*100:+.2f}%"),
        ("cagr", "CAGR", lambda v: f"{v*100:+.2f}%"),
        ("sharpe_ratio", "Sharpe Ratio", lambda v: f"{v:.3f}"),
        ("sortino_ratio", "Sortino Ratio", lambda v: f"{v:.3f}"),
        ("max_drawdown", "Max Drawdown", lambda v: f"{v*100:.2f}%"),
        ("calmar_ratio", "Calmar Ratio", lambda v: f"{v:.3f}"),
        ("win_rate", "Win Rate", lambda v: f"{v*100:.1f}%"),
        ("profit_factor", "Profit Factor", lambda v: f"{v:.2f}"),
        ("var_95", "VaR (95%)", lambda v: f"{v*100:.2f}%"),
        ("n_periods", "Bars Tested", lambda v: f"{int(v):,}"),
    ]

    cards = []
    for key, label, fmt in order:
        val = metrics.get(key)
        if val is None:
            continue
        pos_keys = {"total_return", "cagr", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "win_rate", "profit_factor"}
        neg_keys = {"max_drawdown", "var_95"}
        if key in pos_keys:
            cls = "positive" if val > 0 else "negative"
        elif key in neg_keys:
            cls = "negative" if val < 0 else "positive"
        else:
            cls = "neutral"

        cards.append(html.Div(
            metric_tile(label, fmt(val), value_cls=cls),
            className="metric-tile",
            style={"padding": "16px"},
        ))

    return cards


def layout() -> html.Div:
    return html.Div([
        page_header(
            "Strategy Lab",
            "Configure parameters, run backtests, and review equity curves and fill logs.",
            badge="Backtest",
        ),

        html.Div([
            html.Div([
                html.Div("Configuration", className="controls-panel-title"),

                control_group(
                    "Strategy",
                    dcc.Dropdown(
                        id="sl-strategy",
                        options=STRATEGIES,
                        value="MeanReversionRSI",
                        clearable=False,
                        className="dash-dropdown",
                        style=DROPDOWN_STYLE,
                    ),
                ),
                control_group(
                    "Contract",
                    dcc.Dropdown(
                        id="sl-contract",
                        options=CONTRACTS,
                        value="ES",
                        clearable=False,
                        className="dash-dropdown",
                        style=DROPDOWN_STYLE,
                    ),
                ),

                html.Hr(className="divider"),

                control_group("RSI Period", dcc.Slider(
                    id="sl-rsi-period", min=5, max=30, step=1, value=14,
                    marks={5: "5", 14: "14", 21: "21", 30: "30"},
                    tooltip={"placement": "bottom"},
                )),
                control_group("RSI Oversold", dcc.Slider(
                    id="sl-rsi-oversold", min=15, max=45, step=1, value=30,
                    marks={15: "15", 30: "30", 45: "45"},
                    tooltip={"placement": "bottom"},
                )),
                control_group("RSI Overbought", dcc.Slider(
                    id="sl-rsi-overbought", min=55, max=85, step=1, value=70,
                    marks={55: "55", 70: "70", 85: "85"},
                    tooltip={"placement": "bottom"},
                )),
                control_group("BB Period", dcc.Slider(
                    id="sl-bb-period", min=10, max=50, step=5, value=20,
                    marks={10: "10", 20: "20", 30: "30", 50: "50"},
                    tooltip={"placement": "bottom"},
                )),

                html.Hr(className="divider"),

                control_group("Bars in Test", dcc.Slider(
                    id="sl-nbars", min=200, max=1000, step=100, value=500,
                    marks={200: "200", 500: "500", 1000: "1K"},
                    tooltip={"placement": "bottom"},
                )),
                control_group(
                    "Initial Capital ($)",
                    dcc.Input(id="sl-capital", type="number", value=100_000, min=10_000, step=10_000),
                ),

                primary_button("Run Backtest", "sl-run-btn", className="w-100 mt-2"),
                html.Div(id="sl-run-status", style={"marginTop": "12px", "fontSize": "13px", "color": "#6e6e73"}),
            ], className="controls-panel"),

            html.Div([
                html.Div(id="sl-metrics-row", className="metrics-grid"),
                section_card(
                    "Equity Curve & Trade Signals",
                    "Portfolio value over time with buy and sell markers on the underlying price",
                    dcc.Loading(dcc.Graph(
                        id="sl-equity-chart",
                        config={"displayModeBar": True, "displaylogo": False},
                    )),
                ),
            ]),
        ], className="split-layout"),

        section_card(
            "Fill Log",
            "All executed fills during the backtest",
            html.Div(id="sl-fills-table"),
        ),

        dcc.Store(id="sl-results-store"),
    ])


@callback(
    Output("sl-results-store", "data"),
    Output("sl-run-status", "children"),
    Input("sl-run-btn", "n_clicks"),
    State("sl-strategy", "value"),
    State("sl-contract", "value"),
    State("sl-rsi-period", "value"),
    State("sl-rsi-oversold", "value"),
    State("sl-rsi-overbought", "value"),
    State("sl-bb-period", "value"),
    State("sl-nbars", "value"),
    State("sl-capital", "value"),
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
        store = {
            "metrics": result["metrics"],
            "equity_x": result["equity_curve"]["Date"].astype(str).tolist(),
            "equity_y": result["equity_curve"]["Equity"].tolist(),
            "price_x": result["price_df"]["Date"].astype(str).tolist(),
            "price_y": result["price_df"]["Close"].tolist(),
            "n_fills": len(result["fills"]),
            "fills_data": [
                {
                    "direction": f.direction.value,
                    "qty": f.filled_quantity,
                    "price": f.fill_price,
                    "commission": f.commission,
                    "time": str(f.timestamp)[:16],
                }
                for f in result["fills"][:50]
            ],
        }
        n_fills = len(result["fills"])
        n_bars_run = len(result["equity_curve"])
        return store, f"Complete — {n_bars_run} bars, {n_fills} fills"
    except Exception as exc:
        return None, f"Error: {str(exc)[:80]}"


@callback(
    Output("sl-equity-chart", "figure"),
    Output("sl-metrics-row", "children"),
    Output("sl-fills-table", "children"),
    Input("sl-results-store", "data"),
)
def update_results(store):
    if not store:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            height=300,
            annotations=[dict(
                text="Run a backtest to see results",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#6e6e73", size=15),
                xref="paper", yref="paper",
            )],
        )
        return empty_fig, [], html.P("No fills yet.", className="empty-state")

    metrics = store.get("metrics", {})
    eq_df = pd.DataFrame({"Date": pd.to_datetime(store["equity_x"]), "Equity": store["equity_y"]})
    pr_df = pd.DataFrame({"Date": pd.to_datetime(store["price_x"]), "Close": store["price_y"]})

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
        subplot_titles=["Equity Curve", "Underlying Price"],
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=eq_df["Date"], y=eq_df["Equity"], name="Equity",
        mode="lines", line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(0,113,227,0.06)",
    ), row=1, col=1)

    eq = eq_df["Equity"].to_numpy(dtype=np.float64)
    if len(eq) > 1:
        cummax = np.maximum.accumulate(eq)
        dd = (eq - cummax) / np.where(cummax > 0, cummax, 1.0) * 100
        fig.add_trace(go.Scatter(
            x=eq_df["Date"], y=dd, name="Drawdown %",
            mode="lines", line=dict(color=COLORS["red"], width=1),
            fill="tozeroy", fillcolor="rgba(255,59,48,0.08)",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pr_df["Date"], y=pr_df["Close"], name="Price",
        mode="lines", line=dict(color=COLORS["muted"], width=1),
    ), row=2, col=1)

    fig.update_layout(height=500, hovermode="x unified", legend=dict(orientation="h", y=1.05))

    fills = store.get("fills_data", [])
    if fills:
        rows = []
        for f in fills:
            rows.append(html.Tr([
                html.Td(html.Span(f["direction"], className=f"tag {'tag-green' if f['direction']=='LONG' else 'tag-red'}")),
                html.Td(f"{f['qty']:.0f}"),
                html.Td(f"${f['price']:,.2f}"),
                html.Td(f"${f['commission']:.2f}"),
                html.Td(f["time"], style={"color": "#6e6e73"}),
            ]))
        from app.components import data_table
        table = data_table(
            ["Direction", "Qty", "Price", "Commission", "Time"],
            rows,
        )
    else:
        table = html.P("No fills during this backtest.", className="empty-state")

    return fig, _metrics_cards(metrics), table
