"""
Strategy Lab — AmiBroker-style analysis workspace for futures strategies.

Three analysis modes operate on historical futures data (2000 → today):

* **Backtest**       — single run over the selected date range / periodicity.
* **Optimize**       — exhaustive grid sweep over chosen parameters.
* **Walk-Forward**   — Optuna in-sample optimisation with out-of-sample validation.

Data is sourced from real CME bars (Databento when ``DATABENTO_API_KEY`` is set,
otherwise yfinance daily back to ~2000 with synthetic intraday fill).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

from app.components import (
    control_group,
    data_table,
    metric_tile,
    page_header,
    primary_button,
    section_card,
)
from app.styles import DROPDOWN_STYLE
from app.theme import COLORS
from app.data_service import (
    get_analysis_config,
    run_strategy_analysis,
)
from engine.optimizer import STRATEGY_PARAM_SPACES


STRATEGIES = [
    {"value": "MeanReversionRSI", "label": "Mean Reversion RSI"},
    {"value": "MomentumBreakout", "label": "Momentum Breakout (Donchian)"},
    {"value": "TrendFollowingMACD", "label": "Trend Following MACD"},
]

CONTRACTS = [
    {"value": "ES", "label": "ES — E-mini S&P 500"},
    {"value": "NQ", "label": "NQ — E-mini Nasdaq-100"},
    {"value": "CL", "label": "CL — Crude Oil WTI"},
    {"value": "GC", "label": "GC — Gold"},
    {"value": "NG", "label": "NG — Natural Gas"},
    {"value": "SI", "label": "SI — Silver"},
    {"value": "ZN", "label": "ZN — 10-Year T-Note"},
    {"value": "ZB", "label": "ZB — 30-Year T-Bond"},
]

TIMEFRAMES = [
    {"value": "1d", "label": "Daily"},
    {"value": "4h", "label": "4 Hour"},
    {"value": "1h", "label": "1 Hour"},
    {"value": "30m", "label": "30 Min"},
    {"value": "15m", "label": "15 Min"},
    {"value": "5m", "label": "5 Min"},
    {"value": "1m", "label": "1 Min"},
]

DATA_SOURCES = [
    {"value": "auto", "label": "Auto (Databento → Yahoo)"},
    {"value": "historical", "label": "Historical (Yahoo + synthetic intraday)"},
    {"value": "synthetic", "label": "Synthetic (deterministic)"},
]

PARAM_LABELS = {
    "rsi_period": "RSI Period",
    "rsi_oversold": "RSI Oversold",
    "rsi_overbought": "RSI Overbought",
    "bb_period": "BB Period",
    "bb_std": "BB Std Dev",
    "atr_period": "ATR Period",
    "lookback": "Lookback",
    "sma_short": "SMA Short",
    "sma_long": "SMA Long",
}

OBJECTIVE_LABELS = {
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "calmar_ratio": "Calmar Ratio",
    "total_return": "Total Return",
    "profit_factor": "Profit Factor",
}

_MAX_PLOT_POINTS = 4000


def _objective_options() -> List[Dict[str, str]]:
    cfg = get_analysis_config()
    return [
        {"value": m, "label": OBJECTIVE_LABELS.get(m, m)}
        for m in cfg.analysis.objective_metrics
    ]


def _param_options(strategy: str) -> List[Dict[str, str]]:
    return [
        {"value": p, "label": PARAM_LABELS.get(p, p)}
        for p in STRATEGY_PARAM_SPACES.get(strategy, [])
    ]


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(value: Any, kind: str = "ratio") -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "pct":
        return f"{v * 100:+.2f}%"
    if kind == "ratio":
        if abs(v) >= 1e6:
            return f"{v:.2e}"
        return f"{v:.3f}"
    if kind == "int":
        return f"{int(v):,}"
    return f"{v:.2f}"


def _metrics_cards(metrics: dict) -> list:
    if not metrics:
        return [html.P("Run an analysis to see performance metrics.", className="empty-state")]

    order = [
        ("total_return", "Total Return", "pct"),
        ("cagr", "CAGR", "pct"),
        ("sharpe_ratio", "Sharpe Ratio", "ratio"),
        ("sortino_ratio", "Sortino Ratio", "ratio"),
        ("max_drawdown", "Max Drawdown", "pct"),
        ("calmar_ratio", "Calmar Ratio", "ratio"),
        ("win_rate", "Win Rate", "pct"),
        ("profit_factor", "Profit Factor", "ratio"),
        ("var_95", "VaR (95%)", "pct"),
        ("n_periods", "Bars Tested", "int"),
    ]
    pos_keys = {"total_return", "cagr", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "win_rate", "profit_factor"}
    neg_keys = {"max_drawdown", "var_95"}

    cards = []
    for key, label, kind in order:
        val = metrics.get(key)
        if val is None:
            continue
        if key in pos_keys:
            cls = "positive" if (isinstance(val, (int, float)) and val > 0) else "negative"
        elif key in neg_keys:
            cls = "negative" if (isinstance(val, (int, float)) and val < 0) else "positive"
        else:
            cls = "neutral"
        cards.append(html.Div(
            metric_tile(label, _fmt(val, kind), value_cls=cls),
            className="metric-tile",
            style={"padding": "16px"},
        ))
    return cards


def _coverage_badge(meta: dict) -> html.Div:
    if not meta:
        return html.Div()
    items = []
    pairs = [
        ("Contract", meta.get("symbol")),
        ("Source", meta.get("data_source")),
        ("Range", f"{meta.get('range_from', '?')} → {meta.get('range_to', '?')}"),
        ("Bars", f"{meta.get('bars_used', 0):,}"),
    ]
    for key, val in pairs:
        if val is None:
            continue
        items.append(html.Span([
            html.Span(f"{key}: ", className="badge-key"),
            html.Span(str(val), className="badge-val"),
        ], className="badge-item"))

    real_bars = meta.get("real_bars")
    synth_bars = meta.get("synthetic_bars")
    if real_bars is not None:
        items.append(html.Span([
            html.Span("Real bars: ", className="badge-key"),
            html.Span(f"{real_bars:,}", className="badge-val badge-real"),
        ], className="badge-item"))
    if synth_bars:
        items.append(html.Span([
            html.Span("Synthetic bars: ", className="badge-key"),
            html.Span(f"{synth_bars:,}", className="badge-val badge-synth"),
        ], className="badge-item"))
    if meta.get("bar_cap_applied"):
        items.append(html.Span([
            html.Span("Capped to: ", className="badge-key"),
            html.Span(f"{meta['bar_cap_applied']:,} bars", className="badge-val badge-synth"),
        ], className="badge-item"))

    return html.Div(items, className="fut-coverage-badge")


def _downsample(x: list, y: list) -> tuple:
    n = len(x)
    if n <= _MAX_PLOT_POINTS:
        return x, y
    idx = np.linspace(0, n - 1, _MAX_PLOT_POINTS).astype(int)
    return [x[i] for i in idx], [y[i] for i in idx]


def _equity_figure(eq_x: list, eq_y: list, pr_x: list, pr_y: list, title: str) -> go.Figure:
    eq_x, eq_y = _downsample(eq_x, eq_y)
    pr_x, pr_y = _downsample(pr_x, pr_y)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
        subplot_titles=[title, "Underlying Price"], vertical_spacing=0.12,
    )
    eq_dates = pd.to_datetime(eq_x)
    fig.add_trace(go.Scatter(
        x=eq_dates, y=eq_y, name="Equity", mode="lines",
        line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(0,113,227,0.06)",
    ), row=1, col=1)

    eq = np.array(eq_y, dtype=np.float64)
    if len(eq) > 1:
        cummax = np.maximum.accumulate(eq)
        dd = (eq - cummax) / np.where(cummax > 0, cummax, 1.0) * 100
        fig.add_trace(go.Scatter(
            x=eq_dates, y=dd, name="Drawdown %", mode="lines",
            line=dict(color=COLORS["red"], width=1),
            fill="tozeroy", fillcolor="rgba(255,59,48,0.08)",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(pr_x), y=pr_y, name="Price", mode="lines",
        line=dict(color=COLORS["muted"], width=1),
    ), row=2, col=1)

    fig.update_layout(height=500, hovermode="x unified", legend=dict(orientation="h", y=1.05))
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    cfg = get_analysis_config()
    today = date.today().isoformat()
    start_default = cfg.analysis.history_start_date

    settings_panel = html.Div([
        html.Div("Analysis Settings", className="controls-panel-title"),

        control_group("Symbol", dcc.Dropdown(
            id="sl-contract", options=CONTRACTS, value="ES", clearable=False,
            className="dash-dropdown", style=DROPDOWN_STYLE,
        )),
        control_group("Formula (Strategy)", dcc.Dropdown(
            id="sl-strategy", options=STRATEGIES, value="MeanReversionRSI", clearable=False,
            className="dash-dropdown", style=DROPDOWN_STYLE,
        )),
        control_group("Periodicity", dcc.Dropdown(
            id="sl-timeframe", options=TIMEFRAMES, value=cfg.analysis.default_timeframe,
            clearable=False, className="dash-dropdown", style=DROPDOWN_STYLE,
        )),
        control_group("Data Source", dcc.Dropdown(
            id="sl-data-source", options=DATA_SOURCES, value=cfg.analysis.default_data_source,
            clearable=False, className="dash-dropdown", style=DROPDOWN_STYLE,
        )),

        control_group("Date Range", dcc.DatePickerRange(
            id="sl-date-range",
            min_date_allowed=start_default,
            max_date_allowed=today,
            start_date=start_default,
            end_date=today,
            display_format="YYYY-MM-DD",
            className="sl-date-picker",
        )),

        control_group("Initial Capital ($)", dcc.Input(
            id="sl-capital", type="number", value=100_000, min=10_000, step=10_000,
        )),

        html.Hr(className="divider"),

        # ── Backtest-mode parameters ───────────────────────────────────────────
        html.Div([
            html.Div("Strategy Parameters", className="sl-param-heading"),
            control_group("RSI Period", dcc.Slider(
                id="sl-rsi-period", min=5, max=30, step=1, value=14,
                marks={5: "5", 14: "14", 21: "21", 30: "30"},
                tooltip={"placement": "bottom"},
            )),
            control_group("RSI Oversold", dcc.Slider(
                id="sl-rsi-oversold", min=15, max=45, step=1, value=30,
                marks={15: "15", 30: "30", 45: "45"}, tooltip={"placement": "bottom"},
            )),
            control_group("RSI Overbought", dcc.Slider(
                id="sl-rsi-overbought", min=55, max=85, step=1, value=70,
                marks={55: "55", 70: "70", 85: "85"}, tooltip={"placement": "bottom"},
            )),
            control_group("BB Period", dcc.Slider(
                id="sl-bb-period", min=10, max=50, step=5, value=20,
                marks={10: "10", 20: "20", 30: "30", 50: "50"}, tooltip={"placement": "bottom"},
            )),
        ], id="sl-backtest-block"),

        # ── Optimize / Walk-Forward parameters ─────────────────────────────────
        html.Div([
            html.Div("Optimisation", className="sl-param-heading"),
            control_group("Parameters to Optimise", dcc.Dropdown(
                id="sl-opt-params",
                options=_param_options("MeanReversionRSI"),
                value=["rsi_period", "rsi_oversold"],
                multi=True, className="dash-dropdown", style=DROPDOWN_STYLE,
            )),
            control_group("Objective Metric", dcc.Dropdown(
                id="sl-objective", options=_objective_options(), value="sharpe_ratio",
                clearable=False, className="dash-dropdown", style=DROPDOWN_STYLE,
            )),
            html.Div([
                control_group("Grid Steps / Parameter", dcc.Slider(
                    id="sl-grid-steps", min=2, max=6, step=1, value=cfg.analysis.grid_steps_per_param,
                    marks={2: "2", 3: "3", 4: "4", 5: "5", 6: "6"}, tooltip={"placement": "bottom"},
                )),
            ], id="sl-optimize-sub"),
            html.Div([
                control_group("Walk-Forward Windows", dcc.Slider(
                    id="sl-wfo-windows", min=2, max=8, step=1, value=cfg.analysis.walk_forward_windows,
                    marks={2: "2", 4: "4", 6: "6", 8: "8"}, tooltip={"placement": "bottom"},
                )),
                control_group("Trials / Window", dcc.Slider(
                    id="sl-wfo-trials", min=5, max=50, step=5, value=cfg.analysis.walk_forward_trials,
                    marks={5: "5", 15: "15", 25: "25", 50: "50"}, tooltip={"placement": "bottom"},
                )),
            ], id="sl-wfo-sub", style={"display": "none"}),
        ], id="sl-research-block", style={"display": "none"}),

        primary_button("Run Backtest", "sl-run-btn", className="w-100 mt-2"),
        html.Div(id="sl-run-status", style={"marginTop": "12px", "fontSize": "13px", "color": "#6e6e73"}),
    ], className="controls-panel")

    results_panel = html.Div([
        html.Div(id="sl-coverage", style={"marginBottom": "16px"}),
        dcc.Loading(html.Div(id="sl-results-content"), type="default"),
    ])

    return html.Div([
        page_header(
            "Strategy Lab",
            "Backtest, optimise, and walk-forward validate futures strategies on "
            "historical data from 2000 to today — modelled on AmiBroker's Analysis window.",
            badge="Analysis",
        ),

        dcc.Tabs(
            id="sl-mode", value="backtest", className="charts-mode-tabs", children=[
                dcc.Tab(label="Backtest", value="backtest",
                        className="charts-mode-tab", selected_className="charts-mode-tab--active"),
                dcc.Tab(label="Optimize", value="optimize",
                        className="charts-mode-tab", selected_className="charts-mode-tab--active"),
                dcc.Tab(label="Walk-Forward", value="walk_forward",
                        className="charts-mode-tab", selected_className="charts-mode-tab--active"),
            ],
        ),

        html.Div([settings_panel, results_panel], className="split-layout", style={"marginTop": "20px"}),

        dcc.Store(id="sl-results-store"),
    ])


# ── Mode-driven control visibility ────────────────────────────────────────────

@callback(
    Output("sl-backtest-block", "style"),
    Output("sl-research-block", "style"),
    Output("sl-optimize-sub", "style"),
    Output("sl-wfo-sub", "style"),
    Output("sl-run-btn", "children"),
    Input("sl-mode", "value"),
)
def toggle_mode_controls(mode: str):
    show = {"display": "block"}
    hide = {"display": "none"}
    if mode == "optimize":
        return hide, show, show, hide, "Run Optimization"
    if mode == "walk_forward":
        return hide, show, hide, show, "Run Walk-Forward"
    return show, hide, hide, hide, "Run Backtest"


@callback(
    Output("sl-opt-params", "options"),
    Output("sl-opt-params", "value"),
    Input("sl-strategy", "value"),
)
def update_param_options(strategy: str):
    options = _param_options(strategy or "MeanReversionRSI")
    default = [o["value"] for o in options[:2]]
    return options, default


# ── Run analysis ──────────────────────────────────────────────────────────────

@callback(
    Output("sl-results-store", "data"),
    Output("sl-run-status", "children"),
    Input("sl-run-btn", "n_clicks"),
    State("sl-mode", "value"),
    State("sl-contract", "value"),
    State("sl-strategy", "value"),
    State("sl-timeframe", "value"),
    State("sl-data-source", "value"),
    State("sl-date-range", "start_date"),
    State("sl-date-range", "end_date"),
    State("sl-capital", "value"),
    State("sl-rsi-period", "value"),
    State("sl-rsi-oversold", "value"),
    State("sl-rsi-overbought", "value"),
    State("sl-bb-period", "value"),
    State("sl-opt-params", "value"),
    State("sl-objective", "value"),
    State("sl-grid-steps", "value"),
    State("sl-wfo-windows", "value"),
    State("sl-wfo-trials", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, mode, contract, strategy, timeframe, data_source,
                 start_date, end_date, capital, rsi_period, rsi_oversold,
                 rsi_overbought, bb_period, opt_params, objective,
                 grid_steps, wfo_windows, wfo_trials):
    if not n_clicks:
        return no_update, ""

    try:
        backtest_params = {
            "rsi_period": int(rsi_period or 14),
            "rsi_oversold": float(rsi_oversold or 30),
            "rsi_overbought": float(rsi_overbought or 70),
            "bb_period": int(bb_period or 20),
        }
        result = run_strategy_analysis(
            mode=mode or "backtest",
            symbol=contract or "ES",
            strategy_name=strategy or "MeanReversionRSI",
            timeframe=timeframe or "1d",
            start_date=start_date,
            end_date=end_date,
            data_source=data_source or "auto",
            initial_cash=float(capital or 100_000),
            params=backtest_params,
            optimize_params=opt_params or None,
            objective_metric=objective or "sharpe_ratio",
            grid_steps=int(grid_steps) if grid_steps else None,
            n_windows=int(wfo_windows) if wfo_windows else None,
            n_trials=int(wfo_trials) if wfo_trials else None,
        )
    except Exception as exc:  # pragma: no cover - surfaced to UI
        return {"mode": "error", "error": str(exc)[:200]}, f"Error: {str(exc)[:80]}"

    if result.get("error"):
        return {"mode": "error", "error": result["error"], "coverage": result.get("coverage", {})}, \
            f"Error: {result['error'][:80]}"

    store = _serialize_result(result)
    status = _status_text(store)
    return store, status


def _serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    mode = result.get("mode")
    store: Dict[str, Any] = {"mode": mode, "coverage": result.get("coverage", {})}

    if mode in ("backtest", "optimize"):
        eq = result.get("equity_curve")
        pr = result.get("price_df")
        store["equity_x"] = eq["Date"].astype(str).tolist()
        store["equity_y"] = [float(v) for v in eq["Equity"].tolist()]
        store["price_x"] = pr["Date"].astype(str).tolist()
        store["price_y"] = [float(v) for v in pr["Close"].tolist()]

    if mode == "backtest":
        store["metrics"] = result.get("metrics", {})
        store["params"] = result.get("params", {})
        fills = result.get("fills", [])
        store["n_fills"] = len(fills)
        store["fills_data"] = [
            {
                "direction": f.direction.value,
                "qty": f.filled_quantity,
                "price": f.fill_price,
                "commission": f.commission,
                "time": str(f.timestamp)[:16],
            }
            for f in fills[:100]
        ]
    elif mode == "optimize":
        store["objective_metric"] = result.get("objective_metric")
        store["param_names"] = result.get("param_names", [])
        store["n_combinations"] = result.get("n_combinations")
        store["trials"] = result.get("trials", [])
        store["best_params"] = result.get("best_params", {})
        store["best_metrics"] = result.get("best_metrics", {})
    elif mode == "walk_forward":
        store["objective_metric"] = result.get("objective_metric")
        store["param_names"] = result.get("param_names", [])
        store["n_windows"] = result.get("n_windows")
        store["n_trials"] = result.get("n_trials")
        store["windows"] = result.get("windows", [])
        store["aggregated_oos_metrics"] = result.get("aggregated_oos_metrics", {})

    return store


def _status_text(store: Dict[str, Any]) -> str:
    mode = store.get("mode")
    cov = store.get("coverage", {})
    bars = cov.get("bars_used", 0)
    if mode == "backtest":
        return f"Backtest complete — {bars:,} bars, {store.get('n_fills', 0)} fills"
    if mode == "optimize":
        return f"Optimization complete — {store.get('n_combinations', 0)} combinations over {bars:,} bars"
    if mode == "walk_forward":
        return f"Walk-forward complete — {store.get('n_windows', 0)} windows over {bars:,} bars"
    return "Complete"


# ── Render results ──────────────────────────────────────────────────────────

@callback(
    Output("sl-results-content", "children"),
    Output("sl-coverage", "children"),
    Input("sl-results-store", "data"),
)
def render_results(store):
    if not store:
        return html.Div([
            section_card(
                "Results",
                "Configure the analysis on the left, then run a Backtest, "
                "Optimization, or Walk-Forward study.",
                html.P("No analysis run yet.", className="empty-state"),
            )
        ], **{}), html.Div()

    coverage = _coverage_badge(store.get("coverage", {}))
    mode = store.get("mode")

    if mode == "error":
        return section_card(
            "Analysis Error",
            "The analysis could not be completed.",
            html.P(store.get("error", "Unknown error."), className="empty-state"),
        ), coverage

    if mode == "backtest":
        return _render_backtest(store), coverage
    if mode == "optimize":
        return _render_optimize(store), coverage
    if mode == "walk_forward":
        return _render_walk_forward(store), coverage
    return html.Div(), coverage


def _render_backtest(store) -> html.Div:
    metrics = store.get("metrics", {})
    fig = _equity_figure(
        store["equity_x"], store["equity_y"],
        store["price_x"], store["price_y"], "Equity Curve",
    )

    fills = store.get("fills_data", [])
    if fills:
        rows = [
            html.Tr([
                html.Td(html.Span(f["direction"], className=f"tag {'tag-green' if f['direction']=='LONG' else 'tag-red'}")),
                html.Td(f"{f['qty']:.0f}"),
                html.Td(f"${f['price']:,.2f}"),
                html.Td(f"${f['commission']:.2f}"),
                html.Td(f["time"], style={"color": "#6e6e73"}),
            ])
            for f in fills
        ]
        fills_table = data_table(["Direction", "Qty", "Price", "Commission", "Time"], rows)
    else:
        fills_table = html.P("No fills during this backtest.", className="empty-state")

    return html.Div([
        html.Div(_metrics_cards(metrics), className="metrics-grid"),
        section_card(
            "Equity Curve & Drawdown",
            "Portfolio value and drawdown over the selected period, with the underlying price below.",
            dcc.Graph(figure=fig, config={"displayModeBar": True, "displaylogo": False}),
        ),
        section_card("Fill Log", "Executed fills during the backtest (first 100)", fills_table),
    ])


def _render_optimize(store) -> html.Div:
    objective = store.get("objective_metric", "sharpe_ratio")
    obj_label = OBJECTIVE_LABELS.get(objective, objective)
    param_names = store.get("param_names", [])
    trials = store.get("trials", [])
    best_params = store.get("best_params", {})

    headers = [PARAM_LABELS.get(p, p) for p in param_names] + [
        obj_label, "Total Return", "Max DD", "Win Rate", "Trades"
    ]
    rows = []
    for i, tr in enumerate(trials):
        params = tr.get("params", {})
        m = tr.get("metrics", {})
        is_best = params == best_params
        cells = [html.Td(_fmt_param(params.get(p))) for p in param_names]
        cells += [
            html.Td(html.B(_fmt(tr.get("objective"), "ratio")) if is_best else _fmt(tr.get("objective"), "ratio")),
            html.Td(_fmt(m.get("total_return"), "pct")),
            html.Td(_fmt(m.get("max_drawdown"), "pct")),
            html.Td(_fmt(m.get("win_rate"), "pct")),
            html.Td(_fmt(m.get("n_periods"), "int")),
        ]
        rows.append(html.Tr(cells, className="sl-best-row" if is_best else ""))

    table = data_table(headers, rows, className="sl-results-table")

    best_summary = html.Div([
        html.Span("Best parameters: ", className="badge-key"),
        html.Span(", ".join(f"{PARAM_LABELS.get(k, k)}={_fmt_param(v)}" for k, v in best_params.items()) or "—"),
    ], className="sl-best-summary")

    fig = _equity_figure(
        store["equity_x"], store["equity_y"],
        store["price_x"], store["price_y"], "Best-Parameter Equity Curve",
    )

    return html.Div([
        html.Div(_metrics_cards(store.get("best_metrics", {})), className="metrics-grid"),
        section_card(
            f"Optimization Results — ranked by {obj_label}",
            f"{store.get('n_combinations', 0)} parameter combinations backtested over the full range.",
            html.Div([best_summary, table]),
        ),
        section_card(
            "Best-Parameter Equity Curve",
            "Equity curve produced by the top-ranked parameter set.",
            dcc.Graph(figure=fig, config={"displayModeBar": True, "displaylogo": False}),
        ),
    ])


def _render_walk_forward(store) -> html.Div:
    objective = store.get("objective_metric", "sharpe_ratio")
    obj_label = OBJECTIVE_LABELS.get(objective, objective)
    param_names = store.get("param_names", [])
    windows = store.get("windows", [])

    headers = ["Window", "Best Parameters", f"IS {obj_label}", f"OOS {obj_label}",
               "OOS Return", "OOS Max DD"]
    rows = []
    for w in windows:
        bp = w.get("best_params", {})
        oos_m = w.get("oos_metrics", {})
        bp_str = ", ".join(f"{PARAM_LABELS.get(k, k)}={_fmt_param(v)}" for k, v in bp.items())
        rows.append(html.Tr([
            html.Td(f"#{w.get('window_id', 0) + 1}"),
            html.Td(bp_str, style={"fontSize": "12px"}),
            html.Td(_fmt(w.get("is_objective"), "ratio")),
            html.Td(_fmt(w.get("oos_objective"), "ratio")),
            html.Td(_fmt(oos_m.get("total_return"), "pct")),
            html.Td(_fmt(oos_m.get("max_drawdown"), "pct")),
        ]))
    table = data_table(headers, rows, className="sl-results-table")

    # Per-window IS vs OOS objective bar chart
    win_ids = [f"#{w.get('window_id', 0) + 1}" for w in windows]
    is_vals = [_safe_num(w.get("is_objective")) for w in windows]
    oos_vals = [_safe_num(w.get("oos_objective")) for w in windows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=win_ids, y=is_vals, name="In-Sample", marker_color=COLORS["blue"]))
    fig.add_trace(go.Bar(x=win_ids, y=oos_vals, name="Out-of-Sample", marker_color=COLORS["orange"]))
    fig.update_layout(
        height=380, barmode="group", hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        yaxis_title=obj_label, xaxis_title="Walk-Forward Window",
    )

    agg = store.get("aggregated_oos_metrics", {})
    agg_cards = _metrics_cards(agg)

    return html.Div([
        section_card(
            "Aggregated Out-of-Sample Metrics",
            "Averaged across all walk-forward windows — the unbiased estimate of live performance.",
            html.Div(agg_cards, className="metrics-grid"),
        ),
        section_card(
            f"In-Sample vs Out-of-Sample {obj_label}",
            "Large gaps between IS and OOS bars indicate curve-fitting.",
            dcc.Graph(figure=fig, config={"displayModeBar": True, "displaylogo": False}),
        ),
        section_card(
            "Per-Window Detail",
            f"Best parameters and out-of-sample performance for each of {store.get('n_windows', 0)} windows "
            f"({store.get('n_trials', 0)} Optuna trials per window).",
            table,
        ),
    ])


def _fmt_param(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _safe_num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return v
