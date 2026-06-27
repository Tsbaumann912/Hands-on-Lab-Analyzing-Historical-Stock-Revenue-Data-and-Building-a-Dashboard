"""
Charts page — two-mode charting hub.

Tab 1  «TradingView»   — free TradingView Advanced Chart widget (stocks, crypto,
                         forex, indices). Requires TradingView subscription for
                         real-time CME/NYMEX/COMEX futures data.

Tab 2  «Futures Data»  — server-side Plotly candlestick chart backed by yfinance.
                         Provides free, unrestricted access to 35+ continuous
                         futures contracts across all asset classes with full
                         indicator overlay support. Use this tab for strategy
                         research and backtesting on futures data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Input, Output, State, callback, clientside_callback, dcc, html

from app.theme import COLORS
from app.data_service import (
    FUTURES_CHART_CATALOG,
    TIMEFRAME_CONFIG,
    compute_indicators_for_ui,
    fetch_futures_timeframe_data,
)


# ── TradingView symbol catalogue ──────────────────────────────────────────────

TV_SYMBOLS: list[dict[str, str]] = [
    # Indices
    {"label": "S&P 500 Index",        "value": "SP:SPX"},
    {"label": "NASDAQ 100 Index",     "value": "NASDAQ:NDX"},
    {"label": "Dow Jones Index",      "value": "DJ:DJI"},
    {"label": "Russell 2000",         "value": "TVC:RUT"},
    {"label": "VIX Volatility",       "value": "TVC:VIX"},
    # US Equities
    {"label": "Apple (AAPL)",         "value": "NASDAQ:AAPL"},
    {"label": "Tesla (TSLA)",         "value": "NASDAQ:TSLA"},
    {"label": "NVIDIA (NVDA)",        "value": "NASDAQ:NVDA"},
    {"label": "Microsoft (MSFT)",     "value": "NASDAQ:MSFT"},
    {"label": "Amazon (AMZN)",        "value": "NASDAQ:AMZN"},
    {"label": "Google (GOOGL)",       "value": "NASDAQ:GOOGL"},
    {"label": "Meta (META)",          "value": "NASDAQ:META"},
    {"label": "Berkshire (BRK.B)",    "value": "NYSE:BRK.B"},
    {"label": "JPMorgan (JPM)",       "value": "NYSE:JPM"},
    # Crypto
    {"label": "Bitcoin / USD",        "value": "BINANCE:BTCUSDT"},
    {"label": "Ethereum / USD",       "value": "BINANCE:ETHUSDT"},
    {"label": "Solana / USD",         "value": "BINANCE:SOLUSDT"},
    # Forex
    {"label": "EUR / USD",            "value": "FX:EURUSD"},
    {"label": "GBP / USD",            "value": "FX:GBPUSD"},
    {"label": "USD / JPY",            "value": "FX:USDJPY"},
    {"label": "USD / CHF",            "value": "FX:USDCHF"},
]

TV_INTERVALS: list[dict[str, str]] = [
    {"label": "1m",  "value": "1"},
    {"label": "3m",  "value": "3"},
    {"label": "5m",  "value": "5"},
    {"label": "15m", "value": "15"},
    {"label": "30m", "value": "30"},
    {"label": "1H",  "value": "60"},
    {"label": "2H",  "value": "120"},
    {"label": "4H",  "value": "240"},
    {"label": "1D",  "value": "D"},
    {"label": "1W",  "value": "W"},
    {"label": "1M",  "value": "M"},
]

TV_CHART_STYLES: list[dict[str, str]] = [
    {"label": "Candlestick",    "value": "1"},
    {"label": "Bars",           "value": "0"},
    {"label": "Line",           "value": "2"},
    {"label": "Area",           "value": "3"},
    {"label": "Heikin Ashi",    "value": "8"},
    {"label": "Hollow Candles", "value": "9"},
    {"label": "Baseline",       "value": "10"},
    {"label": "Hi-Lo",          "value": "12"},
]

TV_STUDIES: list[dict[str, str]] = [
    {"label": "Volume",         "value": "Volume@tv-basicstudies"},
    {"label": "RSI (14)",       "value": "RSI@tv-basicstudies"},
    {"label": "MACD",           "value": "MACD@tv-basicstudies"},
    {"label": "Bollinger Bands","value": "BB@tv-basicstudies"},
    {"label": "EMA 20",         "value": "MAExp@tv-basicstudies"},
    {"label": "VWAP",           "value": "VWAP@tv-basicstudies"},
    {"label": "Stochastic",     "value": "Stoch@tv-basicstudies"},
    {"label": "ATR",            "value": "ATR@tv-basicstudies"},
    {"label": "Ichimoku Cloud", "value": "IchimokuCloud@tv-basicstudies"},
    {"label": "Supertrend",     "value": "Supertrend@tv-basicstudies"},
    {"label": "Parabolic SAR",  "value": "PSAR@tv-basicstudies"},
    {"label": "CCI",            "value": "CCI@tv-basicstudies"},
    {"label": "Williams %R",    "value": "WilliamR@tv-basicstudies"},
    {"label": "MFI",            "value": "MFI@tv-basicstudies"},
    {"label": "OBV",            "value": "OBV@tv-basicstudies"},
]


# ── Futures Data catalogue helpers ────────────────────────────────────────────

def _futures_dropdown_options() -> list[dict[str, Any]]:
    """Return grouped dropdown options from FUTURES_CHART_CATALOG."""
    sector_order = ["Index", "Energy", "Metals", "Bonds", "FX", "Ags", "Crypto"]
    groups: dict[str, list[dict[str, str]]] = {s: [] for s in sector_order}
    for sym, meta in FUTURES_CHART_CATALOG.items():
        sector = meta["sector"]
        if sector not in groups:
            groups[sector] = []
        groups[sector].append({"label": meta["label"], "value": sym})

    options: list[dict[str, Any]] = []
    for sector in sector_order:
        if groups[sector]:
            options.append({"label": f"── {sector} ──", "value": f"__hdr_{sector}", "disabled": True})
            options.extend(groups[sector])
    return options


FUTURES_OPTIONS  = _futures_dropdown_options()

TIMEFRAME_OPTIONS: list[dict[str, str]] = [
    {"label": "1 Minute",   "value": "1m"},
    {"label": "5 Minutes",  "value": "5m"},
    {"label": "15 Minutes", "value": "15m"},
    {"label": "30 Minutes", "value": "30m"},
    {"label": "1 Hour",     "value": "1h"},
    {"label": "4 Hours",    "value": "4h"},
    {"label": "Daily",      "value": "1d"},
]

# How many bars to show in the chart (the dataset is always 2000-present;
# this just controls the visible slice for rendering performance).
DISPLAY_BARS_OPTIONS: list[dict] = [
    {"label": "Last 500 bars",    "value": 500},
    {"label": "Last 1 000 bars",  "value": 1_000},
    {"label": "Last 2 000 bars",  "value": 2_000},
    {"label": "Last 5 000 bars",  "value": 5_000},
    {"label": "All (up to 10 k)", "value": 10_000},
]

OVERLAY_OPTIONS: list[dict[str, str]] = [
    {"label": "SMA 20",          "value": "SMA_20"},
    {"label": "SMA 50",          "value": "SMA_50"},
    {"label": "EMA 20",          "value": "EMA_20"},
    {"label": "Bollinger Bands", "value": "BB"},
    {"label": "VWAP",            "value": "VWAP"},
]

SUBPANEL_OPTIONS: list[dict[str, str]] = [
    {"label": "None",      "value": "none"},
    {"label": "RSI (14)",  "value": "RSI_14"},
    {"label": "MACD",      "value": "MACD_line"},
    {"label": "ATR (14)",  "value": "ATR_14"},
    {"label": "OBV",       "value": "OBV"},
]


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([
        # ── Page header ────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H1("Charts", className="page-title"),
                html.P(
                    "Two-mode charting hub — TradingView widget for stocks & crypto, "
                    "plus a free Futures Data mode powered by yfinance for all CME/NYMEX/COMEX/CBOT contracts.",
                    className="page-subtitle",
                ),
            ], className="page-header-text"),
        ], className="page-header"),

        # ── Mode tabs ──────────────────────────────────────────────────────────
        dcc.Tabs(
            id="charts-mode-tabs",
            value="tradingview",
            className="charts-mode-tabs",
            children=[
                dcc.Tab(
                    label="📈  TradingView",
                    value="tradingview",
                    className="charts-mode-tab",
                    selected_className="charts-mode-tab--active",
                ),
                dcc.Tab(
                    label="📊  Futures Data",
                    value="futures",
                    className="charts-mode-tab",
                    selected_className="charts-mode-tab--active",
                ),
            ],
        ),

        # ── TradingView panel ──────────────────────────────────────────────────
        html.Div([

            # Controls bar
            html.Div([
                html.Div([

                    html.Div([
                        html.Label("Symbol", className="ctrl-label"),
                        dcc.Input(
                            id="tv-symbol-input",
                            type="text",
                            value="NASDAQ:AAPL",
                            placeholder="EXCHANGE:TICKER",
                            debounce=True,
                            className="tv-symbol-input",
                        ),
                    ], className="ctrl-group ctrl-group--symbol"),

                    html.Div([
                        html.Label("Quick Pick", className="ctrl-label"),
                        dcc.Dropdown(
                            id="tv-quick-symbol",
                            options=TV_SYMBOLS,
                            placeholder="Popular symbols…",
                            clearable=True,
                            className="tv-dropdown",
                        ),
                    ], className="ctrl-group ctrl-group--quick"),

                    html.Div([
                        html.Label("Interval", className="ctrl-label"),
                        dcc.Dropdown(
                            id="tv-interval",
                            options=TV_INTERVALS,
                            value="D",
                            clearable=False,
                            className="tv-dropdown tv-dropdown--sm",
                        ),
                    ], className="ctrl-group ctrl-group--sm"),

                    html.Div([
                        html.Label("Style", className="ctrl-label"),
                        dcc.Dropdown(
                            id="tv-chart-style",
                            options=TV_CHART_STYLES,
                            value="1",
                            clearable=False,
                            className="tv-dropdown tv-dropdown--sm",
                        ),
                    ], className="ctrl-group ctrl-group--sm"),

                    html.Div([
                        html.Label("Indicators", className="ctrl-label"),
                        dcc.Dropdown(
                            id="tv-studies",
                            options=TV_STUDIES,
                            value=["Volume@tv-basicstudies"],
                            multi=True,
                            placeholder="Add indicators…",
                            className="tv-dropdown tv-dropdown--lg",
                        ),
                    ], className="ctrl-group ctrl-group--lg"),

                    html.Div([
                        html.Label("Theme", className="ctrl-label"),
                        dcc.RadioItems(
                            id="tv-theme",
                            options=[
                                {"label": "Dark",  "value": "dark"},
                                {"label": "Light", "value": "light"},
                            ],
                            value="dark",
                            inline=True,
                            className="tv-theme-toggle",
                            inputClassName="tv-theme-radio",
                            labelClassName="tv-theme-label",
                        ),
                    ], className="ctrl-group ctrl-group--theme"),

                ], className="tv-controls-inner"),
            ], className="tv-controls-bar section-card"),

            # TradingView chart embed
            html.Div([
                html.Div(id="tradingview-chart", className="tv-chart-embed"),
            ], className="tv-chart-wrapper section-card"),

            html.Div([
                html.P([
                    "Charts powered by ",
                    html.A("TradingView", href="https://www.tradingview.com/",
                           target="_blank", rel="noopener noreferrer",
                           className="tv-attribution-link"),
                    ". Stocks, crypto, and forex work without a login. "
                    "CME/NYMEX/COMEX futures require a TradingView data subscription — "
                    "use the Futures Data tab above for free unrestricted futures charting.",
                ], className="tv-attribution-text"),
            ], className="tv-attribution"),

            html.Div(id="tv-widget-sink", style={"display": "none"}),
            dcc.Interval(id="tv-init-interval", interval=800, max_intervals=1),

        ], id="tv-panel", className="charts-panel"),

        # ── Futures Data panel ─────────────────────────────────────────────────
        html.Div([

            # Controls bar
            html.Div([
                html.Div([

                    html.Div([
                        html.Label("Contract", className="ctrl-label"),
                        dcc.Dropdown(
                            id="fut-contract",
                            options=FUTURES_OPTIONS,
                            value="ES=F",
                            clearable=False,
                            className="tv-dropdown tv-dropdown--contract",
                            optionHeight=36,
                        ),
                    ], className="ctrl-group ctrl-group--contract"),

                    html.Div([
                        html.Label("Timeframe", className="ctrl-label"),
                        dcc.Dropdown(
                            id="fut-timeframe",
                            options=TIMEFRAME_OPTIONS,
                            value="1d",
                            clearable=False,
                            className="tv-dropdown tv-dropdown--sm",
                        ),
                    ], className="ctrl-group ctrl-group--sm"),

                    html.Div([
                        html.Label("Display", className="ctrl-label"),
                        dcc.Dropdown(
                            id="fut-display-bars",
                            options=DISPLAY_BARS_OPTIONS,
                            value=2_000,
                            clearable=False,
                            className="tv-dropdown tv-dropdown--sm",
                        ),
                    ], className="ctrl-group ctrl-group--sm"),

                    html.Div([
                        html.Label("Overlays", className="ctrl-label"),
                        dcc.Dropdown(
                            id="fut-overlays",
                            options=OVERLAY_OPTIONS,
                            value=["EMA_20"],
                            multi=True,
                            placeholder="Add overlays…",
                            className="tv-dropdown tv-dropdown--lg",
                        ),
                    ], className="ctrl-group ctrl-group--lg"),

                    html.Div([
                        html.Label("Sub-Panel", className="ctrl-label"),
                        dcc.Dropdown(
                            id="fut-subpanel",
                            options=SUBPANEL_OPTIONS,
                            value="RSI_14",
                            clearable=False,
                            className="tv-dropdown tv-dropdown--sm",
                        ),
                    ], className="ctrl-group ctrl-group--sm"),

                    html.Div([
                        html.Button(
                            "↻ Refresh",
                            id="fut-refresh-btn",
                            className="fut-refresh-btn",
                            n_clicks=0,
                        ),
                    ], className="ctrl-group ctrl-group--btn"),

                ], className="tv-controls-inner"),
            ], className="tv-controls-bar section-card"),

            # Data-coverage badge
            html.Div(id="fut-coverage-badge", className="fut-coverage-badge"),

            # Metrics row
            html.Div(id="fut-metrics-row", className="fut-metrics-row"),

            # Main chart
            html.Div([
                dcc.Loading(
                    id="fut-chart-loading",
                    type="circle",
                    color="#0071e3",
                    children=dcc.Graph(
                        id="fut-chart",
                        config={"displayModeBar": True, "scrollZoom": True,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                "displaylogo": False},
                        style={"height": "660px"},
                        className="fut-chart-graph",
                    ),
                ),
            ], className="tv-chart-wrapper section-card"),

            html.Div([
                html.P([
                    "History: 2000 → present across all 7 timeframes. ",
                    html.Strong("Real bars "), "(solid label) come from Yahoo Finance (yfinance). ",
                    html.Strong("Synthetic bars "), "(dashed label) are generated by resampling the real "
                    "daily series via vectorised GBM — statistically calibrated to the same "
                    "volatility and trend but without intraday tick structure. "
                    "For live tick-level data connect a Databento key in ",
                    html.Code("config/default.yaml"), ".",
                ], className="tv-attribution-text"),
            ], className="tv-attribution"),

        ], id="fut-panel", className="charts-panel", style={"display": "none"}),

    ], className="page-content charts-page")


# ── Tab visibility callback ───────────────────────────────────────────────────

@callback(
    Output("tv-panel",  "style"),
    Output("fut-panel", "style"),
    Input("charts-mode-tabs", "value"),
)
def switch_panel(tab: str) -> tuple[dict, dict]:
    if tab == "futures":
        return {"display": "none"}, {}
    return {}, {"display": "none"}


# ── TradingView widget clientside callback ────────────────────────────────────

clientside_callback(
    """
    function(symbolInput, quickSymbol, interval, chartStyle, studies, theme, _tick) {

        var symbol  = (quickSymbol && quickSymbol.length > 0)
                      ? quickSymbol : (symbolInput || "NASDAQ:AAPL");
        var studies = studies || [];
        var theme   = theme   || "dark";
        var bg      = theme === "dark" ? "#131722" : "#ffffff";

        var cfg = {
            container_id:        "tradingview-chart",
            autosize:            true,
            symbol:              symbol,
            interval:            interval || "D",
            timezone:            "Etc/UTC",
            theme:               theme,
            style:               chartStyle || "1",
            locale:              "en",
            toolbar_bg:          bg,
            enable_publishing:   false,
            allow_symbol_change: true,
            studies:             studies,
            show_popup_button:   true,
            popup_width:         "1200",
            popup_height:        "700",
            save_image:          true,
            hide_side_toolbar:   false,
            withdateranges:      true,
        };

        function buildWidget() {
            var el = document.getElementById("tradingview-chart");
            if (!el) { return; }
            el.innerHTML = "";
            if (window.TradingView) { new window.TradingView.widget(cfg); }
        }

        if (window.TradingView) {
            buildWidget();
        } else {
            var old = document.getElementById("tv-script-tag");
            if (old) { old.remove(); }
            var s   = document.createElement("script");
            s.id    = "tv-script-tag";
            s.src   = "https://s3.tradingview.com/tv.js";
            s.onload = function() { setTimeout(buildWidget, 200); };
            document.head.appendChild(s);
        }
        return "";
    }
    """,
    Output("tv-widget-sink", "children"),
    [
        Input("tv-symbol-input",  "value"),
        Input("tv-quick-symbol",  "value"),
        Input("tv-interval",      "value"),
        Input("tv-chart-style",   "value"),
        Input("tv-studies",       "value"),
        Input("tv-theme",         "value"),
        Input("tv-init-interval", "n_intervals"),
    ],
)


# ── Futures Data chart server-side callback ───────────────────────────────────

def _build_futures_figure(
    df: pd.DataFrame,
    symbol: str,
    overlays: list[str],
    subpanel: str,
) -> go.Figure:
    """Build an Apple-themed Plotly OHLCV figure with optional overlays and sub-panel."""
    inds = compute_indicators_for_ui(df)
    name = FUTURES_CHART_CATALOG.get(symbol, {}).get("name", symbol)

    has_sub = subpanel and subpanel != "none" and subpanel in inds

    rows    = 3 if has_sub else 2
    heights = [0.60, 0.20, 0.20] if has_sub else [0.75, 0.25]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.04,
    )

    # ── Candlestick ────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name=name,
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        increasing_fillcolor=COLORS["green"],
        decreasing_fillcolor=COLORS["red"],
        line_width=1,
    ), row=1, col=1)

    # ── Overlay indicators ────────────────────────────────────────────────────
    _overlay_styles: dict[str, dict] = {
        "SMA_20": {"color": COLORS["blue"],   "dash": "solid",  "width": 1.5, "name": "SMA 20"},
        "SMA_50": {"color": COLORS["gold"],   "dash": "solid",  "width": 1.5, "name": "SMA 50"},
        "EMA_20": {"color": COLORS["cyan"],   "dash": "solid",  "width": 1.5, "name": "EMA 20"},
        "VWAP":   {"color": COLORS["purple"], "dash": "dot",    "width": 1.5, "name": "VWAP"},
    }
    for ov in (overlays or []):
        if ov == "BB" and "BB_upper" in inds:
            for band, lbl in [("BB_upper", "BB +2σ"), ("BB_middle", "BB Mid"), ("BB_lower", "BB −2σ")]:
                if band in inds:
                    fig.add_trace(go.Scatter(
                        x=df["Date"], y=inds[band],
                        name=lbl, mode="lines",
                        line=dict(color=COLORS["muted"], width=0.8, dash="dot"),
                        showlegend=(band == "BB_upper"),
                    ), row=1, col=1)
        elif ov in _overlay_styles and ov in inds:
            s = _overlay_styles[ov]
            fig.add_trace(go.Scatter(
                x=df["Date"], y=inds[ov],
                name=s["name"], mode="lines",
                line=dict(color=s["color"], width=s["width"], dash=s["dash"]),
            ), row=1, col=1)

    # ── Volume bar chart ──────────────────────────────────────────────────────
    close_arr = df["Close"].to_numpy(dtype=np.float64)
    vol_colors = np.where(
        np.diff(np.concatenate([[close_arr[0]], close_arr])) >= 0,
        COLORS["green"],
        COLORS["red"],
    )
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"],
        name="Volume",
        marker_color=vol_colors,
        marker_line_width=0,
        opacity=0.6,
        showlegend=False,
    ), row=2, col=1)

    # ── Sub-panel indicator ───────────────────────────────────────────────────
    if has_sub:
        sub_arr = inds[subpanel]
        if subpanel == "RSI_14":
            fig.add_trace(go.Scatter(
                x=df["Date"], y=sub_arr, name="RSI (14)", mode="lines",
                line=dict(color=COLORS["blue"], width=1.5),
            ), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color=COLORS["red"],   row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color=COLORS["green"], row=3, col=1)
            fig.update_yaxes(range=[0, 100], row=3, col=1)

        elif subpanel == "MACD_line" and "MACD_signal" in inds and "MACD_hist" in inds:
            hist = inds["MACD_hist"]
            fig.add_trace(go.Bar(
                x=df["Date"], y=hist, name="MACD Hist",
                marker_color=np.where(hist >= 0, COLORS["green"], COLORS["red"]),
                marker_line_width=0, opacity=0.6, showlegend=False,
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df["Date"], y=sub_arr, name="MACD",
                mode="lines", line=dict(color=COLORS["blue"], width=1.5),
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df["Date"], y=inds["MACD_signal"], name="Signal",
                mode="lines", line=dict(color=COLORS["orange"], width=1.5, dash="dot"),
            ), row=3, col=1)

        elif subpanel == "ATR_14":
            fig.add_trace(go.Scatter(
                x=df["Date"], y=sub_arr, name="ATR (14)", mode="lines",
                line=dict(color=COLORS["orange"], width=1.5),
            ), row=3, col=1)

        elif subpanel == "OBV":
            fig.add_trace(go.Scatter(
                x=df["Date"], y=sub_arr, name="OBV", mode="lines",
                line=dict(color=COLORS["purple"], width=1.5),
            ), row=3, col=1)

    # ── Layout polish ─────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"{name}  ({symbol})", x=0.02, xanchor="left"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=52, r=20, t=52, b=32),
        hovermode="x unified",
        dragmode="pan",
    )

    # Remove axis titles; keep grid subtle
    for i in range(1, rows + 1):
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)", row=i, col=1)

    return fig


def _metrics_tiles(df: pd.DataFrame, symbol: str) -> list:
    """Build the five KPI metric tiles for the Futures Data header."""
    if df.empty:
        return []

    close   = df["Close"].to_numpy(dtype=np.float64)
    vol     = df["Volume"].to_numpy(dtype=np.float64)
    last    = float(close[-1])
    prev    = float(close[-2]) if len(close) > 1 else last
    chg_pct = (last / prev - 1.0) * 100.0
    hi52    = float(np.nanmax(close))
    lo52    = float(np.nanmin(close))
    avg_vol = float(np.nanmean(vol[-20:])) if len(vol) >= 20 else float(np.nanmean(vol))

    chg_color = COLORS["green"] if chg_pct >= 0 else COLORS["red"]
    chg_sign  = "+" if chg_pct >= 0 else ""

    def _tile(title: str, value: str, sub: str = "", color: str = "") -> html.Div:
        return html.Div([
            html.Span(title, className="fut-metric-title"),
            html.Span(value, className="fut-metric-value",
                      style={"color": color} if color else {}),
            html.Span(sub,   className="fut-metric-sub") if sub else None,
        ], className="fut-metric-tile")

    name = FUTURES_CHART_CATALOG.get(symbol, {}).get("name", symbol)

    return [
        _tile("Contract",  name),
        _tile("Last Price", f"{last:,.3f}"),
        _tile("Change",    f"{chg_sign}{chg_pct:.2f}%", color=chg_color),
        _tile("Period High", f"{hi52:,.3f}"),
        _tile("Period Low",  f"{lo52:,.3f}"),
        _tile("Avg Volume (20d)", f"{avg_vol:,.0f}"),
    ]


def _coverage_badge(meta: dict, total_bars: int) -> list:
    """Build the data-coverage info strip shown above the chart."""
    tf        = meta.get("timeframe", "1d")
    real_bars = meta.get("real_bars", 0)
    synth_bars= meta.get("synthetic_bars", 0)
    real_from = meta.get("real_from", "N/A")
    real_to   = meta.get("real_to",   "N/A")
    synth_from= meta.get("synthetic_from", "N/A")
    synth_to  = meta.get("synthetic_to",   "N/A")

    cfg        = TIMEFRAME_CONFIG.get(tf, {})
    max_days   = cfg.get("max_real_days", "∞")
    ivl_label  = cfg.get("yf_interval", tf)

    return [
        html.Div([
            html.Span("Total history: ", className="badge-key"),
            html.Span(f"{total_bars:,} bars  ·  2000 → present", className="badge-val"),
        ], className="badge-item"),
        html.Div([
            html.Span("● Real  ", className="badge-key badge-real"),
            html.Span(
                f"{real_bars:,} bars  ({real_from} → {real_to}, "
                f"yfinance {ivl_label}, last {max_days}d)",
                className="badge-val",
            ),
        ], className="badge-item"),
        html.Div([
            html.Span("◌ Synthetic  ", className="badge-key badge-synth"),
            html.Span(
                f"{synth_bars:,} bars  ({synth_from} → {synth_to}, "
                "daily-resampled GBM)",
                className="badge-val",
            ),
        ], className="badge-item") if synth_bars > 0 else None,
    ]


@callback(
    Output("fut-chart",          "figure"),
    Output("fut-metrics-row",    "children"),
    Output("fut-coverage-badge", "children"),
    Input("fut-contract",        "value"),
    Input("fut-timeframe",       "value"),
    Input("fut-display-bars",    "value"),
    Input("fut-overlays",        "value"),
    Input("fut-subpanel",        "value"),
    Input("fut-refresh-btn",     "n_clicks"),
)
def update_futures_chart(
    symbol: str,
    timeframe: str,
    display_bars: int,
    overlays: list[str],
    subpanel: str,
    _refresh: int,
) -> tuple[go.Figure, list, list]:
    symbol       = symbol    or "ES=F"
    timeframe    = timeframe or "1d"
    display_bars = int(display_bars or 2_000)
    overlays     = overlays  or []
    subpanel     = subpanel  or "none"

    df, meta = fetch_futures_timeframe_data(symbol, timeframe, start_year=2000)

    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available — check connection or try another symbol.",
            margin=dict(l=52, r=20, t=52, b=32),
        )
        return empty_fig, [], []

    # Trim to the most recent display_bars for chart performance
    display_df = df.tail(display_bars).reset_index(drop=True)

    fig     = _build_futures_figure(display_df, symbol, overlays, subpanel)
    metrics = _metrics_tiles(display_df, symbol)
    badge   = _coverage_badge(meta, len(df))
    return fig, metrics, badge
