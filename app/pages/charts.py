"""
Charts page — TradingView Advanced Chart widget integration.

Replaces the former Stocks page and gives users access to TradingView's
full charting suite: 100+ indicators, all intervals, all asset classes,
drawing tools, and more — no API key required (free TradingView widget).
"""

from __future__ import annotations

from dash import Input, Output, clientside_callback, dcc, html


# ── Symbol catalogues ──────────────────────────────────────────────────────────

POPULAR_SYMBOLS: list[dict[str, str]] = [
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
    # Futures
    {"label": "ES1! — S&P 500 Fut.", "value": "CME_MINI:ES1!"},
    {"label": "NQ1! — Nasdaq Fut.",  "value": "CME_MINI:NQ1!"},
    {"label": "CL1! — Crude Oil",    "value": "NYMEX:CL1!"},
    {"label": "GC1! — Gold",         "value": "COMEX:GC1!"},
    {"label": "SI1! — Silver",       "value": "COMEX:SI1!"},
    {"label": "ZN1! — 10Y T-Note",  "value": "CBOT:ZN1!"},
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

INTERVALS: list[dict[str, str]] = [
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

CHART_STYLES: list[dict[str, str]] = [
    {"label": "Candlestick",      "value": "1"},
    {"label": "Bars",             "value": "0"},
    {"label": "Line",             "value": "2"},
    {"label": "Area",             "value": "3"},
    {"label": "Heikin Ashi",      "value": "8"},
    {"label": "Hollow Candles",   "value": "9"},
    {"label": "Baseline",         "value": "10"},
    {"label": "Hi-Lo",            "value": "12"},
]

STUDIES: list[dict[str, str]] = [
    {"label": "Volume",              "value": "Volume@tv-basicstudies"},
    {"label": "RSI (14)",            "value": "RSI@tv-basicstudies"},
    {"label": "MACD",                "value": "MACD@tv-basicstudies"},
    {"label": "Bollinger Bands",     "value": "BB@tv-basicstudies"},
    {"label": "EMA 20",              "value": "MAExp@tv-basicstudies"},
    {"label": "VWAP",                "value": "VWAP@tv-basicstudies"},
    {"label": "Stochastic",          "value": "Stoch@tv-basicstudies"},
    {"label": "ATR",                 "value": "ATR@tv-basicstudies"},
    {"label": "Ichimoku Cloud",      "value": "IchimokuCloud@tv-basicstudies"},
    {"label": "Supertrend",          "value": "Supertrend@tv-basicstudies"},
    {"label": "Parabolic SAR",       "value": "PSAR@tv-basicstudies"},
    {"label": "CCI",                 "value": "CCI@tv-basicstudies"},
    {"label": "Williams %R",         "value": "WilliamR@tv-basicstudies"},
    {"label": "MFI",                 "value": "MFI@tv-basicstudies"},
    {"label": "OBV",                 "value": "OBV@tv-basicstudies"},
]


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([
        # ── Page header ────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H1("Charts", className="page-title"),
                html.P(
                    "TradingView Advanced Charts — real-time data, 100+ indicators, "
                    "all asset classes. Type any symbol or pick from the quick list.",
                    className="page-subtitle",
                ),
            ], className="page-header-text"),
        ], className="page-header"),

        # ── Controls bar ───────────────────────────────────────────────────────
        html.Div([
            html.Div([

                # Symbol text input
                html.Div([
                    html.Label("Symbol", className="ctrl-label"),
                    dcc.Input(
                        id="tv-symbol-input",
                        type="text",
                        value="NASDAQ:AAPL",
                        placeholder="EXCHANGE:TICKER — e.g. NASDAQ:AAPL",
                        debounce=True,
                        className="tv-symbol-input",
                    ),
                ], className="ctrl-group ctrl-group--symbol"),

                # Quick-pick dropdown
                html.Div([
                    html.Label("Quick Pick", className="ctrl-label"),
                    dcc.Dropdown(
                        id="tv-quick-symbol",
                        options=POPULAR_SYMBOLS,
                        placeholder="Popular symbols…",
                        clearable=True,
                        className="tv-dropdown",
                    ),
                ], className="ctrl-group ctrl-group--quick"),

                # Interval
                html.Div([
                    html.Label("Interval", className="ctrl-label"),
                    dcc.Dropdown(
                        id="tv-interval",
                        options=INTERVALS,
                        value="D",
                        clearable=False,
                        className="tv-dropdown tv-dropdown--sm",
                    ),
                ], className="ctrl-group ctrl-group--sm"),

                # Chart style
                html.Div([
                    html.Label("Style", className="ctrl-label"),
                    dcc.Dropdown(
                        id="tv-chart-style",
                        options=CHART_STYLES,
                        value="1",
                        clearable=False,
                        className="tv-dropdown tv-dropdown--sm",
                    ),
                ], className="ctrl-group ctrl-group--sm"),

                # Indicators / studies (multi-select)
                html.Div([
                    html.Label("Indicators", className="ctrl-label"),
                    dcc.Dropdown(
                        id="tv-studies",
                        options=STUDIES,
                        value=["Volume@tv-basicstudies"],
                        multi=True,
                        placeholder="Add indicators…",
                        className="tv-dropdown tv-dropdown--lg",
                    ),
                ], className="ctrl-group ctrl-group--lg"),

                # Theme toggle
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

        # ── TradingView chart ──────────────────────────────────────────────────
        html.Div([
            html.Div(
                id="tradingview-chart",
                className="tv-chart-embed",
            ),
        ], className="tv-chart-wrapper section-card"),

        # ── Attribution note ───────────────────────────────────────────────────
        html.Div([
            html.P([
                "Charts powered by ",
                html.A(
                    "TradingView",
                    href="https://www.tradingview.com/",
                    target="_blank",
                    rel="noopener noreferrer",
                    className="tv-attribution-link",
                ),
                ". Use the symbol search box or Quick Pick to explore any asset. "
                "Click any indicator name inside the chart to configure its parameters.",
            ], className="tv-attribution-text"),
        ], className="tv-attribution"),

        # Hidden sink for clientside callback return value
        html.Div(id="tv-widget-sink", style={"display": "none"}),

        # One-shot interval fires 800 ms after page load to ensure the widget
        # initialises even if the first callback ran before the DOM was ready.
        dcc.Interval(id="tv-init-interval", interval=800, max_intervals=1),

    ], className="page-content charts-page")


# ── Clientside callback — initialise / update the TradingView widget ──────────
#
# The TradingView free widget is loaded from TradingView's CDN (tv.js).
# We destroy the old instance and recreate it whenever any control changes.
# This runs entirely in the browser — no server round-trip needed.

clientside_callback(
    """
    function(symbolInput, quickSymbol, interval, chartStyle, studies, theme, _tick) {

        // Resolve the symbol to display
        var symbol = (quickSymbol && quickSymbol.length > 0)
                     ? quickSymbol
                     : (symbolInput || "NASDAQ:AAPL");

        var studiesArr = studies || [];
        var themeVal   = theme || "dark";
        var bgColor    = themeVal === "dark" ? "#131722" : "#ffffff";

        var config = {
            container_id:       "tradingview-chart",
            autosize:           true,
            symbol:             symbol,
            interval:           interval || "D",
            timezone:           "Etc/UTC",
            theme:              themeVal,
            style:              chartStyle || "1",
            locale:             "en",
            toolbar_bg:         bgColor,
            enable_publishing:  false,
            allow_symbol_change: true,
            studies:            studiesArr,
            show_popup_button:  true,
            popup_width:        "1200",
            popup_height:       "700",
            save_image:         true,
            hide_side_toolbar:  false,
            withdateranges:     true,
            details:            false,
            hotlist:            false,
            calendar:           false
        };

        function createWidget() {
            var container = document.getElementById("tradingview-chart");
            if (!container) { return; }
            container.innerHTML = "";
            if (window.TradingView) {
                new window.TradingView.widget(config);
            }
        }

        if (window.TradingView) {
            createWidget();
        } else {
            var existing = document.getElementById("tv-script-tag");
            if (existing) { existing.remove(); }
            var s = document.createElement("script");
            s.id  = "tv-script-tag";
            s.src = "https://s3.tradingview.com/tv.js";
            s.onload = function() { setTimeout(createWidget, 200); };
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
