# Quantitative Futures Trading Terminal

A professional-grade, event-driven quantitative trading research and strategy development terminal for CME futures markets. Built with a decoupled, modular architecture designed to be extended with Cursor AI.

## Architecture

```
futures-terminal/
├── .cursor/rules/quant-standards.mdc  # AI coding rules for Cursor
├── .cursorignore                       # Exclude sensitive data from indexing
├── config/
│   ├── default.yaml                   # All tuneable parameters (no hardcoding)
│   └── optuna.yaml                    # Hyperparameter search bounds
├── core/          # System-wide: enums, event bus, config, data models
├── data/          # Databento/CME data ingestion + async WebSocket live feed
├── indicators/    # Vectorised NumPy indicators (RSI, MACD, BB, ATR, VWAP…)
├── strategies/    # Abstract Strategy base class + 3 concrete implementations
├── engine/        # Backtester, metrics engine, walk-forward optimiser (Optuna)
├── risk/          # RiskManager firewall with 4 hard circuit-breaker rules
├── portfolio/     # Deterministic futures portfolio state tracker
├── brokers/       # PaperBroker + Alpaca live execution adapter
└── tests/         # 125 pytest tests covering all modules
```

## Quick Start

```bash
# 1. Install web dependencies
pip install -r requirements-web.txt

# 2. Launch the desktop application (browser opens automatically)
python3 wsgi.py
# OR
./run.sh

# The app opens at: http://127.0.0.1:8050
```

## Desktop App (double-click to open)

Install a **QuantTerminal** icon on your desktop, then open it like any other
application — it launches in its own window (no browser tabs, no address bar):

```bash
# One-time setup: creates the desktop shortcut for your OS
python3 install_desktop_app.py
```

| OS | What gets created |
|----|-------------------|
| **Windows** | `QuantTerminal.lnk` shortcut on the Desktop (launches silently via `pythonw`) |
| **macOS** | `QuantTerminal.app` bundle on the Desktop |
| **Linux** | `QuantTerminal.desktop` on the Desktop + an entry in the app menu |

### Windows laptops — one-click setup

No command line needed. On the Windows machine:

1. Install Python 3 from [python.org/downloads](https://www.python.org/downloads/windows/)
   — tick **"Add python.exe to PATH"** during install (skip if already installed).
2. Download / clone this repository (GitHub → **Code → Download ZIP** → extract).
3. Double-click **`install_windows.bat`** in the project folder. It installs the
   dependencies and puts a **QuantTerminal** icon on your desktop.
4. Double-click **QuantTerminal** on the desktop to open the terminal. It opens
   in its own app window via Microsoft Edge (preinstalled on Windows 10/11) or
   Chrome — no browser tabs or address bar.

`QuantTerminal.bat` in the project folder launches the app directly and can be
used as a fallback if shortcut creation is blocked by policy.

### Open from other laptops on your network

To use QuantTerminal from a Windows laptop while the server runs on another
machine (it binds to all interfaces on port 8050):

```bash
python3 wsgi.py                 # on the host machine
```

Then on the laptop, browse to `http://<host-ip>:8050` (find the host IP with
`ipconfig` / `ip addr`). Allow port 8050 through the host firewall if prompted.
Alternatively `./start-public.sh` exposes the app on a public Cloudflare URL
that works from anywhere.

You can also launch the window directly without a shortcut:

```bash
python3 desktop.py
```

The launcher starts the local server (picking a free port if 8050 is busy),
waits for it to become healthy, then opens the app window using the best
available backend: a native window via [pywebview](https://pywebview.flowrl.com)
if installed (`pip install pywebview`), otherwise a Chrome/Edge/Chromium
app-mode window, otherwise the default browser. Closing the window shuts the
server down cleanly. Window size, port, and title are configurable under the
`app:` section of `config/default.yaml`.

## Public URL (Cloudflare Tunnel)

Open QuantTerminal in any browser:

**https://pts-instructor-almost-temperatures.trycloudflare.com**

Start the app and tunnel together:

```bash
./start-public.sh
```

Or start them separately:

```bash
python3 wsgi.py    # local server on :8050
./expose.sh        # Cloudflare quick tunnel only
```

The canonical URL is stored in `PUBLIC_URL`. Cloudflare quick tunnels stay at the same address while `cloudflared` keeps running; if you restart the tunnel, run `./expose.sh` and update `PUBLIC_URL` with the new link.

### Desktop Application Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Market overview, watchlist KPI cards, 1-year normalised performance chart, CME futures snapshot |
| **Stock Research** | Interactive OHLCV candlestick + EMA overlays; quarterly revenue vs share price dual-panel chart (from the Final Assignment notebook) for TSLA, GME, and any ticker |
| **Futures Terminal** | CME contract price chart with full overlay suite (SMA/EMA/BB/VWAP) + sub-panel indicators (RSI/MACD/ATR/OBV) |
| **Indicator Explorer** | 5-panel synchronised chart: Price+BB+EMA → RSI → MACD → ATR → OBV |
| **Strategy Lab** | Configure strategy parameters with sliders, run backtests, view equity curve + fills table + all performance metrics |
| **Risk Console** | Drawdown speedometer gauge, risk-limit utilisation bars, equity history, open positions table |

### Credentials (optional)

```bash
cp .env.example .env
# Add DATABENTO_API_KEY, ALPACA_API_KEY for live data
# App works fully with synthetic data if keys are absent
```

### Running Tests

```bash
python3 -m pytest tests/ -v
# 125 tests, all pass
```

## Modules

### `core/`
- **`enums.py`** — `Direction`, `OrderType`, `OrderStatus`, `AssetClass`, `EventType`
- **`events.py`** — Pub-sub `EventBus` (sync + async handlers, fault-isolated)
- **`config.py`** — Hierarchical YAML config loader with typed dataclasses
- **`models.py`** — `Bar`, `Tick`, `Signal`, `Order`, `Fill` dataclasses

### `data/`
- **`historical.py`** — `DatabentoHistoricalClient` with Parquet caching
- **`live.py`** — `LiveTickStream` (asyncio WebSocket, auto-reconnect)
- **`transforms.py`** — Continuous contract roll (Panama method), OHLCV normalisation

### `indicators/`
All functions return `None` (never raise) when the warm-up window is unsatisfied.

| Module | Indicators |
|--------|-----------|
| `momentum.py` | RSI, MACD, Stochastic Oscillator |
| `trend.py` | SMA, EMA, WMA, SuperTrend |
| `volatility.py` | ATR, Bollinger Bands, Historical Volatility, Keltner Channels |
| `volume.py` | OBV, VWAP, Volume Oscillator |

### `strategies/`
- **`base.py`** — Abstract `Strategy` + `BarBuffer` (vectorised rolling buffer)
- **`mean_reversion.py`** — RSI + Bollinger Band mean-reversion (`MeanReversionRSI`)
- **`momentum.py`** — Donchian channel breakout with volume confirmation (`MomentumBreakout`)
- **`trend_following.py`** — MACD crossover + SuperTrend dual-confirmation (`TrendFollowingMACD`)

### `engine/`
- **`backtest.py`** — Bar-by-bar event-driven `BacktestEngine`
- **`metrics.py`** — Sharpe, Sortino, Max Drawdown, Calmar, VaR, CVaR
- **`optimizer.py`** — `WalkForwardOptimizer` using Optuna (prevents curve-fitting)
- **`live.py`** — `LiveEngine` with `KeyboardInterrupt` panic-button

### `risk/`
`RiskManager` enforces 4 hard rules before routing any signal to the broker:
1. **Drawdown Circuit Breaker** — halts trading if equity < peak × (1 - max_dd_pct)
2. **Position Sizing Limit** — caps single trade notional to % of equity
3. **Max Open Positions** — rejects new entries when count ≥ limit
4. **Default Stop/TP Application** — applies ATR-based levels if strategy omits them

### `portfolio/`
`Portfolio` uses **futures margin accounting** (not full-notional):
- Commission + slippage deducted immediately on fill
- Realised P&L credited to cash on close
- `total_equity = cash + unrealised_pnl`
- Continuous equity curve snapshots for drawdown tracking

### `brokers/`
- **`PaperBroker`** — Local simulation with configurable slippage + commission
- **`AlpacaBroker`** — Live/paper execution via `alpaca-py` SDK

## Configuration

All parameters live in `config/default.yaml`. Never hardcode values in source:

```yaml
risk:
  max_daily_drawdown_pct: 0.03    # 3 % daily drawdown circuit breaker
  max_position_size_pct: 0.10     # max 10 % equity per trade
  max_open_positions: 5

portfolio:
  initial_cash: 100000.0
  commission_per_contract: 2.25
  contract_multiplier: 50.0       # ES point value = $50
```

## Using Cursor AI with This Terminal

This project is designed to be extended via Cursor's Composer (`Cmd+I`):

### Example prompts:

**Add a new strategy:**
> "In `strategies/`, create a new `PairsTradingSpread` class inheriting from `Strategy`. It should trade the ES/NQ spread using a z-score of rolling 60-bar correlation. Use only vectorised NumPy operations."

**Run walk-forward optimisation:**
> "In `engine/optimizer.py`, add a method to plot OOS Sharpe ratios across WFO windows using matplotlib. Export the chart to `logs/wfo_results.png`."

**Add a new indicator:**
> "In `indicators/`, add a `vwap_bands` function that computes VWAP ± N standard deviations of price from VWAP, following the existing pattern in `volatility.py`."

## Project Retrieval (QuantTermProj)

The project is saved as tag **`QuantTermProj-v1.0.0`** and a portable bundle file `QuantTermProj-v1.0.0.bundle` in the repo root. See **`RESTORE.md`** for full instructions.

```bash
# Clone the project into a fresh QuantTermProj repo
git clone --branch cursor/quant-futures-terminal-9269 \
  https://github.com/Tsbaumann912/Hands-on-Lab-Analyzing-Historical-Stock-Revenue-Data-and-Building-a-Dashboard \
  QuantTermProj
cd QuantTermProj
git remote set-url origin https://github.com/YOUR_USERNAME/QuantTermProj
git push -u origin HEAD:main
```

## Security

- API keys are loaded from environment variables only (never hardcoded)
- `.env` and `*.key` files are excluded from git and Cursor indexing
- Raw data files (`data/raw/`, `*.parquet`) are excluded from indexing
- Implement IP allowlisting for all API keys before live trading

## License

MIT
