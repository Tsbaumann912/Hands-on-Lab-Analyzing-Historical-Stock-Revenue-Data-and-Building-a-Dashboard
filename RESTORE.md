# QuantTermProj — Restore & Retrieval Guide

This file documents how to restore this project into a fresh `QuantTermProj` repository.

---

## Option 1 — Clone directly from the tagged commit (recommended)

The entire project is saved as tag `QuantTermProj-v1.0.0` on:

```
https://github.com/Tsbaumann912/Hands-on-Lab-Analyzing-Historical-Stock-Revenue-Data-and-Building-a-Dashboard
```

### Clone only the QuantTermProj branch into a new repo

```bash
# 1. Create a new repo on GitHub named "QuantTermProj" (via github.com/new)

# 2. Clone the working branch from the source repo
git clone --branch cursor/quant-futures-terminal-9269 \
  https://github.com/Tsbaumann912/Hands-on-Lab-Analyzing-Historical-Stock-Revenue-Data-and-Building-a-Dashboard \
  QuantTermProj

cd QuantTermProj

# 3. Point it at your new repo and push
git remote set-url origin https://github.com/YOUR_USERNAME/QuantTermProj
git push -u origin HEAD:main

# Done — the full project is now in QuantTermProj
```

---

## Option 2 — Restore from the git bundle file

A self-contained bundle `QuantTermProj-v1.0.0.bundle` is stored in the repo root.

```bash
# Clone from the bundle
git clone QuantTermProj-v1.0.0.bundle QuantTermProj
cd QuantTermProj

# Check out the working branch
git checkout cursor/quant-futures-terminal-9269

# Point at new remote and push
git remote set-url origin https://github.com/YOUR_USERNAME/QuantTermProj
git push --all
git push --tags
```

---

## What's in the project

```
QuantTermProj/
├── app.py                   ← launch point:  python3 app.py
├── run.sh                   ← one-click launcher
├── requirements.txt
├── config/
│   ├── default.yaml         ← all tunable parameters
│   └── optuna.yaml          ← hyperparameter search bounds
├── core/                    ← enums, event bus, config, data models
├── data/                    ← Databento ingestion + live WebSocket
├── indicators/              ← RSI, MACD, BB, ATR, SMA/EMA, VWAP, OBV …
├── strategies/              ← MeanReversionRSI, MomentumBreakout, TrendFollowingMACD
├── engine/                  ← BacktestEngine, WalkForwardOptimizer, LiveEngine, metrics
├── risk/                    ← RiskManager (4 circuit-breaker rules)
├── portfolio/               ← futures margin accounting
├── brokers/                 ← PaperBroker, AlpacaBroker
├── app/                     ← Dash desktop application (6 pages)
│   ├── pages/
│   │   ├── dashboard.py
│   │   ├── stock_research.py   ← Final Assignment integration (TSLA/GME)
│   │   ├── futures_terminal.py
│   │   ├── indicator_explorer.py
│   │   ├── strategy_lab.py
│   │   └── risk_console.py
└── tests/                   ← 125 unit tests + runtime_check.py
```

---

## Quickstart after restore

```bash
pip install -r requirements.txt
cp .env.example .env          # add API keys (optional)
python3 -m pytest tests/ -v   # verify: 125 tests pass
python3 app.py                # launch at http://127.0.0.1:8050
```

---

## Stable tag

```
QuantTermProj-v1.0.0  →  commit 8681032
```

Retrieve any time:
```bash
git fetch --tags
git checkout QuantTermProj-v1.0.0
```

---

## Prompts to continue development with a Cursor agent

Copy any of these into Cursor Composer (Cmd+I) to pick up from where we left off:

**Add real Databento data:**
> "In `data/historical.py`, connect the `DatabentoHistoricalClient` to my DATABENTO_API_KEY env variable and fetch ES 1-minute OHLCV bars for 2023. Cache them to `data/cache/`. Then update the Strategy Lab page to use real data instead of synthetic."

**Add a new strategy:**
> "In `strategies/`, create a `MeanReversionZScore` class that inherits from `Strategy`. It should compute a 60-bar rolling z-score of price deviation from the mean and enter long when z < -2 and short when z > +2. Add it to the Strategy Lab dropdown."

**Add walk-forward optimisation to the UI:**
> "In `app/pages/strategy_lab.py`, add a 'Walk-Forward Optimise' button below the Run Backtest button. When clicked, call `WalkForwardOptimizer.run()` with the current parameters. Display per-window OOS Sharpe ratios as a bar chart and the aggregated OOS metrics table."

**Connect live Alpaca paper trading:**
> "In `engine/live.py` and `brokers/alpaca_broker.py`, wire up a live paper-trading session. The Live Engine should subscribe to 1-minute bars via Alpaca's streaming API and route signals through the RiskManager to AlpacaBroker. Add a 'Go Live (Paper)' button to the Risk Console page."
