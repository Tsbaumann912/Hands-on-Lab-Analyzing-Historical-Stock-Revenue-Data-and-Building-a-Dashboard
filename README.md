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
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env with your Databento + Alpaca API keys

# 3. Run the full test suite
python3 -m pytest tests/ -v

# 4. Run a backtest (example)
python3 -c "
from core.config import Config
from engine.backtest import BacktestEngine
from strategies.mean_reversion import MeanReversionRSI
from tests.conftest import make_bars

config = Config.from_yaml('config/default.yaml')
engine = BacktestEngine(config, MeanReversionRSI)
bars = make_bars(n=500)   # replace with real data
result = engine.run({'ES.c.0': bars})
print(result.summary())
"
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

## Security

- API keys are loaded from environment variables only (never hardcoded)
- `.env` and `*.key` files are excluded from git and Cursor indexing
- Raw data files (`data/raw/`, `*.parquet`) are excluded from indexing
- Implement IP allowlisting for all API keys before live trading

## License

MIT
