# Platinum Ensemble CTA

Standalone systematic trading research and backtest project for **NYMEX Platinum (PL)** futures.

This package is **not** part of QuantTerminal. It has zero imports from QuantTerminal and can be copied or published as its own repository.

## What it does

Combines five economically motivated forecast sleeves into one vol-targeted ensemble:

| Sleeve | Idea |
|--------|------|
| **A TSMOM** | Multi-horizon time-series momentum (SA/auto/ETF news under-reaction) |
| **B Carry** | Curve roll / convenience-yield premium |
| **C PL–GC RV** | Gold–platinum relative-value mean reversion |
| **D Inventory-trend** | Warehouse/ETF inventory draws confirming price trend |
| **E Macro fade** | Mean-reversion only when gold/USD do not confirm the move |

Robustness features: disagreement flattening, trend-over-fade gate, EWMA volatility targeting, fractional Kelly caps, sleeve ablation, purged walk-forward, and Deflated Sharpe helpers.

## Quick start

```bash
cd platinum_ensemble
pip install -r requirements.txt
pytest -q
python -m platinum_ensemble.cli backtest --synthetic --plot-summary
python -m platinum_ensemble.cli validate --synthetic
python -m platinum_ensemble.cli wfo --start 2008-01-01 --cash 350000000
python scripts/eda_research.py
```

Institutional WFO (`wfo`) runs **anchored** and **rolling** walk-forward on PL from
`validation.data_start` (default 2008-01-01) at `$350M`, and gates stitched OOS on
Sharpe &gt; 0, UPI &gt; 0, CAGR &gt; 0, and Max DD &lt; 30%.

## Layout

```
platinum_ensemble/
  config/default.yaml      # all tuneable parameters
  platinum_ensemble/
    data/                  # features, synthetic & yfinance loaders
    forecasts/             # sleeves A–E
    blend/                 # weights, FDM, disagreement gate
    sizing/                # vol target → contracts
    risk/                  # circuit breakers, caps
    engine/                # backtest loop
    validation/            # ablation, WFA, DSR
    strategy.py            # PlatinumEnsemble orchestrator
    cli.py                 # command-line entry
  tests/
```

## Design notes

- Explicit type hints; `from __future__ import annotations` in every module.
- Vectorised NumPy for all hot-path numerics (no row loops over DataFrames).
- Futures P&L uses notional: `price × 50 × contracts`.
- Warm-up returns flat / NaN forecasts — never raises on short history.
- PL slippage defaults to 2 ticks (thinner than gold).
