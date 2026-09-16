# Silver Ensemble CTA

Standalone systematic trading research and backtest project for **COMEX Silver (SI)** futures.

This package is **not** part of QuantTerminal. It has zero imports from QuantTerminal and can be copied or published as its own repository.

## What it does

Combines five economically motivated forecast sleeves into one vol-targeted ensemble:

| Sleeve | Idea |
|--------|------|
| **A TSMOM** | Multi-horizon time-series momentum (21/63/126/252d) |
| **B Carry** | Curve roll / convenience-yield premium |
| **C Basis-momentum** | Change in futures basis |
| **D Inventory-trend** | Inventory draws/builds confirming price trend |
| **E GS-ratio / real-yield fade** | Gold–silver relative value + fade when real yields and USD rise |

Robustness features: disagreement flattening, trend-over-fade gate, EWMA volatility targeting, fractional Kelly caps, sleeve ablation, purged walk-forward, and Deflated Sharpe helpers.

Contract economics: **5,000 oz/contract**, tick 0.005 ($25), commission from `config/default.yaml`.

## Quick start

```bash
cd silver_ensemble
pip install -r requirements.txt
pip install -e .
pytest -q
python -m silver_ensemble.cli backtest --synthetic --plot-summary
python -m silver_ensemble.cli validate --synthetic
```

Live Yahoo proxy (network):

```bash
python -m silver_ensemble.cli backtest --period 5y --plot-summary
```

## Layout

```
silver_ensemble/
  config/default.yaml      # all tuneable parameters
  silver_ensemble/
    data/                  # features, synthetic & yfinance loaders
    forecasts/             # sleeves A–E
    blend/                 # weights, FDM, disagreement gate
    sizing/                # vol target → contracts
    risk/                  # circuit breakers, caps
    engine/                # backtest loop
    validation/            # ablation, WFA, DSR
    strategy.py            # SilverEnsemble orchestrator
    cli.py                 # command-line entry
  tests/
  RESEARCH.md
```

## Design notes

- Explicit type hints; `from __future__ import annotations` in every module.
- Vectorised NumPy for all hot-path numerics (no DataFrame row loops in forecasts).
- Futures P&L uses notional: `price × 5_000 × contracts`.
- Warm-up returns flat / NaN forecasts — never raises on short history.
