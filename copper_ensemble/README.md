# Copper Ensemble CTA

Standalone systematic trading research and backtest project for **COMEX Copper (HG)** futures.

This package is **not** part of QuantTerminal. It has zero imports from QuantTerminal and can be copied or published as its own repository.

## What it does

Combines five economically motivated forecast sleeves into one vol-targeted ensemble:

| Sleeve | Idea |
|--------|------|
| **A TSMOM** | Multi-horizon time-series momentum |
| **B Carry** | Curve roll / convenience-yield premium |
| **C Basis-momentum** | Change in futures basis |
| **D Inventory-trend** | Inventory draws/builds confirming price trend |
| **E Macro fade** | Mean-reversion only when macro does not confirm the move |

Robustness features: disagreement flattening, trend-over-fade gate, EWMA volatility targeting, fractional Kelly caps, sleeve ablation, purged walk-forward, and Deflated Sharpe helpers.

## Quick start

```bash
cd copper_ensemble
pip install -r requirements.txt
pytest -q
python -m copper_ensemble.cli backtest --synthetic --plot-summary
python -m copper_ensemble.cli validate --synthetic

# Web dashboard (default http://127.0.0.1:8060)
python -m copper_ensemble.web
```

## Layout

```
copper_ensemble/
  config/default.yaml      # all tuneable parameters
  copper_ensemble/
    data/                  # features, synthetic & yfinance loaders
    forecasts/             # sleeves A–E
    blend/                 # weights, FDM, disagreement gate
    sizing/                # vol target → contracts
    risk/                  # circuit breakers, caps
    engine/                # backtest loop
    validation/            # ablation, WFA, DSR
    strategy.py            # CopperEnsemble orchestrator
    cli.py                 # command-line entry
  tests/
```

## Design notes

- Explicit type hints; `from __future__ import annotations` in every module.
- Vectorised NumPy for all hot-path numerics (no row loops).
- Futures P&L uses notional: `price × 25_000 × contracts`.
- Warm-up returns flat / NaN forecasts — never raises on short history.
