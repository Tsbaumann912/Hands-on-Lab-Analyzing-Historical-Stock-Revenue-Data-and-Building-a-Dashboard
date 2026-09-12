# Copper Ensemble CTA

Standalone systematic trading research and backtest project for **COMEX Copper (HG)** futures.

This package is **not** part of QuantTerminal. It has zero imports from QuantTerminal and can be copied or published as its own repository.

## Download

Pre-built release zip (production book: all calendar years green, max DD ≤ 30%):

- [`releases/copper-ensemble-allgreen-dd30-v0.2.0.zip`](releases/copper-ensemble-allgreen-dd30-v0.2.0.zip)
- Install guide: [`DOWNLOAD.md`](DOWNLOAD.md)

```bash
unzip copper-ensemble-allgreen-dd30-v0.2.0.zip
cd copper-ensemble-allgreen-dd30-v0.2.0
pip install -r requirements.txt && pip install -e .
python -m copper_ensemble.cli backtest --synthetic
```

Rebuild the zip anytime with `bash scripts/build_release.sh`.

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
python -m copper_ensemble.cli optimize --start 2008-01-01 --mode candidates
```

## Market walk-forward (2008 → now)

```bash
# Yahoo HG=F from 2008-01-01 through today (curve/inventory/PMI are proxies)
python -m copper_ensemble.cli validate --start 2008-01-01
python -m copper_ensemble.cli backtest --start 2008-01-01 --plot-summary
# Anti-overfit config selection among a fixed economic menu:
python -m copper_ensemble.cli optimize --start 2008-01-01 --mode candidates
```

In the web UI choose **Yahoo HG 2008→now** and click **Run backtest + validate**.
Long histories (≥3000 bars) use 8 walk-forward windows automatically.

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
