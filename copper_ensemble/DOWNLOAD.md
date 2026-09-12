# Download & install — Copper Ensemble CTA

Standalone COMEX Copper (HG) ensemble strategy package.
Production book: **all calendar years profitable** and **max drawdown ≤ 30%**
on Yahoo HG 2008→now (see `RESEARCH.md`).

## Package contents

| Path | Purpose |
|------|---------|
| `config/default.yaml` | Production parameters (all-green + DD≤30%) |
| `copper_ensemble/` | Strategy, engine, risk, sizing, forecasts, CLI, web UI |
| `tests/` | Pytest suite |
| `scripts/` | Search / research helpers |
| `RESEARCH.md` | Methodology, ceilings, and verified metrics |

## Install (from zip)

```bash
unzip copper-ensemble-allgreen-dd30-v0.2.0.zip
cd copper-ensemble-allgreen-dd30-v0.2.0
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Run

```bash
# Synthetic smoke test (no network)
python3 -m copper_ensemble.cli backtest --synthetic --plot-summary

# Full Yahoo HG history from 2008
python3 -m copper_ensemble.cli backtest --start 2008-01-01 --plot-summary

# Tests
pytest -q

# Local web dashboard (http://127.0.0.1:8060)
copper-ensemble-web
# or: python3 -m copper_ensemble.web
```

## $350M account note

Default `portfolio.initial_cash` is `$500,000` with `risk.max_contracts: 28`.
For a like-for-like risk profile at **$350,000,000**, scale contracts with capital
(×700 → `max_contracts: 19600`) or the book is capacity-starved. See research notes.

## Disclaimer

Research software only. Not investment advice. Past backtests do not guarantee
future results. Futures trading involves substantial risk of loss.
