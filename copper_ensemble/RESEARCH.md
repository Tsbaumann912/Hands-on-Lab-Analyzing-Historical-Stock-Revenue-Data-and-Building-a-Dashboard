# Research notes — COMEX Copper ensemble (standalone project)

This package implements a CTA-style copper algorithm that is **not** part of QuantTerminal.

## Economic rationale (sleeves)

1. **TSMOM** — Under-reaction to China/growth/supply news creates multi-horizon drift (AQR/Moskowitz).
2. **Carry** — Theory of storage: low inventories → high convenience yield → backwardation → positive roll for longs (Gorton–Hayashi–Rouwenhorst; Koijen et al.).
3. **Basis-momentum** — Changes in curve slope capture inventory/hedging shocks (Boons–Prado).
4. **Inventory-trend** — Physical draws confirming price trend reduce false breakouts.
5. **Macro fade** — Fade stretched moves only when PMI/USD do not confirm; disabled when trend and tightness agree.

## Robust combination

- Majority-sign consensus (opposing sleeves dropped, not averaged away).
- Disagreement flatten when agreement < `agreement_min`.
- Forecast diversification multiplier (FDM).
- EWMA volatility targeting + fractional Kelly + hard contract/leverage caps.
- Promotion: ablation, purged walk-forward, Deflated Sharpe.

## Run

```bash
cd copper_ensemble
pip install -e .
pytest -q
python -m copper_ensemble.cli backtest --synthetic --plot-summary
python -m copper_ensemble.cli validate --synthetic
```
