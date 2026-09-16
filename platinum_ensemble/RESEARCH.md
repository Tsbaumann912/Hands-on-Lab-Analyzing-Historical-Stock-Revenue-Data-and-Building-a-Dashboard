# Research notes — NYMEX Platinum ensemble (standalone project)

This package implements a CTA-style platinum algorithm that is **not** part of QuantTerminal.

## Market structure

- **Contract:** NYMEX PL, 50 troy oz, tick $0.10/oz ($5), physical delivery.
- **Supply:** ~70–80% mined in South Africa → power/strike shocks create persistent trends.
- **Demand:** autocatalyst + industrial + jewellery + volatile ETF/investment flow.
- **Macro:** USD real rates and gold’s monetary bid spill into PGMs with lag; PL often cheap vs GC historically.

## Economic rationale (sleeves)

1. **TSMOM** — Under-reaction then herding over-reaction to SA supply / auto / ETF news (AQR/Moskowitz).
2. **Carry** — Theory of storage: low inventories → high convenience yield → backwardation → positive roll for longs.
3. **PL–GC RV** — Gold ≈ monetary demand; platinum ≈ industrial/autocatalyst. Temporary wedges mean-revert; kill sleeve if EV/diesel breaks cointegration.
4. **Inventory-trend** — Physical draws confirming price trend reduce false breakouts in a thin market.
5. **Macro fade** — Fade stretched moves only when gold impulse / USD do not confirm; disabled when trend and tightness agree.

## Math summary

- TSMOM: \(\mathrm{sign}(r_{t-X:t})\cdot(\sigma_{\mathrm{target}}/\sigma_t)\)
- Carry: \(R_{\mathrm{futures}} \approx R_{\mathrm{spot}} + R_{\mathrm{roll}}\)
- RV: \(s_t=\ln P^{\mathrm{PL}}-\beta\ln P^{\mathrm{GC}}\), \(z=(s-\mu)/\sigma\)
- Sizing: \(n=\mathrm{round}(w\cdot\mathrm{Equity}\cdot\sigma_{\mathrm{target}}/(P\cdot 50\cdot\sigma))\)

## Robust combination

- Majority-sign consensus (opposing sleeves dropped, not averaged away).
- Disagreement flatten when agreement < `agreement_min`.
- Forecast diversification multiplier (FDM).
- EWMA volatility targeting + fractional Kelly + hard contract/leverage caps.
- Promotion: ablation, purged walk-forward, Deflated Sharpe.

## Run

```bash
cd platinum_ensemble
pip install -e .
pytest -q
python -m platinum_ensemble.cli backtest --synthetic --plot-summary
python -m platinum_ensemble.cli validate --synthetic
```
