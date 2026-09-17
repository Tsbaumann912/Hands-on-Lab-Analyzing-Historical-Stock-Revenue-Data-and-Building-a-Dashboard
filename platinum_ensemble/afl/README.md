# AmiBroker AFL — Platinum Ensemble CTA

`PlatinumEnsemble_CTA.afl` is the AmiBroker Formula Language port of the locked
Python `platinum_ensemble` strategy (inventory + macro-fade, long-only, year
profit-lock, vol targeting).

## Import

1. Open AmiBroker → Analysis → Formula Editor → Open `PlatinumEnsemble_CTA.afl`
2. Tools → Verify Syntax
3. Apply as a chart formula and/or Automatic Analysis system
4. Database must include:
   - Platinum continuous (chart symbol)
   - Gold: `GC` (ParamStr override available)
   - USD: `DX` (ParamStr override available)
5. Set Initial Equity to `350000000` for parity with the Python institutional run

## Notes

- Inventory uses the same price-zscore proxy as the Python yfinance loader
  (optional OpenInterest toggle).
- Year profit-lock is implemented with an equity-proxy bar loop (Approximation
  of the Python engine lock; AmiBroker CBT can refine account-equity lock).
- Locked defaults match `config/default.yaml` after the all-years-green
  re-optimisation.
