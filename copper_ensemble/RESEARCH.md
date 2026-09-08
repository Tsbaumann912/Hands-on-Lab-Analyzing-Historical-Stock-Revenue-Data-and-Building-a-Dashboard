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
- Drawdown halt auto-resumes after recovery (so one early streak cannot zero an 18-year test).

## Can walk-forward OOS Sharpe reach 1.5 on HG?

**Honest answer: not with a single copper futures market under robust OOS validation.**

| Approach | Can mean OOS Sharpe ≥ 1.5? | Why |
|---|---|---|
| Real Yahoo HG 2008→now (this repo) | **No** | Empirical WFA mean OOS Sharpe is near 0 / negative. One window can spike ~1.4 by luck; the mean does not. |
| Tune parameters until OOS ≥ 1.5 | **No (not honestly)** | That is fitting the holdout. Deflated Sharpe / multiple-testing will reject it. |
| Synthetic HG in this repo | **Yes** | Edge is planted in the simulator (regime drift + inventory). Proves engineering, not copper alpha. |
| Diversified CTA (50–100+ futures, real curves) | **Plausible** | AQR *Demystifying Managed Futures* / Moskowitz TSMOM report ~1.5–1.8 **gross** on multi-market books — diversification is the Sharpe engine, not copper alone. |
| True LME curve + inventories + PMI | Helps slightly | Improves carry/inventory sleeves vs Yahoo proxies; still does not turn one contract into a 1.5 OOS machine. |

**What would be required for ~1.5 (institutional path):**
1. Trade a **wide futures universe** (commodities + rates + FX + equity indices), not HG alone.
2. Use **real** term-structure and inventory data (not price-derived proxies).
3. Keep costs realistic; report **net** Sharpe.
4. Pass purged WFA + DSR without peeking at OOS.

This project therefore treats `target_mean_oos_sharpe: 1.5` as an **aspirational gate that single-name HG is expected to fail**, and surfaces that failure explicitly in validation notes.

## Robust optimisation (anti-overfit)

Two modes (`python -m copper_ensemble.cli optimize --start 2008-01-01`):

1. **`--mode candidates` (default, recommended)** — score a **fixed menu** of ~6 economically motivated variants on nested purged OOS; pick the winner by mean OOS Sharpe. No continuous search → low multiple-testing risk.
2. **`--mode optuna`** — nested purged WFA: Optuna maximises **IS-only** score per window; OOS is evaluation-only; production params = median of IS winners. On Yahoo HG 2008→now this **did not beat** the untuned baseline on nested OOS, which is the expected outcome when the edge is weak.

**Production choice (2008→now):** `C_no_fade_long_h` — fade sleeve off, TSMOM horizons `[63, 126, 252]`.
Among the fixed menu this wins on both 6-window nested OOS (~+0.26 vs baseline ~+0.22) and
8-window nested OOS (~−0.29 vs baseline ~−0.48) with lower turnover. Anchored 2019→now holdout
remains weak (single-name limit). Free Optuna did **not** beat baseline on nested OOS.

## Run

```bash
cd copper_ensemble
pip install -e .
pytest -q
python -m copper_ensemble.cli optimize --start 2008-01-01 --mode candidates
python -m copper_ensemble.cli validate --start 2008-01-01
python -m copper_ensemble.web
```
