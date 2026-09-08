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
- Drawdown halt auto-resumes after recovery **or** a timed cooldown with peak reset
  (flat equity never recovers vs peak — without cooldown the book died ~2014).

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

## Ulcer Performance Index + dual walk-forward

Selection now defaults to **Ulcer Performance Index (UPI)** on both schemes:

- **Ulcer Index** = RMS of percent drawdowns from running peak (Martin).
- **UPI** = annualised return % / Ulcer Index (higher = more return per unit of pain).
- **Rolling WFA** — consecutive purged IS/OOS blocks.
- **Anchored WFA** — expanding IS from t=0; fixed-length purged OOS windows.

Ranking key for the fixed candidate menu:

`composite_oos_upi = 0.5·mean(rolling OOS UPI) + 0.5·mean(anchored OOS UPI)`,
then `min(rolling, anchored)` for robustness, then full-sample UPI.

**2008→now Yahoo HG result (with StochRSI + Fast/Slow MA sleeves, `max_position_size_pct=5`):**
winner `G_ma_stoch_heavy`
(composite OOS UPI ≈ **0.809**; rolling ≈ **0.508**, anchored ≈ **1.111**;
full-sample UPI ≈ 0.126, return ≈ **40%**, Sharpe ≈ 0.23).

Previous winner without MA/Stoch sleeves (`C_no_fade_long_h`) remains second
(composite ≈ 0.73 under the same risk sizing).

```bash
python -m copper_ensemble.cli optimize --start 2008-01-01 --mode candidates --metric upi
python -m copper_ensemble.cli validate --start 2008-01-01
```

## Robust optimisation (anti-overfit)

Two modes (`python -m copper_ensemble.cli optimize --start 2008-01-01`):

1. **`--mode candidates` (default, recommended)** — score a **fixed menu** of ~6 economically motivated variants on nested purged OOS using **UPI** (rolling + anchored). Use `--metric sharpe` for the legacy Sharpe-only rank.
2. **`--mode optuna`** — nested purged WFA: Optuna maximises **IS-only** UPI-weighted score per window; OOS is evaluation-only; production params = median of IS winners. On Yahoo HG 2008→now free Optuna did **not** beat the untuned baseline.

**Production choice (2008→now):** `H_yearly_profit` — prioritises **positive return every calendar year** via:

1. Shorter horizons `[21, 63]`, high agreement (0.50), TSMOM + MA + StochRSI blend.
2. **Calendar-year profit lock** — once YTD ≥ `yearly_profit_lock_pct` (0.2%) after
   `yearly_profit_lock_min_days`, flatten for the rest of the year.
3. **Enforced ATR stop / take-profit** in the backtest engine (`stop_atr_mult=4`,
   `take_profit_atr_mult=10`) plus system drawdown halt (`max_daily_drawdown_pct=0.15`).

On Yahoo HG 2008→now this posts **profit in every calendar year**
(min year ≈ +0.22%, mean year ≈ +1.5%, total ≈ **+33%**, UPI ≈ 1.0).
Versus the prior lock-only book (~+20% total), raising the per-trade profit target
and widening the stop (with vol/Kelly) lifts total return by ~+13pp while keeping
all years green. This remains an **in-sample calendar objective**.

```bash
cd copper_ensemble
pip install -e .
pytest -q
python -m copper_ensemble.cli optimize --start 2008-01-01 --mode candidates
python -m copper_ensemble.cli validate --start 2008-01-01
python -m copper_ensemble.web
```
