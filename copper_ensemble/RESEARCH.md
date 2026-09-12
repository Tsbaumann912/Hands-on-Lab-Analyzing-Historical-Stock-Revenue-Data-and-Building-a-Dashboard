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

**Production choice (2008→now):** `K_allgreen_dd30` — **every calendar year profitable** and **max drawdown ≤ 30%**, maximizing mean yearly / total return under those caps via:

1. Horizons `[21, 63]`, agreement 0.50, blend `tsmom=0.55` / `ma_cross=0.25` / `stoch_rsi=0.20`.
2. **Yearly profit lock ON** at 0.1% after 2 sessions (+ Nov protect) — required for all-green years.
3. Risk budget: `vol_target_annual=0.32`, `max_position_size_pct=7`, `max_contracts=28`,
   `max_daily_drawdown_pct=0.25`, ATR stop 4 / TP 25, Kelly 1.0, buffer 2.5.

### All-green years + max DD ≤ 30% (current production)

| Constraint | Result |
|---|---|
| Target | all calendar years > 0 **and** max DD ≤ **30%**, max mean/total |
| Production (`K_allgreen_dd30`) | mean yearly ≈ **+3.70%**, total ≈ **+97.6%**, CAGR ≈ **+3.72%**, **0** losing years, max DD ≈ **29.4%** |
| Prior all-green (`H_yearly_profit`) | mean yearly ≈ **+3.0%**, total ≈ **+74%**, max DD ≈ **26.5%** |
| Unlocked ≥20% book (`J_mean20_annual`) | mean yearly ≈ **+23.3%**, total ≈ **+303%**, **10** red years, max DD ≈ **76%** |
| Unlocked ≥15% book (`I_mean15_annual`) | mean yearly ≈ **+21.4%**, total ≈ **+169%**, max DD ≈ **85%** |
| HG buy&hold 2008→now | mean yearly ≈ **10%**, **7** losing years |

**Honest ceiling:** under all-green + DD ≤ 30% on single-name HG, mean yearly stays ~**3–4%**. Higher arithmetic means (≥15%/≥20%) require unlocking the yearly lock and accepting red years plus much deeper drawdowns.

### Can mean calendar-year return reach 20% or 30% with all years green?

**No — not on single-name COMEX HG under this methodology with DD ≤ 30%.**

| Constraint | Result |
|---|---|
| Mean ≥ 20% + all-green + DD ≤ 30% | **Not found** |
| Mean ≥ 30% + all-green | **Not found** — high means come with ruin years |
| Best all-green + DD ≤ 30% found | mean yearly ≈ **+3.70%**, total ≈ **+97.6%**, min year ≈ **+0.04%**, max DD ≈ **29.4%** |

Legacy candidates `H_yearly_profit`, `I_mean15_annual`, and `J_mean20_annual` remain in the menu for comparison.

```bash
cd copper_ensemble
pip install -e .
pytest -q
python -m copper_ensemble.cli optimize --start 2008-01-01 --mode candidates
python -m copper_ensemble.cli validate --start 2008-01-01
python -m copper_ensemble.web
```
