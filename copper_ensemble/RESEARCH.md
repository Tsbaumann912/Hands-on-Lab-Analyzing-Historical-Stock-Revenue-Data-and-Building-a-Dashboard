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

**Production choice (2008→now):** `L_maxprofit_dd30` — maximises total / mean yearly
profit subject to **max system DD < 30%**, while keeping the calendar-year profit lock:

1. Shorter horizons `[21, 63]`, high agreement (0.50), TSMOM + MA + StochRSI blend
   (`0.50 / 0.25 / 0.25`), `vol_target_annual=0.30`, Kelly `1.0`.
2. **Calendar-year profit lock** — once YTD ≥ `yearly_profit_lock_pct` (0.05%) after
   `yearly_profit_lock_min_days=2`, flatten for the rest of the year (Nov protect on).
3. **Enforced ATR stop / take-profit** (`stop_atr_mult=4`, `take_profit_atr_mult=20`)
   plus system drawdown halt (`max_daily_drawdown_pct=0.25`), `max_position_size_pct=6`,
   `max_contracts=28`.

Full-sample Yahoo HG 2008→now: total ≈ **+105%**, mean yearly ≈ **+3.9%**,
max DD ≈ **29.9%**, **0** losing calendar years. Search:
`scripts/pin_maxprofit_dd30.py`, artifacts `copper_maxprofit_dd30_*.json`.

Legacy `H_yearly_profit` remains in the candidate menu (~+74% total / ~26.5% DD).

### Can mean calendar-year return ≥ 20% with max system DD < 30%?

**No — not on single-name COMEX HG under this methodology (ATR TP/stop + DD halt + ensemble ± yearly lock).**

Searched **1,050+** lock-off/lock-on configs and **384** soft-lock configs on Yahoo HG 2008→now:

| Constraint | Result |
|---|---|
| Target | arithmetic mean calendar-year ≥ **20%** **and** path max DD **< 30%** |
| Hits found | **0** |
| Best mean with DD < 30% | ≈ **+4.1%** (tight yearly lock required) |
| Best mean with DD < 50% | ≈ **+15.9%** (still short of 20%) |
| Lowest DD among mean ≥ 20% | ≈ **89%** (ruin path; lock off, high vol/pos) |
| Soft yearly lock (5–25%) | still **0** dual hits; mean≥20% remains ~99%+ DD |

**Pareto read:** under DD < 30% the yearly lock caps mean yearly near ~3–4%. Unlocking (or softening) the lock is what lifts arithmetic mean toward/above 20%, but path drawdowns blow out to ~90%+. Same knobs cannot clear both gates together on this single-name HG sample.

Production remains the **DD-aware yearly-lock book** (all-green / low DD), not an unmet 20%+DD<30 dual target. Search artifacts: `scripts/probe_mean20_dd30.py`, `/opt/cursor/artifacts/copper_mean20_dd30_probe.json`.

### Can mean calendar-year return reach 30%?

**No — not on single-name COMEX HG under this methodology.**

| Constraint | Result |
|---|---|
| Target | arithmetic mean calendar-year return ≥ **30%** |
| Same methodology | all years green + ATR TP/stop + DD halt + yearly lock + `max_position_size_pct=5` / 20 contracts |
| HG buy&hold 2008→now | mean yearly ≈ **10%**, **7** losing years |
| Aggressive leverage search (vol ≤ 2, pos ≤ 50×, lock ≥ 30%) | **0** all-green configs; high means come with ruin years |
| Best all-green ceiling found (older H book) | mean yearly ≈ **3.0%**, total ≈ **+74%**, min year ≈ **+0.28%** |
| Best max-profit under DD < 30% (production) | mean yearly ≈ **3.9%**, total ≈ **+105%**, max DD ≈ **29.9%** |

Pushing for 30% mean yearly requires leverage that breaks the DD / all-green constraints (and often the account). Production is the **max-profit DD<30%** book found with the same knobs (TP/stop/DD/lock/vol/Kelly), not the unmet 20%/30% mean targets.

On Yahoo HG 2008→now this posts **profit in every calendar year**
(min year ≈ +0.37%, mean year ≈ **+3.9%**, total ≈ **+105%**, max DD ≈ **29.9%**).
This remains an **in-sample calendar objective**.

```bash
cd copper_ensemble
pip install -e .
pytest -q
python -m copper_ensemble.cli optimize --start 2008-01-01 --mode candidates
python -m copper_ensemble.cli validate --start 2008-01-01
python -m copper_ensemble.web
```
