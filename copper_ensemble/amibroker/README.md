# AmiBroker AFL — Copper Ensemble CTA

File: [`CopperEnsembleCTA.afl`](CopperEnsembleCTA.afl)

Port of the production Python book `L_allgreen_maxprofit`
(`copper_ensemble/config/default.yaml`) into AmiBroker Formula Language.

## Install

1. Open AmiBroker → **Analysis** → **Formula Editor** (or Charts → Formula Editor).
2. File → Open → select `CopperEnsembleCTA.afl`.
3. Apply to a **COMEX Copper** daily series (Yahoo `HG=F` continuous or
   back-adjusted HG). Set **PointValue = 25000** (full-size HG, USD/lb quotes).
4. AA Settings: Periodicity = Daily, Futures mode on, initial equity $500,000,
   trade at **Close**, delays = 0, “Activate stops immediately” **unchecked**.

## What is ported

| Python module | AFL |
|---|---|
| TSMOM / MA-cross / StochRSI forecasts | Sleeve forecasts |
| Majority blend + agreement gate | Blend block |
| Forecast buffer 2.5 | Stateful `for` loop |
| Vol-target sizing (fixed $500k) | `TargetContracts` / `PositionSize` |
| ATR(14) stop 4× / TP 20× | `ApplyStop` |
| Yearly lock 0.5% + Nov protect + DD halt 45% | Custom backtester phase |

## Parameters

All production knobs are `Param(...)` defaults — tweak in the Parameters dialog
without editing code.

## Parity caveats

- Sizing uses **fixed** `$500,000` (matches Python), not live equity growth.
- `max_position_size_pct = 15` means **15× leverage**, not 15%.
- RSI is **SMA-based** (matches Python), not Wilder `RSI()`.
- Zero-weight sleeves omitted from agreement (small difference vs Python).
- Continuous-contract rolls differ across data vendors.
