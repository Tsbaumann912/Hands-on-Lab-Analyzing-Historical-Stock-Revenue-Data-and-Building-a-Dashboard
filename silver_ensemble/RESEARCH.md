# Research notes — COMEX Silver ensemble (standalone project)

This package implements a CTA-style silver algorithm that is **not** part of QuantTerminal.

## Institutional workflow (recreated here)

1. **Economic hypothesis** — who pays the premium and why it persists.
2. **Features** — continuous SI returns, front–next carry/basis, inventory proxy, gold–silver log ratio, DXY 20d return, real-yield change proxy.
3. **Forecasts** — map each sleeve to Carver-scale \(f \in [-20, 20]\) with target mean \(|f|\approx 10\).
4. **Blend** — majority-sign consensus, disagreement flatten, forecast diversification multiplier (FDM).
5. **Size** — EWMA vol targeting on SI notional \(P \times 5000 \times N\).
6. **Validate** — ablation, Deflated Sharpe, plus **anchored and rolling** institutional walk-forward gated on Sharpe, UPI, CAGR, and max DD < 30%.

```
hypothesis → features → forecast → vol-target size → cost-aware backtest → anchored/rolling WFO → paper → live
```

## Economic rationale (sleeves)

1. **TSMOM** — Under-reaction to deficit, lease-rate, Fed/real-yield, and industrial news creates multi-horizon drift (Moskowitz–Ooi–Pedersen 2012). SI’s thinner book vs gold amplifies persistence.
2. **Carry** — Theory of storage: low inventories → high convenience yield → backwardation → positive roll for longs (Gorton–Hayashi–Rouwenhorst; Koijen et al. *Carry*). Structural physical deficits and ETP demand support scarcity regimes (Silver Institute World Silver Survey).
3. **Basis-momentum** — Changes in curve slope capture inventory/hedging shocks faster than level carry (Boons–Prado style). Useful around COMEX delivery/FND transitions.
4. **Inventory-trend** — Physical draws confirming price trend reduce false breakouts when paper OI dwarfs registered metal. Curve alone ≠ squeeze; confirm with stocks (AgAu evidence-stack practice).
5. **GS-ratio / real-yield fade** — Silver’s dual monetary/industrial identity:
   - Extreme \(\log(P_{GC}/P_{SI})\) mean-reverts or signals relative value.
   - Rising real yields + rising USD raise the opportunity cost of non-yielding metal → fade stretched SI z-scores.
   - Disabled when TSMOM and inventory tightness agree (structural trend gate).
6. **StochRSI + Fast/Slow MA** — Trend from SMA(fast) vs SMA(slow); entries confirmed by Stochastic RSI pullbacks (oversold in uptrend / overbought in downtrend).

## Applied math

**Carry (curve slope proxy):**

\[
\mathrm{Carry}_t = \frac{F^{(1)}_t - F^{(2)}_t}{F^{(2)}_t}
\]

**TSMOM horizon vote:** \(\mathrm{sign}(\sum_{i=0}^{h-1} r_{t-i})\) averaged over \(h \in \{21,63,126,252\}\), scaled ×10, clipped to ±20.

**Vol targeting:**

\[
N_t = \mathrm{round}\!\left(\frac{f^\star_t}{10}\cdot\frac{\sigma_{\mathrm{target}}\,E_t}{\sigma_t\,P_t\,M}\right),\quad M=5000
\]

**Gold–silver ratio:** \(R_t = \log P^{GC}_t - \log P^{SI}_t\), z-scored over `ratio_lookback`.

**Deflated Sharpe:** Bailey–López de Prado correction for selection bias under \(N\) trials and non-normal returns (`validation/`).

**Ulcer Index / UPI (Peter Martin):**

\[
D_t = 100 \times \left(\frac{E_t}{\mathrm{peak}_t} - 1\right),\quad
UI = \sqrt{\mathrm{mean}(D_t^2)},\quad
UPI = \frac{\mathrm{CAGR}}{UI/100}
\]

## Institutional walk-forward ($350M, 2008–now)

Config defaults: `portfolio.initial_cash = 350_000_000`, `risk.max_contracts = 2500`,
`max_daily_drawdown_pct = 0.30`, data from `backtest.data_start = 2008-01-01`.

| Mode | In-sample | Out-of-sample | Step |
|------|-----------|---------------|------|
| **Rolling** | Fixed 3y (756 bars) | Next 1y (252 bars) | Slide start by 1y |
| **Anchored** | From first bar, expanding to OOS − purge | Next 1y | OOS advances by 1y; IS always starts at \(t_0\) |

Purge gap between IS and OOS: `purge_bars` (default 5). Fixed strategy parameters (no per-window Optuna).

**Mode pass** requires stitched OOS equity to satisfy **all**:
`sharpe ≥ min_oos_sharpe`, `upi ≥ min_oos_upi`, `cagr ≥ min_oos_cagr`, and
`max_drawdown < 0.30`. Overall pass requires **both** modes (`require_both_modes: true`).

```bash
python -m silver_ensemble.cli validate-wfo --start 2008-01-01
python -m silver_ensemble.cli validate-wfo --synthetic
python -m silver_ensemble.cli optimize --start 2008-01-01 --trials 80
```

## Optimization (calendar-year profitability)

Objective: **maximize average calendar-year equity return** subject to
**max system drawdown &lt; 30%** (`silver_ensemble/optimize.py`, search space in
`config/optuna.yaml`).

Tunable knobs: sleeve weights (incl. `stoch_ma`), MA/StochRSI periods, vol target, Kelly fraction,
agreement/buffer/fade thresholds, ATR stops, leverage caps. **`max_position_size_pct` is fixed at
0.05 (5% of equity)**. Circuit-breaker halt is set to 29% so realised max DD stays strictly below
the 30% gate.

CLI:

```bash
python -m silver_ensemble.cli optimize --start 2008-01-01 --trials 80 --max-dd 0.30
```

## Robust combination

- Majority-sign consensus (opposing sleeves dropped, not averaged away).
- Disagreement flatten when agreement < `agreement_min`.
- Forecast diversification multiplier (FDM).
- EWMA volatility targeting + fractional Kelly + hard contract/leverage caps.
- Promotion: ablation, purged walk-forward, Deflated Sharpe, institutional anchored/rolling WFO.

## Data notes / proxies

- Continuous SI from yfinance `SI=F` (or synthetic panel offline).
- Curve: when full COMEX chain unavailable, basis proxied from rolling return drift — replace with true \(F_1,F_2\) when Databento/CME feed is available.
- Inventory: synthetic AR(1) offline; yfinance path uses inverse price z-score as a **demo proxy** — replace with CME registered stocks for production.
- Real yield: `^TNX` daily change as stand-in when TIPS series unavailable.
- Do **not** feed Panama-adjusted prices into carry; trend uses continuous close, carry uses near/next fields.

## Run

```bash
cd silver_ensemble
pip install -e .
pytest -q
python -m silver_ensemble.cli backtest --synthetic --plot-summary
python -m silver_ensemble.cli validate --synthetic
python -m silver_ensemble.cli validate-wfo --start 2008-01-01
```

## Selected references

- Moskowitz, Ooi, Pedersen (2012), *Time Series Momentum*.
- Koijen, Moskowitz, Pedersen, Vrugt, *Carry* (NBER w19325).
- Gorton, Hayashi, Rouwenhorst, *Fundamentals of Commodity Futures Returns*.
- Bailey & López de Prado, *The Deflated Sharpe Ratio*.
- Carver, *Advanced Futures Trading Strategies* (forecast scale / FDM practice).
- Silver Institute, *World Silver Survey*; AgAu COMEX market-structure evidence stack.
