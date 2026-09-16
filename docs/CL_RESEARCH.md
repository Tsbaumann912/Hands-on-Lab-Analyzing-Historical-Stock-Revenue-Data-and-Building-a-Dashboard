# Light Crude Oil (CL / WTI) — Quant Research Workflow

Standalone research note for formulating and launching automated CL futures
strategies in QuantTerminal. Mirrors institutional CTA / commodity risk-premia
practice and the sleeve pattern in `copper_ensemble/`.

## Industry workflow (recreate this)

1. **Economic hypothesis** — one sentence tied to storage, hedging, or under-reaction.
2. **Data** — continuous front for P&L (Panama/ratio roll); raw curve (F1…Fn) for carry.
3. **Lagged features** — signal at \(t\) uses only info known at \(t\); trade next bar.
4. **Skill check** — Information Coefficient \(IC=\mathrm{corr}(\mathrm{signal}_t,r_{t+1})\) and ICIR.
5. **Sizing** — volatility targeting; optional nonlinear reaction \(R(z)=z\,e^{0.5(1-z^2)}\).
6. **Costs** — commission, fees, slippage, financing, roll. CL notional = price × 1000 × contracts.
7. **Validation** — purged walk-forward, parameter plateaus, Deflated Sharpe (trial count).
8. **Launch** — paper → small size → scale; kill if rolling IC → 0 or regime break.

Oil is regime-sensitive (normal backwardation → structural contango → post-shale).
Prefer post-2016 samples for live relevance (Bouchouev & Zuo, GCARD 2020).

## Applied math

| Formula | Use |
|---------|-----|
| \(F(t,T)=S_t e^{(r+u-\delta)(T-t)}\) | Cost-of-carry; \(\delta\) = convenience yield |
| \(\mathrm{Carry}\approx\frac{F_1-F_n}{F_n}\cdot\frac{365}{\Delta T}\) | Ex-ante carry (AQR / Koijen et al.) |
| \(z=(C_t-\mathrm{MA}_N(C))/\hat\sigma_C\) | Carry-momentum |
| \(w=\sigma^\*/\hat\sigma\) | Vol targeting |
| \(s=\Delta\mathrm{Stocks}-\mathbb{E}[\Delta\mathrm{Stocks}]\) | EIA inventory surprise |
| DSR | Sharpe corrected for multiple testing + fat tails |

## Micro / macro rationale for CL

- **Micro:** Inventories set convenience yield. Scarcity → backwardation → positive roll
  for longs; surplus → contango → negative roll. Inventory hedgers sell futures when
  contango pays storage and buy back when it does not.
- **Macro:** Demand (growth, China, USD); supply (OPEC+, shale, geopolitics). Signed
  carry, not long-only, survives regime flips.
- **Weekly EIA/API:** Unexpected builds → nearby futures down (and vice versa);
  announcement window is short (~25 minutes of elevated activity).

## Strategies implemented

| Class | Edge | Priority |
|-------|------|----------|
| `CLCarryCurve` | Sign of \(F_1-F_n\) (storage carry) | High |
| `CLCarryMomentum` | Momentum on carry (inventory flip early-warning) | Highest |
| `CLVolTargetTSMOM` | Multi-horizon TSMOM + reaction sizing | Overlay |
| `CLInventoryConfirm` | Carry only when inventory surprise agrees | Confirm |

### Data constraint

Without multi-expiry Databento curve data, carry uses a **research proxy**: implied
back-month from front via rolling return basis (same pattern as `copper_ensemble`
Yahoo path). Production carry requires real F1–Fn. Inventory uses an optional
injected series or a price-implied proxy when EIA is unavailable — never fabricated
“surprises” labelled as official EIA.

## Anchored + rolling walk-forward (promotion)

Confirmation filters (must agree with carry-momentum, else FLAT):

- **Fast MA / Slow MA** — price trend alignment (`fast_ma_period`, `slow_ma_period`)
- **Stochastic RSI** — avoid longs when StochRSI %K is overbought; avoid shorts when
  oversold (`stoch_rsi_*` knobs)

Re-validate after indicator changes:

```bash
python3 scripts/validate_cl_wfo.py \
  --strategy CLCarryMomentum \
  --start 2008-01-01 \
  --capital 350000000 \
  --modes anchored,rolling \
  --trials 15
```

| Mode | IS | OOS | Notes |
|------|----|-----|-------|
| **Anchored** | Expanding from 2008, min 3y | 1y steps | Start date fixed |
| **Rolling** | Fixed 5y | 1y, slide 1y | `purge_bars=5` embargo |

**Ulcer metrics** (in `engine/metrics.py`):

\[
\mathrm{UI}=\sqrt{\mathrm{mean}(D_t^2)},\quad
\mathrm{UPI}=\mathrm{CAGR}/(\mathrm{UI}+\varepsilon)
\]

where \(D_t\) is percentage drawdown from peak equity.

**Hard OOS gates** (stitched OOS equity, both modes must pass):

- `sharpe_ratio > 0`
- `ulcer_performance_index > 0`
- `cagr > 0`
- `abs(max_drawdown) < 0.30`

Config: `cl_validation` in `config/default.yaml`. Report: `docs/CL_WFO_REPORT.md`.

## Sources

- Gorton, Hayashi, Rouwenhorst — *Fundamentals of Commodity Futures Returns* (NBER w13249)
- Koijen, Moskowitz, Pedersen, Vrugt — *Carry*
- Bouchouev & Zuo — Oil risk premia under changing regimes (GCARD)
- Rebellion Research — Systematic Energy Trading (WTI carry-momentum)
- Quantpedia — Continuous futures methodology
- Bailey & López de Prado — Deflated Sharpe Ratio
- Martin & McCann — Ulcer Index / Ulcer Performance Index
