# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum` (carry-momentum **+ Stochastic RSI + Fast/Slow MA**)
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Methodology: same anchored + rolling WFO as prior CL validation; gates Sharpe>0, UPI>0, CAGR>0, |MaxDD|<30%
- Optuna: 15 trials / IS window; OOS warm-up from prior IS bars
- **Overall pass:** True

## Anchored WFO

- Windows: 15
- Gates passed: True
- Failures: —
- Sharpe: 0.3829
- UPI: 0.2923
- CAGR: 0.008506
- Max DD: -0.071354
- Final equity: 397217739.509092

## Rolling WFO

- Windows: 13
- Gates passed: True
- Failures: —
- Sharpe: 0.1878
- UPI: 0.1263
- CAGR: 0.004045
- Max DD: -0.074209
- Final equity: 368783116.9830121
