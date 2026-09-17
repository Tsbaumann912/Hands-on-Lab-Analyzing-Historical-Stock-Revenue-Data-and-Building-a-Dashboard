# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Optuna objective: `cagr`
- Position sizing: uncapped (ATR/Kelly only)
- System MaxDD gate: < 30%
- Calendar-year returns: ≥60% non-losing years; worst year ≥ -8%
- OOS-fold profitability: not required
- **Overall pass:** True

## Anchored WFO

- Windows: 15
- Gates passed: True
- Failures: —
- Sharpe: 0.4693
- UPI: 2.0622
- CAGR: 0.009275
- Total return: 0.147899
- Max DD: -0.030644
- Final equity: 401764568.8738184
- Calendar-year returns: 2011: +1.03%, 2012: +1.35%, 2013: +0.46%, 2014: +0.00%, 2015: +0.76%, 2016: +1.17%, 2017: -0.14%, 2018: +1.78%, 2019: +1.50%, 2020: +1.61%, 2021: +2.03%, 2022: +0.21%, 2023: -0.26%, 2024: +0.32%, 2025: +4.32%

## Rolling WFO

- Windows: 13
- Gates passed: True
- Failures: —
- Sharpe: 0.2954
- UPI: 0.4909
- CAGR: 0.007287
- Total return: 0.098576
- Max DD: -0.078319
- Final equity: 384501451.737038
- Calendar-year returns: 2013: +0.39%, 2014: +0.06%, 2015: +1.87%, 2016: +1.99%, 2017: +1.17%, 2018: +1.93%, 2019: -1.09%, 2020: +2.04%, 2021: +2.04%, 2022: +0.33%, 2023: -1.15%, 2024: -0.00%, 2025: +2.11%
