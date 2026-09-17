# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Optuna objective: `cagr`
- Position sizing: uncapped (ATR/Kelly only)
- System MaxDD gate: < 30%
- Calendar-year returns: required (stitched calendar years ≥ 0)
- OOS-fold profitability: not required
- **Overall pass:** False

## Anchored WFO

- Windows: 15
- Gates passed: False
- Failures: ['year_2012_return=-0.007364 < -0.001', 'year_2023_return=-0.024350 < -0.001']
- Sharpe: 0.3395
- UPI: 0.6771
- CAGR: 0.00756
- Total return: 0.119107
- Max DD: -0.066847
- Final equity: 391687451.7132487
- Calendar-year returns: 2011: +3.57%, 2012: -0.74%, 2013: +1.31%, 2014: -0.01%, 2015: +0.51%, 2016: +1.60%, 2017: +0.21%, 2018: +0.90%, 2019: +2.23%, 2020: +1.85%, 2021: +1.98%, 2022: +0.20%, 2023: -2.44%, 2024: +2.42%, 2025: +0.00%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['year_2019_return=-0.014434 < -0.001', 'year_2022_return=-0.014334 < -0.001', 'year_2024_return=-0.009164 < -0.001']
- Sharpe: 0.2121
- UPI: 0.3993
- CAGR: 0.004773
- Total return: 0.063598
- Max DD: -0.043735
- Final equity: 372259171.9308285
- Calendar-year returns: 2013: +0.38%, 2014: +0.19%, 2015: +0.90%, 2016: +1.63%, 2017: -0.07%, 2018: +1.89%, 2019: -1.44%, 2020: +1.59%, 2021: +2.02%, 2022: -1.43%, 2023: +1.04%, 2024: -0.92%, 2025: +0.00%
