# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Optuna objective: `cagr`
- Position sizing: uncapped (ATR/Kelly only)
- System MaxDD gate: < 30%
- Calendar-year returns: ≥80% non-losing years; worst year ≥ -7%
- OOS-fold profitability: not required
- **Overall pass:** False

## Anchored WFO

- Windows: 15
- Gates passed: False
- Failures: ['calendar_year_pass_fraction=0.733 < 0.8']
- Sharpe: 0.0379
- UPI: 0.0153
- CAGR: 0.00059
- Total return: 0.008854
- Max DD: -0.077198
- Final equity: 353098825.6948612
- Calendar-year returns: 2011: -6.30%, 2012: -0.09%, 2013: +0.03%, 2014: +0.47%, 2015: +0.77%, 2016: +1.67%, 2017: -0.37%, 2018: +0.90%, 2019: +0.79%, 2020: +1.53%, 2021: +2.05%, 2022: +1.14%, 2023: +0.16%, 2024: +0.75%, 2025: -0.20%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['calendar_year_pass_fraction=0.462 < 0.8']
- Sharpe: 0.2765
- UPI: 0.2259
- CAGR: 0.004415
- Total return: 0.058694
- Max DD: -0.070764
- Final equity: 370542890.6343197
- Calendar-year returns: 2013: +0.78%, 2014: -1.03%, 2015: +1.81%, 2016: +1.65%, 2017: -0.35%, 2018: +2.79%, 2019: -0.37%, 2020: +1.74%, 2021: +2.05%, 2022: -1.10%, 2023: -2.06%, 2024: -0.51%, 2025: -0.21%
