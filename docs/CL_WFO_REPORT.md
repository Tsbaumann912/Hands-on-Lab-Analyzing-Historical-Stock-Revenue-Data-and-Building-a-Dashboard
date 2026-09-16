# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Optuna objective: `cagr`
- Position sizing: uncapped (ATR/Kelly only)
- System MaxDD gate: < 30%
- Calendar-year profitability: required (every evaluable calendar year > 0)
- **Overall pass:** False

## Anchored WFO

- Windows: 15
- Gates passed: False
- Failures: ['year_2012_return=-0.000308 <= 0.0', 'year_2017_return=-0.003545 <= 0.0', 'year_2022_return=-0.002823 <= 0.0', 'year_2023_return=-0.002453 <= 0.0']
- Sharpe: 0.5051
- UPI: 2.0025
- CAGR: 0.00974
- Total return: 0.155821
- Max DD: -0.03446
- Final equity: 404537493.48385245
- Calendar-year returns: 2011: +3.42%, 2012: -0.03%, 2013: +0.46%, 2014: +0.70%, 2015: +0.76%, 2016: +1.57%, 2017: -0.35%, 2018: +1.87%, 2019: +2.42%, 2020: +1.52%, 2021: +1.98%, 2022: -0.28%, 2023: -0.25%, 2024: +1.06%, 2025: +1.01%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['year_2017_return=-0.002745 <= 0.0']
- Sharpe: 0.4395
- UPI: 0.7648
- CAGR: 0.010366
- Total return: 0.14286
- Max DD: -0.071236
- Final equity: 400001005.96904033
- Calendar-year returns: 2013: +0.99%, 2014: +0.33%, 2015: +1.20%, 2016: +1.70%, 2017: -0.27%, 2018: +0.81%, 2019: +0.45%, 2020: +1.79%, 2021: +1.98%, 2022: +1.67%, 2023: +0.96%, 2024: +0.33%, 2025: +2.76%
