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
- Failures: ['year_2012_return=-0.001722 <= 0.0', 'year_2014_return=0.000000 <= 0.0', 'year_2022_return=-0.001858 <= 0.0', 'year_2023_return=-0.032515 <= 0.0']
- Sharpe: 0.248
- UPI: 0.3397
- CAGR: 0.005442
- Total return: 0.084471
- Max DD: -0.071131
- Final equity: 379564838.1792832
- Calendar-year returns: 2011: +0.13%, 2012: -0.17%, 2013: +0.41%, 2014: +0.00%, 2015: +0.78%, 2016: +1.16%, 2017: +0.20%, 2018: +1.49%, 2019: +0.31%, 2020: +2.00%, 2021: +2.01%, 2022: -0.19%, 2023: -3.25%, 2024: +1.06%, 2025: +2.76%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['sharpe_ratio=-0.0745 <= 0.0', 'ulcer_performance_index=-0.0422 <= 0.0', 'cagr=-0.002454 <= 0.0', 'year_2017_return=-0.108520 <= 0.0', 'year_2022_return=-0.013832 <= 0.0', 'year_2023_return=-0.019370 <= 0.0', 'year_2025_return=-0.011956 <= 0.0']
- Sharpe: -0.0745
- UPI: -0.0422
- CAGR: -0.002454
- Total return: -0.031311
- Max DD: -0.151205
- Final equity: 339041005.3818707
- Calendar-year returns: 2013: +0.95%, 2014: +1.62%, 2015: +0.58%, 2016: +1.68%, 2017: -10.85%, 2018: +1.90%, 2019: +2.74%, 2020: +1.92%, 2021: +1.99%, 2022: -1.38%, 2023: -1.94%, 2024: +0.31%, 2025: -1.20%
