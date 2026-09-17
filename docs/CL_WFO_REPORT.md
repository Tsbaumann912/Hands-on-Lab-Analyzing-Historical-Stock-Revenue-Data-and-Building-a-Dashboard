# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Optuna objective: `cagr`
- Position sizing: uncapped (ATR/Kelly only)
- System MaxDD gate: < 30%
- Calendar-year profitability: required (every evaluable calendar year ≥ 0)
- **Overall pass:** False

## Anchored WFO

- Windows: 15
- Gates passed: False
- Failures: ['year_2012_return=-0.002329 < 0.0', 'year_2014_return=-0.002138 < 0.0', 'year_2019_return=-0.001994 < 0.0']
- Sharpe: 0.3636
- UPI: 1.3332
- CAGR: 0.006082
- Total return: 0.094831
- Max DD: -0.03187
- Final equity: 383190729.6997355
- Calendar-year returns: 2011: +0.66%, 2012: -0.23%, 2013: +1.45%, 2014: -0.21%, 2015: +0.79%, 2016: +1.68%, 2017: +0.02%, 2018: +0.36%, 2019: -0.20%, 2020: +1.71%, 2021: +2.02%, 2022: +0.06%, 2023: +1.03%, 2024: +1.23%, 2025: +0.00%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['year_2017_return=-0.002179 < 0.0', 'year_2019_return=-0.021428 < 0.0', 'year_2022_return=-0.003210 < 0.0', 'year_2024_return=-0.008951 < 0.0']
- Sharpe: 0.4611
- UPI: 0.8044
- CAGR: 0.006506
- Total return: 0.087597
- Max DD: -0.028812
- Final equity: 380659067.2243045
- Calendar-year returns: 2013: +0.91%, 2014: +2.23%, 2015: +1.83%, 2016: +1.55%, 2017: -0.22%, 2018: +1.90%, 2019: -2.14%, 2020: +1.63%, 2021: +2.05%, 2022: -0.32%, 2023: +1.02%, 2024: -0.90%, 2025: +0.21%
