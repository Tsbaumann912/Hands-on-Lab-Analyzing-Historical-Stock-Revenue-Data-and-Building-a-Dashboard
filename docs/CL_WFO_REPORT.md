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
- Failures: ['year_2023_return=-0.034058 < 0.0']
- Sharpe: 0.2657
- UPI: 0.2813
- CAGR: 0.006113
- Total return: 0.095334
- Max DD: -0.091401
- Final equity: 383367014.398064
- Calendar-year returns: 2011: +0.81%, 2012: +0.48%, 2013: +1.29%, 2014: +1.43%, 2015: +0.78%, 2016: +1.21%, 2017: +0.34%, 2018: +0.56%, 2019: +1.73%, 2020: +1.96%, 2021: +2.05%, 2022: +0.22%, 2023: -3.41%, 2024: +0.81%, 2025: +0.22%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['year_2017_return=-0.000164 < 0.0', 'year_2019_return=-0.003573 < 0.0', 'year_2022_return=-0.003667 < 0.0', 'year_2023_return=-0.016024 < 0.0', 'year_2025_return=-0.004069 < 0.0']
- Sharpe: 0.3379
- UPI: 0.4197
- CAGR: 0.004379
- Total return: 0.058207
- Max DD: -0.040602
- Final equity: 370372462.2044797
- Calendar-year returns: 2013: +0.32%, 2014: +0.90%, 2015: +1.82%, 2016: +1.71%, 2017: -0.02%, 2018: +0.67%, 2019: -0.36%, 2020: +2.00%, 2021: +2.05%, 2022: -0.37%, 2023: -1.60%, 2024: +0.25%, 2025: -0.41%
