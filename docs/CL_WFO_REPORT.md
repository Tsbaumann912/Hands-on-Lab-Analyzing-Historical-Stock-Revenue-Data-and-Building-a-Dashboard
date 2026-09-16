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
- Failures: ['year_2011_return=-0.041935 < 0.0', 'year_2012_return=-0.001699 < 0.0', 'year_2017_return=-0.000628 < 0.0', 'year_2022_return=-0.005944 < 0.0']
- Sharpe: 0.1124
- UPI: 0.1033
- CAGR: 0.002183
- Total return: 0.033116
- Max DD: -0.063245
- Final equity: 361590437.55245036
- Calendar-year returns: 2011: -4.19%, 2012: -0.17%, 2013: +0.80%, 2014: +1.04%, 2015: +0.76%, 2016: +1.21%, 2017: -0.06%, 2018: +1.54%, 2019: +0.51%, 2020: +1.71%, 2021: +2.00%, 2022: -0.59%, 2023: +0.99%, 2024: +0.03%, 2025: +0.00%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['year_2014_return=-0.000080 < 0.0', 'year_2017_return=-0.002515 < 0.0', 'year_2019_return=-0.020282 < 0.0', 'year_2024_return=-0.056131 < 0.0', 'year_2025_return=-0.001963 < 0.0']
- Sharpe: 0.0409
- UPI: 0.0273
- CAGR: 0.000637
- Total return: 0.008278
- Max DD: -0.089002
- Final equity: 352897185.07779837
- Calendar-year returns: 2013: +0.96%, 2014: -0.01%, 2015: +0.57%, 2016: +1.63%, 2017: -0.25%, 2018: +1.86%, 2019: -2.03%, 2020: +1.96%, 2021: +2.04%, 2022: +0.33%, 2023: +1.03%, 2024: -5.61%, 2025: -0.20%
