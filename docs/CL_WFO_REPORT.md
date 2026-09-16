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
- Failures: ['year_2011_return=-0.042553 <= 0.0', 'year_2013_return=-0.000609 <= 0.0', 'year_2017_return=-0.050750 <= 0.0', 'year_2019_return=-0.035144 <= 0.0', 'year_2021_return=-0.015549 <= 0.0', 'year_2023_return=-0.018530 <= 0.0', 'year_2024_return=-0.026460 <= 0.0', 'year_2025_return=-0.011460 <= 0.0']
- Sharpe: 0.1623
- UPI: 0.1109
- CAGR: 0.005009
- Total return: 0.077513
- Max DD: -0.111356
- Final equity: 377129482.5083067
- Calendar-year returns: 2011: -4.26%, 2012: +1.82%, 2013: -0.06%, 2014: +3.14%, 2015: +6.83%, 2016: +2.27%, 2017: -5.07%, 2018: +0.29%, 2019: -3.51%, 2020: +10.46%, 2021: -1.55%, 2022: +4.66%, 2023: -1.85%, 2024: -2.65%, 2025: -1.15%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['year_2016_return=-0.028802 <= 0.0', 'year_2017_return=-0.020128 <= 0.0', 'year_2024_return=-0.050398 <= 0.0']
- Sharpe: 0.4633
- UPI: 0.37
- CAGR: 0.014569
- Total return: 0.205963
- Max DD: -0.089756
- Final equity: 422086900.51268303
- Calendar-year returns: 2013: +2.20%, 2014: +4.59%, 2015: +1.76%, 2016: -2.88%, 2017: -2.01%, 2018: +0.66%, 2019: +2.00%, 2020: +8.53%, 2021: +0.38%, 2022: +5.05%, 2023: +3.47%, 2024: -5.04%, 2025: +1.27%
