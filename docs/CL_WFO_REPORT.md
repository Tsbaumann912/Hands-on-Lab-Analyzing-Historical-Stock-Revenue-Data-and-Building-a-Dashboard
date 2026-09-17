# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Optuna objective: `cagr`
- Position sizing: uncapped (ATR/Kelly only)
- System MaxDD gate: < 30%
- Calendar-year returns: ≥70% non-losing years; worst year ≥ -8%
- OOS-fold profitability: not required
- **Overall pass:** False

## Anchored WFO

- Windows: 15
- Gates passed: True
- Failures: —
- Sharpe: 0.3232
- UPI: 1.0378
- CAGR: 0.005702
- Total return: 0.088662
- Max DD: -0.030269
- Final equity: 381031750.1737973
- Calendar-year returns: 2011: +1.15%, 2012: +0.00%, 2013: +1.34%, 2014: +0.03%, 2015: +0.76%, 2016: +1.19%, 2017: +0.20%, 2018: +1.41%, 2019: -0.53%, 2020: +1.57%, 2021: +2.04%, 2022: -0.83%, 2023: +0.99%, 2024: +0.46%, 2025: +0.00%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['calendar_year_pass_fraction=0.615 < 0.7']
- Sharpe: 0.1201
- UPI: 0.0945
- CAGR: 0.001852
- Total return: 0.024251
- Max DD: -0.061403
- Final equity: 358487884.6977742
- Calendar-year returns: 2013: +0.37%, 2014: +0.00%, 2015: +1.80%, 2016: +1.69%, 2017: -0.27%, 2018: -0.45%, 2019: +1.54%, 2020: +2.04%, 2021: +2.05%, 2022: -0.82%, 2023: -2.09%, 2024: -1.46%, 2025: +0.25%
