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
- Failures: ['year_2011_return=-0.031277 <= 0.0', 'year_2019_return=-0.007438 <= 0.0', 'year_2023_return=-0.002876 <= 0.0']
- Sharpe: 0.2484
- UPI: 0.3454
- CAGR: 0.005676
- Total return: 0.088241
- Max DD: -0.093071
- Final equity: 380884298.5576097
- Calendar-year returns: 2011: -3.13%, 2012: +0.02%, 2013: +1.32%, 2014: +2.44%, 2015: +0.77%, 2016: +1.59%, 2017: +0.30%, 2018: +1.42%, 2019: -0.74%, 2020: +1.69%, 2021: +2.04%, 2022: +0.90%, 2023: -0.29%, 2024: +0.73%, 2025: +0.77%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['year_2025_return=-0.004083 <= 0.0']
- Sharpe: 0.7578
- UPI: 2.6863
- CAGR: 0.012236
- Total return: 0.170549
- Max DD: -0.028859
- Final equity: 409692107.64263535
- Calendar-year returns: 2013: +0.78%, 2014: +3.33%, 2015: +1.81%, 2016: +1.99%, 2017: +0.41%, 2018: +1.90%, 2019: +1.52%, 2020: +1.82%, 2021: +2.02%, 2022: +0.06%, 2023: +0.98%, 2024: +0.27%, 2025: -0.41%
