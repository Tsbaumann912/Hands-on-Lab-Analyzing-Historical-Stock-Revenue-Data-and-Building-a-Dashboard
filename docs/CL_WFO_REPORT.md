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
- Failures: ['year_2023_return=-0.028592 <= 0.0']
- Sharpe: 0.3544
- UPI: 0.4118
- CAGR: 0.008633
- Total return: 0.137046
- Max DD: -0.106241
- Final equity: 397966012.91037077
- Calendar-year returns: 2011: +1.03%, 2012: +0.84%, 2013: +1.46%, 2014: +1.56%, 2015: +0.76%, 2016: +1.66%, 2017: +0.64%, 2018: +1.42%, 2019: +0.95%, 2020: +1.52%, 2021: +1.97%, 2022: +1.83%, 2023: -2.86%, 2024: +0.87%, 2025: +0.56%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['year_2022_return=-0.002194 <= 0.0', 'year_2023_return=-0.013359 <= 0.0']
- Sharpe: 0.6208
- UPI: 1.5356
- CAGR: 0.01021
- Total return: 0.140582
- Max DD: -0.035654
- Final equity: 399203741.92365056
- Calendar-year returns: 2013: +1.25%, 2014: +1.66%, 2015: +1.80%, 2016: +2.35%, 2017: +0.25%, 2018: +1.60%, 2019: +0.80%, 2020: +1.56%, 2021: +1.99%, 2022: -0.22%, 2023: -1.34%, 2024: +0.32%, 2025: +2.90%
