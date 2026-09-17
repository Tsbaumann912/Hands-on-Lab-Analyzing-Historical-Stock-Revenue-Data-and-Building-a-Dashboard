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
- Failures: ['ulcer_performance_index=-0.0833 <= 0.0', 'cagr=-0.018798 <= 0.0', 'abs(max_drawdown)=0.896016 >= 0.3', 'year_2011_return=-0.016207 < 0.0', 'year_2012_return=-0.062074 < 0.0', 'year_2014_return=-0.014428 < 0.0', 'year_2016_return=-0.104421 < 0.0', 'year_2018_return=-0.177372 < 0.0', 'year_2020_return=-0.104467 < 0.0', 'year_2022_return=-0.027460 < 0.0', 'year_2023_return=-0.004330 < 0.0', 'year_2024_return=-0.063483 < 0.0']
- Sharpe: 0.1775
- UPI: -0.0833
- CAGR: -0.018798
- Total return: -0.246872
- Max DD: -0.896016
- Final equity: 263594635.85990587
- Calendar-year returns: 2011: -1.62%, 2012: -6.21%, 2013: +3.55%, 2014: -1.44%, 2015: +4.72%, 2016: -10.44%, 2017: +0.71%, 2018: -17.74%, 2019: +5.18%, 2020: -10.45%, 2021: +17.90%, 2022: -2.75%, 2023: -0.43%, 2024: -6.35%, 2025: +3.42%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['abs(max_drawdown)=0.952953 >= 0.3', 'year_2016_return=-0.070538 < 0.0', 'year_2018_return=-0.161266 < 0.0', 'year_2020_return=-0.113090 < 0.0', 'year_2022_return=-0.034560 < 0.0', 'year_2023_return=-0.023295 < 0.0', 'year_2025_return=-0.007929 < 0.0']
- Sharpe: 0.2553
- UPI: 0.0396
- CAGR: 0.007798
- Total return: 0.105813
- Max DD: -0.952953
- Final equity: 387034643.8734867
- Calendar-year returns: 2013: +0.35%, 2014: +25.44%, 2015: +5.85%, 2016: -7.05%, 2017: +6.67%, 2018: -16.13%, 2019: +1.30%, 2020: -11.31%, 2021: +21.06%, 2022: -3.46%, 2023: -2.33%, 2024: +0.24%, 2025: -0.79%
