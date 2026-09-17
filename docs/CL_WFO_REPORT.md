# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Optuna objective: `cagr`
- Position sizing: uncapped (ATR/Kelly only)
- System MaxDD gate: < 30%
- Calendar-year returns: ≥80% non-losing years; worst year ≥ -5%
- OOS-fold profitability: not required
- **Overall pass:** False

## Anchored WFO

- Windows: 15
- Gates passed: True
- Failures: —
- Sharpe: 0.1986
- UPI: 0.3067
- CAGR: 0.004909
- Total return: 0.075911
- Max DD: -0.097016
- Final equity: 376568994.6780328
- Calendar-year returns: 2011: +0.18%, 2012: +0.00%, 2013: +1.49%, 2014: +0.28%, 2015: +0.50%, 2016: +1.22%, 2017: +0.02%, 2018: +0.93%, 2019: +2.13%, 2020: +1.89%, 2021: +2.03%, 2022: +0.29%, 2023: -1.32%, 2024: +1.21%, 2025: -3.38%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['worst_calendar_year=-0.064797 < -0.05']
- Sharpe: 0.1696
- UPI: 0.1935
- CAGR: 0.00373
- Total return: 0.049386
- Max DD: -0.099741
- Final equity: 367285140.4073683
- Calendar-year returns: 2013: +0.32%, 2014: +0.36%, 2015: +1.84%, 2016: +1.57%, 2017: +0.05%, 2018: +2.72%, 2019: +0.42%, 2020: +1.92%, 2021: +2.04%, 2022: -0.17%, 2023: +0.00%, 2024: -6.48%, 2025: +2.74%
