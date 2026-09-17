# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Optuna objective: `cagr`
- Position sizing: uncapped (ATR/Kelly only)
- System MaxDD gate: < 30%
- Calendar-year returns: required (stitched calendar years ≥ 0)
- OOS-fold profitability: not required
- **Overall pass:** False

## Anchored WFO

- Windows: 15
- Gates passed: False
- Failures: ['sharpe_ratio=-0.0466 <= 0.0', 'ulcer_performance_index=-0.0294 <= 0.0', 'cagr=-0.001559 <= 0.0', 'year_2011_return=-0.073994 < -0.005', 'year_2014_return=-0.010139 < -0.005', 'year_2023_return=-0.048337 < -0.005']
- Sharpe: -0.0466
- UPI: -0.0294
- CAGR: -0.001559
- Total return: -0.023042
- Max DD: -0.090817
- Final equity: 341935131.0788249
- Calendar-year returns: 2011: -7.40%, 2012: -0.10%, 2013: +0.43%, 2014: -1.01%, 2015: +0.76%, 2016: +1.57%, 2017: +0.00%, 2018: +2.09%, 2019: +1.53%, 2020: +1.85%, 2021: +2.05%, 2022: +1.78%, 2023: -4.83%, 2024: +0.85%, 2025: -0.18%

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['sharpe_ratio=-0.2816 <= 0.0', 'ulcer_performance_index=-0.1245 <= 0.0', 'cagr=-0.006066 <= 0.0', 'year_2019_return=-0.008283 < -0.005', 'year_2022_return=-0.014879 < -0.005', 'year_2023_return=-0.019796 < -0.005', 'year_2024_return=-0.103285 < -0.005']
- Sharpe: -0.2816
- UPI: -0.1245
- CAGR: -0.006066
- Total return: -0.075758
- Max DD: -0.146125
- Final equity: 323484565.05800104
- Calendar-year returns: 2013: +0.00%, 2014: +1.31%, 2015: +1.80%, 2016: +1.64%, 2017: +0.00%, 2018: +0.00%, 2019: -0.83%, 2020: +2.04%, 2021: +2.05%, 2022: -1.49%, 2023: -1.98%, 2024: -10.33%, 2025: -0.18%
