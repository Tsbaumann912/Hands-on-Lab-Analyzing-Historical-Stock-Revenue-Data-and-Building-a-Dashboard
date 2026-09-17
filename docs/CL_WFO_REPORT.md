# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Optuna objective: `cagr`
- Position sizing: uncapped (ATR/Kelly only)
- System MaxDD gate: < 30%
- Calendar-year returns: reported (not gated)
- OOS-fold profitability: required (each OOS fold ≥ 0)
- **Overall pass:** False

## Anchored WFO

- Windows: 15
- Gates passed: False
- Failures: ['oos_window_2_return=-0.026813 < 0.0', 'oos_window_6_return=-0.121603 < 0.0', 'oos_window_7_return=-0.005841 < 0.0', 'oos_window_12_return=-0.012918 < 0.0', 'oos_window_13_return=-0.048667 < 0.0', 'oos_window_14_return=-0.033047 < 0.0']
- Sharpe: 0.331
- UPI: 0.2416
- CAGR: 0.023707
- Total return: 0.419155
- Max DD: -0.233616
- Final equity: 496704318.6101938
- Calendar-year returns: —

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['oos_window_3_return=-0.033943 < 0.0', 'oos_window_4_return=-0.129255 < 0.0', 'oos_window_5_return=-0.075165 < 0.0', 'oos_window_8_return=-0.031041 < 0.0', 'oos_window_12_return=-0.031221 < 0.0']
- Sharpe: 0.2611
- UPI: 0.1597
- CAGR: 0.018088
- Total return: 0.261273
- Max DD: -0.275753
- Final equity: 441445461.4098505
- Calendar-year returns: —
