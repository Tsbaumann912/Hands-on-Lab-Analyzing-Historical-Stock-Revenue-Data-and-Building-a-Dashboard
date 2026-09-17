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
- Failures: ['oos_window_0_return=-0.078978 < 0.0', 'oos_window_2_return=-0.017180 < 0.0', 'oos_window_4_return=-0.013378 < 0.0', 'oos_window_6_return=-0.142762 < 0.0', 'oos_window_7_return=-0.040133 < 0.0', 'oos_window_10_return=-0.064538 < 0.0', 'oos_window_14_return=-0.039119 < 0.0']
- Sharpe: 0.2437
- UPI: 0.1621
- CAGR: 0.016594
- Total return: 0.278762
- Max DD: -0.271336
- Final equity: 447566629.60049075
- Calendar-year returns: —

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['ulcer_performance_index=-0.014 <= 0.0', 'cagr=-0.002231 <= 0.0', 'abs(max_drawdown)=0.320737 >= 0.3', 'oos_window_3_return=-0.062078 < 0.0', 'oos_window_4_return=-0.142047 < 0.0', 'oos_window_5_return=-0.053533 < 0.0', 'oos_window_8_return=-0.087076 < 0.0', 'oos_window_10_return=-0.022831 < 0.0', 'oos_window_11_return=-0.021760 < 0.0', 'oos_window_12_return=-0.048316 < 0.0']
- Sharpe: 0.0143
- UPI: -0.014
- CAGR: -0.002231
- Total return: -0.028505
- Max DD: -0.320737
- Final equity: 340023264.51151794
- Calendar-year returns: —
