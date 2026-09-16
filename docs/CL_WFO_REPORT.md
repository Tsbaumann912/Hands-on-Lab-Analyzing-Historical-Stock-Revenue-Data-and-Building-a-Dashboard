# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- **Overall pass:** False

## Anchored WFO

- Windows: 15
- Gates passed: False
- Failures: ['sharpe_ratio=-0.2723 <= 0.0', 'ulcer_performance_index=-0.002 <= 0.0', 'cagr=-1.0 <= 0.0', 'abs(max_drawdown)=27344.031937 >= 0.3']
- Sharpe: -0.2723
- UPI: -0.002
- CAGR: -1.0
- Max DD: -27344.031937
- Final equity: -1.7246307853158792e+61

## Rolling WFO

- Windows: 13
- Gates passed: False
- Failures: ['sharpe_ratio=-0.4641 <= 0.0', 'abs(max_drawdown)=3279.570649 >= 0.3']
- Sharpe: -0.4641
- UPI: 5612.8458
- CAGR: 855375.463536
- Max DD: -3279.570649
- Final equity: 2.2703295709173847e+85
