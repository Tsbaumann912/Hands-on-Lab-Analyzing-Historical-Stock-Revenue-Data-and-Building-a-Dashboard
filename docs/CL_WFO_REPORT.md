# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- **Overall pass:** False

## Anchored WFO

- Windows: 15
- Gates passed: False
- Failures: ['sharpe_ratio=-0.4644 <= 0.0', 'ulcer_performance_index=-0.0836 <= 0.0', 'cagr=-0.000179 <= 0.0']
- Sharpe: -0.4644
- UPI: -0.0836
- CAGR: -0.000179
- Max DD: -0.003333
- Final equity: 349063085.52642393

## Rolling WFO

- Windows: 13
- Gates passed: True
- Failures: —
- Sharpe: 0.3569
- UPI: 0.6294
- CAGR: 0.000151
- Max DD: -0.000601
- Final equity: 350684700.971722
