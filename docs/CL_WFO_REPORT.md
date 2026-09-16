# CL Walk-Forward Validation Report

- Strategy: `CLCarryMomentum`
- Capital: $350,000,000
- Start: 2008-01-01
- Data: {'n_bars': 4706, 'start': '2008-01-02T00:00:00+00:00', 'end': '2026-09-16T00:00:00+00:00', 'source': 'yfinance_CL=F'}
- Position sizing: uncapped (ATR/Kelly only)
- System MaxDD gate: < 30%
- **Overall pass:** True

## Anchored WFO

- Windows: 15
- Gates passed: True
- Failures: —
- Sharpe: 0.4736
- UPI: 0.519
- CAGR: 0.014995
- Max DD: -0.079605
- Final equity: 437159735.1573555

## Rolling WFO

- Windows: 13
- Gates passed: True
- Failures: —
- Sharpe: 0.2509
- UPI: 0.1467
- CAGR: 0.007787
- Max DD: -0.111154
- Final equity: 386978961.2652076
