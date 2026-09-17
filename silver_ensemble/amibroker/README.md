# AmiBroker (AFL) port

Downloadable formula: [`SilverEnsemble_SI.afl`](SilverEnsemble_SI.afl)

Port of the Python `silver_ensemble` CTA for COMEX Silver (SI):

- Sleeves: TSMOM, carry/basis (proxy), inventory (Aux1/proxy), fade (Foreign gold/DXY/TNX), StochRSI+MA
- Majority-sign blend, agreement gate, forecast buffer, vol-targeted contract sizing
- Futures mode: `PointValue = 5000`, ATR stop / profit via `ApplyStop`
- Defaults match `config/default.yaml` (all-years-positive optimization)

Open in AmiBroker Formula Editor → Send to Analysis. Set Positions to **Long and Short**.
