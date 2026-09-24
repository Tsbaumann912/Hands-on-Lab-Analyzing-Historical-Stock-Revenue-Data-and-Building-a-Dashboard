# Module 02 — Trend Following & Momentum (S01–S04)

**Edge hypothesis:** Prices exhibit serial dependence; gains from large
trending moves outweigh frequent small losses (positive skew of trade P&L).

**Fails when:** Choppy, mean-reverting regimes; high correlation shocks;
transaction costs on frequent whipsaws.

---

## S01 — Dual moving-average / EMA crossover

### Mechanics

Go long when a fast MA crosses above a slow MA; go short (or flat) on the
opposite cross. Variants: SMA vs EMA, triple MA, “ribbon” filters.

### Where it works

Liquid futures (equity indices, rates, commodities) and major FX pairs on
daily/weekly bars. Classic CTA building block.

### Risk notes

- Always-in systems take every whipsaw; add a flat zone or ADX filter (S04).
- Volatility-scale size; raw fixed-contract sizing over-risks quiet markets.

### Videos & links

- Investopedia: [Moving Average Strategies](https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp)
- Babypips: [Moving Averages](https://www.babypips.com/learn/forex/elementary) (Grade 4)
- CME / CTA context: search CME Institute “trend following” webinars on [cmegroup.com/education](https://www.cmegroup.com/education.html)
- YouTube (CME Group): browse [@cmegroup](https://www.youtube.com/@cmegroup) for trend/managed futures explainers

### QuantTerminal hook

`strategies/trend_following.py` — systematic trend module used in Strategy Lab.

---

## S02 — Donchian / Turtle-style channel breakout

### Mechanics

Enter on N-day high/low breakouts (classic Turtle: 20-day entry, 10-day exit;
or 55-day long-term system). Pyramid on further breakouts; risk a fixed % of
equity per unit.

### Where it works

Diversified futures portfolios — the original Turtle markets (currencies,
rates, commodities, metals). Less reliable on single noisy FX pairs alone.

### Risk notes

- Large open risk during cascading trends; use unit sizing and max units.
- False breakouts in ranges dominate trade count.

### Videos & links

- Original Turtle rules summary (many reprints): search “Original Turtle Trading Rules PDF”
- Investopedia: [Donchian Channel](https://www.investopedia.com/terms/d/donchianchannels.asp)
- Babypips: [Trading Breakouts](https://www.babypips.com/learn/forex/high-school) (Grade 10)

---

## S03 — Time-series momentum (TSMOM)

### Mechanics

Academic / quant form: signal = past return over lookback *L* (e.g. 1–12
months). Position long if past return > 0 else short; size by inverse
volatility. Cross-asset portfolios (Moskowitz, Ooi, Pedersen style).

### Where it works

Broad futures universes; also FX as currency futures. Horizon matters —
short lookbacks trade more noise.

### Risk notes

- Crash risk when momentum crashes (crowded unwind).
- Skip the most recent month in some equity formulations; futures variants differ.

### Videos & links

- AQR / academic overview: [Time Series Momentum (paper page via Google Scholar)](https://scholar.google.com/scholar?q=Time+Series+Momentum+Moskowitz)
- QuantConnect research (carry + trend): [Combined Carry and Trend](https://www.quantconnect.com/research/16001/combined-carry-and-trend/)
- Investopedia: [Momentum Investing](https://www.investopedia.com/terms/m/momentum.asp)

### QuantTerminal hook

`strategies/momentum.py`

---

## S04 — ADX-filtered directional trend

### Mechanics

Trade trend systems (S01/S02) only when ADX (or similar trend-strength meter)
exceeds a threshold; stand aside in ranges. Direction from +DI/−DI or MA slope.

### Where it works

Reduces whipsaw in FX ranges and equity index congestions; does not create
edge alone — it is a **regime filter**.

### Risk notes

- ADX lags; you often miss the first part of a move.
- Thresholds must be tuned per asset/timeframe — avoid overfit.

### Videos & links

- Babypips: [ADX](https://www.babypips.com/learn/forex/how-to-use-adx)
- Investopedia: [Average Directional Index](https://www.investopedia.com/terms/a/adx.asp)
- Babypips: [Market Environment](https://www.babypips.com/learn/forex/high-school) (Grade 11)

---

## Family practice

1. Backtest S01 and S03 on the same futures panel with identical vol targeting.
2. Measure correlation of daily P&L — often high within the trend family.
3. Add S04 filter; compare trade count, max DD, and Sharpe *after costs*.
