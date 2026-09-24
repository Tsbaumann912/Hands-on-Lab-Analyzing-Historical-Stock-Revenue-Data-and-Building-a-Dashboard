# Module 03 — Mean Reversion & Range (S05–S08)

**Edge hypothesis:** Prices oscillate around a local equilibrium (VWAP,
moving mean, fair value); extremes revert more often than they continue
*in that regime*.

**Fails when:** Strong trends, gap openings, structural breaks, thin liquidity.

---

## S05 — RSI / Stochastic fade

### Mechanics

Fade overbought/oversold oscillator readings (e.g. RSI > 70 short, < 30 long)
with confirmation from support/resistance or a short MA. Exit at mid-band or
opposite threshold.

### Where it works

Range-bound FX pairs and equity indices in consolidation; shorter timeframes
need tighter costs control.

### Risk notes

- Oscillators stay pinned in trends — use a trend filter or hard stop.
- High win rate can hide large left-tail losses.

### Videos & links

- Babypips: [RSI](https://www.babypips.com/learn/forex/how-to-use-rsi) · [Stochastic](https://www.babypips.com/learn/forex/how-to-use-the-stochastic-indicator)
- Investopedia: [Relative Strength Index](https://www.investopedia.com/terms/r/rsi.asp)

---

## S06 — Bollinger Band mean reversion

### Mechanics

Enter when price closes outside ±kσ bands; target the middle band (SMA).
Variants: %B, BandWidth filters, Keltner confirmation.

### Where it works

Mean-reverting commodities and FX in quiet vol; poor in breakout regimes
(see S11 for the opposite trade).

### Risk notes

- Walking the bands is a classic trap — require regime filter.
- Widen stops beyond “just inside the band.”

### Videos & links

- Babypips: [Bollinger Bands](https://www.babypips.com/learn/forex/how-to-use-bollinger-bands)
- Investopedia: [Bollinger Band®](https://www.investopedia.com/terms/b/bollingerbands.asp)

### QuantTerminal hook

`strategies/mean_reversion.py`

---

## S07 — VWAP / session mean reversion (intraday)

### Mechanics

Fade extensions away from session VWAP (or anchored VWAP) during liquid
hours; cover into VWAP. Often combined with inventory/time-of-day filters.

### Where it works

Liquid futures (ES, NQ, CL) and major FX during London/NY. Requires low
latency relative to holding period and tight spreads.

### Risk notes

- News spikes invalidate VWAP magnets.
- Overnight gaps — flat before events if you lack edge.

### Videos & links

- Investopedia: [VWAP](https://www.investopedia.com/terms/v/vwap.asp)
- CME education on order types / day trading readiness: [Things to Know Before Trading](https://www.cmegroup.com/education/courses/things-to-know-before-trading-cme-futures.html)

---

## S08 — Pairs / relative-value arbitrage

### Mechanics

Trade the spread between related instruments (e.g. gold vs silver futures,
EURUSD vs EURJPY residual, crack-related books). Z-score of spread; long
cheap / short rich; exit at mean.

### Where it works

Economically linked futures and FX crosses. Needs cointegration / stable
hedge ratio research — correlation alone is not enough.

### Risk notes

- Spread can “run away” when the relationship breaks (regime shift).
- Legging risk and margin on both sides.

### Videos & links

- Investopedia: [Pairs Trading](https://www.investopedia.com/terms/p/pairstrade.asp)
- TradeStation: [Introduction to Futures Spread Trading](https://www.tradestation.com/learn/futures-education-center/an-introduction-to-futures-spread-trading/)
- Babypips: [Currency Correlations](https://www.babypips.com/learn/forex) (Risk / correlations modules)

---

## Family practice

1. Compare S06 with and without an ADX < threshold filter.
2. For S08, plot rolling hedge ratio stability — if unstable, do not trade.
