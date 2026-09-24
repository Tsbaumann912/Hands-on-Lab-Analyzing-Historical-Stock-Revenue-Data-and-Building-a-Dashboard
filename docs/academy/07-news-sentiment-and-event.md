# Module 07 — News, Events & Sentiment (S21–S22)

**Edge hypothesis:** Scheduled and unscheduled information moves prices
faster than it is fully discounted; positioning extremes amplify moves.

**Fails when:** “Priced in” data, fakeouts, widened spreads, broker
restrictions around news.

---

## S21 — Scheduled news / event trading

### Mechanics

Trade market reactions to CPI, NFP, FOMC, GDP, central-bank speakers, EIA.
Styles: (a) straddle/strangle via options before event; (b) directional
bias from surprise vs consensus; (c) fade the initial spike; (d) follow
through after consolidation.

### Where it works

FX majors and index/rate futures with tight spreads *outside* the print;
during the print, slippage dominates retail edge.

### Risk notes

- Spreads explode at release — backtests that ignore this lie.
- Many brokers widen or halt — know your venue.
- Prefer waiting for the first 1–5 minutes of discovery unless you have
  institutional infra.

### Videos & links

- Babypips: [Understanding and Trading the News](https://www.babypips.com/learn/forex)
- Babypips: [Economic Data and Market Reactions](https://www.babypips.com/learn/forex)
- CME: economic event education via [cmegroup.com/education](https://www.cmegroup.com/education.html)
- Calendar tools: Babypips economic calendar · Forex Factory calendar

### QuantTerminal hook

News/sentiment APIs in `app/news_service.py` (headline sentiment via VADER)
for research context — not a turnkey news scalper.

---

## S22 — Positioning & sentiment (COT, risk-on/off)

### Mechanics

Use CFTC Commitment of Traders (COT) extremes, risk sentiment indices,
volatility fear gauges, or equity–FX cross-asset cues. Contrarian when
specs are extremely one-sided; or trend-align with commercial hedging flows
depending on framework.

### Where it works

Weekly/daily horizons on FX futures and commodities with reliable COT
coverage. Poor for scalping.

### Risk notes

- COT is lagged (released Fridays for prior Tuesday).
- Extremes can persist.

### Videos & links

- CFTC COT: [cftc.gov/MarketReports/CommitmentsofTraders](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)
- Babypips: [Sentiment Analysis](https://www.babypips.com/learn/forex)
- Investopedia: [Commitment of Traders](https://www.investopedia.com/terms/c/commitmentoftraders.asp)
- Babypips: [Using Equities to Trade FX](https://www.babypips.com/learn/forex)

---

## Family practice

1. Paper-trade three CPI releases with a *post*-spike rule only; record
   slippage assumptions honestly.
2. Overlay COT net specs on a currency futures chart for 2 years; mark
   extremes vs subsequent 4-week returns.
