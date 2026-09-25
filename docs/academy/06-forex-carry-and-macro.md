# Module 06 — Forex Carry & Macro (S17–S20)

**Edge hypothesis:** Currency excess returns compensate for interest
differentials, purchasing-power gaps, and persistent macro regimes — not only
from chart patterns.

**Fails when:** Risk-off unwind of carry (2008-style), sudden policy shifts,
interventions.

---

## S17 — Classic FX carry trade

### Mechanics

Long high-yielding currencies / short low-yielding currencies (often G10 or
EM basket). Hold and collect the interest differential (swap or via FX
futures curve). Equal-vol or equal-notional variants; sometimes long top-3 /
short bottom-3 ranked by deposit rates.

### Where it works

Calm risk-on regimes with stable rate differentials. Poor during equity
crashes when high-yielders sell off together.

### Risk notes

- Left-tail crashes; use vol targeting and risk-off overlays (equity trend
  filter, VIX).
- Funding liquidity risk.

### Videos & links

- Babypips: [Carry Trade](https://www.babypips.com/learn/forex) (Macro / Carry modules)
- Investopedia: [Carry Trade](https://www.investopedia.com/terms/c/carrytrade.asp)
- Academic-style overview: [Momentum and Carry in FX (thesis PDF)](https://thesis.eur.nl/pub/67782/579706.pdf)

---

## S18 — Cross-sectional FX momentum

### Mechanics

Rank currencies by past excess returns; long winners / short losers. Often
combined with carry because correlation between the two is imperfect —
diversification benefit.

### Where it works

Liquid G10 and selected EM; implementation via spot baskets or currency
futures.

### Risk notes

- Momentum crashes after sharp reversals.
- Transaction costs on frequent rebalances.

### Videos & links

- Babypips: [Intermarket / USD Index modules](https://www.babypips.com/learn/forex)
- Investopedia: [Currency Momentum](https://www.investopedia.com/articles/forex/08/currency-momentum.asp)
- QuantPedia / research summaries on FX momentum (search QuantPedia FX momentum)

---

## S19 — PPP / real-exchange-rate value

### Mechanics

Fade currencies rich/cheap vs purchasing-power parity or real effective
exchange rate (REER) bands. Long-horizon (months–years); often used as a
slow value overlay, not a day-trade signal.

### Where it works

Macro discretionary and slow systematic value books. Weak as a standalone
short-horizon strategy.

### Risk notes

- “Markets can remain irrational” for years — need survival capital.
- PPP estimates depend on basket methodology.

### Videos & links

- Investopedia: [Purchasing Power Parity](https://www.investopedia.com/terms/p/purchasingpowerparity.asp)
- OECD / IMF REER data portals (search “IMF real effective exchange rate”)
- Babypips: [Macro Fundamentals](https://www.babypips.com/learn/forex)

---

## S20 — Forward-rate bias & rate differentials

### Mechanics

Related to carry: trade the tendency for the forward rate to be a biased
predictor of future spot (forward premium puzzle). Use yield curve and
central-bank path expectations.

### Where it works

G10 with clear hiking/cutting cycles; event weeks around central banks.

### Risk notes

- Policy surprises dominate technicals around FOMC/ECB.
- Spreads compress when markets price cuts early.

### Videos & links

- Babypips: [Economic Data and Market Reactions](https://www.babypips.com/learn/forex)
- Federal Reserve education: [federalreserve.gov](https://www.federalreserve.gov/)
- CME FedWatch / rates education: [cmegroup.com](https://www.cmegroup.com/markets/interest-rates.html)

---

## Family practice

1. Build a weekly carry rank of G10; track returns in risk-on vs risk-off
   months (use equity trend as proxy).
2. Correlate S17 and S18 weekly P&L — expect imperfect correlation.
