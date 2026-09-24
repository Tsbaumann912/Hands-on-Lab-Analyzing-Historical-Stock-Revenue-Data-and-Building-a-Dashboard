# Module 05 — Futures Spreads, Curve & Seasonality (S12–S16)

**Edge hypothesis:** Relative prices along the futures curve (or between
related commodities) are driven by storage, inventory, seasonality, and
hedging flows — often more stable than outright direction.

**Fails when:** Delivery squeezes, structural market changes, extreme weather,
policy shocks.

---

## S12 — Calendar (intra-commodity) spreads

### Mechanics

Long one expiry, short another of the same root (e.g. CL Z vs CL F). Trade
the differential. Lower margin via exchange spread credits.

### Where it works

Energies, ags, metals with meaningful term structure. Prop-style relative
value desks live here.

### Risk notes

- Stops belong on the **spread**, not on one leg.
- Near-expiry liquidity and delivery mechanics matter.

### Videos & links

- TradeStation: [Futures Spread Trading intro](https://www.tradestation.com/learn/futures-education-center/an-introduction-to-futures-spread-trading/)
- EDHEC note: [Commodity Futures Strategies — Trend & Calendar Spreads (PDF)](https://www.premiacap.com/publications/EDHEC_Working_Paper_Commodity_Futures_Trading_Strategies%20Jan%202017.pdf)
- CME product calendars / spread pages via [cmegroup.com](https://www.cmegroup.com/)

---

## S13 — Intercommodity spreads (crack, crush, …)

### Mechanics

Trade economically linked products: crack (crude vs gasoline/heating oil),
crush (soybeans vs meal/oil), gold–silver, treasury curve spreads, equity
index spreads (ES vs NQ).

### Where it works

When the processing / substitution relationship is stable. Often combines
fundamental inventory views with technical entry on the spread chart.

### Risk notes

- Weights must match contract multipliers and barrel/bushel economics.
- Correlation can break in crises.

### Videos & links

- Investopedia: [Crack Spread](https://www.investopedia.com/terms/c/crackspread.asp)
- Investopedia: [Crush Spread](https://www.investopedia.com/terms/c/crushspread.asp)
- NexusFi overview: [Spread Trading in Futures](https://nexusfi.com/a/strategies/spread-trading)

---

## S14 — Curve carry (contango / backwardation)

### Mechanics

Harvest roll yield: in backwardation, long front continuity may earn positive
roll; in deep contango, short-biased or deferred structures may be preferred.
Systematic “carry” forecasts on the futures curve (see Carver-style multi-
forecast CTAs).

### Where it works

Commodity and some FX futures portfolios when combined with trend (divergent
+ convergent mix).

### Risk notes

- Contango can steepen violently (longs bleed).
- Roll methodology in data must match live rolls.

### Videos & links

- QuantConnect: [Combined Carry and Trend](https://www.quantconnect.com/research/16001/combined-carry-and-trend/)
- Investopedia: [Contango](https://www.investopedia.com/terms/c/contango.asp) · [Backwardation](https://www.investopedia.com/terms/b/backwardation.asp)
- CME education on contract rolls / calendars: [cmegroup.com/education](https://www.cmegroup.com/education.html)

---

## S15 — Seasonality & inventory cycles

### Mechanics

Trade recurring seasonal patterns (natural gas winter, ag planting/harvest,
equity “Sell in May” style — treat equity seasonality skeptically). Overlay
inventory reports (EIA, USDA) for confirmation.

### Where it works

Ags and energies with physical seasonality; weaker in pure financial futures.

### Risk notes

- Climate and policy change historical patterns.
- Sample size of “same week next year” is tiny — easy to overfit.

### Videos & links

- CME commodity product insights / seasonality tools: [cmegroup.com](https://www.cmegroup.com/)
- EIA energy reports: [eia.gov](https://www.eia.gov/)
- USDA WASDE: [usda.gov](https://www.usda.gov/)

---

## S16 — Hedge / basis trades

### Mechanics

Commercial hedgers lock prices (producer short hedge, consumer long hedge).
Speculators trade the **basis** (cash − futures) converging to delivery.
Educational focus: understand *why* flow exists — it creates the other side
of many speculative markets.

### Where it works

Physical commodity markets with identifiable cash markets.

### Risk notes

- Basis risk is real; hedges are imperfect.
- Delivery location and grade differentials.

### Videos & links

- CME: [Introduction to Futures](https://www.cmegroup.com/education/courses/introduction-to-futures.html) (hedgers vs speculators)
- Investopedia: [Basis](https://www.investopedia.com/terms/b/basis.asp) · [Hedging](https://www.investopedia.com/terms/h/hedge.asp)

---

## Family practice

1. Build a spread chart (front − deferred); apply S06 z-score rules to the
   spread series, not the outright.
2. Measure margin with vs without exchange spread credits.
