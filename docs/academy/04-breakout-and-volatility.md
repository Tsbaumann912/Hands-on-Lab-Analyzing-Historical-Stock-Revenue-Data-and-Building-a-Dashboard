# Module 04 — Breakouts & Volatility Expansion (S09–S11)

**Edge hypothesis:** When price escapes a compression zone with rising
participation, the next move tends to continue (information arrival /
stop cascades).

**Fails when:** Fakeouts, low-volume breaks, news fades, exhausted trends.

---

## S09 — Opening-range breakout (ORB)

### Mechanics

Define the high/low of the first N minutes after the cash or futures open.
Buy break of range high / sell break of low; stop at opposite side or mid.
Time stop by end of session.

### Where it works

Index and energy futures around RTH open; FX around London open. Session
definition is critical.

### Risk notes

- First break often fails — some systems wait for retest.
- Avoid ORB on major scheduled releases inside the range window.

### Videos & links

- Babypips: [Trading Breakouts and Fakeouts](https://www.babypips.com/learn/forex/high-school) (Grade 10)
- Investopedia: [Opening Range](https://www.investopedia.com/terms/o/openingrange.asp)
- CME Group YouTube: search “day trading futures” on [@cmegroup](https://www.youtube.com/@cmegroup)

---

## S10 — ATR / volatility breakout

### Mechanics

Entry when price moves more than *k × ATR* from a reference (prior close,
N-day high). Position size inverse to ATR. Classic “volatility channel”
systems (e.g. chandelier exits).

### Where it works

Diversified futures; adapts to each market’s noise level better than fixed
point stops.

### Risk notes

- ATR expands *after* big moves — lag can widen risk.
- Parameter *k* and lookback need robustness checks.

### Videos & links

- Investopedia: [Average True Range](https://www.investopedia.com/terms/a/atr.asp)
- Investopedia: [Chandelier Exit](https://www.investopedia.com/articles/technical/03/031103.asp)

---

## S11 — Squeeze release (Bollinger + Keltner)

### Mechanics

When Bollinger Bands contract inside Keltner Channels (squeeze), wait for
bands to expand outside Keltner; trade in the direction of the break with
momentum confirmation (e.g. momentum oscillator).

### Where it works

Equities, index futures, and FX pairs that alternate quiet/active regimes.

### Risk notes

- Direction of release can be wrong — use a directional filter.
- Multiple false expansions in news weeks.

### Videos & links

- Babypips: [Bollinger Bands](https://www.babypips.com/learn/forex/how-to-use-bollinger-bands) · [Keltner Channels](https://www.babypips.com/learn/forex/how-to-use-keltner-channels)
- Investopedia: [Keltner Channel](https://www.investopedia.com/terms/k/keltnerchannel.asp)
- Search YouTube: “TTM Squeeze explained” (verify educator quality; prefer exchange/educator channels)

---

## Family practice

1. Log every ORB trade’s volume vs 20-day average volume at break.
2. Compare fixed-point stops vs ATR stops on the same breakout rules.
