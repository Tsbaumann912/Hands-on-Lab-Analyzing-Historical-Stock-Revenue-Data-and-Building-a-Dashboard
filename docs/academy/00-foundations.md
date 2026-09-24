# Module 00 — Market Foundations

Before strategies, understand the playground. Futures and forex share leverage
and continuous trading, but their **contract design, funding, and risk** differ.

## Learning objectives

- Contrast exchange-traded futures with OTC spot FX / forwards / NDFs
- Explain margin, mark-to-market, and notional exposure
- Account for futures **rolls** and FX **swap points / carry**
- Map major sessions (Asia, London, New York) and liquidity windows

## Futures essentials

A futures contract is a standardized, exchange-cleared agreement to buy/sell an
underlying (index, rate, commodity, FX) at a future date for a price agreed
today. P&L is marked to market daily against the clearinghouse.

Key ideas:

- **Multiplier** — dollar value per point (e.g. ES ≈ $50/point). Equity risk is
  *price × multiplier × contracts*, not price alone.
- **Initial / maintenance margin** — performance bond, not a down payment.
- **Settlement** — cash or physical; know the product specs.
- **Continuous contracts** — historical series stitch successive expiries;
  roll methods (last trade, volume, open interest) change backtest results.

### Video — futures basics

- CME Institute: [Introduction to Futures](https://www.cmegroup.com/education/courses/introduction-to-futures.html)
- CME video course overview: [Introduction to Futures Video Course](https://news.cqg.com/blogs/2022/02/introduction-futures-video-course)
- CME Group YouTube channel (search “Introduction to Futures”): [youtube.com/@cmegroup](https://www.youtube.com/@cmegroup)

### Web resources

- [CME Education hub](https://www.cmegroup.com/education.html)
- [Things to Know Before Trading CME Futures](https://www.cmegroup.com/education/courses/things-to-know-before-trading-cme-futures.html)
- Investopedia: [Futures Contract Definition](https://www.investopedia.com/terms/f/futurescontract.asp)

## Forex essentials

Spot FX is typically traded OTC via brokers as a currency pair (base/quote).
Profit comes from appreciation of the long currency versus the short currency,
plus overnight **swap / rollover** interest differential.

Key ideas:

- **Pip / pipette** — smallest quoted move (varies by pair).
- **Lot size** — standard, mini, micro; position size sets pip value.
- **Leverage** — broker-dependent; retail FX often offers high gearing.
- **Sessions** — Tokyo, London, New York overlaps drive volatility.

### Video — forex basics

- Babypips School of Pipsology (start here): [Learn Forex](https://www.babypips.com/learn/forex)
- Babypips Preschool / Kindergarten modules (embedded lessons + quizzes)

### Web resources

- [Babypips — What is Forex?](https://www.babypips.com/learn/forex)
- Investopedia: [Forex Trading](https://www.investopedia.com/terms/f/forex.asp)
- BIS Triennial Survey (market size / structure): [bis.org](https://www.bis.org/statistics/rpfx23.htm)

## Futures FX vs spot FX

| Feature | FX futures (e.g. 6E) | Spot FX |
|---------|----------------------|---------|
| Venue | Exchange / cleared | OTC broker |
| Counterparty | Clearinghouse | Broker |
| Expiry | Yes — must roll | Continuous |
| Transparency | Central limit order book | Variable |
| Carry | Embedded in futures curve | Explicit swap |

Many systematic FX strategies are implemented on **currency futures** for
cleaner data and clearing.

## Checkpoint

1. Why can two continuous futures series of the same root ticker yield
   different backtest Sharpe ratios?
2. If EURUSD rises 50 pips and you are long 1 standard lot, what roughly
   happens to P&L in USD account terms?
3. Name one risk unique to physical-delivery commodity futures that equity
   index futures do not share.
