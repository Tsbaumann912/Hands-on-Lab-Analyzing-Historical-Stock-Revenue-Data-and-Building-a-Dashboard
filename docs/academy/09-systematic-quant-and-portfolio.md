# Module 09 — Systematic Quant & Multi-Strategy Portfolios (S26–S30)

**Edge hypothesis:** Robust rules + risk engineering + diversification across
uncorrelated return streams beat discretionary single-market heroics.

**Fails when:** Overfitting, ignored costs, correlated “diversification,”
operational failures.

---

## S26 — Walk-forward optimized CTA

### Mechanics

Optimize parameters on in-sample windows; trade the next out-of-sample
window; roll forward. Used for trend/momentum CTAs across futures.

### Where it works

Multi-market futures portfolios with enough history. Single-market WFO is
fragile.

### Risk notes

- Optimization bias still exists — keep search spaces small
  (`config/optuna.yaml` in this repo).
- Prefer stability of OOS equity over peak IS Sharpe.

### Videos & links

- QuantTerminal Strategy Lab: `/strategy-lab` (backtest / optimize / walk-forward)
- Investopedia: [Walk-Forward Optimization](https://www.investopedia.com/terms/w/walk-forward-optimization-wfo.asp) (concept)
- Optuna docs: [https://optuna.org](https://optuna.org)

---

## S27 — Multi-timeframe confluence systems

### Mechanics

Require agreement across horizons (e.g. weekly trend, daily pullback, hourly
trigger). Reduces trade count; improves average trade if filters are causal.

### Where it works

FX swing trading and futures position trading.

### Risk notes

- Too many filters → zero trades and curve-fit stories.
- Define timeframes in advance; do not cherry-pick after the fact.

### Videos & links

- Babypips: [Multiple Time Frame Analysis](https://www.babypips.com/learn/forex/high-school) (Grade 12)
- Babypips: [Build Your Own Trading System](https://www.babypips.com/learn/forex)

---

## S28 — Volatility targeting / risk overlays

### Mechanics

Not a directional signal: scale all positions so portfolio or sleeve vol
matches a target; cut risk when realized vol spikes; optional trailing
drawdown circuit breakers.

### Where it works

Every systematic book. Often the largest Sharpe improvement vs naive sizing.

### Risk notes

- Vol estimates lag; crashes still hurt.
- Targeting too low may overtrade costs when scaling up/down.

### Videos & links

- Investopedia: [Volatility](https://www.investopedia.com/terms/v/volatility.asp)
- Risk parity / vol targeting primers (AQR public papers — search “AQR risk parity”)
- QuantTerminal Risk console: `/risk`

---

## S29 — Regime classification systems

### Mechanics

Classify markets as trend/range/high-vol/low-vol (rules or ML); enable the
matching strategy family (trend vs mean reversion). Meta-model sitting above
S01–S11.

### Where it works

Multi-strategy desks; research platforms.

### Risk notes

- Misclassification is costly at regime transitions.
- ML without economic priors overfits easily — prefer simple features first.

### Videos & links

- Babypips: [Market Environment](https://www.babypips.com/learn/forex/high-school) (Grade 11)
- Investopedia: [Market Regimes](https://www.investopedia.com/articles/trading/09/regime-switching.asp) (related concepts)

---

## S30 — Multi-strategy portfolio construction

### Mechanics

Run several low-correlation strategies (e.g. futures trend + FX carry +
calendar carry); risk-budget each sleeve; rebalance. Measure diversification
with rolling and stress correlations — not labels alone.

### Where it works

The professional end-state of this course. One market, one setup is a job;
a portfolio of edges is a business.

### Risk notes

- “Different indicators, same beta” is fake diversification.
- Capacity and operational complexity grow with sleeve count.

### Videos & links

- QuantConnect: [Combined Carry and Trend](https://www.quantconnect.com/research/16001/combined-carry-and-trend/)
- QuantPedia: [Multi Strategy Management](https://quantpedia.com/multi-strategy-management-for-your-portfolio/)
- Babypips: [Developing Your Own Trading Plan](https://www.babypips.com/learn/forex)

---

## Capstone project

1. Select **one** trend (S01–S03), **one** mean-reversion or carry (S06 or
   S17), and **one** risk overlay (S28).
2. Backtest each sleeve separately with costs.
3. Combine with equal risk budgets; report combined max DD vs average of
   singles.
4. Write a one-page trading plan: markets, horizon, kill-switch, review cadence.

Then open QuantTerminal **Strategies** and **Risk** pages to operationalize
what you can systematize today.
