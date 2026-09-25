# Module 01 — Risk, Costs & Execution

Strategies fail more often from **position sizing and costs** than from a
wrong indicator. Treat risk as the first strategy.

## Learning objectives

- Size positions from volatility and account equity, not gut feel
- Include commission, exchange fees, slippage, and financing in expectancy
- Define max drawdown, daily loss limits, and kill-switches
- Distinguish research expectancy from live execution quality

## Core risk toolkit

### 1. Risk per trade

Common rule of thumb (not gospel): risk a small fixed fraction of equity
(e.g. 0.25–1%) to the stop. Futures: convert stop distance in points ×
multiplier × contracts into dollars.

### 2. Volatility targeting

Scale contracts so each position targets similar daily variance
(`contracts ∝ target_vol / (σ × multiplier)`). CTA programs almost always
volatility-scale.

### 3. Correlation & concentration

Five “different” FX pairs can be one USD risk bet. Aggregate by factor
(USD, rates, energy, risk-on) not by ticker count.

### 4. Transaction costs (mandatory in futures research)

Expectancy after costs:

`E = win_rate × avg_win − loss_rate × avg_loss − costs_per_round_trip`

Costs stack: commission + exchange/clearing + half-spread slippage +
roll/financing.

## Psychology & process

- Pre-commit rules; journal every discretionary override
- Separate research mode from live mode
- Avoid revenge trading after a stop-out
- Prefer fewer robust edges over many fragile ones

## Video resources

- CME: [Introduction to Risk Management](https://www.cmegroup.com/education.html) (search CME Institute risk courses)
- Babypips Graduate modules: [Risk Management](https://www.babypips.com/learn/forex) (Risk Management section)
- Van Tharp–style position sizing concepts (overview articles): [Investopedia — Position Sizing](https://www.investopedia.com/articles/trading/09/determine-position-size.asp)

## Web resources

- Investopedia: [Risk Management](https://www.investopedia.com/terms/r/riskmanagement.asp)
- CME: [Learn to trade futures](https://www.cmegroup.com/trading/why-futures/learn-to-trade-futures.html)
- QuantTerminal Risk page in-app: `/risk`

## Checkpoint

1. Two strategies both make +2R average when they win. Strategy A wins 40%
   with 2× wider stops than B. Which needs larger sample size to estimate
   expectancy reliably?
2. List four cost components you must model for CME ES round-trips.
3. Why can high win-rate mean-reversion systems still have poor Sharpe after
   costs?
