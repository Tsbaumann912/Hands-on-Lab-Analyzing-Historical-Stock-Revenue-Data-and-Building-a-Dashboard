"""Comprehensive runtime integrity checks — run directly with python3."""

import sys
import os
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

bugs = []

def ok(msg):  print(f"  ✓ {msg}")
def fail(msg):
    print(f"  ✗ {msg}")
    bugs.append(msg)

# ── 1. Indicators ─────────────────────────────────────────────────────────────
print("[1] Indicators — edge cases")
from indicators.momentum import rsi, macd
from indicators.trend import sma, ema, supertrend
from indicators.volatility import atr, bollinger_bands, keltner_channels
from indicators.volume import obv, vwap

rng = np.random.default_rng(0)
c = 4500 + np.cumsum(rng.normal(0, 5, 200))
h = c + rng.uniform(1, 5, 200)
l = c - rng.uniform(1, 5, 200)
v = rng.uniform(1000, 10000, 200)

try:
    assert rsi(c[:5], 14) is None;              ok("rsi returns None on short input")
    r = rsi(c, 14)
    assert len(r) == 200;                        ok("rsi correct output length")
    valid = r[~np.isnan(r)]
    assert valid.min() >= 0 and valid.max() <= 100; ok("rsi values in [0, 100]")
    assert np.all(np.isnan(r[:14]));             ok("rsi first 14 values are NaN (warm-up)")
    assert macd(c[:30]) is None;                 ok("macd returns None on short input")
    m = macd(c)
    assert len(m.macd_line) == 200;              ok("macd output length correct")
    mask = ~(np.isnan(m.macd_line) | np.isnan(m.signal_line))
    assert np.allclose(m.histogram[mask], (m.macd_line - m.signal_line)[mask], atol=1e-9)
    ok("macd histogram == macd_line - signal_line")
    bb = bollinger_bands(c)
    valid_bb = ~np.isnan(bb.upper)
    assert np.all(bb.upper[valid_bb] >= bb.lower[valid_bb]); ok("bollinger upper >= lower")
    a = atr(h, l, c)
    assert np.all(a[~np.isnan(a)] > 0);         ok("ATR always positive")
    s20 = sma(np.arange(1.0, 11), 5)
    assert abs(s20[-1] - 8.0) < 1e-9;           ok("SMA value correct")
    e20 = ema(c, 20)
    assert len(e20) == 200;                      ok("EMA length correct")
    o = obv(c, v)
    assert len(o) == 200;                        ok("OBV length correct")
    vw = vwap(h, l, c, v)
    assert np.all(vw[~np.isnan(vw)] > 0);       ok("VWAP positive")
    assert supertrend(h, l, c) is not None;      ok("SuperTrend computes successfully")
    assert keltner_channels(h, l, c) is not None; ok("Keltner Channels computes successfully")
except AssertionError as e:
    fail(f"Indicator assertion failed: {e}")
except Exception as e:
    fail(f"Indicator error: {e}")

# ── 2. Portfolio accounting ───────────────────────────────────────────────────
print("\n[2] Portfolio — accounting integrity")
from core.config import Config
from core.enums import Direction
from core.models import Fill
from portfolio.portfolio import Portfolio

def _fill(d, qty, price, comm=0.0, slip=0.0):
    return Fill(order_id="t", symbol="ES.c.0", direction=d,
                filled_quantity=qty, fill_price=price,
                commission=comm, slippage=slip,
                timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc))

cfg = Config()
initial = cfg.portfolio.initial_cash

# Open: no notional deduction
p = Portfolio(cfg)
p.process_fill(_fill(Direction.LONG, 1, 4500.0))
if abs(p.cash - initial) < 0.01: ok("Open position: no notional deducted (margin model)")
else: fail(f"Open drained cash: expected {initial}, got {p.cash}")

# MTM gain
p.mark_to_market({"ES.c.0": 4600.0}, datetime(2023, 1, 2, tzinfo=timezone.utc))
if abs(p.unrealised_pnl - 5000.0) < 0.01: ok("MTM long gain = (4600-4500)*1*50 = 5000")
else: fail(f"MTM gain wrong: {p.unrealised_pnl}")

if p.current_drawdown == 0.0: ok("Zero drawdown at new peak")
else: fail(f"Non-zero drawdown at peak: {p.current_drawdown}")

# Close
p.process_fill(_fill(Direction.SHORT, 1, 4600.0))
if abs(p.cash - (initial + 5000.0)) < 0.01: ok("Close: realised PnL credited to cash")
else: fail(f"Cash after close: expected {initial+5000}, got {p.cash}")
if len(p.open_positions) == 0: ok("Position cleared after full close")
else: fail("Position not cleared after close")

# Round-trip with commission
p2 = Portfolio(cfg)
p2.process_fill(_fill(Direction.LONG, 1, 4500.0, comm=2.25))
p2.process_fill(_fill(Direction.SHORT, 1, 4550.0, comm=2.25))
expected2 = initial + (4550 - 4500) * 50 - 2 * 2.25
if abs(p2.total_equity - expected2) < 0.01:
    ok(f"Round-trip equity correct: ${p2.total_equity:,.2f}")
else:
    fail(f"Round-trip equity: expected {expected2:.2f}, got {p2.total_equity:.2f}")

# Short round-trip
p3 = Portfolio(cfg)
p3.process_fill(_fill(Direction.SHORT, 1, 4500.0))
p3.process_fill(_fill(Direction.LONG, 1, 4400.0))
expected3 = initial + (4500 - 4400) * 50
if abs(p3.total_equity - expected3) < 0.01:
    ok(f"Short round-trip equity: ${p3.total_equity:,.2f}")
else:
    fail(f"Short round-trip: expected {expected3}, got {p3.total_equity}")

# Partial close
p4 = Portfolio(cfg)
p4.process_fill(_fill(Direction.LONG, 3, 4500.0))
p4.process_fill(_fill(Direction.SHORT, 2, 4510.0))
assert len(p4.open_positions) == 1
rem_qty = abs(p4.open_positions["ES.c.0"].quantity)
if abs(rem_qty - 1.0) < 0.01: ok("Partial close: 1 contract remains")
else: fail(f"Partial close remnant qty: expected 1, got {rem_qty}")

# ── 3. RiskManager ───────────────────────────────────────────────────────────
print("\n[3] RiskManager — circuit breakers")
from risk.risk_manager import RiskManager, ViolationType
from core.models import Signal

def _sig(d=Direction.LONG):
    return Signal("ES.c.0", d, 0.7, datetime(2023, 1, 1, tzinfo=timezone.utc),
                  "test", metadata={"price": 4500.0})

p5 = Portfolio(cfg)
rm = RiskManager(cfg, p5)
dec = rm.evaluate(_sig())
if dec.approved: ok("Normal signal approved")
else: fail(f"Normal signal rejected: {dec.violations}")

# Drawdown breach
p5._peak_equity = 100_000.0
p5._cash = 90_000.0
dec2 = rm.evaluate(_sig())
if not dec2.approved and dec2.violations[0].violation_type == ViolationType.DRAWDOWN_BREACH:
    ok("Drawdown circuit breaker fires at >3% DD")
else:
    fail(f"Drawdown CB didn't fire. approved={dec2.approved}")

# Halt persists
dec3 = rm.evaluate(_sig())
if not dec3.approved and dec3.violations[0].violation_type == ViolationType.TRADING_HALTED:
    ok("Halt persists across subsequent signals")
else:
    fail("Halt didn't persist")

# Resume
rm.resume_trading()
p5._cash = 100_000.0; p5._peak_equity = 100_000.0
dec4 = rm.evaluate(_sig())
if dec4.approved: ok("Resume clears halt flag")
else: fail("Resume didn't clear halt")

# FLAT always approved
dec5 = rm.evaluate(_sig(Direction.FLAT))
if dec5.approved and dec5.suggested_quantity == 0.0:
    ok("FLAT signal always approved with qty=0")
else:
    fail(f"FLAT rejected or qty={dec5.suggested_quantity}")

# Stop/TP defaults applied
rm2 = RiskManager(cfg, Portfolio(cfg))
rm2.update_atr("ES.c.0", 20.0)
dec6 = rm2.evaluate(_sig())
if dec6.adjusted_signal.stop_loss is not None and dec6.adjusted_signal.take_profit is not None:
    ok("Default ATR stop/TP applied when missing from signal")
else:
    fail("Default stops not applied")

# ── 4. BacktestEngine ────────────────────────────────────────────────────────
print("\n[4] BacktestEngine — end-to-end smoke test")
from engine.backtest import BacktestEngine
from engine.metrics import compute_metrics
from strategies.mean_reversion import MeanReversionRSI
from strategies.momentum import MomentumBreakout
from strategies.trend_following import TrendFollowingMACD
from core.models import Bar
from core.enums import AssetClass

def _make_bars(n=300, symbol="ES.c.0", seed=42):
    rng2 = np.random.default_rng(seed)
    prices = 5200 + np.cumsum(rng2.normal(0, 5, n))
    base = datetime(2023, 1, 2, 9, 30, tzinfo=timezone.utc)
    return [
        Bar(symbol, base + timedelta(minutes=i),
            float(p), float(p + 3), float(p - 3), float(p),
            float(rng2.uniform(500, 5000)), AssetClass.FUTURES)
        for i, p in enumerate(prices)
    ]

for strat_cls in [MeanReversionRSI, MomentumBreakout, TrendFollowingMACD]:
    name = strat_cls.__name__
    try:
        engine = BacktestEngine(cfg, strat_cls)
        result = engine.run({"ES.c.0": _make_bars(300)})
        assert len(result.equity_curve) > 0
        assert "sharpe_ratio" in result.metrics
        ok(f"{name}: {len(result.equity_curve)} equity pts, {len(result.fills)} fills, Sharpe={result.metrics['sharpe_ratio']:.3f}")
    except Exception as e:
        fail(f"{name} backtest error: {e}")

# Metrics integrity
eq = np.linspace(100_000, 130_000, 252)
m = compute_metrics(eq)
assert m["total_return"] > 0 and m["sharpe_ratio"] > 0 and m["max_drawdown"] <= 0
ok("compute_metrics: positive return series yields positive Sharpe")

# ── 5. Data service ───────────────────────────────────────────────────────────
print("\n[5] Data service — integrity")
from app.data_service import get_synthetic_futures_bars, compute_indicators_for_ui, run_backtest_for_ui

for sym, start in [("ES", 5200), ("NQ", 18000), ("CL", 78), ("GC", 2300)]:
    df = get_synthetic_futures_bars(sym, n=300)
    assert len(df) == 300
    assert (df["High"] >= df["Close"]).all(), f"{sym}: High < Close"
    assert (df["Low"] <= df["Close"]).all(), f"{sym}: Low > Close"
    ok(f"get_synthetic_futures_bars({sym}): OHLCV constraints hold")

inds = compute_indicators_for_ui(get_synthetic_futures_bars("ES", 300))
for k in ["RSI_14", "MACD_line", "BB_upper", "ATR_14", "SMA_20", "OBV", "VWAP"]:
    assert k in inds, f"Missing: {k}"
    assert len(inds[k]) == 300, f"{k} wrong length"
ok(f"compute_indicators_for_ui: all indicators present, lengths correct")

bt = run_backtest_for_ui("ES", "MeanReversionRSI", n_bars=200)
assert len(bt["equity_curve"]) > 0
assert "sharpe_ratio" in bt["metrics"]
ok(f"run_backtest_for_ui: returns {len(bt['equity_curve'])} equity pts")

# ── 6. App pages render ───────────────────────────────────────────────────────
print("\n[6] App pages — layout render")
import app.theme  # noqa
from app.pages import dashboard, stock_research, futures_terminal, strategy_lab, risk_console, indicator_explorer

for name, mod in [
    ("dashboard", dashboard),
    ("stock_research", stock_research),
    ("futures_terminal", futures_terminal),
    ("strategy_lab", strategy_lab),
    ("risk_console", risk_console),
    ("indicator_explorer", indicator_explorer),
]:
    try:
        mod.layout()
        ok(f"{name} renders without error")
    except Exception as e:
        fail(f"{name} render error: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 55)
if bugs:
    print(f"BUGS FOUND ({len(bugs)}):")
    for b in bugs:
        print(f"  ✗ {b}")
else:
    print("ALL RUNTIME CHECKS PASSED — zero bugs found.")
print("=" * 55)
