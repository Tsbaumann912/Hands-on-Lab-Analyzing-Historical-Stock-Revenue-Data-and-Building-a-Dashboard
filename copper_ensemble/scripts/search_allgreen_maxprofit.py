#!/usr/bin/env python3
"""Maximize total / mean yearly return subject to ALL calendar years profitable.

No hard max-DD cap — only the all-green constraint (neg==0 and min_yr>0).
Yearly profit lock stays ON (required historically for all-green).

Writes:
  /opt/cursor/artifacts/copper_allgreen_maxprofit_winner.json
  /opt/cursor/artifacts/copper_allgreen_maxprofit_search.json
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from copper_ensemble.data import dataframe_to_bars, load_yfinance_hg
from copper_ensemble.engine import BacktestEngine, calendar_year_returns
from copper_ensemble.models import clone_config, load_config

warnings.filterwarnings("ignore")
OUT = Path("/opt/cursor/artifacts")
OUT.mkdir(parents=True, exist_ok=True)

bars = dataframe_to_bars(load_yfinance_hg("HG=F", start="2008-01-01"), "HG")
base = load_config()
ts = [b.timestamp for b in bars]
print(f"bars={len(bars)}", flush=True)

W: Dict[str, Dict[str, float]] = {
    "w50": {
        "tsmom": 0.50,
        "carry": 0.0,
        "basis_mom": 0.0,
        "inventory": 0.0,
        "fade": 0.0,
        "ma_cross": 0.25,
        "stoch_rsi": 0.25,
    },
    "w55": {
        "tsmom": 0.55,
        "carry": 0.0,
        "basis_mom": 0.0,
        "inventory": 0.0,
        "fade": 0.0,
        "ma_cross": 0.25,
        "stoch_rsi": 0.20,
    },
}


def run(p: Dict[str, Any]) -> Dict[str, Any]:
    eo = {
        "vol_target_annual": float(p["vol"]),
        "kelly_fraction": 1.0,
        "buffer_forecast": 2.5,
        "agreement_min": 0.50,
        "horizons_days": [21, 63],
        "weights": dict(p["w"]),
        "take_profit_atr_mult": float(p["tp"]),
        "stop_atr_mult": float(p["stop"]),
        "fast_ma_period": 8,
        "slow_ma_period": 34,
    }
    ro = {
        "max_position_size_pct": float(p["pos"]),
        "max_contracts": int(p["mc"]),
        "max_leverage": float(max(5.0, p["pos"])),
        "max_daily_drawdown_pct": float(p["dd_halt"]),
        "yearly_profit_lock_enabled": True,
        "yearly_profit_lock_pct": float(p["lock_pct"]),
        "yearly_profit_lock_min_days": int(p["md"]),
        "yearly_nov_protect": bool(p["nov"]),
        "halt_on_breach": True,
        "halt_cooldown_bars": 21,
    }
    cfg = clone_config(base, ensemble_overrides=eo, risk_overrides=ro)
    result = BacktestEngine(cfg).run(bars)
    yearly = calendar_year_returns(result.equity_curve, ts)
    vals = np.asarray(list(yearly.values()), dtype=float)
    return {
        "mean_yr": float(vals.mean()),
        "min_yr": float(vals.min()),
        "neg": int((vals < 0).sum()),
        "tot": float(result.metrics["total_return"]),
        "maxdd": float(result.metrics["max_drawdown"]),
        "cagr": float(result.metrics.get("cagr", 0.0)),
        "sharpe": float(result.metrics.get("sharpe", 0.0)),
        "upi": float(result.metrics.get("upi", 0.0)),
        "yearly": {str(k): float(v) for k, v in yearly.items()},
    }


base_p: Dict[str, Any] = {
    "vol": 0.30,
    "pos": 6.0,
    "mc": 28,
    "dd_halt": 0.25,
    "tp": 20.0,
    "stop": 4.0,
    "lock_pct": 0.0005,
    "md": 2,
    "nov": True,
    "w": W["w50"],
    "wkey": "w50",
}
base_m = run(base_p)
print("BASE", {k: base_m[k] for k in ("tot", "mean_yr", "min_yr", "maxdd", "neg")}, flush=True)

rows: List[Dict[str, Any]] = []
t0 = time.time()
n = 0

# Compact polish around the known all-green high-vol frontier.
for vol, pos, mc, dd_halt, tp, stop, lock_pct, md, nov, wkey in itertools.product(
    [0.75, 0.80, 0.85],
    [15.0, 18.0, 20.0],
    [60, 80],
    [0.45, 0.55],
    [20.0, 22.0],
    [4.0, 5.0],
    [0.005, 0.006, 0.008],
    [1, 2],
    [True, False],
    ["w50", "w55"],
):
    n += 1
    p = {
        "vol": vol,
        "pos": pos,
        "mc": mc,
        "dd_halt": dd_halt,
        "tp": tp,
        "stop": stop,
        "lock_pct": lock_pct,
        "md": md,
        "nov": nov,
        "w": W[wkey],
        "wkey": wkey,
    }
    m = run(p)
    if m["neg"] == 0 and m["min_yr"] > 0 and m["tot"] > 0:
        rows.append({**p, **m})
    if n % 40 == 0 and rows:
        best = max(rows, key=lambda r: (r["tot"], r["mean_yr"]))
        print(
            f"n={n} ok={len(rows)} best_tot={best['tot']:.2%} mean={best['mean_yr']:.3%} "
            f"min={best['min_yr']:.3%} dd={best['maxdd']:.3%} vol={best['vol']} "
            f"pos={best['pos']} lock={best['lock_pct']} t={time.time()-t0:.0f}s",
            flush=True,
        )

assert rows, "no all-green configs found"
rows.sort(key=lambda r: (r["tot"], r["mean_yr"], r["cagr"], -r["maxdd"]), reverse=True)
winner = rows[0]
verify = run(winner)
assert verify["neg"] == 0 and verify["min_yr"] > 0
print(
    "FINAL",
    {k: verify[k] for k in ["tot", "mean_yr", "min_yr", "maxdd", "neg", "cagr"]},
    flush=True,
)
print(
    "PARAMS",
    {
        k: winner[k]
        for k in [
            "vol",
            "pos",
            "mc",
            "dd_halt",
            "tp",
            "stop",
            "lock_pct",
            "md",
            "nov",
            "wkey",
        ]
    },
    flush=True,
)
print(
    f"LIFT tot={verify['tot']-base_m['tot']:+.2%} mean={verify['mean_yr']-base_m['mean_yr']:+.3%}",
    flush=True,
)

payload = {
    "baseline": {k: base_m[k] for k in base_m if k != "yearly"},
    "constraint": "every calendar year profitable; no hard DD cap",
    "objective": "maximize total_return then mean_yr",
    "n_evaluated": n,
    "n_ok": len(rows),
    "top15": [{k: v for k, v in r.items() if k != "yearly"} for r in rows[:15]],
    "winner": {
        **{k: v for k, v in winner.items() if k != "yearly"},
        **{k: verify[k] for k in verify if k != "yearly"},
    },
    "verify": {k: verify[k] for k in verify if k != "yearly"},
    "yearly": verify["yearly"],
}
(OUT / "copper_allgreen_maxprofit_winner.json").write_text(
    json.dumps(payload["winner"], indent=2)
)
(OUT / "copper_allgreen_maxprofit_search.json").write_text(json.dumps(payload, indent=2))
print("wrote artifacts", flush=True)
