#!/usr/bin/env python3
"""Compact max-profit search under max DD < 30% (HG 2008→now)."""
from __future__ import annotations

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

W = {
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
    "w60": {
        "tsmom": 0.60,
        "carry": 0.0,
        "basis_mom": 0.0,
        "inventory": 0.0,
        "fade": 0.0,
        "ma_cross": 0.25,
        "stoch_rsi": 0.15,
    },
}


def run(p: Dict[str, Any]) -> Dict[str, Any]:
    eo = {
        "vol_target_annual": float(p["vol"]),
        "kelly_fraction": float(p["k"]),
        "buffer_forecast": float(p["buf"]),
        "agreement_min": float(p["agr"]),
        "horizons_days": list(p["h"]),
        "weights": dict(p["w"]),
        "take_profit_atr_mult": float(p["tp"]),
        "stop_atr_mult": float(p["stop"]),
        "fast_ma_period": int(p["fast"]),
        "slow_ma_period": int(p["slow"]),
    }
    ro = {
        "max_position_size_pct": float(p["pos"]),
        "max_contracts": int(p["mc"]),
        "max_leverage": float(max(5.0, p["pos"])),
        "max_daily_drawdown_pct": float(p["dd_halt"]),
        "yearly_profit_lock_enabled": bool(p["lock"]),
        "yearly_profit_lock_pct": float(p["lock_pct"]),
        "yearly_profit_lock_min_days": int(p["md"]),
        "yearly_nov_protect": bool(p["nov"]),
        "halt_on_breach": True,
        "halt_cooldown_bars": int(p["cd"]),
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


r0 = BacktestEngine(clone_config(base)).run(bars)
yr0 = calendar_year_returns(r0.equity_curve, ts)
v0 = np.asarray(list(yr0.values()), float)
base_m = {
    "mean_yr": float(v0.mean()),
    "min_yr": float(v0.min()),
    "neg": int((v0 < 0).sum()),
    "tot": float(r0.metrics["total_return"]),
    "maxdd": float(r0.metrics["max_drawdown"]),
    "cagr": float(r0.metrics.get("cagr", 0.0)),
}
print("BASE", base_m, flush=True)

rows: List[Dict[str, Any]] = []
t0 = time.time()
n = 0

# ~7*4*2*3*3*2*3 = 3024 → too many; keep ~504
grid: List[Dict[str, Any]] = []
for vol in [0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34]:
    for pos in [5.5, 6.0, 6.5, 7.0]:
        for mc in [24, 28]:
            for dd_halt in [0.22, 0.25, 0.28]:
                for tp in [20.0, 25.0, 30.0]:
                    for lock_pct in [0.0005, 0.001]:
                        for wkey in ["w50", "w55", "w60"]:
                            grid.append(
                                {
                                    "vol": vol,
                                    "pos": pos,
                                    "mc": mc,
                                    "dd_halt": dd_halt,
                                    "tp": tp,
                                    "stop": 4.0,
                                    "lock_pct": lock_pct,
                                    "wkey": wkey,
                                    "agr": 0.50,
                                    "buf": 2.5,
                                    "k": 1.0,
                                    "h": [21, 63],
                                    "w": W[wkey],
                                    "fast": 8,
                                    "slow": 34,
                                    "lock": True,
                                    "md": 2,
                                    "nov": True,
                                    "cd": 21,
                                }
                            )

print(f"grid={len(grid)}", flush=True)
for p in grid:
    n += 1
    m = run(p)
    if m["maxdd"] < 0.30 and m["tot"] > 0:
        rows.append({**p, **m})
    if n % 40 == 0:
        if rows:
            best = max(rows, key=lambda r: (r["tot"], r["mean_yr"]))
            print(
                f"n={n} ok={len(rows)} best_tot={best['tot']:.2%} "
                f"mean={best['mean_yr']:.3%} dd={best['maxdd']:.3%} "
                f"vol={best['vol']} pos={best['pos']} tp={best['tp']} "
                f"{best['wkey']} t={time.time()-t0:.0f}s",
                flush=True,
            )
        else:
            print(f"n={n} ok=0 t={time.time()-t0:.0f}s", flush=True)

rows.sort(key=lambda r: (r["tot"], r["mean_yr"], r["cagr"], -r["maxdd"]), reverse=True)
seeds = rows[:5]
seen = set()
for s in seeds:
    for vol in [s["vol"] - 0.01, s["vol"] - 0.005, s["vol"], s["vol"] + 0.005, s["vol"] + 0.01]:
        if not (0.26 <= vol <= 0.36):
            continue
        for pos in [s["pos"] - 0.5, s["pos"] - 0.25, s["pos"], s["pos"] + 0.25, s["pos"] + 0.5]:
            if not (4.5 <= pos <= 8.5):
                continue
            for mc in [max(16, s["mc"] - 4), s["mc"], s["mc"] + 4]:
                for dd_halt in [0.24, 0.25, 0.26, 0.27, 0.28]:
                    for tp in [s["tp"] - 5, s["tp"], s["tp"] + 5]:
                        if tp < 15:
                            continue
                        for stop in [3.5, 4.0, 4.5]:
                            for lock_pct in [0.0005, 0.001, 0.0015]:
                                for wkey in ["w50", "w55", "w60"]:
                                    key = (
                                        round(vol, 4),
                                        round(pos, 3),
                                        mc,
                                        dd_halt,
                                        tp,
                                        stop,
                                        lock_pct,
                                        wkey,
                                    )
                                    if key in seen:
                                        continue
                                    if hash(key) % 3 != 0:
                                        continue
                                    seen.add(key)
                                    n += 1
                                    p = {
                                        "vol": float(vol),
                                        "pos": float(pos),
                                        "mc": int(mc),
                                        "dd_halt": float(dd_halt),
                                        "tp": float(tp),
                                        "stop": float(stop),
                                        "lock_pct": float(lock_pct),
                                        "wkey": wkey,
                                        "agr": 0.50,
                                        "buf": 2.5,
                                        "k": 1.0,
                                        "h": [21, 63],
                                        "w": W[wkey],
                                        "fast": 8,
                                        "slow": 34,
                                        "lock": True,
                                        "md": 2,
                                        "nov": True,
                                        "cd": 21,
                                    }
                                    m = run(p)
                                    if m["maxdd"] < 0.30 and m["tot"] > 0:
                                        rows.append({**p, **m})
                                    if n % 40 == 0 and rows:
                                        best = max(rows, key=lambda r: (r["tot"], r["mean_yr"]))
                                        print(
                                            f"n={n} ok={len(rows)} best_tot={best['tot']:.2%} "
                                            f"mean={best['mean_yr']:.3%} dd={best['maxdd']:.3%} "
                                            f"vol={best['vol']} pos={best['pos']} "
                                            f"t={time.time()-t0:.0f}s",
                                            flush=True,
                                        )

rows.sort(key=lambda r: (r["tot"], r["mean_yr"], r["cagr"], -r["maxdd"]), reverse=True)
print(f"DONE n={n} ok={len(rows)} t={time.time()-t0:.1f}s", flush=True)
print("TOP12:", flush=True)
for r in rows[:12]:
    print(
        f" tot={r['tot']:.2%} mean={r['mean_yr']:.3%} dd={r['maxdd']:.3%} neg={r['neg']} "
        f"vol={r['vol']} pos={r['pos']} mc={r['mc']} halt={r['dd_halt']} "
        f"tp={r['tp']} stop={r['stop']} lock={r['lock_pct']} {r['wkey']}",
        flush=True,
    )

if not rows:
    raise SystemExit("no feasible configs")

winner = rows[0]
verify = run(winner)
assert verify["maxdd"] < 0.30
assert abs(verify["tot"] - winner["tot"]) < 1e-9
print("VERIFY", {k: verify[k] for k in ["tot", "mean_yr", "maxdd", "cagr", "neg", "min_yr"]}, flush=True)
print("WEIGHTS", winner["w"], flush=True)
print(
    f"LIFT tot {verify['tot']-base_m['tot']:+.2%} mean {verify['mean_yr']-base_m['mean_yr']:+.3%}",
    flush=True,
)

payload = {
    "baseline": base_m,
    "n_evaluated": n,
    "n_ok": len(rows),
    "top15": [{k: v for k, v in r.items() if k != "yearly"} for r in rows[:15]],
    "winner": {k: v for k, v in {**winner, **verify}.items() if k != "yearly"},
    "verify": {k: verify[k] for k in verify if k != "yearly"},
    "yearly": verify["yearly"],
}
(OUT / "copper_maxprofit_dd30_winner.json").write_text(
    json.dumps({**{k: v for k, v in winner.items() if k != "yearly"}, **verify}, indent=2)
)
(OUT / "copper_maxprofit_dd30_search.json").write_text(json.dumps(payload, indent=2))
print("wrote artifacts", flush=True)
