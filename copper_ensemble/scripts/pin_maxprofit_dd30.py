#!/usr/bin/env python3
from __future__ import annotations

import json
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
print("bars", len(bars), flush=True)

W50 = {
    "tsmom": 0.50,
    "carry": 0.0,
    "basis_mom": 0.0,
    "inventory": 0.0,
    "fade": 0.0,
    "ma_cross": 0.25,
    "stoch_rsi": 0.25,
}
W55 = {
    "tsmom": 0.55,
    "carry": 0.0,
    "basis_mom": 0.0,
    "inventory": 0.0,
    "fade": 0.0,
    "ma_cross": 0.25,
    "stoch_rsi": 0.20,
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
    r = BacktestEngine(cfg).run(bars)
    yr = calendar_year_returns(r.equity_curve, ts)
    v = np.asarray(list(yr.values()), float)
    return {
        "mean_yr": float(v.mean()),
        "min_yr": float(v.min()),
        "neg": int((v < 0).sum()),
        "tot": float(r.metrics["total_return"]),
        "maxdd": float(r.metrics["max_drawdown"]),
        "cagr": float(r.metrics.get("cagr", 0.0)),
        "sharpe": float(r.metrics.get("sharpe", 0.0)),
        "upi": float(r.metrics.get("upi", 0.0)),
        "yearly": {str(k): float(x) for k, x in yr.items()},
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

cands: List[Dict[str, Any]] = []
for vol in [0.29, 0.30, 0.31, 0.32]:
    for pos in [5.5, 6.0, 6.5, 7.0]:
        for mc in [24, 28, 32]:
            for dd_halt in [0.22, 0.25, 0.28]:
                for tp in [20.0, 25.0, 30.0]:
                    for lock_pct in [0.0005, 0.001, 0.002]:
                        for w, wkey in [(W50, "w50"), (W55, "w55")]:
                            cands.append(
                                dict(
                                    vol=vol,
                                    pos=pos,
                                    mc=mc,
                                    dd_halt=dd_halt,
                                    tp=tp,
                                    stop=4.0,
                                    lock_pct=lock_pct,
                                    w=w,
                                    wkey=wkey,
                                    agr=0.5,
                                    buf=2.5,
                                    k=1.0,
                                    h=[21, 63],
                                    fast=8,
                                    slow=34,
                                    lock=True,
                                    md=2,
                                    nov=True,
                                    cd=21,
                                )
                            )

print("n_cands", len(cands), flush=True)
rows: List[Dict[str, Any]] = []
for i, p in enumerate(cands, 1):
    m = run(p)
    if m["maxdd"] < 0.30 and m["tot"] > 0:
        rows.append({**p, **m})
    if i % 50 == 0:
        if rows:
            b = max(rows, key=lambda r: (r["tot"], r["mean_yr"]))
            print(
                f"n={i} ok={len(rows)} best={b['tot']:.2%} mean={b['mean_yr']:.3%} "
                f"dd={b['maxdd']:.3%} vol={b['vol']} pos={b['pos']} mc={b['mc']} "
                f"halt={b['dd_halt']} tp={b['tp']} lock={b['lock_pct']} {b['wkey']}",
                flush=True,
            )
        else:
            print(f"n={i} ok=0", flush=True)

rows.sort(key=lambda r: (r["tot"], r["mean_yr"], r["cagr"], -r["maxdd"]), reverse=True)
print("TOP15:", flush=True)
for r in rows[:15]:
    print(
        f" tot={r['tot']:.2%} mean={r['mean_yr']:.3%} dd={r['maxdd']:.3%} neg={r['neg']} "
        f"vol={r['vol']} pos={r['pos']} mc={r['mc']} halt={r['dd_halt']} tp={r['tp']} "
        f"stop={r['stop']} lock={r['lock_pct']} {r['wkey']}",
        flush=True,
    )

winner = rows[0]
verify = run(winner)
assert verify["maxdd"] < 0.30
assert abs(verify["tot"] - winner["tot"]) < 1e-9
print("VERIFY", {k: verify[k] for k in ["tot", "mean_yr", "maxdd", "cagr", "neg", "min_yr"]}, flush=True)
print("WEIGHTS", winner["w"], flush=True)
print(
    f"LIFT tot={verify['tot']-base_m['tot']:+.2%} mean={verify['mean_yr']-base_m['mean_yr']:+.3%}",
    flush=True,
)


def clean(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k == "w":
            out[k] = dict(v)
        elif k == "h":
            out[k] = list(v)
        elif k == "yearly":
            out[k] = dict(v)
        else:
            out[k] = v
    return out


payload = {
    "baseline": base_m,
    "n_evaluated": len(cands),
    "n_ok": len(rows),
    "top15": [clean({k: v for k, v in r.items() if k != "yearly"}) for r in rows[:15]],
    "winner": clean({**{k: v for k, v in winner.items() if k != "yearly"}, **verify}),
    "verify": {k: verify[k] for k in verify if k != "yearly"},
    "yearly": verify["yearly"],
    "constraint": "max_drawdown < 0.30",
    "objective": "maximize total_return then mean_yr",
}
(OUT / "copper_maxprofit_dd30_winner.json").write_text(json.dumps(payload["winner"], indent=2))
(OUT / "copper_maxprofit_dd30_search.json").write_text(json.dumps(payload, indent=2))
print("wrote artifacts", flush=True)
