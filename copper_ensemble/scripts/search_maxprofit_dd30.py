#!/usr/bin/env python3
"""Maximize total / mean yearly return subject to max DD < 30% (HG 2008→now)."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

W_MENU: Dict[str, Dict[str, float]] = {
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
H_MENU: Dict[str, List[int]] = {
    "h21": [21, 63],
    "h21126": [21, 63, 126],
}


def evaluate(eo: Dict[str, Any], ro: Dict[str, Any]) -> Dict[str, Any]:
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


def pack(p: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        "yearly_profit_lock_pct": float(p.get("lock_pct", 0.001)),
        "yearly_profit_lock_min_days": int(p.get("md", 2)),
        "yearly_nov_protect": bool(p.get("nov", True)),
        "halt_on_breach": True,
        "halt_cooldown_bars": int(p.get("cd", 21)),
    }
    return eo, ro


def run_params(p: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate(*pack(p))


base_m = evaluate({}, {})
print("BASE", {k: base_m[k] for k in ("mean_yr", "tot", "maxdd", "neg", "cagr")}, flush=True)

seed = {
    "vol": 0.32,
    "k": 1.0,
    "buf": 2.5,
    "agr": 0.50,
    "h": [21, 63],
    "w": W_MENU["w55"],
    "tp": 25.0,
    "stop": 4.0,
    "fast": 8,
    "slow": 34,
    "pos": 7.0,
    "mc": 28,
    "dd_halt": 0.25,
    "lock": True,
    "lock_pct": 0.001,
    "md": 2,
    "nov": True,
    "cd": 21,
    "wkey": "w55",
    "hkey": "h21",
}
seed_m = run_params(seed)
print("SEED", {k: seed_m[k] for k in ("mean_yr", "tot", "maxdd", "neg", "cagr")}, flush=True)

rows: List[Dict[str, Any]] = []
t0 = time.time()
n = 0
seen = set()


def consider(p: Dict[str, Any], m: Dict[str, Any]) -> None:
    if m["maxdd"] < 0.30 and m["tot"] > 0:
        rows.append({**p, **m})


def eval_once(p: Dict[str, Any]) -> None:
    global n
    key = (
        round(p["vol"], 4),
        round(p["pos"], 3),
        int(p["mc"]),
        round(p["dd_halt"], 4),
        round(p["tp"], 2),
        round(p["stop"], 2),
        round(float(p.get("lock_pct", 0.001)), 5),
        bool(p["lock"]),
        int(p.get("md", 2)),
        p.get("wkey"),
        p.get("hkey"),
        round(float(p.get("k", 1.0)), 3),
        round(float(p.get("buf", 2.5)), 2),
        int(p.get("cd", 21)),
    )
    if key in seen:
        return
    seen.add(key)
    n += 1
    m = run_params(p)
    consider(p, m)
    if n % 25 == 0:
        if rows:
            best = max(rows, key=lambda r: (r["tot"], r["mean_yr"], r["cagr"]))
            print(
                f"n={n} ok={len(rows)} best_tot={best['tot']:.2%} "
                f"mean={best['mean_yr']:.3%} dd={best['maxdd']:.3%} "
                f"vol={best['vol']} pos={best['pos']} t={time.time()-t0:.0f}s",
                flush=True,
            )
        else:
            print(f"n={n} ok=0 t={time.time()-t0:.0f}s", flush=True)


# Phase 1: compact locked neighborhood around known all-green frontier
# 8*5*3*4*3*2*3*3*2 = 25920 too big → keep core dims only: ~8*5*3*3*2*3*2 = 4320 → still big
# Target ~400 evals.
for vol in [0.28, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35]:
    for pos in [5.5, 6.0, 6.5, 7.0, 7.5]:
        for mc in [24, 28, 32]:
            for dd_halt in [0.20, 0.22, 0.25, 0.28]:
                for tp in [20.0, 25.0, 30.0]:
                    for stop in [4.0]:
                        for lock_pct in [0.0005, 0.001, 0.002]:
                            for wkey in ["w50", "w55", "w60"]:
                                for hkey in ["h21", "h21126"]:
                                    key = (vol, pos, mc, dd_halt, tp, lock_pct, wkey, hkey)
                                    if hash(key) % 7 != 0:
                                        continue
                                    eval_once(
                                        {
                                            "vol": vol,
                                            "pos": pos,
                                            "mc": mc,
                                            "dd_halt": dd_halt,
                                            "tp": tp,
                                            "stop": stop,
                                            "agr": 0.50,
                                            "buf": 2.5,
                                            "k": 1.0,
                                            "h": H_MENU[hkey],
                                            "w": W_MENU[wkey],
                                            "fast": 8,
                                            "slow": 34,
                                            "lock": True,
                                            "lock_pct": lock_pct,
                                            "md": 2,
                                            "nov": True,
                                            "cd": 21,
                                            "wkey": wkey,
                                            "hkey": hkey,
                                        }
                                    )

print(f"PHASE1 done n={n} ok={len(rows)}", flush=True)
eval_once(seed)

# Unlock / soft-lock screen
for lock, lock_pct, md, vol, pos, dd_halt, tp in [
    (False, 0.2, 1, 0.12, 2.0, 0.15, 25.0),
    (False, 0.2, 1, 0.15, 2.5, 0.18, 25.0),
    (False, 0.2, 1, 0.18, 3.0, 0.18, 20.0),
    (False, 0.2, 1, 0.20, 3.5, 0.20, 25.0),
    (True, 0.05, 5, 0.28, 5.0, 0.22, 25.0),
    (True, 0.03, 3, 0.32, 6.0, 0.25, 25.0),
    (True, 0.01, 2, 0.34, 7.0, 0.25, 25.0),
]:
    eval_once(
        {
            "vol": vol,
            "pos": pos,
            "mc": 28,
            "dd_halt": dd_halt,
            "tp": tp,
            "stop": 4.0,
            "agr": 0.50,
            "buf": 2.5,
            "k": 1.0,
            "h": [21, 63],
            "w": W_MENU["w55"],
            "fast": 8,
            "slow": 34,
            "lock": lock,
            "lock_pct": lock_pct,
            "md": md,
            "nov": True,
            "cd": 21,
            "wkey": "w55",
            "hkey": "h21",
        }
    )

print(f"PHASE1b done n={n} ok={len(rows)}", flush=True)

rows.sort(key=lambda r: (r["tot"], r["mean_yr"], r["cagr"], -r["maxdd"]), reverse=True)
seeds = rows[:6]

# Phase 2: local refine of top seeds (dense, no hash skip)
for seed_row in seeds:
    for vol in [
        seed_row["vol"] - 0.015,
        seed_row["vol"] - 0.01,
        seed_row["vol"] - 0.005,
        seed_row["vol"],
        seed_row["vol"] + 0.005,
        seed_row["vol"] + 0.01,
        seed_row["vol"] + 0.015,
    ]:
        if not (0.26 <= vol <= 0.38):
            continue
        for pos in [
            seed_row["pos"] - 0.75,
            seed_row["pos"] - 0.5,
            seed_row["pos"] - 0.25,
            seed_row["pos"],
            seed_row["pos"] + 0.25,
            seed_row["pos"] + 0.5,
            seed_row["pos"] + 0.75,
        ]:
            if pos < 4.0 or pos > 9.0:
                continue
            for mc in sorted({max(16, seed_row["mc"] - 4), seed_row["mc"], seed_row["mc"] + 4}):
                for dd_halt in sorted(
                    {0.22, 0.24, 0.25, 0.26, 0.27, 0.28, float(seed_row["dd_halt"])}
                ):
                    for tp in sorted({seed_row["tp"] - 5, seed_row["tp"], seed_row["tp"] + 5}):
                        if tp < 15:
                            continue
                        for stop in [3.5, 4.0, 4.5]:
                            for lock_pct in sorted(
                                {
                                    0.0005,
                                    0.001,
                                    0.0015,
                                    float(seed_row.get("lock_pct", 0.001)),
                                }
                            ):
                                for wkey in ["w50", "w55", "w60"]:
                                    for hkey in ["h21", "h21126"]:
                                        for md in [1, 2]:
                                            key = (
                                                round(vol, 4),
                                                round(pos, 3),
                                                mc,
                                                round(dd_halt, 4),
                                                tp,
                                                stop,
                                                round(lock_pct, 5),
                                                wkey,
                                                hkey,
                                                md,
                                            )
                                            if hash(key) % 4 != 0:
                                                continue
                                            eval_once(
                                                {
                                                    "vol": float(vol),
                                                    "pos": float(pos),
                                                    "mc": int(mc),
                                                    "dd_halt": float(dd_halt),
                                                    "tp": float(tp),
                                                    "stop": float(stop),
                                                    "agr": 0.50,
                                                    "buf": 2.5,
                                                    "k": 1.0,
                                                    "h": H_MENU[hkey],
                                                    "w": W_MENU[wkey],
                                                    "fast": 8,
                                                    "slow": 34,
                                                    "lock": True,
                                                    "lock_pct": float(lock_pct),
                                                    "md": int(md),
                                                    "nov": True,
                                                    "cd": 21,
                                                    "wkey": wkey,
                                                    "hkey": hkey,
                                                }
                                            )

rows.sort(key=lambda r: (r["tot"], r["mean_yr"], r["cagr"], -r["maxdd"]), reverse=True)
print(f"DONE n={n} ok={len(rows)} t={time.time()-t0:.1f}s", flush=True)
print("TOP15:", flush=True)
for r in rows[:15]:
    print(
        f" tot={r['tot']:.2%} mean={r['mean_yr']:.3%} cagr={r['cagr']:.3%} "
        f"dd={r['maxdd']:.3%} neg={r['neg']} vol={r['vol']} pos={r['pos']} "
        f"mc={r['mc']} halt={r['dd_halt']} tp={r['tp']} stop={r['stop']} "
        f"lock={r['lock']}/{r.get('lock_pct')} md={r.get('md')} "
        f"{r.get('wkey')}/{r.get('hkey')}",
        flush=True,
    )

if not rows:
    raise SystemExit("no DD<30 profitable configs found")

winner = rows[0]
verify = run_params(winner)
print("WINNER", {k: winner[k] for k in winner if k not in ("yearly", "w")}, flush=True)
print("WEIGHTS", winner["w"], flush=True)
print(
    "VERIFY",
    {k: verify[k] for k in ["mean_yr", "tot", "maxdd", "cagr", "neg", "min_yr"]},
    flush=True,
)
assert verify["maxdd"] < 0.30
assert abs(verify["tot"] - winner["tot"]) < 1e-9

payload = {
    "baseline": {k: base_m[k] for k in base_m if k != "yearly"},
    "seed": {k: v for k, v in {**seed, **seed_m}.items() if k != "yearly"},
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
print(
    f"LIFT vs base tot {verify['tot']-base_m['tot']:+.2%} "
    f"mean {verify['mean_yr']-base_m['mean_yr']:+.3%}",
    flush=True,
)
