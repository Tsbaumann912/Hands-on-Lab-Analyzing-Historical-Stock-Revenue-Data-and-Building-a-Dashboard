#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np

from copper_ensemble.data import dataframe_to_bars, load_yfinance_hg
from copper_ensemble.engine import BacktestEngine, calendar_year_returns
from copper_ensemble.models import clone_config, load_config

warnings.filterwarnings("ignore")

OUT = Path("/opt/cursor/artifacts")
OUT.mkdir(parents=True, exist_ok=True)

bars = dataframe_to_bars(load_yfinance_hg("HG=F", start="2008-01-01"), "HG")
base = load_config()
print(f"bars={len(bars)}", flush=True)

W60 = {
    "tsmom": 0.60,
    "carry": 0.0,
    "basis_mom": 0.0,
    "inventory": 0.0,
    "fade": 0.0,
    "ma_cross": 0.25,
    "stoch_rsi": 0.15,
}
W70 = {
    "tsmom": 0.70,
    "carry": 0.0,
    "basis_mom": 0.0,
    "inventory": 0.0,
    "fade": 0.0,
    "ma_cross": 0.20,
    "stoch_rsi": 0.10,
}


def run(eo: dict, ro: dict) -> dict:
    cfg = clone_config(base, ensemble_overrides=eo, risk_overrides=ro)
    result = BacktestEngine(cfg).run(bars)
    yearly = calendar_year_returns(result.equity_curve, [b.timestamp for b in bars])
    vals = np.asarray(list(yearly.values()), dtype=float)
    return {
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "neg": int((vals < 0).sum()),
        "tot": float(result.metrics["total_return"]),
        "dd": float(result.metrics["max_drawdown"]),
        "cagr": float(result.metrics.get("cagr", 0.0)),
        "yearly": {str(k): float(v) for k, v in yearly.items()},
    }


grid = []
for vol, pos, dd_halt, lock, lock_pct, w in itertools.product(
    [0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.0],
    [5, 8, 12, 15, 20],
    [0.12, 0.18, 0.22, 0.25, 0.28],
    [False, True],
    [0.002, 0.10],
    [W60, W70],
):
    if (not lock) and lock_pct != 0.10:
        continue
    grid.append((vol, pos, dd_halt, lock, lock_pct, w))

print(f"grid={len(grid)}", flush=True)
rows = []
for i, (vol, pos, dd_halt, lock, lock_pct, w) in enumerate(grid, 1):
    metrics = run(
        {
            "vol_target_annual": float(vol),
            "kelly_fraction": 1.0,
            "buffer_forecast": 2.0,
            "agreement_min": 0.50,
            "horizons_days": [21, 63],
            "weights": w,
            "take_profit_atr_mult": 20.0,
            "stop_atr_mult": 4.0,
            "fast_ma_period": 8,
            "slow_ma_period": 34,
        },
        {
            "max_position_size_pct": float(pos),
            "max_contracts": int(max(28, pos * 4)),
            "max_leverage": max(8.0, float(pos)),
            "max_daily_drawdown_pct": float(dd_halt),
            "yearly_profit_lock_enabled": bool(lock),
            "yearly_profit_lock_pct": float(lock_pct),
            "yearly_profit_lock_min_days": 2,
            "yearly_nov_protect": bool(lock),
            "halt_on_breach": True,
            "halt_cooldown_bars": 21,
        },
    )
    row = {
        "vol": vol,
        "pos": pos,
        "dd_halt": dd_halt,
        "lock": lock,
        "lock_pct": lock_pct,
        "w": w["tsmom"],
        **metrics,
        "ok": metrics["mean"] >= 0.20 and metrics["dd"] < 0.30,
    }
    rows.append(row)
    if i % 40 == 0 or row["ok"]:
        under = [r for r in rows if r["dd"] < 0.30]
        best = max((r["mean"] for r in under), default=float("nan"))
        peak = max(rows, key=lambda r: r["mean"])
        print(
            f"i={i}/{len(grid)} hits={sum(r['ok'] for r in rows)} "
            f"best@dd<30={best:.3%} peak_mean={peak['mean']:.3%}@dd={peak['dd']:.3%}",
            flush=True,
        )

ok = [r for r in rows if r["ok"]]
under = sorted(
    [r for r in rows if r["dd"] < 0.30],
    key=lambda r: (r["mean"], r["tot"]),
    reverse=True,
)
ge20 = sorted([r for r in rows if r["mean"] >= 0.20], key=lambda r: r["dd"])
print(
    f"DONE hits={len(ok)} under30={len(under)} mean>=20={len(ge20)}",
    flush=True,
)
print("TOP under30:", flush=True)
for r in under[:15]:
    print(
        f" mean={r['mean']:.3%} dd={r['dd']:.3%} tot={r['tot']:.1%} neg={r['neg']} "
        f"lock={r['lock']}@{r['lock_pct']} vol={r['vol']} pos={r['pos']} halt={r['dd_halt']}",
        flush=True,
    )
print("lowest DD among mean>=20:", flush=True)
for r in ge20[:12]:
    print(
        f" dd={r['dd']:.3%} mean={r['mean']:.3%} tot={r['tot']:.1%} "
        f"lock={r['lock']}@{r['lock_pct']} vol={r['vol']} pos={r['pos']} halt={r['dd_halt']}",
        flush=True,
    )
frontier = {}
for cap in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
    cand = [r for r in rows if r["dd"] < cap]
    if not cand:
        print(f"cap<{cap}:none", flush=True)
        frontier[str(cap)] = None
        continue
    best = max(cand, key=lambda x: x["mean"])
    frontier[str(cap)] = best
    print(
        f"cap<{cap:.0%}: mean={best['mean']:.3%} dd={best['dd']:.3%} "
        f"lock={best['lock']} vol={best['vol']} pos={best['pos']} halt={best['dd_halt']}",
        flush=True,
    )

payload = {
    "n": len(rows),
    "hits": ok[:20],
    "top_under30": under[:25],
    "lowest_dd_mean20": ge20[:15],
    "frontier": frontier,
}
(OUT / "copper_mean20_dd30_probe.json").write_text(json.dumps(payload, indent=2))
print("wrote", OUT / "copper_mean20_dd30_probe.json", flush=True)
