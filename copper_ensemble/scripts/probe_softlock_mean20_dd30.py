#!/usr/bin/env python3
from __future__ import annotations

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
print("bars", len(bars), flush=True)
W = {
    "tsmom": 0.60,
    "carry": 0.0,
    "basis_mom": 0.0,
    "inventory": 0.0,
    "fade": 0.0,
    "ma_cross": 0.25,
    "stoch_rsi": 0.15,
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


prod = run({}, {})
print("PROD", {k: prod[k] for k in ["mean", "dd", "tot", "neg", "cagr"]}, flush=True)

best_dd30 = run(
    {
        "vol_target_annual": 0.55,
        "kelly_fraction": 1.0,
        "buffer_forecast": 2.0,
        "agreement_min": 0.50,
        "horizons_days": [21, 63],
        "weights": W,
        "take_profit_atr_mult": 20.0,
        "stop_atr_mult": 4.0,
        "fast_ma_period": 8,
        "slow_ma_period": 34,
    },
    {
        "max_position_size_pct": 5.0,
        "max_contracts": 28,
        "max_leverage": 8.0,
        "max_daily_drawdown_pct": 0.12,
        "yearly_profit_lock_enabled": True,
        "yearly_profit_lock_pct": 0.002,
        "yearly_profit_lock_min_days": 2,
        "yearly_nov_protect": True,
        "halt_on_breach": True,
        "halt_cooldown_bars": 21,
    },
)
print(
    "BEST_DD30",
    {k: best_dd30[k] for k in ["mean", "dd", "tot", "neg", "cagr"]},
    flush=True,
)

rows = []
i = 0
for vol in [0.55, 0.70, 0.85, 1.0]:
    for pos in [8, 10, 12, 15]:
        for halt in [0.22, 0.25, 0.28]:
            for lock_pct in [0.08, 0.12, 0.18, 0.25]:
                for lock_days in [1, 5]:
                    i += 1
                    metrics = run(
                        {
                            "vol_target_annual": float(vol),
                            "kelly_fraction": 1.0,
                            "buffer_forecast": 2.0,
                            "agreement_min": 0.50,
                            "horizons_days": [21, 63],
                            "weights": W,
                            "take_profit_atr_mult": 20.0,
                            "stop_atr_mult": 4.0,
                            "fast_ma_period": 8,
                            "slow_ma_period": 34,
                        },
                        {
                            "max_position_size_pct": float(pos),
                            "max_contracts": int(max(30, pos * 4)),
                            "max_leverage": max(8.0, float(pos)),
                            "max_daily_drawdown_pct": float(halt),
                            "yearly_profit_lock_enabled": True,
                            "yearly_profit_lock_pct": float(lock_pct),
                            "yearly_profit_lock_min_days": int(lock_days),
                            "yearly_nov_protect": False,
                            "halt_on_breach": True,
                            "halt_cooldown_bars": 21,
                        },
                    )
                    row = {
                        "vol": vol,
                        "pos": pos,
                        "halt": halt,
                        "lock_pct": lock_pct,
                        "lock_days": lock_days,
                        **metrics,
                        "ok": metrics["mean"] >= 0.20 and metrics["dd"] < 0.30,
                    }
                    rows.append(row)
                    if i % 40 == 0 or row["ok"]:
                        under = [r for r in rows if r["dd"] < 0.30]
                        best = max((r["mean"] for r in under), default=float("nan"))
                        print(
                            f"i={i} hits={sum(r['ok'] for r in rows)} best@dd<30={best:.3%}",
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
    f"DONE soft n={len(rows)} hits={len(ok)} under30={len(under)} ge20={len(ge20)}",
    flush=True,
)
print("soft TOP under30:", flush=True)
for r in under[:10]:
    print(
        f" mean={r['mean']:.3%} dd={r['dd']:.3%} tot={r['tot']:.1%} neg={r['neg']} "
        f"lock@{r['lock_pct']} d={r['lock_days']} vol={r['vol']} pos={r['pos']} halt={r['halt']}",
        flush=True,
    )
print("soft lowest DD mean>=20:", flush=True)
for r in ge20[:8]:
    print(
        f" dd={r['dd']:.3%} mean={r['mean']:.3%} tot={r['tot']:.1%} "
        f"lock@{r['lock_pct']} d={r['lock_days']} vol={r['vol']} pos={r['pos']} halt={r['halt']}",
        flush=True,
    )
for cap in [0.30, 0.40, 0.50, 0.60]:
    cand = [r for r in rows if r["dd"] < cap]
    if not cand:
        print(f"cap<{cap}:none", flush=True)
        continue
    best = max(cand, key=lambda x: x["mean"])
    print(
        f"soft cap<{cap:.0%}: mean={best['mean']:.3%} dd={best['dd']:.3%} "
        f"lock@{best['lock_pct']} vol={best['vol']} pos={best['pos']}",
        flush=True,
    )

(OUT / "copper_mean20_dd30_softlock.json").write_text(
    json.dumps(
        {
            "prod": prod,
            "best_dd30_candidate": best_dd30,
            "hits": ok[:10],
            "top_under30": under[:15],
            "lowest_dd_mean20": ge20[:10],
        },
        indent=2,
    )
)
print("wrote softlock", flush=True)
