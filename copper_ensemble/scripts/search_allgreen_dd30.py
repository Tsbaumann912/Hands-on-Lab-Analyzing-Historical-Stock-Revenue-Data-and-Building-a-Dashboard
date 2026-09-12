#!/usr/bin/env python3
"""Search all-green calendar years + max DD <= 30%, maximize mean yearly / total return."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from copper_ensemble.data import dataframe_to_bars, load_yfinance_hg
from copper_ensemble.engine import BacktestEngine, calendar_year_returns
from copper_ensemble.models import clone_config, load_config

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/opt/cursor/artifacts")

WEIGHTS: Dict[int, Dict[str, float]] = {
    40: {
        "tsmom": 0.40,
        "carry": 0.0,
        "basis_mom": 0.0,
        "inventory": 0.0,
        "fade": 0.0,
        "ma_cross": 0.30,
        "stoch_rsi": 0.30,
    },
    50: {
        "tsmom": 0.50,
        "carry": 0.0,
        "basis_mom": 0.0,
        "inventory": 0.0,
        "fade": 0.0,
        "ma_cross": 0.25,
        "stoch_rsi": 0.25,
    },
    55: {
        "tsmom": 0.55,
        "carry": 0.0,
        "basis_mom": 0.0,
        "inventory": 0.0,
        "fade": 0.0,
        "ma_cross": 0.25,
        "stoch_rsi": 0.20,
    },
    60: {
        "tsmom": 0.60,
        "carry": 0.0,
        "basis_mom": 0.0,
        "inventory": 0.0,
        "fade": 0.0,
        "ma_cross": 0.25,
        "stoch_rsi": 0.15,
    },
    70: {
        "tsmom": 0.70,
        "carry": 0.0,
        "basis_mom": 0.0,
        "inventory": 0.0,
        "fade": 0.0,
        "ma_cross": 0.20,
        "stoch_rsi": 0.10,
    },
}


def run_one(cfg_base: Any, bars: List[Any], p: Dict[str, Any]) -> Dict[str, Any]:
    c = clone_config(
        cfg_base,
        ensemble_overrides={
            "vol_target_annual": float(p["vol"]),
            "kelly_fraction": float(p["k"]),
            "buffer_forecast": float(p["buf"]),
            "agreement_min": float(p["agr"]),
            "horizons_days": list(p["h"]),
            "weights": WEIGHTS[int(p["w"])],
            "take_profit_atr_mult": float(p["tp"]),
            "stop_atr_mult": float(p["stop"]),
            "fast_ma_period": int(p.get("fast", 8)),
            "slow_ma_period": int(p.get("slow", 34)),
        },
        risk_overrides={
            "max_position_size_pct": float(p["pos"]),
            "max_contracts": int(p["mc"]),
            "max_leverage": float(max(5.0, p["pos"])),
            "max_daily_drawdown_pct": float(p["dd"]),
            "yearly_profit_lock_enabled": bool(p["lock_en"]),
            "yearly_profit_lock_pct": float(p["lock"]),
            "yearly_profit_lock_min_days": int(p.get("md", 2)),
            "yearly_nov_protect": bool(p.get("nov", True)),
        },
    )
    r = BacktestEngine(c).run(bars)
    yr = calendar_year_returns(r.equity_curve, [b.timestamp for b in bars])
    vals = np.asarray(list(yr.values()), dtype=np.float64)
    m = r.metrics
    return {
        **p,
        "weights": WEIGHTS[int(p["w"])],
        "mean_yr": float(vals.mean()),
        "min_yr": float(vals.min()),
        "neg": int((vals < 0).sum()),
        "tot": float(m["total_return"]),
        "maxdd": float(m["max_drawdown"]),
        "cagr": float(m["cagr"]),
        "sharpe": float(m.get("sharpe", 0.0)),
        "upi": float(m.get("upi", 0.0)),
        "yearly": {str(k): float(v) for k, v in yr.items()},
        "metrics": {k: float(v) for k, v in m.items()},
    }


def score(r: Dict[str, Any]) -> Tuple:
    ok = r["neg"] == 0 and r["min_yr"] > 0.0 and r["maxdd"] <= 0.30 and r["tot"] > 0.0
    return (
        1 if ok else 0,
        r["mean_yr"] if ok else -999.0,
        r["tot"] if ok else -999.0,
        -r["maxdd"] if ok else -999.0,
        r["min_yr"] if ok else -999.0,
        r["cagr"] if ok else -999.0,
    )


def build_todo() -> List[Dict[str, Any]]:
    """Lean candidate set (~700) around known all-green / DD<=30 regime."""
    todo: List[Dict[str, Any]] = []

    # Prior all-green production-like seed
    todo.append(
        dict(
            vol=0.30,
            pos=5.0,
            dd=0.15,
            tp=25.0,
            stop=4.0,
            agr=0.50,
            h=[21, 63],
            w=50,
            k=0.75,
            buf=2.5,
            lock_en=True,
            lock=0.002,
            md=2,
            nov=True,
            mc=20,
            fast=8,
            slow=34,
        )
    )

    # Core neighborhood (lock ON) — keep product small
    for vol in (0.25, 0.28, 0.30, 0.32, 0.35, 0.38):
        for pos in (4.0, 5.0, 6.0, 7.0):
            for dd in (0.12, 0.15, 0.20, 0.25, 0.30):
                for tp, stop in ((20.0, 4.0), (25.0, 4.0), (30.0, 4.0), (40.0, 5.0)):
                    for lock in (0.001, 0.002, 0.003, 0.005):
                        for agr in (0.45, 0.50, 0.55):
                            for w in (50, 55, 60):
                                for h in ([21, 63],):
                                    for k in (0.75, 1.0):
                                        for buf in (2.5,):
                                            for md in (2,):
                                                for nov in (True,):
                                                    todo.append(
                                                        dict(
                                                            vol=vol,
                                                            pos=pos,
                                                            dd=dd,
                                                            tp=tp,
                                                            stop=stop,
                                                            agr=agr,
                                                            h=list(h),
                                                            w=w,
                                                            k=k,
                                                            buf=buf,
                                                            lock_en=True,
                                                            lock=lock,
                                                            md=md,
                                                            nov=nov,
                                                            mc=int(min(30, max(12, pos * 4))),
                                                            fast=8,
                                                            slow=34,
                                                        )
                                                    )

    # Extra: raise TP / tweak lock/nov around winners
    for vol in (0.30, 0.32, 0.35):
        for pos in (5.0, 6.0, 7.0):
            for dd in (0.15, 0.20, 0.25):
                for tp, stop in ((25.0, 4.0), (30.0, 4.0), (40.0, 4.0), (50.0, 5.0)):
                    for lock in (0.002, 0.003, 0.004):
                        for agr in (0.50, 0.55):
                            for w in (50, 55):
                                for nov in (True, False):
                                    for md in (1, 2):
                                        for h in ([21, 63], [10, 21, 63]):
                                            todo.append(
                                                dict(
                                                    vol=vol,
                                                    pos=pos,
                                                    dd=dd,
                                                    tp=tp,
                                                    stop=stop,
                                                    agr=agr,
                                                    h=list(h),
                                                    w=w,
                                                    k=0.75,
                                                    buf=2.5,
                                                    lock_en=True,
                                                    lock=lock,
                                                    md=md,
                                                    nov=nov,
                                                    mc=int(min(30, max(12, pos * 4))),
                                                    fast=8,
                                                    slow=34,
                                                )
                                            )

    seen = set()
    out: List[Dict[str, Any]] = []
    for p in todo:
        key = json.dumps(p, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)

    # Cap if somehow large
    if len(out) > 900:
        rng = np.random.default_rng(7)
        seeds = out[:1]
        rest = out[1:]
        out = seeds + list(rng.choice(rest, size=899, replace=False))
    return out


def main() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    bars = dataframe_to_bars(load_yfinance_hg("HG=F", start="2008-01-01"), "HG")
    print(f"bars {len(bars)}", flush=True)

    todo = build_todo()
    print(f"todo {len(todo)}", flush=True)

    best: Optional[Dict[str, Any]] = None
    hits: List[Dict[str, Any]] = []
    for i, p in enumerate(todo):
        try:
            r = run_one(cfg, bars, p)
        except Exception as exc:  # noqa: BLE001
            print(f"ERR {exc}", flush=True)
            continue
        r["_sc"] = score(r)
        if best is None or r["_sc"] > best["_sc"]:
            best = r
            print(
                f"BEST mean={r['mean_yr']*100:.2f}% neg={r['neg']} min={r['min_yr']*100:.2f}% "
                f"tot={r['tot']*100:.1f}% maxdd={r['maxdd']*100:.1f}% cagr={r['cagr']*100:.2f}% "
                f"vol={r['vol']} pos={r['pos']} dd={r['dd']} tp={r['tp']} stop={r['stop']} "
                f"lock={r['lock']} md={r['md']} nov={r['nov']} agr={r['agr']} h={r['h']} "
                f"w={r['w']} k={r['k']} buf={r['buf']}",
                flush=True,
            )
        if r["neg"] == 0 and r["min_yr"] > 0 and r["maxdd"] <= 0.30 and r["tot"] > 0:
            hits.append(r)
        if (i + 1) % 50 == 0:
            print(f"progress {i+1}/{len(todo)} hits={len(hits)}", flush=True)

    hits.sort(key=lambda x: x["_sc"], reverse=True)
    print(f"HITS {len(hits)}", flush=True)
    for r in hits[:20]:
        print(
            f"mean={r['mean_yr']*100:.2f}% neg={r['neg']} min={r['min_yr']*100:.2f}% "
            f"tot={r['tot']*100:.1f}% maxdd={r['maxdd']*100:.1f}% cagr={r['cagr']*100:.2f}% "
            f"vol={r['vol']} pos={r['pos']} dd={r['dd']} tp={r['tp']} stop={r['stop']} "
            f"lock={r['lock']} md={r['md']} nov={r['nov']} agr={r['agr']} h={r['h']} "
            f"w={r['w']} k={r['k']} buf={r['buf']}",
            flush=True,
        )

    winner = hits[0] if hits else best
    assert winner is not None
    out = {k: v for k, v in winner.items() if not k.startswith("_")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "copper_allgreen_dd30_winner.json").write_text(json.dumps(out, indent=2))
    (OUT / "copper_allgreen_dd30_top20.json").write_text(
        json.dumps(
            [{k: v for k, v in r.items() if not k.startswith("_")} for r in hits[:20]],
            indent=2,
        )
    )
    print(
        f"WROTE mean={out['mean_yr']} tot={out['tot']} maxdd={out['maxdd']} neg={out['neg']}",
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
