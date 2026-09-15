#!/usr/bin/env python3
"""Paper-trade the production copper ensemble at institutional capital.

Default: $350M account, Yahoo HG daily bars from 2008-01-01 → now.

At large AUM, ``max_contracts`` must scale with capital or the book is
capacity-starved (tiny % returns). Use ``--scale-contracts`` (default) for
like-for-like %-risk vs the $500k production config, or ``--no-scale-contracts``
to keep the absolute production contract cap.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

from copper_ensemble.data import dataframe_to_bars, load_yfinance_hg
from copper_ensemble.engine import BacktestEngine, calendar_year_returns
from copper_ensemble.models import clone_config, load_config

warnings.filterwarnings("ignore")


def _run(
    *,
    cash: float,
    max_contracts: int,
    start: str,
    out: Path,
    label: str,
) -> Dict[str, Any]:
    base = load_config()
    cfg = clone_config(
        base,
        risk_overrides={
            "max_contracts": int(max_contracts),
            "max_leverage": float(max(base.risk.max_leverage, base.risk.max_position_size_pct)),
        },
    )
    cfg = replace(cfg, portfolio=replace(cfg.portfolio, initial_cash=float(cash)))

    bars = dataframe_to_bars(load_yfinance_hg("HG=F", start=start), "HG")
    result = BacktestEngine(cfg).run(bars)
    equity = np.asarray(result.equity_curve, dtype=float)
    ts = [b.timestamp for b in bars]
    yearly = calendar_year_returns(equity, ts)
    vals = np.asarray(list(yearly.values()), dtype=float)
    positions = np.asarray(getattr(result, "positions", []), dtype=float)
    max_abs = float(np.nanmax(np.abs(positions))) if positions.size else float("nan")

    payload: Dict[str, Any] = {
        "label": label,
        "start": start,
        "n_bars": len(bars),
        "initial_cash": float(cash),
        "max_contracts": int(max_contracts),
        "max_position_size_pct": float(cfg.risk.max_position_size_pct),
        "max_leverage": float(cfg.risk.max_leverage),
        "vol_target_annual": float(cfg.ensemble.vol_target_annual),
        "end_equity": float(equity[-1]),
        "pnl": float(equity[-1] - cash),
        "total_return": float(result.metrics["total_return"]),
        "cagr": float(result.metrics.get("cagr", 0.0)),
        "mean_yearly": float(vals.mean()),
        "min_yearly": float(vals.min()),
        "losing_years": int((vals < 0).sum()),
        "n_years": int(len(vals)),
        "max_drawdown": float(result.metrics["max_drawdown"]),
        "sharpe": float(result.metrics.get("sharpe", 0.0)),
        "upi": float(result.metrics.get("upi", 0.0)),
        "n_fills": int(result.metrics.get("n_fills", 0)),
        "max_abs_contracts": max_abs,
        "yearly": {str(k): float(v) for k, v in yearly.items()},
        "metrics": {k: float(v) for k, v in result.metrics.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"copper_paper_{label}.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cash", type=float, default=350_000_000.0)
    p.add_argument("--start", default="2008-01-01")
    p.add_argument(
        "--scale-contracts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale max_contracts with cash / production AUM (default: true)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts"),
    )
    args = p.parse_args()

    base = load_config()
    base_cash = float(base.portfolio.initial_cash)
    scale = float(args.cash) / base_cash
    if args.scale_contracts:
        mc = int(round(base.risk.max_contracts * scale))
        label = f"{int(args.cash/1e6)}m_scaled"
    else:
        mc = int(base.risk.max_contracts)
        label = f"{int(args.cash/1e6)}m_starved"

    print(
        f"paper cash=${args.cash:,.0f} scale={scale:.1f}x max_contracts={mc} "
        f"scale_contracts={args.scale_contracts}",
        flush=True,
    )
    payload = _run(
        cash=float(args.cash),
        max_contracts=mc,
        start=args.start,
        out=args.out,
        label=label,
    )
    print(
        f"tot={payload['total_return']*100:.2f}% mean_yr={payload['mean_yearly']*100:.2f}% "
        f"maxDD={payload['max_drawdown']*100:.2f}% pnl=${payload['pnl']:,.0f} "
        f"end=${payload['end_equity']:,.0f} losing_years={payload['losing_years']}",
        flush=True,
    )
    print(f"wrote {args.out / f'copper_paper_{label}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
