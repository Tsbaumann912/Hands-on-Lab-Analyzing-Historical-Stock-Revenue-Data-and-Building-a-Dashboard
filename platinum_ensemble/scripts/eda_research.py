#!/usr/bin/env python3
"""
EDA for NYMEX Platinum research — lock hypotheses before optimisation.

Runs offline on synthetic data by default; pass --yfinance for live PL=F / GC=F.
Writes a JSON summary under platinum_ensemble/artifacts/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platinum_ensemble.data import dataframe_to_bars, load_yfinance_pl, make_synthetic_pl
from platinum_ensemble.forecasts import compute_all_forecasts
from platinum_ensemble.models import load_config
from platinum_ensemble.data import build_feature_matrix


def _acf(x: np.ndarray, lags: int = 20) -> list[float]:
    x = x[np.isfinite(x)]
    if x.size < lags + 5:
        return []
    x = x - np.mean(x)
    var = float(np.dot(x, x))
    if var < 1e-18:
        return [0.0] * lags
    out: list[float] = []
    for k in range(1, lags + 1):
        out.append(float(np.dot(x[:-k], x[k:]) / var))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Platinum research EDA")
    parser.add_argument("--yfinance", action="store_true")
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(ROOT / "config" / "default.yaml")
    if args.yfinance:
        df = load_yfinance_pl(cfg.contract.yfinance_ticker, cfg.ensemble.gold_ticker)
    else:
        df = make_synthetic_pl(n_days=args.days, seed=args.seed)

    bars = dataframe_to_bars(df, "PL")
    feats = build_feature_matrix(bars, cfg)
    forecasts = compute_all_forecasts(feats, cfg.ensemble)

    rets = feats["returns"]
    rets_clean = rets[np.isfinite(rets)]
    acf = _acf(rets_clean, 20)

    # PL/GC log ratio stationarity proxy: fraction of time |z| > 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.clip(feats["close"], 1e-12, None)) - np.log(
            np.clip(feats["gold_close"], 1e-12, None)
        )
    mu = np.nanmean(ratio)
    sd = np.nanstd(ratio)
    z = (ratio - mu) / (sd + 1e-12)
    extreme_frac = float(np.nanmean(np.abs(z) > 1.5))

    summary = {
        "n_bars": len(bars),
        "source": "yfinance" if args.yfinance else "synthetic",
        "pl_last": float(feats["close"][-1]),
        "gold_last": float(feats["gold_close"][-1]),
        "return_mean_ann": float(np.nanmean(rets_clean) * 252),
        "return_vol_ann": float(np.nanstd(rets_clean, ddof=1) * np.sqrt(252)),
        "acf_lags_1_5_10_20": {
            "1": acf[0] if len(acf) > 0 else None,
            "5": acf[4] if len(acf) > 4 else None,
            "10": acf[9] if len(acf) > 9 else None,
            "20": acf[19] if len(acf) > 19 else None,
        },
        "pl_gc_extreme_z_frac": extreme_frac,
        "mean_carry": float(np.nanmean(feats["carry"])),
        "sleeve_nan_frac": {
            k: float(np.mean(~np.isfinite(v))) for k, v in forecasts.items()
        },
        "hypotheses_locked": [
            "A TSMOM: positive short-lag ACF supports momentum on PL",
            "B Carry: nonzero mean carry implies roll-yield exposure",
            "C PL-GC RV: nonzero extreme_z_frac implies tradable wedges",
            "D Inventory: use warehouse/ETF proxy confirmation",
            "E Macro fade: only when gold/USD do not confirm",
        ],
    }

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "eda_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
