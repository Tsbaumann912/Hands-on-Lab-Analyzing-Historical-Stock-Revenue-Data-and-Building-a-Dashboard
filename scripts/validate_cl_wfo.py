#!/usr/bin/env python3
"""
Anchored + rolling walk-forward validation for CL energy strategies.

Example
-------
python3 scripts/validate_cl_wfo.py \\
  --strategy CLCarryMomentum \\
  --start 2008-01-01 \\
  --capital 350000000 \\
  --modes anchored,rolling \\
  --trials 25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Repo root on sys.path when invoked as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import CLValidationConfig, Config
from core.enums import AssetClass
from core.models import Bar
from engine.optimizer import STRATEGY_PARAM_SPACES, build_search_space_from_yaml
from engine.walk_forward import (
    GateThresholds,
    ModeValidationReport,
    WalkForwardValidator,
    years_to_bars,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("risk.risk_manager").setLevel(logging.ERROR)
logging.getLogger("brokers.paper").setLevel(logging.ERROR)
logging.getLogger("portfolio.portfolio").setLevel(logging.ERROR)
logging.getLogger("strategies.base").setLevel(logging.WARNING)
logger = logging.getLogger("validate_cl_wfo")

CL_STRATEGIES = {
    "CLCarryCurve",
    "CLCarryMomentum",
    "CLVolTargetTSMOM",
    "CLInventoryConfirm",
}


def _apply_cl_contract(cfg: Config) -> None:
    spec = cfg.contracts.get("CL")
    if spec is None:
        cfg.portfolio.contract_multiplier = 1000.0
        cfg.portfolio.tick_size = 0.01
        cfg.portfolio.tick_value = 10.0
        return
    cfg.portfolio.contract_multiplier = spec.contract_multiplier
    cfg.portfolio.tick_size = spec.tick_size
    cfg.portfolio.tick_value = spec.tick_value
    cfg.portfolio.commission_per_contract = spec.commission_per_contract


def _apply_validation_risk(cfg: Config, v: CLValidationConfig) -> None:
    cfg.risk.max_daily_drawdown_pct = float(v.risk_max_daily_drawdown_pct)
    cfg.risk.halt_on_breach = bool(v.risk_halt_on_breach)
    cfg.risk.max_position_size_pct = float(v.risk_max_position_size_pct)
    cfg.risk.max_leverage = float(v.risk_max_leverage)
    # Futures-style Sharpe / Sortino for Optuna + stitched OOS metrics.
    cfg.backtest.risk_free_rate = float(getattr(v, "metrics_risk_free_rate", 0.0))
    if hasattr(cfg.risk, "risk_fraction"):
        cfg.risk.risk_fraction = float(getattr(v, "risk_fraction", 0.01))


def load_cl_daily_bars(
    start_date: str,
    symbol: str = "CL.c.0",
    multiplier: float = 1000.0,
) -> Tuple[List[Bar], Dict[str, Any]]:
    """Download daily CL=F from Yahoo (fallback: synthetic) and convert to bars."""
    from app.data_service import _download_futures_history

    start_year = int(start_date[:4])
    df = _download_futures_history("CL=F", start_year)
    df = df.sort_values("Date").reset_index(drop=True)
    start_ts = pd.Timestamp(start_date)
    df = df[pd.to_datetime(df["Date"]) >= start_ts].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No CL history on/after {start_date}")

    bars: List[Bar] = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["Date"]).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=ts,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0) or 0),
                asset_class=AssetClass.FUTURES,
                contract_multiplier=multiplier,
            )
        )
    meta = {
        "n_bars": len(bars),
        "start": bars[0].timestamp.isoformat(),
        "end": bars[-1].timestamp.isoformat(),
        "source": "yfinance_CL=F",
    }
    return bars, meta


def _strategy_cls(name: str):
    from app.data_service import STRATEGY_REGISTRY

    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy {name!r}")
    return cls


def run_validation(
    cfg: Config,
    strategy_name: str,
    modes: Sequence[str],
    *,
    synthetic: bool = False,
    n_trials: Optional[int] = None,
    output_dir: Path,
) -> Dict[str, Any]:
    v = cfg.cl_validation
    _apply_cl_contract(cfg)
    _apply_validation_risk(cfg, v)
    cfg.portfolio.initial_cash = float(v.capital)

    symbol = "CL.c.0"
    if synthetic:
        from tests.conftest import make_bars

        # Long synthetic path covering multi-year WFO folds.
        n = years_to_bars(v.rolling_is_years + 8)
        raw = make_bars(
            n=n,
            symbol=symbol,
            start_price=75.0,
            volatility=1.2,
            seed=2008,
        )
        # Override floor: rebuild without ES $100 clip for CL.
        from datetime import timedelta

        rng = np.random.default_rng(2008)
        prices = 75.0 + np.cumsum(rng.normal(0.01, 0.9, n))
        prices = np.clip(prices, 20.0, None)
        base = datetime(2008, 1, 1, tzinfo=timezone.utc)
        bars = []
        for i, px in enumerate(prices):
            c = float(px)
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=base + timedelta(days=i),
                    open=c,
                    high=c + 0.5,
                    low=c - 0.5,
                    close=c,
                    volume=5000.0,
                    asset_class=AssetClass.FUTURES,
                    contract_multiplier=cfg.portfolio.contract_multiplier,
                )
            )
        meta = {"n_bars": len(bars), "source": "synthetic"}
    else:
        bars, meta = load_cl_daily_bars(
            v.start_date,
            symbol=symbol,
            multiplier=cfg.portfolio.contract_multiplier,
        )

    bar_map = {symbol: bars}
    param_names = STRATEGY_PARAM_SPACES.get(strategy_name, [])
    search_space = build_search_space_from_yaml(param_names)
    if not search_space:
        raise RuntimeError(f"Empty Optuna search space for {strategy_name}")

    thresholds = GateThresholds(
        max_dd_limit=float(v.max_dd_limit),
        min_sharpe=float(v.min_sharpe),
        min_upi=float(v.min_upi),
        min_cagr=float(v.min_cagr),
        require_all_years_profitable=bool(
            getattr(v, "require_all_years_profitable", False)
        ),
        require_oos_windows_profitable=bool(
            getattr(v, "require_oos_windows_profitable", False)
        ),
        min_year_return=float(getattr(v, "min_year_return", 0.0)),
        min_year_bars=int(getattr(v, "min_year_bars", 2)),
        min_year_pass_fraction=float(getattr(v, "min_year_pass_fraction", 1.0)),
        max_year_loss=float(getattr(v, "max_year_loss", -1.0)),
    )
    objective = str(getattr(v, "objective_metric", "cagr") or "cagr")
    validator = WalkForwardValidator(
        cfg,
        _strategy_cls(strategy_name),
        search_space,
        objective_metric=objective,
        thresholds=thresholds,
    )

    trials = int(n_trials if n_trials is not None else v.n_trials)
    timeout = int(v.timeout_seconds)
    ppy = int(v.periods_per_year)

    reports: Dict[str, ModeValidationReport] = {}
    for mode in modes:
        mode_l = mode.strip().lower()
        logger.info("Running %s WFO (%d trials) …", mode_l, trials)
        if mode_l == "anchored":
            report = validator.run_mode(
                bar_map,
                "anchored",
                n_trials=trials,
                timeout=timeout,
                min_is_bars=years_to_bars(v.anchored_min_is_years, ppy),
                oos_bars=years_to_bars(v.oos_years, ppy),
                step_bars=years_to_bars(v.step_years, ppy),
            )
        elif mode_l == "rolling":
            report = validator.run_mode(
                bar_map,
                "rolling",
                n_trials=trials,
                timeout=timeout,
                is_bars=years_to_bars(v.rolling_is_years, ppy),
                oos_bars=years_to_bars(v.rolling_oos_years, ppy),
                step_bars=years_to_bars(v.step_years, ppy),
                purge_bars=int(v.purge_bars),
            )
        else:
            raise ValueError(f"Unsupported mode {mode!r}")
        reports[mode_l] = report
        logger.info(
            "%s: windows=%d passed=%s failures=%s stitched=%s",
            mode_l,
            len(report.wfo.windows),
            report.gates.passed,
            report.gates.failures,
            report.stitched_metrics,
        )

    overall_pass = all(r.gates.passed for r in reports.values()) and len(reports) > 0
    payload: Dict[str, Any] = {
        "strategy": strategy_name,
        "capital": float(v.capital),
        "start_date": v.start_date,
        "data": meta,
        "objective_metric": str(getattr(v, "objective_metric", "cagr")),
        "risk": {
            "max_position_size_pct": float(cfg.risk.max_position_size_pct),
            "max_position_size_uncapped": float(cfg.risk.max_position_size_pct) <= 0.0,
            "max_dd_limit": float(v.max_dd_limit),
            "max_daily_drawdown_pct": float(cfg.risk.max_daily_drawdown_pct),
            "halt_on_breach": bool(cfg.risk.halt_on_breach),
            "require_all_years_profitable": bool(
                getattr(v, "require_all_years_profitable", False)
            ),
            "require_oos_windows_profitable": bool(
                getattr(v, "require_oos_windows_profitable", False)
            ),
            "min_year_return": float(getattr(v, "min_year_return", 0.0)),
            "min_year_pass_fraction": float(
                getattr(v, "min_year_pass_fraction", 1.0)
            ),
            "max_year_loss": float(getattr(v, "max_year_loss", -1.0)),
        },
        "overall_pass": overall_pass,
        "modes": {k: r.to_dict() for k, r in reports.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cl_wfo_validation.json"
    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    md_path = output_dir / "CL_WFO_REPORT.md"
    md_path.write_text(_format_markdown(payload), encoding="utf-8")
    logger.info("Wrote %s and %s", json_path, md_path)
    return payload


def _format_markdown(payload: Dict[str, Any]) -> str:
    risk = payload.get("risk") or {}
    size_pct = risk.get("max_position_size_pct")
    size_note = (
        "uncapped (ATR/Kelly only)"
        if risk.get("max_position_size_uncapped")
        else f"{size_pct:.0%} equity notional cap"
    )
    year_gate = (
        f"≥{float(risk.get('min_year_pass_fraction', 1.0)):.0%} non-losing years; "
        f"worst year ≥ {float(risk.get('max_year_loss', -1.0)):.0%}"
        if risk.get("require_all_years_profitable")
        else "reported (not gated)"
    )
    oos_gate = (
        "required (each OOS fold ≥ 0)"
        if risk.get("require_oos_windows_profitable")
        else "not required"
    )
    lines = [
        "# CL Walk-Forward Validation Report",
        "",
        f"- Strategy: `{payload['strategy']}`",
        f"- Capital: ${payload['capital']:,.0f}",
        f"- Start: {payload['start_date']}",
        f"- Data: {payload.get('data')}",
        f"- Optuna objective: `{payload.get('objective_metric', 'sharpe_ratio')}`",
        f"- Position sizing: {size_note}",
        f"- System MaxDD gate: < {float(risk.get('max_dd_limit', 0.30)):.0%}",
        f"- Calendar-year returns: {year_gate}",
        f"- OOS-fold profitability: {oos_gate}",
        f"- **Overall pass:** {payload['overall_pass']}",
        "",
    ]
    for mode, report in payload["modes"].items():
        gates = report["gates"]
        m = report.get("stitched_metrics") or {}
        year_rets = report.get("calendar_year_returns") or gates.get(
            "calendar_year_returns"
        ) or {}
        year_line = ", ".join(
            f"{y}: {float(r):+.2%}" for y, r in sorted(year_rets.items(), key=lambda kv: int(kv[0]))
        ) or "—"
        lines.extend(
            [
                f"## {mode.title()} WFO",
                "",
                f"- Windows: {report['n_windows']}",
                f"- Gates passed: {gates['passed']}",
                f"- Failures: {gates.get('failures') or '—'}",
                f"- Sharpe: {m.get('sharpe_ratio')}",
                f"- UPI: {m.get('ulcer_performance_index')}",
                f"- CAGR: {m.get('cagr')}",
                f"- Total return: {m.get('total_return')}",
                f"- Max DD: {m.get('max_drawdown')}",
                f"- Final equity: {report.get('stitched_equity_final')}",
                f"- Calendar-year returns: {year_line}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default=None, help="CL strategy class name")
    p.add_argument("--start", default=None, help="ISO start date (YYYY-MM-DD)")
    p.add_argument("--capital", type=float, default=None, help="Account size USD")
    p.add_argument(
        "--modes",
        default="anchored,rolling",
        help="Comma-separated: anchored,rolling",
    )
    p.add_argument("--trials", type=int, default=None, help="Optuna trials per IS window")
    p.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config",
    )
    p.add_argument(
        "--output-dir",
        default="docs",
        help="Directory for JSON + markdown report",
    )
    p.add_argument(
        "--artifacts-dir",
        default="/opt/cursor/artifacts",
        help="Also copy report artifacts here when writable",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic CL bars (no network)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = Config.from_yaml(args.config)
    v = cfg.cl_validation
    if args.strategy:
        v.strategy = args.strategy
    if args.start:
        v.start_date = args.start
    if args.capital is not None:
        v.capital = float(args.capital)
    if args.trials is not None:
        v.n_trials = int(args.trials)

    if v.strategy not in CL_STRATEGIES:
        logger.error("Strategy must be one of %s", sorted(CL_STRATEGIES))
        return 2

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out = Path(args.output_dir)
    payload = run_validation(
        cfg,
        v.strategy,
        modes,
        synthetic=bool(args.synthetic),
        n_trials=args.trials,
        output_dir=out,
    )

    artifacts = Path(args.artifacts_dir)
    try:
        artifacts.mkdir(parents=True, exist_ok=True)
        for name in ("cl_wfo_validation.json", "CL_WFO_REPORT.md"):
            src = out / name
            if src.exists():
                (artifacts / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write artifacts to %s: %s", artifacts, exc)

    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
