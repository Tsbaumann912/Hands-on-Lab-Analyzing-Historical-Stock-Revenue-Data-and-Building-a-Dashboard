"""CLI for the standalone silver ensemble project."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from silver_ensemble.data import dataframe_to_bars, load_yfinance_si, make_synthetic_si
from silver_ensemble.engine import BacktestEngine
from silver_ensemble.models import load_config
from silver_ensemble.validation import (
    institutional_report_to_dict,
    run_institutional_wfo,
    validate_ensemble,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("silver_ensemble")


def _load_bars(args: argparse.Namespace):
    cfg = load_config(args.config)
    if getattr(args, "synthetic", False):
        df = make_synthetic_si(n_days=args.days, seed=args.seed)
    else:
        start = getattr(args, "start", None)
        end = getattr(args, "end", None)
        if start is None and hasattr(cfg.backtest, "data_start"):
            # Prefer config data_start for institutional runs when flag omitted
            start = None
        period = getattr(args, "period", "5y")
        df = load_yfinance_si(
            cfg.contract.yfinance_ticker,
            period=period,
            start=start,
            end=end,
        )
    bars = dataframe_to_bars(df, symbol=cfg.contract.symbol)
    return cfg, bars


def _load_bars_wfo(args: argparse.Namespace):
    """Load bars for institutional WFO — default start from config (2008-01-01)."""
    cfg = load_config(args.config)
    if args.synthetic:
        # Need ≥ is_years + several oos years of synthetic history
        n = max(args.days, (cfg.backtest.is_years + 8) * 252)
        df = make_synthetic_si(n_days=n, seed=args.seed)
    else:
        start = args.start or cfg.backtest.data_start
        df = load_yfinance_si(
            cfg.contract.yfinance_ticker,
            period=args.period,
            start=start,
            end=args.end,
        )
    bars = dataframe_to_bars(df, symbol=cfg.contract.symbol)
    return cfg, bars


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg, bars = _load_bars(args)
    result = BacktestEngine(cfg).run(bars)
    print(json.dumps(result.metrics, indent=2))
    if args.plot_summary:
        eq = result.equity_curve
        logger.info(
            "equity %.2f → %.2f | fills=%s | final_pos=%.0f",
            eq[0],
            eq[-1],
            len(result.fills),
            result.positions[-1],
        )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg, bars = _load_bars(args)
    report = validate_ensemble(cfg, bars)
    payload = {
        "full": report.full,
        "deflated_sharpe": report.deflated_sharpe,
        "oos_retention": report.oos_retention,
        "passed": report.passed,
        "notes": report.notes,
        "ablations": report.ablations,
        "walk_forward": [
            {
                "window": w.window,
                "is_sharpe": w.is_sharpe,
                "oos_sharpe": w.oos_sharpe,
                "oos_max_dd": w.oos_max_dd,
            }
            for w in report.walk_forward
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0 if report.passed else 1


def cmd_validate_wfo(args: argparse.Namespace) -> int:
    cfg, bars = _load_bars_wfo(args)
    logger.info(
        "Institutional WFO: account=$%.0f bars=%d start=%s",
        cfg.portfolio.initial_cash,
        len(bars),
        getattr(args, "start", None) or cfg.backtest.data_start,
    )
    report = run_institutional_wfo(cfg, bars)
    payload = institutional_report_to_dict(report)
    print(json.dumps(payload, indent=2))
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="silver_ensemble", description="Standalone SI silver ensemble CTA")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "default.yaml"),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Run ensemble backtest")
    bt.add_argument("--synthetic", action="store_true")
    bt.add_argument("--days", type=int, default=1500)
    bt.add_argument("--seed", type=int, default=42)
    bt.add_argument("--period", default="5y")
    bt.add_argument("--start", default=None, help="YYYY-MM-DD (overrides period)")
    bt.add_argument("--end", default=None)
    bt.add_argument("--plot-summary", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    val = sub.add_parser("validate", help="Ablation + legacy block WFA + DSR report")
    val.add_argument("--synthetic", action="store_true")
    val.add_argument("--days", type=int, default=1500)
    val.add_argument("--seed", type=int, default=42)
    val.add_argument("--period", default="5y")
    val.add_argument("--start", default=None)
    val.add_argument("--end", default=None)
    val.set_defaults(func=cmd_validate)

    wfo = sub.add_parser(
        "validate-wfo",
        help="Anchored + rolling institutional WFO (Sharpe/UPI/CAGR/maxDD<30%)",
    )
    wfo.add_argument("--synthetic", action="store_true")
    wfo.add_argument("--days", type=int, default=3000, help="Synthetic length (auto-raised if short)")
    wfo.add_argument("--seed", type=int, default=42)
    wfo.add_argument("--period", default="max", help="Used only when --start is omitted")
    wfo.add_argument(
        "--start",
        default=None,
        help="YYYY-MM-DD (default: config backtest.data_start = 2008-01-01)",
    )
    wfo.add_argument("--end", default=None)
    wfo.set_defaults(func=cmd_validate_wfo)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
