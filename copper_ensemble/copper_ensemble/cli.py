"""CLI for the standalone copper ensemble project."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from copper_ensemble.data import dataframe_to_bars, load_yfinance_hg, make_synthetic_hg
from copper_ensemble.engine import BacktestEngine
from copper_ensemble.models import load_config
from copper_ensemble.validation import validate_ensemble

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("copper_ensemble")


def _load_bars(args: argparse.Namespace):
    cfg = load_config(args.config)
    if args.synthetic:
        df = make_synthetic_hg(n_days=args.days, seed=args.seed)
    else:
        df = load_yfinance_hg(cfg.contract.yfinance_ticker, period=args.period)
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="copper_ensemble", description="Standalone HG copper ensemble CTA")
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
    bt.add_argument("--plot-summary", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    val = sub.add_parser("validate", help="Ablation + WFA + DSR report")
    val.add_argument("--synthetic", action="store_true")
    val.add_argument("--days", type=int, default=1500)
    val.add_argument("--seed", type=int, default=42)
    val.add_argument("--period", default="5y")
    val.set_defaults(func=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
