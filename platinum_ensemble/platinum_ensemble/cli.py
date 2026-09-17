"""CLI for the standalone platinum ensemble project."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from platinum_ensemble.data import dataframe_to_bars, load_yfinance_pl, make_synthetic_pl
from platinum_ensemble.engine import BacktestEngine
from platinum_ensemble.models import load_config
from platinum_ensemble.validation import (
    institutional_report_to_dict,
    run_institutional_wfo,
    validate_ensemble,
)
from platinum_ensemble.optimize import (
    optimize_result_to_dict,
    run_calendar_optimize,
    write_optimized_yaml,
)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("platinum_ensemble")


def _load_bars(args: argparse.Namespace):
    cfg = load_config(args.config)
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    if getattr(args, "synthetic", False):
        df = make_synthetic_pl(n_days=args.days, seed=args.seed)
    else:
        kwargs = {
            "ticker": cfg.contract.yfinance_ticker,
            "gold_ticker": cfg.ensemble.gold_ticker,
        }
        if start:
            kwargs["start"] = start
            if end:
                kwargs["end"] = end
        else:
            kwargs["period"] = getattr(args, "period", "5y")
        df = load_yfinance_pl(**kwargs)
    bars = dataframe_to_bars(df, symbol=cfg.contract.symbol)
    if getattr(args, "cash", None) is not None:
        cfg = replace(cfg, portfolio=replace(cfg.portfolio, initial_cash=float(args.cash)))
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
                "oos_cagr": w.oos_cagr,
                "oos_upi": w.oos_upi,
            }
            for w in report.walk_forward
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0 if report.passed else 1


def cmd_wfo(args: argparse.Namespace) -> int:
    """Anchored + rolling institutional WFO ($350M default, 2008–now)."""
    cfg = load_config(args.config)
    start = args.start or cfg.validation.data_start
    cash = float(args.cash) if args.cash is not None else cfg.portfolio.initial_cash

    if args.synthetic:
        df = make_synthetic_pl(n_days=args.days, seed=args.seed)
    else:
        df = load_yfinance_pl(
            cfg.contract.yfinance_ticker,
            gold_ticker=cfg.ensemble.gold_ticker,
            start=start,
            end=args.end,
        )
    bars = dataframe_to_bars(df, symbol=cfg.contract.symbol)
    report = run_institutional_wfo(cfg, bars, cash=cash)
    payload = institutional_report_to_dict(report)
    print(json.dumps(payload, indent=2))
    logger.info(
        "WFO overall=%s | anchored=%s rolling=%s | account=$%.0f | bars=%d",
        report.passed,
        report.anchored.passed,
        report.rolling.passed,
        report.account_size,
        report.n_bars,
    )
    return 0 if report.passed else 1


def cmd_optimize(args: argparse.Namespace) -> int:
    """Re-optimise for mean calendar-year return with Max DD < 30%."""
    cfg = load_config(args.config)
    start = args.start or cfg.validation.data_start
    if args.cash is not None:
        cfg = replace(cfg, portfolio=replace(cfg.portfolio, initial_cash=float(args.cash)))

    if args.synthetic:
        df = make_synthetic_pl(n_days=args.days, seed=args.seed)
    else:
        df = load_yfinance_pl(
            cfg.contract.yfinance_ticker,
            gold_ticker=cfg.ensemble.gold_ticker,
            start=start,
            end=args.end,
        )
    bars = dataframe_to_bars(df, symbol=cfg.contract.symbol)
    optuna_path = args.optuna or str(Path(args.config).resolve().parent / "optuna.yaml")
    result = run_calendar_optimize(
        cfg,
        bars,
        optuna_path=optuna_path,
        n_trials=args.trials,
        max_dd_limit=args.max_dd,
    )
    payload = optimize_result_to_dict(result)
    print(json.dumps(payload, indent=2))

    if args.write_config:
        out = Path(args.write_config)
        write_optimized_yaml(Path(args.config), result, out)
        logger.info("Wrote optimised config → %s", out)

    logger.info(
        "optimize mean_yr_IS=%.2f%% maxDD_IS=%.2f%% | mean_yr_OOS=%.2f%% maxDD_OOS=%.2f%% | full mean_yr=%.2f%% maxDD=%.2f%%",
        100.0 * result.is_calendar.mean_return,
        100.0 * float(result.is_metrics.get("max_drawdown", 0.0)),
        100.0 * result.oos_calendar.mean_return,
        100.0 * float(result.oos_metrics.get("max_drawdown", 0.0)),
        100.0 * result.full_calendar.mean_return,
        100.0 * float(result.full_metrics.get("max_drawdown", 0.0)),
    )
    full_ok = float(result.full_metrics.get("max_drawdown", 1.0)) < args.max_dd
    all_green = result.full_calendar.n_negative == 0 and result.full_calendar.n_years > 0
    return 0 if full_ok and all_green and result.full_calendar.mean_return > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="platinum_ensemble", description="Standalone PL platinum ensemble CTA")
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
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--cash", type=float, default=None)
    bt.add_argument("--plot-summary", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    val = sub.add_parser("validate", help="Ablation + WFA + DSR report")
    val.add_argument("--synthetic", action="store_true")
    val.add_argument("--days", type=int, default=1500)
    val.add_argument("--seed", type=int, default=42)
    val.add_argument("--period", default="5y")
    val.add_argument("--start", default=None)
    val.add_argument("--end", default=None)
    val.add_argument("--cash", type=float, default=None)
    val.set_defaults(func=cmd_validate)

    wfo = sub.add_parser(
        "wfo",
        help="Anchored + rolling institutional WFO (Sharpe/UPI/CAGR/MaxDD gates)",
    )
    wfo.add_argument("--synthetic", action="store_true")
    wfo.add_argument("--yfinance", action="store_true", help="Use Yahoo PL=F (default unless --synthetic)")
    wfo.add_argument("--days", type=int, default=2500, help="Synthetic bar count")
    wfo.add_argument("--seed", type=int, default=42)
    wfo.add_argument("--start", default=None, help="Data start YYYY-MM-DD (default from config)")
    wfo.add_argument("--end", default=None)
    wfo.add_argument("--cash", type=float, default=None, help="Account size (default 350e6 from config)")
    wfo.set_defaults(func=cmd_wfo)

    opt = sub.add_parser(
        "optimize",
        help="Re-optimise for mean calendar-year return with Max DD < 30%",
    )
    opt.add_argument("--synthetic", action="store_true")
    opt.add_argument("--days", type=int, default=2500)
    opt.add_argument("--seed", type=int, default=42)
    opt.add_argument("--start", default=None)
    opt.add_argument("--end", default=None)
    opt.add_argument("--cash", type=float, default=None)
    opt.add_argument("--trials", type=int, default=None)
    opt.add_argument("--max-dd", type=float, default=0.30)
    opt.add_argument("--optuna", default=None, help="Path to optuna.yaml")
    opt.add_argument(
        "--write-config",
        default=None,
        help="Write best params into this YAML path (e.g. config/default.yaml)",
    )
    opt.set_defaults(func=cmd_optimize)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
