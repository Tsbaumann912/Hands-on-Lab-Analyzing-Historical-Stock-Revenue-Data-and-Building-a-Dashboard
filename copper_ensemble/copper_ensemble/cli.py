"""CLI for the standalone copper ensemble project."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from copper_ensemble.data import dataframe_to_bars, load_yfinance_hg, make_synthetic_hg
from copper_ensemble.engine import BacktestEngine
from copper_ensemble.models import load_config
from copper_ensemble.optimize import (
    nested_walk_forward_optimize,
    params_to_config,
    robust_params_to_yaml_ensemble,
)
from copper_ensemble.validation import validate_ensemble

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("copper_ensemble")


def _load_bars(args: argparse.Namespace):
    cfg = load_config(args.config)
    if args.synthetic:
        df = make_synthetic_hg(n_days=args.days, seed=args.seed)
    else:
        start = getattr(args, "start", None)
        end = getattr(args, "end", None)
        period = getattr(args, "period", None)
        # Default market validate/backtest without period/start → 2008→now
        if start is None and (period is None or period == ""):
            start = "2008-01-01"
            period = None
        df = load_yfinance_hg(
            cfg.contract.yfinance_ticker,
            period=None if start else (period or "5y"),
            start=start,
            end=end,
        )
        logger.info(
            "loaded HG market bars=%s start=%s end=%s",
            len(df),
            df.index.min(),
            df.index.max(),
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
        "mean_oos_sharpe": report.mean_oos_sharpe,
        "mean_is_sharpe": report.mean_is_sharpe,
        "target_oos_met": report.target_oos_met,
        "passed": report.passed,
        "notes": report.notes,
        "n_bars": len(bars),
        "date_start": str(bars[0].timestamp) if bars else None,
        "date_end": str(bars[-1].timestamp) if bars else None,
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


def _report_to_dict(report: Any) -> Dict[str, Any]:
    return {
        "baseline_mean_oos_sharpe": report.baseline_mean_oos_sharpe,
        "mean_oos_sharpe": report.mean_oos_sharpe,
        "median_oos_sharpe": report.median_oos_sharpe,
        "mean_is_sharpe": report.mean_is_sharpe,
        "robust_params": report.robust_params,
        "robust_full_metrics": report.robust_full_metrics,
        "robust_wfa": report.robust_wfa,
        "holdout": report.holdout,
        "windows": [
            {
                "window": w.window,
                "is_sharpe": w.is_sharpe,
                "oos_sharpe": w.oos_sharpe,
                "oos_max_dd": w.oos_max_dd,
                "oos_total_return": w.oos_total_return,
                "best_params": w.best_params,
            }
            for w in report.windows
        ],
        "notes": report.notes,
        "search_space_path": report.search_space_path,
    }


def cmd_optimize(args: argparse.Namespace) -> int:
    """Nested purged WFA Optuna — IS-only tuning, OOS evaluation, median robust params."""
    cfg, bars = _load_bars(args)
    space_path = args.optuna_config
    report = nested_walk_forward_optimize(
        cfg,
        bars,
        space_path=space_path,
        n_trials=args.trials,
        n_windows=args.windows,
        seed=args.seed,
    )
    payload = _report_to_dict(report)
    payload["n_bars"] = len(bars)
    payload["date_start"] = str(bars[0].timestamp) if bars else None
    payload["date_end"] = str(bars[-1].timestamp) if bars else None

    if args.apply and report.robust_params:
        config_path = Path(args.config)
        with config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}
        ens = robust_params_to_yaml_ensemble(cfg, report.robust_params)
        # Preserve comments-free merge for ensemble block
        raw["ensemble"] = {**raw.get("ensemble", {}), **ens}
        # Keep explanatory keys
        raw["ensemble"]["weights"] = ens["weights"]
        with config_path.open("w") as fh:
            yaml.safe_dump(raw, fh, sort_keys=False, default_flow_style=False)
        logger.info("Applied robust ensemble params to %s", config_path)
        payload["applied_config"] = str(config_path)
        # Quick post-apply validate metrics under new config
        new_cfg = params_to_config(cfg, report.robust_params)
        from copper_ensemble.validation import validate_ensemble as _val

        v = _val(new_cfg, bars)
        payload["post_apply_validation"] = {
            "full": v.full,
            "mean_oos_sharpe": v.mean_oos_sharpe,
            "mean_is_sharpe": v.mean_is_sharpe,
            "passed": v.passed,
            "notes": v.notes[:6],
            "walk_forward": [
                {
                    "window": w.window,
                    "is_sharpe": w.is_sharpe,
                    "oos_sharpe": w.oos_sharpe,
                    "oos_max_dd": w.oos_max_dd,
                }
                for w in v.walk_forward
            ],
        }

    print(json.dumps(payload, indent=2, default=str))
    # Non-zero only if optimization produced no windows
    return 0 if report.windows else 1


def _add_data_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--days", type=int, default=1500)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument(
        "--period",
        default=None,
        help="Yahoo period (e.g. 5y, max). Ignored when --start is set.",
    )
    sp.add_argument(
        "--start",
        default=None,
        help="Yahoo start date YYYY-MM-DD (default for market mode: 2008-01-01).",
    )
    sp.add_argument("--end", default=None, help="Yahoo end date YYYY-MM-DD (default: today).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="copper_ensemble", description="Standalone HG copper ensemble CTA")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "default.yaml"),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Run ensemble backtest")
    _add_data_args(bt)
    bt.add_argument("--plot-summary", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    val = sub.add_parser("validate", help="Ablation + WFA + DSR report")
    _add_data_args(val)
    val.set_defaults(func=cmd_validate)

    opt = sub.add_parser(
        "optimize",
        help="Nested purged WFA Optuna (IS-only tune; OOS evaluate; no holdout peeking)",
    )
    _add_data_args(opt)
    opt.add_argument(
        "--optuna-config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "optuna.yaml"),
        help="Search-space YAML (bounds only; never hardcode in code)",
    )
    opt.add_argument("--trials", type=int, default=None, help="Override trials per window")
    opt.add_argument("--windows", type=int, default=None, help="Override WFA window count")
    opt.add_argument(
        "--apply",
        action="store_true",
        help="Write median IS-selected robust params into --config (not OOS-cherry-picked)",
    )
    opt.set_defaults(func=cmd_optimize)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
