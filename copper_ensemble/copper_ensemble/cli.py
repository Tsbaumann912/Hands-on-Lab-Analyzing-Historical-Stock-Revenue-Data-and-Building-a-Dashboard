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
    select_pre_specified_candidates,
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
        "mean_oos_upi": report.mean_oos_upi,
        "mean_anchored_oos_upi": report.mean_anchored_oos_upi,
        "composite_oos_upi": report.composite_oos_upi,
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
                "scheme": w.scheme,
                "is_sharpe": w.is_sharpe,
                "oos_sharpe": w.oos_sharpe,
                "is_upi": w.is_upi,
                "oos_upi": w.oos_upi,
                "oos_ulcer_index": w.oos_ulcer_index,
                "oos_max_dd": w.oos_max_dd,
            }
            for w in report.walk_forward
        ],
        "walk_forward_anchored": [
            {
                "window": w.window,
                "scheme": w.scheme,
                "is_sharpe": w.is_sharpe,
                "oos_sharpe": w.oos_sharpe,
                "is_upi": w.is_upi,
                "oos_upi": w.oos_upi,
                "oos_ulcer_index": w.oos_ulcer_index,
                "oos_max_dd": w.oos_max_dd,
            }
            for w in report.walk_forward_anchored
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
    """Nested purged WFA Optuna or discrete candidate selection (anti-overfit)."""
    cfg, bars = _load_bars(args)
    mode = getattr(args, "mode", "candidates")

    if mode == "candidates":
        winner, ranked = select_pre_specified_candidates(
            cfg,
            bars,
            n_windows=args.windows,
            metric=getattr(args, "metric", "upi"),
        )
        payload: Dict[str, Any] = {
            "mode": "candidates",
            "metric": getattr(args, "metric", "upi"),
            "winner": winner.name,
            "winner_composite_oos_upi": winner.composite_oos_upi,
            "winner_rolling_mean_oos_upi": winner.rolling_mean_oos_upi,
            "winner_anchored_mean_oos_upi": winner.anchored_mean_oos_upi,
            "winner_min_scheme_oos_upi": winner.min_scheme_oos_upi,
            "winner_full_upi": winner.full_upi,
            "winner_mean_oos_sharpe": winner.mean_oos_sharpe,
            "winner_median_oos_sharpe": winner.median_oos_sharpe,
            "winner_full": winner.full,
            "winner_holdout_sharpe": winner.holdout_sharpe,
            "winner_holdout_upi": winner.holdout_upi,
            "winner_ensemble": winner.ensemble,
            "candidates": [
                {
                    "name": r.name,
                    "composite_oos_upi": r.composite_oos_upi,
                    "rolling_mean_oos_upi": r.rolling_mean_oos_upi,
                    "anchored_mean_oos_upi": r.anchored_mean_oos_upi,
                    "min_scheme_oos_upi": r.min_scheme_oos_upi,
                    "full_upi": r.full_upi,
                    "full_ulcer_index": r.full.get("ulcer_index"),
                    "full_sharpe": r.full.get("sharpe"),
                    "full_return": r.full.get("total_return"),
                    "n_fills": r.full.get("n_fills"),
                    "mean_oos_sharpe": r.mean_oos_sharpe,
                    "holdout_upi": r.holdout_upi,
                    "holdout_sharpe": r.holdout_sharpe,
                    "rolling_oos_upis": r.rolling_oos_upis,
                    "anchored_oos_upis": r.anchored_oos_upis,
                }
                for r in ranked
            ],
            "notes": [
                "Candidates are pre-specified economic variants (not free Optuna).",
                "Default ranking uses Ulcer Performance Index on rolling + anchored purged OOS.",
                "composite_oos_upi = 0.5*rolling_mean + 0.5*anchored_mean; "
                "tie-break with min(scheme) for robustness.",
            ],
            "n_bars": len(bars),
            "date_start": str(bars[0].timestamp) if bars else None,
            "date_end": str(bars[-1].timestamp) if bars else None,
        }
        if args.apply:
            from copper_ensemble.models import clone_config

            config_path = Path(args.config)
            with config_path.open() as fh:
                raw = yaml.safe_load(fh) or {}
            raw["ensemble"] = {**raw.get("ensemble", {}), **winner.ensemble}
            with config_path.open("w") as fh:
                yaml.safe_dump(raw, fh, sort_keys=False, default_flow_style=False)
            logger.info("Applied winner %s to %s", winner.name, config_path)
            payload["applied_config"] = str(config_path)
            from copper_ensemble.validation import validate_ensemble as _val

            v = _val(clone_config(cfg, ensemble_overrides=winner.ensemble), bars)
            payload["post_apply_validation"] = {
                "full": v.full,
                "mean_oos_sharpe": v.mean_oos_sharpe,
                "mean_oos_upi": v.mean_oos_upi,
                "mean_anchored_oos_upi": v.mean_anchored_oos_upi,
                "composite_oos_upi": v.composite_oos_upi,
                "mean_is_sharpe": v.mean_is_sharpe,
                "passed": v.passed,
                "notes": v.notes[:8],
                "walk_forward": [
                    {
                        "window": w.window,
                        "scheme": w.scheme,
                        "is_sharpe": w.is_sharpe,
                        "oos_sharpe": w.oos_sharpe,
                        "oos_upi": w.oos_upi,
                        "oos_max_dd": w.oos_max_dd,
                    }
                    for w in v.walk_forward
                ],
                "walk_forward_anchored": [
                    {
                        "window": w.window,
                        "oos_upi": w.oos_upi,
                        "oos_sharpe": w.oos_sharpe,
                    }
                    for w in v.walk_forward_anchored
                ],
            }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    # mode == optuna
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
    payload["mode"] = "optuna"
    payload["n_bars"] = len(bars)
    payload["date_start"] = str(bars[0].timestamp) if bars else None
    payload["date_end"] = str(bars[-1].timestamp) if bars else None

    if args.apply and report.robust_params:
        config_path = Path(args.config)
        with config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}
        ens = robust_params_to_yaml_ensemble(cfg, report.robust_params)
        raw["ensemble"] = {**raw.get("ensemble", {}), **ens}
        raw["ensemble"]["weights"] = ens["weights"]
        with config_path.open("w") as fh:
            yaml.safe_dump(raw, fh, sort_keys=False, default_flow_style=False)
        logger.info("Applied robust ensemble params to %s", config_path)
        payload["applied_config"] = str(config_path)
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
        help="Anti-overfit optimize: discrete candidates (default) or nested Optuna",
    )
    _add_data_args(opt)
    opt.add_argument(
        "--mode",
        choices=["candidates", "optuna"],
        default="candidates",
        help="candidates=fixed economic menu (recommended); optuna=IS-only nested WFA",
    )
    opt.add_argument(
        "--optuna-config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "optuna.yaml"),
        help="Search-space YAML for --mode optuna",
    )
    opt.add_argument("--trials", type=int, default=None, help="Override trials per window")
    opt.add_argument("--windows", type=int, default=None, help="Override WFA window count")
    opt.add_argument(
        "--metric",
        choices=["upi", "sharpe"],
        default="upi",
        help="Candidate ranking metric (default: Ulcer Performance Index dual WFA)",
    )
    opt.add_argument(
        "--apply",
        action="store_true",
        help="Write selected ensemble into --config",
    )
    opt.set_defaults(func=cmd_optimize)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
