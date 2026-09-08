"""Shared models and configuration for the standalone copper ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import yaml


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True)
class Bar:
    """OHLCV bar plus optional curve / inventory / macro features."""

    symbol: str
    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float
    near_price: Optional[float] = None
    next_price: Optional[float] = None
    inventory: Optional[float] = None
    china_pmi: Optional[float] = None
    usd_ret_20d: Optional[float] = None


@dataclass
class Signal:
    """Trade intent produced by the ensemble."""

    symbol: str
    direction: Direction
    strength: float
    timestamp: Any
    strategy_name: str = "CopperEnsemble"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    suggested_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")


@dataclass(frozen=True)
class ContractConfig:
    symbol: str
    name: str
    yfinance_ticker: str
    multiplier: float
    tick_size: float
    tick_value: float
    commission_per_contract: float
    slippage_ticks: int


@dataclass(frozen=True)
class EnsembleConfig:
    horizons_days: List[int]
    weights: Mapping[str, float]
    forecast_cap: float
    agreement_min: float
    fdm_cap: float
    vol_target_annual: float
    ewma_vol_com_days: int
    fade_z: float
    buffer_forecast: float
    kelly_fraction: float
    inventory_sma: int
    inventory_delta_days: int
    basis_mom_lookback: int
    price_z_lookback: int
    atr_period: int
    stop_atr_mult: float
    take_profit_atr_mult: float


@dataclass(frozen=True)
class RiskConfig:
    max_daily_drawdown_pct: float
    max_position_size_pct: float
    max_leverage: float
    max_contracts: int
    halt_on_breach: bool
    halt_cooldown_bars: int = 21


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float


@dataclass(frozen=True)
class BacktestConfig:
    walk_forward_windows: int
    in_sample_ratio: float
    purge_bars: int
    walk_forward_windows_long: int = 8
    long_history_bars: int = 3000


@dataclass(frozen=True)
class ValidationConfig:
    dsr_pass_threshold: float
    oos_retention_min: float
    target_mean_oos_sharpe: float = 1.5


@dataclass(frozen=True)
class Config:
    contract: ContractConfig
    ensemble: EnsembleConfig
    risk: RiskConfig
    portfolio: PortfolioConfig
    backtest: BacktestConfig
    validation: ValidationConfig
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


def _mapping(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    val = d.get(key, {})
    if not isinstance(val, dict):
        raise TypeError(f"config key {key!r} must be a mapping")
    return val


def rebuild_weights(
    base: Mapping[str, float],
    tsmom_weight: float,
    enable_fade: bool,
) -> Dict[str, float]:
    """
    Rebuild sleeve weights with a target TSMOM share.

    Remaining mass is split across non-TSMOM sleeves in proportion to ``base``
    (fade may be zeroed and its mass redistributed).
    """
    tsmom_w = float(np.clip(tsmom_weight, 0.05, 0.95))
    others = {k: float(v) for k, v in base.items() if k != "tsmom"}
    if not enable_fade:
        others["fade"] = 0.0
    other_sum = float(sum(others.values()))
    residual = max(1.0 - tsmom_w, 0.0)
    if other_sum <= 1e-12:
        # fallback equal split among non-fade sleeves
        keys = [k for k in others if k != "fade" or enable_fade]
        if not keys:
            keys = list(others.keys())
        each = residual / max(len(keys), 1)
        out = {k: (each if k in keys else 0.0) for k in others}
    else:
        out = {k: residual * (v / other_sum) for k, v in others.items()}
    out["tsmom"] = tsmom_w
    # numerical renorm
    s = float(sum(out.values()))
    if s <= 0:
        raise ValueError("weights must sum positive")
    return {k: float(v) / s for k, v in out.items()}


def clone_config(
    cfg: Config,
    *,
    ensemble_overrides: Optional[Mapping[str, Any]] = None,
    risk_overrides: Optional[Mapping[str, Any]] = None,
) -> Config:
    """Return a new ``Config`` with selected frozen fields replaced."""
    e = cfg.ensemble
    eo = dict(ensemble_overrides or {})
    new_e = EnsembleConfig(
        horizons_days=list(eo.get("horizons_days", e.horizons_days)),
        weights=dict(eo.get("weights", e.weights)),
        forecast_cap=float(eo.get("forecast_cap", e.forecast_cap)),
        agreement_min=float(eo.get("agreement_min", e.agreement_min)),
        fdm_cap=float(eo.get("fdm_cap", e.fdm_cap)),
        vol_target_annual=float(eo.get("vol_target_annual", e.vol_target_annual)),
        ewma_vol_com_days=int(eo.get("ewma_vol_com_days", e.ewma_vol_com_days)),
        fade_z=float(eo.get("fade_z", e.fade_z)),
        buffer_forecast=float(eo.get("buffer_forecast", e.buffer_forecast)),
        kelly_fraction=float(eo.get("kelly_fraction", e.kelly_fraction)),
        inventory_sma=int(eo.get("inventory_sma", e.inventory_sma)),
        inventory_delta_days=int(eo.get("inventory_delta_days", e.inventory_delta_days)),
        basis_mom_lookback=int(eo.get("basis_mom_lookback", e.basis_mom_lookback)),
        price_z_lookback=int(eo.get("price_z_lookback", e.price_z_lookback)),
        atr_period=int(eo.get("atr_period", e.atr_period)),
        stop_atr_mult=float(eo.get("stop_atr_mult", e.stop_atr_mult)),
        take_profit_atr_mult=float(eo.get("take_profit_atr_mult", e.take_profit_atr_mult)),
    )
    r = cfg.risk
    ro = dict(risk_overrides or {})
    new_r = RiskConfig(
        max_daily_drawdown_pct=float(ro.get("max_daily_drawdown_pct", r.max_daily_drawdown_pct)),
        max_position_size_pct=float(ro.get("max_position_size_pct", r.max_position_size_pct)),
        max_leverage=float(ro.get("max_leverage", r.max_leverage)),
        max_contracts=int(ro.get("max_contracts", r.max_contracts)),
        halt_on_breach=bool(ro.get("halt_on_breach", r.halt_on_breach)),
        halt_cooldown_bars=int(ro.get("halt_cooldown_bars", r.halt_cooldown_bars)),
    )
    return Config(
        contract=cfg.contract,
        ensemble=new_e,
        risk=new_r,
        portfolio=cfg.portfolio,
        backtest=cfg.backtest,
        validation=cfg.validation,
        raw=cfg.raw,
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load YAML config; defaults to ``config/default.yaml`` next to the package root."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    c = _mapping(raw, "contract")
    e = _mapping(raw, "ensemble")
    r = _mapping(raw, "risk")
    p = _mapping(raw, "portfolio")
    b = _mapping(raw, "backtest")
    v = _mapping(raw, "validation")

    return Config(
        contract=ContractConfig(
            symbol=str(c["symbol"]),
            name=str(c["name"]),
            yfinance_ticker=str(c["yfinance_ticker"]),
            multiplier=float(c["multiplier"]),
            tick_size=float(c["tick_size"]),
            tick_value=float(c["tick_value"]),
            commission_per_contract=float(c["commission_per_contract"]),
            slippage_ticks=int(c["slippage_ticks"]),
        ),
        ensemble=EnsembleConfig(
            horizons_days=[int(x) for x in e["horizons_days"]],
            weights={str(k): float(val) for k, val in dict(e["weights"]).items()},
            forecast_cap=float(e["forecast_cap"]),
            agreement_min=float(e["agreement_min"]),
            fdm_cap=float(e["fdm_cap"]),
            vol_target_annual=float(e["vol_target_annual"]),
            ewma_vol_com_days=int(e["ewma_vol_com_days"]),
            fade_z=float(e["fade_z"]),
            buffer_forecast=float(e["buffer_forecast"]),
            kelly_fraction=float(e["kelly_fraction"]),
            inventory_sma=int(e["inventory_sma"]),
            inventory_delta_days=int(e["inventory_delta_days"]),
            basis_mom_lookback=int(e["basis_mom_lookback"]),
            price_z_lookback=int(e["price_z_lookback"]),
            atr_period=int(e["atr_period"]),
            stop_atr_mult=float(e["stop_atr_mult"]),
            take_profit_atr_mult=float(e["take_profit_atr_mult"]),
        ),
        risk=RiskConfig(
            max_daily_drawdown_pct=float(r["max_daily_drawdown_pct"]),
            max_position_size_pct=float(r["max_position_size_pct"]),
            max_leverage=float(r["max_leverage"]),
            max_contracts=int(r["max_contracts"]),
            halt_on_breach=bool(r["halt_on_breach"]),
            halt_cooldown_bars=int(r.get("halt_cooldown_bars", 21)),
        ),
        portfolio=PortfolioConfig(initial_cash=float(p["initial_cash"])),
        backtest=BacktestConfig(
            walk_forward_windows=int(b["walk_forward_windows"]),
            in_sample_ratio=float(b["in_sample_ratio"]),
            purge_bars=int(b["purge_bars"]),
            walk_forward_windows_long=int(b.get("walk_forward_windows_long", 8)),
            long_history_bars=int(b.get("long_history_bars", 3000)),
        ),
        validation=ValidationConfig(
            dsr_pass_threshold=float(v["dsr_pass_threshold"]),
            oos_retention_min=float(v["oos_retention_min"]),
            target_mean_oos_sharpe=float(v.get("target_mean_oos_sharpe", 1.5)),
        ),
        raw=raw,
    )
