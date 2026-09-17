"""Shared models and configuration for the standalone silver ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True)
class Bar:
    """OHLCV bar plus optional curve / inventory / macro / gold features."""

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
    gold_close: Optional[float] = None
    real_yield_chg: Optional[float] = None
    usd_ret_20d: Optional[float] = None


@dataclass
class Signal:
    """Trade intent produced by the ensemble."""

    symbol: str
    direction: Direction
    strength: float
    timestamp: Any
    strategy_name: str = "SilverEnsemble"
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
    ratio_lookback: int
    atr_period: int
    stop_atr_mult: float
    take_profit_atr_mult: float
    # StochRSI + MA sleeve
    rsi_period: int
    stoch_rsi_period: int
    stoch_rsi_smooth: int
    ma_fast: int
    ma_slow: int
    stoch_oversold: float
    stoch_overbought: float


@dataclass(frozen=True)
class RiskConfig:
    max_daily_drawdown_pct: float
    max_position_size_pct: float
    max_leverage: float
    max_contracts: int
    halt_on_breach: bool
    ytd_loss_halt_pct: float = 0.01


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float
    collateral_yield_annual: float = 0.02


@dataclass(frozen=True)
class BacktestConfig:
    walk_forward_windows: int
    in_sample_ratio: float
    purge_bars: int
    data_start: str
    is_years: int
    oos_years: int
    step_years: int


@dataclass(frozen=True)
class ValidationConfig:
    dsr_pass_threshold: float
    oos_retention_min: float
    max_drawdown_gate: float
    min_oos_sharpe: float
    min_oos_cagr: float
    min_oos_upi: float
    require_both_modes: bool


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
            ratio_lookback=int(e.get("ratio_lookback", 63)),
            atr_period=int(e["atr_period"]),
            stop_atr_mult=float(e["stop_atr_mult"]),
            take_profit_atr_mult=float(e["take_profit_atr_mult"]),
            rsi_period=int(e.get("rsi_period", 14)),
            stoch_rsi_period=int(e.get("stoch_rsi_period", 14)),
            stoch_rsi_smooth=int(e.get("stoch_rsi_smooth", 3)),
            ma_fast=int(e.get("ma_fast", 10)),
            ma_slow=int(e.get("ma_slow", 40)),
            stoch_oversold=float(e.get("stoch_oversold", 0.20)),
            stoch_overbought=float(e.get("stoch_overbought", 0.80)),
        ),
        risk=RiskConfig(
            max_daily_drawdown_pct=float(r["max_daily_drawdown_pct"]),
            max_position_size_pct=float(r["max_position_size_pct"]),
            max_leverage=float(r["max_leverage"]),
            max_contracts=int(r["max_contracts"]),
            halt_on_breach=bool(r["halt_on_breach"]),
            ytd_loss_halt_pct=float(r.get("ytd_loss_halt_pct", 0.01)),
        ),
        portfolio=PortfolioConfig(
            initial_cash=float(p["initial_cash"]),
            collateral_yield_annual=float(p.get("collateral_yield_annual", 0.02)),
        ),
        backtest=BacktestConfig(
            walk_forward_windows=int(b.get("walk_forward_windows", 4)),
            in_sample_ratio=float(b.get("in_sample_ratio", 0.70)),
            purge_bars=int(b.get("purge_bars", 5)),
            data_start=str(b.get("data_start", "2008-01-01")),
            is_years=int(b.get("is_years", 3)),
            oos_years=int(b.get("oos_years", 1)),
            step_years=int(b.get("step_years", 1)),
        ),
        validation=ValidationConfig(
            dsr_pass_threshold=float(v.get("dsr_pass_threshold", 0.95)),
            oos_retention_min=float(v.get("oos_retention_min", 0.60)),
            max_drawdown_gate=float(v.get("max_drawdown_gate", 0.30)),
            min_oos_sharpe=float(v.get("min_oos_sharpe", 0.0)),
            min_oos_cagr=float(v.get("min_oos_cagr", 0.0)),
            min_oos_upi=float(v.get("min_oos_upi", 0.0)),
            require_both_modes=bool(v.get("require_both_modes", True)),
        ),
        raw=raw,
    )
