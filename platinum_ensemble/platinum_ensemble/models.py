"""Shared models and configuration for the standalone platinum ensemble."""

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
    """OHLCV bar plus optional curve / inventory / gold / macro features."""

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
    gold_ret_20d: Optional[float] = None
    usd_ret_20d: Optional[float] = None


@dataclass
class Signal:
    """Trade intent produced by the ensemble."""

    symbol: str
    direction: Direction
    strength: float
    timestamp: Any
    strategy_name: str = "PlatinumEnsemble"
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
    rv_lookback: int
    rv_entry_z: float
    price_z_lookback: int
    atr_period: int
    stop_atr_mult: float
    take_profit_atr_mult: float
    gold_ticker: str
    short_scale: float = 1.0
    # Flatten for the rest of a calendar year once YTD return hits this level.
    year_profit_lock_pct: float = 0.0
    year_profit_lock_min_days: int = 5
    # Only keep inventory forecasts that agree in sign with the fade sleeve.
    inventory_require_fade_agree: bool = False


@dataclass(frozen=True)
class RiskConfig:
    max_daily_drawdown_pct: float
    max_position_size_pct: float
    max_leverage: float
    max_contracts: int
    halt_on_breach: bool


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float
    # Interest on unencumbered cash (institutional financing proxy).
    cash_interest_annual: float = 0.0
    # Initial margin as a fraction of futures notional for idle-cash calc.
    margin_fraction: float = 0.12


@dataclass(frozen=True)
class BacktestConfig:
    walk_forward_windows: int
    in_sample_ratio: float
    purge_bars: int
    anchored_initial_is_bars: int = 756
    oos_bars: int = 252
    rolling_is_bars: int = 756
    rolling_step_bars: int = 252


@dataclass(frozen=True)
class ValidationConfig:
    dsr_pass_threshold: float
    oos_retention_min: float
    max_oos_drawdown: float = 0.30
    min_oos_sharpe: float = 0.0
    min_oos_cagr: float = 0.0
    min_oos_upi: float = 0.0
    data_start: str = "2008-01-01"

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
            rv_lookback=int(e["rv_lookback"]),
            rv_entry_z=float(e["rv_entry_z"]),
            price_z_lookback=int(e["price_z_lookback"]),
            atr_period=int(e["atr_period"]),
            stop_atr_mult=float(e["stop_atr_mult"]),
            take_profit_atr_mult=float(e["take_profit_atr_mult"]),
            gold_ticker=str(e.get("gold_ticker", "GC=F")),
            short_scale=float(e.get("short_scale", 1.0)),
            year_profit_lock_pct=float(e.get("year_profit_lock_pct", 0.0)),
            year_profit_lock_min_days=int(e.get("year_profit_lock_min_days", 5)),
            inventory_require_fade_agree=bool(e.get("inventory_require_fade_agree", False)),
        ),
        risk=RiskConfig(
            max_daily_drawdown_pct=float(r["max_daily_drawdown_pct"]),
            max_position_size_pct=float(r["max_position_size_pct"]),
            max_leverage=float(r["max_leverage"]),
            max_contracts=int(r["max_contracts"]),
            halt_on_breach=bool(r["halt_on_breach"]),
        ),
        portfolio=PortfolioConfig(
            initial_cash=float(p["initial_cash"]),
            cash_interest_annual=float(p.get("cash_interest_annual", 0.0)),
            margin_fraction=float(p.get("margin_fraction", 0.12)),
        ),
        backtest=BacktestConfig(
            walk_forward_windows=int(b["walk_forward_windows"]),
            in_sample_ratio=float(b["in_sample_ratio"]),
            purge_bars=int(b["purge_bars"]),
            anchored_initial_is_bars=int(b.get("anchored_initial_is_bars", 756)),
            oos_bars=int(b.get("oos_bars", 252)),
            rolling_is_bars=int(b.get("rolling_is_bars", 756)),
            rolling_step_bars=int(b.get("rolling_step_bars", 252)),
        ),
        validation=ValidationConfig(
            dsr_pass_threshold=float(v["dsr_pass_threshold"]),
            oos_retention_min=float(v["oos_retention_min"]),
            max_oos_drawdown=float(v.get("max_oos_drawdown", 0.30)),
            min_oos_sharpe=float(v.get("min_oos_sharpe", 0.0)),
            min_oos_cagr=float(v.get("min_oos_cagr", 0.0)),
            min_oos_upi=float(v.get("min_oos_upi", 0.0)),
            data_start=str(v.get("data_start", "2008-01-01")),
        ),
        raw=raw,
    )
