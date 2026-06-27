"""Hierarchical configuration loader (YAML → dataclass) for the terminal."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    provider: str = "databento"
    dataset: str = "GLBX.MDP3"            # CME Globex MDP3
    schema: str = "ohlcv-1m"
    symbols: List[str] = field(default_factory=lambda: ["ES.c.0", "NQ.c.0"])
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    cache_dir: str = "data/cache"
    databento_api_key: str = field(default_factory=lambda: os.getenv("DATABENTO_API_KEY", ""))
    cme_api_key: str = field(default_factory=lambda: os.getenv("CME_API_KEY", ""))
    websocket_url: str = "wss://ws.databento.com/v0"


@dataclass
class IndicatorConfig:
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    sma_short: int = 20
    sma_long: int = 50
    atr_period: int = 14
    volume_ma_period: int = 20


@dataclass
class StrategyConfig:
    name: str = "MeanReversionRSI"
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    entry_z_score: float = 2.0
    exit_z_score: float = 0.5
    lookback: int = 50
    signal_strength_scaling: bool = True


@dataclass
class RiskConfig:
    max_daily_drawdown_pct: float = 0.03        # 3 % of peak equity
    max_position_size_pct: float = 0.10         # 10 % of capital per trade
    max_open_positions: int = 5
    default_stop_loss_atr_mult: float = 2.0
    default_take_profit_atr_mult: float = 4.0
    max_leverage: float = 10.0
    halt_on_breach: bool = True


@dataclass
class PortfolioConfig:
    initial_cash: float = 100_000.0
    base_currency: str = "USD"
    commission_per_contract: float = 2.25       # USD, typical CME rate
    slippage_ticks: int = 1
    tick_size: float = 0.25                     # ES tick size
    tick_value: float = 12.50                   # ES tick value in USD
    contract_multiplier: float = 50.0           # ES multiplier


@dataclass
class BrokerConfig:
    provider: str = "alpaca"
    paper_trading: bool = True
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    order_timeout_seconds: int = 30


@dataclass
class BacktestConfig:
    engine: str = "vectorbt"
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    walk_forward_windows: int = 5
    in_sample_ratio: float = 0.70
    optuna_trials: int = 100
    optuna_timeout_seconds: int = 300
    metrics: List[str] = field(
        default_factory=lambda: ["sharpe", "sortino", "max_drawdown", "calmar"]
    )


@dataclass
class AnalysisConfig:
    """Tuneable bounds for the Strategies-tab analysis workspace."""

    history_start_date: str = "2000-01-01"
    default_timeframe: str = "1d"
    default_data_source: str = "auto"
    max_backtest_bars: int = 60_000
    max_optimize_bars: int = 30_000
    grid_steps_per_param: int = 4
    max_grid_combinations: int = 96
    objective_metric: str = "sharpe_ratio"
    objective_metrics: List[str] = field(
        default_factory=lambda: [
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "total_return",
            "profit_factor",
        ]
    )
    walk_forward_windows: int = 4
    in_sample_ratio: float = 0.70
    walk_forward_trials: int = 25
    walk_forward_timeout_seconds: int = 90


@dataclass
class ContractSpec:
    """CME futures contract economics used for notional P&L accounting."""

    symbol: str
    name: str
    yfinance_ticker: str
    contract_multiplier: float = 50.0
    tick_size: float = 0.25
    tick_value: float = 12.50
    commission_per_contract: float = 2.25
    synthetic_start_price: float = 4500.0


@dataclass
class MarketIntelligenceAssetConfig:
    symbol: str
    label: str
    yfinance_ticker: str
    color: str = "#0071e3"


@dataclass
class MarketIntelligenceConfig:
    history_start: str = "2000-01-01"
    cache_ttl_seconds: int = 300
    news_limit: int = 25
    assets: List[MarketIntelligenceAssetConfig] = field(default_factory=list)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    contracts: Dict[str, ContractSpec] = field(default_factory=dict)
    market_intelligence: MarketIntelligenceConfig = field(
        default_factory=MarketIntelligenceConfig
    )
    log_level: str = "INFO"

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def _parse_contracts(cls, raw: Dict[str, Any]) -> Dict[str, ContractSpec]:
        contracts_raw = raw.get("contracts", {}) or {}
        specs: Dict[str, ContractSpec] = {}
        for symbol, spec in contracts_raw.items():
            specs[symbol] = ContractSpec(symbol=symbol, **spec)
        return specs

    @classmethod
    def _parse_market_intelligence(cls, raw: Dict[str, Any]) -> MarketIntelligenceConfig:
        mi_raw = raw.get("market_intelligence", {})
        assets_raw = mi_raw.get("assets", [])
        assets = [
            MarketIntelligenceAssetConfig(**asset)
            for asset in assets_raw
        ]
        return MarketIntelligenceConfig(
            history_start=mi_raw.get("history_start", "2000-01-01"),
            cache_ttl_seconds=mi_raw.get("cache_ttl_seconds", 300),
            news_limit=mi_raw.get("news_limit", 25),
            assets=assets,
        )

    @classmethod
    def from_yaml(cls, path: str | Path = "config/default.yaml") -> "Config":
        """Load a ``Config`` from a YAML file, merging with defaults."""
        path = Path(path)
        if not path.exists():
            return cls()

        with path.open() as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        return cls(
            data=DataConfig(**raw.get("data", {})),
            indicators=IndicatorConfig(**raw.get("indicators", {})),
            strategy=StrategyConfig(**raw.get("strategy", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            portfolio=PortfolioConfig(**raw.get("portfolio", {})),
            broker=BrokerConfig(**raw.get("broker", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            analysis=AnalysisConfig(**raw.get("analysis", {})),
            contracts=cls._parse_contracts(raw),
            market_intelligence=cls._parse_market_intelligence(raw),
            log_level=raw.get("log_level", "INFO"),
        )

    def to_yaml(self, path: str | Path = "config/default.yaml") -> None:
        """Serialise the current config to YAML."""
        import dataclasses

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = dataclasses.asdict(self)
        with path.open("w") as fh:
            yaml.dump(raw, fh, default_flow_style=False, sort_keys=False)
