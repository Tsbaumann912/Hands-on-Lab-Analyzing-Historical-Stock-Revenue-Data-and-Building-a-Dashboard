"""System-wide enumerations for the futures trading terminal."""

from __future__ import annotations

from enum import Enum


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class AssetClass(str, Enum):
    FUTURES = "FUTURES"
    EQUITY = "EQUITY"
    CRYPTO = "CRYPTO"
    FX = "FX"
    OPTIONS = "OPTIONS"


class EventType(str, Enum):
    BAR = "BAR"
    TICK = "TICK"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    RISK_BREACH = "RISK_BREACH"
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    HEARTBEAT = "HEARTBEAT"
