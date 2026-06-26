"""Async WebSocket live tick stream — supports Databento and CME native feeds."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Callable, List, Optional

from core.models import Tick

logger = logging.getLogger(__name__)

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:  # pragma: no cover
    HAS_WEBSOCKETS = False


TickCallback = Callable[[Tick], None]


class LiveTickStream:
    """
    Non-blocking WebSocket client that subscribes to live tick data and
    dispatches each tick to registered callbacks on the event bus.

    Usage::

        stream = LiveTickStream(
            url="wss://ws.databento.com/v0",
            api_key=os.getenv("DATABENTO_API_KEY"),
            symbols=["ES.c.0", "NQ.c.0"],
        )
        stream.add_callback(lambda tick: print(tick))
        asyncio.run(stream.start())
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        dataset: str = "GLBX.MDP3",
        schema: str = "mbp-1",
        reconnect_delay: float = 5.0,
    ) -> None:
        self._url = url
        self._api_key = api_key or os.getenv("DATABENTO_API_KEY", "")
        self._symbols = symbols or []
        self._dataset = dataset
        self._schema = schema
        self._reconnect_delay = reconnect_delay
        self._callbacks: List[TickCallback] = []
        self._running = False

    # ── Registration ──────────────────────────────────────────────────────────

    def add_callback(self, cb: TickCallback) -> None:
        self._callbacks.append(cb)

    def remove_callback(self, cb: TickCallback) -> None:
        try:
            self._callbacks.remove(cb)
        except ValueError:
            pass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect and stream ticks indefinitely, reconnecting on disconnect."""
        if not HAS_WEBSOCKETS:
            raise RuntimeError(
                "websockets library not installed. Run: pip install websockets"
            )
        self._running = True
        logger.info("LiveTickStream starting for %s …", self._symbols)

        while self._running:
            try:
                await self._connect_and_stream()
            except (OSError, ConnectionError) as exc:
                logger.warning("WebSocket disconnected: %s — reconnecting in %ss", exc, self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
            except asyncio.CancelledError:
                break

        logger.info("LiveTickStream stopped.")

    async def stop(self) -> None:
        self._running = False

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _connect_and_stream(self) -> None:
        """Establish WebSocket connection, subscribe, and dispatch messages."""
        auth_header = {"Authorization": f"Bearer {self._api_key}"}

        async with websockets.connect(self._url, extra_headers=auth_header) as ws:  # type: ignore[attr-defined]
            await self._subscribe(ws)
            async for raw_msg in ws:
                if not self._running:
                    break
                tick = self._parse_message(raw_msg)
                if tick is not None:
                    self._dispatch(tick)

    async def _subscribe(self, ws: object) -> None:
        """Send subscription message for configured symbols and schema."""
        sub_msg = {
            "action": "subscribe",
            "dataset": self._dataset,
            "schema": self._schema,
            "symbols": self._symbols,
        }
        await ws.send(json.dumps(sub_msg))  # type: ignore[attr-defined]
        logger.debug("Subscribed: %s", sub_msg)

    def _parse_message(self, raw: str | bytes) -> Optional[Tick]:
        """Parse a raw WebSocket message into a ``Tick`` dataclass."""
        try:
            msg = json.loads(raw)
            if msg.get("type") not in ("mbp-1", "mbo", "trades"):
                return None

            ts = datetime.fromtimestamp(
                msg["ts_event"] / 1_000_000_000, tz=timezone.utc
            )
            return Tick(
                symbol=msg.get("symbol", "UNKNOWN"),
                timestamp=ts,
                bid=float(msg.get("bid_px_00", msg.get("price", 0)) / 1e9),
                ask=float(msg.get("ask_px_00", msg.get("price", 0)) / 1e9),
                bid_size=float(msg.get("bid_sz_00", 0)),
                ask_size=float(msg.get("ask_sz_00", 0)),
                last=float(msg.get("price", 0) / 1e9),
                last_size=float(msg.get("size", 0)),
            )
        except (KeyError, ValueError, TypeError):
            logger.debug("Could not parse message: %s", str(raw)[:200])
            return None

    def _dispatch(self, tick: Tick) -> None:
        for cb in self._callbacks:
            try:
                cb(tick)
            except Exception:
                logger.exception("Tick callback raised an exception.")


async def iter_ticks(
    stream: LiveTickStream,
    max_ticks: Optional[int] = None,
) -> AsyncIterator[Tick]:
    """
    Async generator that yields ticks from *stream*.

    Useful for testing and scripted consumption without callbacks.
    """
    received: List[Tick] = []
    event = asyncio.Event()

    def _collect(tick: Tick) -> None:
        received.append(tick)
        event.set()

    stream.add_callback(_collect)
    count = 0
    try:
        while max_ticks is None or count < max_ticks:
            await event.wait()
            event.clear()
            while received:
                yield received.pop(0)
                count += 1
    finally:
        stream.remove_callback(_collect)
