"""Lightweight publish-subscribe event bus for decoupled inter-module communication."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from core.enums import EventType

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Immutable envelope for all system events."""

    event_type: EventType
    timestamp: datetime
    payload: Any
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"timestamp must be datetime, got {type(self.timestamp)}")


# Synchronous handler signature
SyncHandler = Callable[[Event], None]
# Async handler signature
AsyncHandler = Callable[[Event], "asyncio.coroutine"]


class EventBus:
    """
    Thread-safe, optionally async publish-subscribe event bus.

    Usage::

        bus = EventBus()
        bus.subscribe(EventType.BAR, my_handler)
        bus.publish(Event(EventType.BAR, datetime.now(timezone.utc), bar_data))
    """

    def __init__(self) -> None:
        self._sync_handlers: Dict[EventType, List[SyncHandler]] = defaultdict(list)
        self._async_handlers: Dict[EventType, List[AsyncHandler]] = defaultdict(list)
        self._event_log: List[Event] = []
        self._log_events: bool = False

    # ── Registration ──────────────────────────────────────────────────────────

    def subscribe(self, event_type: EventType, handler: SyncHandler) -> None:
        """Register a synchronous handler for *event_type*."""
        self._sync_handlers[event_type].append(handler)

    def subscribe_async(self, event_type: EventType, handler: AsyncHandler) -> None:
        """Register an async coroutine handler for *event_type*."""
        self._async_handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: SyncHandler) -> None:
        try:
            self._sync_handlers[event_type].remove(handler)
        except ValueError:
            logger.warning("Handler %s not found for %s", handler, event_type)

    # ── Publishing ────────────────────────────────────────────────────────────

    def publish(self, event: Event) -> None:
        """Synchronously dispatch *event* to all registered handlers."""
        if self._log_events:
            self._event_log.append(event)

        for handler in list(self._sync_handlers[event.event_type]):
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Handler %s raised while processing %s", handler, event.event_type
                )

    async def publish_async(self, event: Event) -> None:
        """Asynchronously dispatch *event* to all async handlers concurrently."""
        self.publish(event)  # fire sync handlers first

        async_tasks = [
            h(event) for h in self._async_handlers[event.event_type]
        ]
        if async_tasks:
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.exception("Async handler raised: %s", result)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def enable_event_log(self) -> None:
        self._log_events = True

    @property
    def event_log(self) -> List[Event]:
        return list(self._event_log)

    def clear_log(self) -> None:
        self._event_log.clear()

    def subscriber_count(self, event_type: EventType) -> int:
        return len(self._sync_handlers[event_type]) + len(
            self._async_handlers[event_type]
        )
