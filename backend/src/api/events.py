"""
global event bus for broadcasting tui events to connected sse clients.
uses asyncio queues — no external dependencies.
"""

from __future__ import annotations

import asyncio
from collections import deque
from uuid import uuid4
from loguru import logger
from dataclasses import dataclass
from typing import TypedDict

MAX_QUEUE_SIZE = 1000
MAX_RECENT_EVENTS = 100


class BusEventProperties(TypedDict, total=False):
    """Properties dict can have any string keys with arbitrary values."""

    ...


@dataclass
class BusEvent:
    type: str
    properties: BusEventProperties
    sequence_nr: int


class EventBus:
    """asyncio-based broadcast bus. each subscriber gets its own queue."""

    def __init__(self) -> None:
        self._subscribers: dict[str, asyncio.Queue[BusEvent]] = {}
        self._sequence: int = 0
        self._recent: deque[BusEvent] = deque(maxlen=MAX_RECENT_EVENTS)
        self._dropped_count: int = 0
        self._lock = asyncio.Lock()

    def subscribe(self) -> tuple[str, asyncio.Queue[BusEvent]]:
        sub_id = str(uuid4())
        queue: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._subscribers[sub_id] = queue
        return sub_id, queue

    def unsubscribe(self, sub_id: str) -> None:
        _ = self._subscribers.pop(sub_id, None)

    async def publish(self, event: BusEventProperties) -> None:  # type: ignore[reportExplicitAny]
        async with self._lock:
            self._sequence += 1
            seq = self._sequence
        bus_event = BusEvent(
            type=event.get("type", "unknown"),
            properties=event.get("properties", {}),
            sequence_nr=seq,
        )
        self._recent.append(bus_event)
        for queue in self._subscribers.values():
            try:
                queue.put_nowait(bus_event)
            except asyncio.QueueFull:
                self._dropped_count += 1
                logger.warning(
                    "Slow consumer: dropping event "
                    f"{event.get('type', 'unknown')} "
                    f"(subscriber queue full, maxsize={MAX_QUEUE_SIZE}, "
                    f"total_dropped={self._dropped_count})"
                )

    def replay_missed(self, after_id: int) -> list[BusEvent]:
        """Return events with sequence_nr > after_id for Last-Event-ID replay."""
        return [e for e in self._recent if e.sequence_nr > after_id]

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


event_bus = EventBus()
