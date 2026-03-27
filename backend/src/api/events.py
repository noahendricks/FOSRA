"""
global event bus for broadcasting tui events to connected sse clients.
uses asyncio queues — no external dependencies.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4
from loguru import logger
from dataclasses import dataclass
from typing import Any

MAX_QUEUE_SIZE = 1000
MAX_RECENT_EVENTS = 100


@dataclass
class BusEvent:
    type: str
    properties: dict[str, Any]
    sequence_nr: int


class EventBus:
    """asyncio-based broadcast bus. each subscriber gets its own queue."""

    def __init__(self) -> None:
        self._subscribers: dict[str, asyncio.Queue[BusEvent]] = {}
        self._sequence: int = 0
        self._recent: list[BusEvent] = []
        self._lock = asyncio.Lock()

    def subscribe(self) -> tuple[str, asyncio.Queue[BusEvent]]:
        sub_id = str(uuid4())
        queue: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._subscribers[sub_id] = queue
        return sub_id, queue

    def unsubscribe(self, sub_id: str) -> None:
        self._subscribers.pop(sub_id, None)

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._lock:
            self._sequence += 1
            seq = self._sequence
        bus_event = BusEvent(
            type=event.get("type", "unknown"),
            properties=event.get("properties", {}),
            sequence_nr=seq,
        )
        self._recent.append(bus_event)
        if len(self._recent) > MAX_RECENT_EVENTS:
            self._recent.pop(0)
        for queue in self._subscribers.values():
            try:
                queue.put_nowait(bus_event)
            except asyncio.QueueFull:
                logger.warning(
                    "Slow consumer: dropping event "
                    f"{event.get('type', 'unknown')} "
                    f"(subscriber queue full, maxsize={MAX_QUEUE_SIZE})"
                )

    def replay_missed(self, after_id: int) -> list[BusEvent]:
        """Return events with sequence_nr > after_id for Last-Event-ID replay."""
        return [e for e in self._recent if e.sequence_nr > after_id]

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


event_bus = EventBus()
