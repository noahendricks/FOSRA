"""
global event bus for broadcasting tui events to connected sse clients.
uses asyncio queues — no external dependencies.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

MAX_QUEUE_SIZE = 1000


class EventBus:
    """asyncio-based broadcast bus. each subscriber gets its own queue."""

    def __init__(self) -> None:
        self._subscribers: dict[str, asyncio.Queue] = {}

    def subscribe(self) -> tuple[str, asyncio.Queue]:
        sub_id = str(uuid4())
        queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._subscribers[sub_id] = queue
        return sub_id, queue

    def unsubscribe(self, sub_id: str) -> None:
        self._subscribers.pop(sub_id, None)

    async def publish(self, event: dict) -> None:
        for queue in self._subscribers.values():
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


event_bus = EventBus()
