"""Async-native event bus.

Replaces jarvis's `_event_listeners` list + `run_coroutine_threadsafe` bridge
(needed there because voice/keyboard ran on plain threads). Everything in
Luna runs on one asyncio event loop, so publish/subscribe can be plain
asyncio.Queue fan-out: no thread-safety machinery, no dropped callbacks
swallowed by a bare `except Exception: pass`.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    STATE_CHANGED = "state_changed"
    WAKE_DETECTED = "wake_detected"
    TRANSCRIPT = "transcript"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_DENIED = "tool_denied"
    ERROR = "error"
    MUTE_CHANGED = "mute_changed"
    PROVIDER_CHANGED = "provider_changed"
    CONNECTED = "connected"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "data": self.data, "timestamp": self.timestamp}


class EventBus:
    """Pub/sub for pipeline observability. Every WebSocket client and every
    internal observer gets its own queue, so one slow subscriber can't block
    publishers or other subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []

    def subscribe(self, maxsize: int = 256) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def publish(self, event: Event) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # drop for a backed-up subscriber rather than block publishers

    def publish_type(self, event_type: EventType, **data: Any) -> None:
        self.publish(Event(type=event_type, data=data))
