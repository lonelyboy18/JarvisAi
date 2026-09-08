"""LunaState + StateMachine.

Every state change is validated against an explicit transition table and
published to the EventBus, so barge-in / interrupts / errors are observable
by the frontend instead of being a silent side effect buried in pipeline
code. Transitions are serialized with an asyncio.Lock since the normal
pipeline flow and a barge-in interrupt can both try to transition
concurrently from different coroutines.
"""
from __future__ import annotations

import asyncio
from enum import StrEnum

from luna_core.events import Event, EventBus, EventType


class LunaState(StrEnum):
    IDLE = "idle"
    WAKE = "wake"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


# ERROR and INTERRUPTED are exceptional — reachable from any state.
_ALWAYS_REACHABLE = {LunaState.ERROR, LunaState.INTERRUPTED}

_TRANSITIONS: dict[LunaState, set[LunaState]] = {
    LunaState.IDLE: {LunaState.WAKE},
    # WAKE -> THINKING directly supports typed/WebSocket "activate" input,
    # which already has text and skips the record-audio step (see
    # VoicePipeline.activate_with_text) — a stand-in for voice activation
    # while wake-word detection is disabled.
    LunaState.WAKE: {LunaState.LISTENING, LunaState.THINKING, LunaState.IDLE},
    LunaState.LISTENING: {LunaState.THINKING, LunaState.IDLE},
    LunaState.THINKING: {LunaState.EXECUTING, LunaState.SPEAKING, LunaState.IDLE},
    LunaState.EXECUTING: {LunaState.THINKING, LunaState.SPEAKING, LunaState.IDLE},
    LunaState.SPEAKING: {LunaState.IDLE},
    LunaState.INTERRUPTED: {LunaState.IDLE},
    LunaState.ERROR: {LunaState.IDLE},
}

_BUSY_STATES = frozenset({LunaState.LISTENING, LunaState.THINKING, LunaState.EXECUTING, LunaState.SPEAKING})


class InvalidTransitionError(Exception):
    def __init__(self, current: LunaState, target: LunaState) -> None:
        super().__init__(f"Cannot transition from {current.value} to {target.value}")
        self.current = current
        self.target = target


class StateMachine:
    def __init__(self, event_bus: EventBus, initial: LunaState = LunaState.IDLE) -> None:
        self._bus = event_bus
        self._state = initial
        self._lock = asyncio.Lock()

    @property
    def current(self) -> LunaState:
        return self._state

    @property
    def is_busy(self) -> bool:
        """True while EXECUTING or SPEAKING (or getting there) — used by the
        wake-word listener to decide barge-in vs. a normal new listen cycle."""
        return self._state in _BUSY_STATES

    async def transition_to(self, target: LunaState, reason: str | None = None) -> None:
        async with self._lock:
            if target not in _ALWAYS_REACHABLE and target not in _TRANSITIONS.get(self._state, set()):
                raise InvalidTransitionError(self._state, target)
            previous = self._state
            self._state = target
        self._bus.publish(Event(
            type=EventType.STATE_CHANGED,
            data={"from": previous.value, "to": target.value, "reason": reason},
        ))
