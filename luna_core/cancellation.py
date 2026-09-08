"""Shared cancellation signal for the voice pipeline and conversation loop.

One CancellationToken is created per turn. Every major step (STT recording,
LLM call, TTS playback) calls .check() between chunks/iterations; a set
token raises LunaCancelled, which unwinds cleanly back to the pipeline's
top-level handler. Barge-in and stop-command detection both just call
.cancel() — they don't need to know what step is currently running.
"""
from __future__ import annotations

import asyncio


class LunaCancelled(Exception):
    """Raised when a running turn is cancelled (barge-in, stop command, error)."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise LunaCancelled()

    async def wait(self) -> None:
        await self._event.wait()
