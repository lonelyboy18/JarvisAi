"""MicGate — asyncio-native mic pause/resume.

The wake-word listener and the STT recorder must never open the microphone
device at the same time. Both are async functions that `await
asyncio.to_thread(...)` per audio chunk rather than blocking the event loop
for a whole read loop, so a plain asyncio.Event checked directly between
awaits is a correct, race-free coordination primitive here — there's no
second OS thread to bridge to, unlike jarvis's threading.Event version.
"""
from __future__ import annotations

import asyncio


class MicGate:
    def __init__(self) -> None:
        self._resumed = asyncio.Event()
        self._resumed.set()  # not paused by default

    @property
    def is_paused(self) -> bool:
        return not self._resumed.is_set()

    def pause(self) -> None:
        self._resumed.clear()

    def resume(self) -> None:
        self._resumed.set()

    async def wait_until_resumed(self) -> None:
        await self._resumed.wait()
