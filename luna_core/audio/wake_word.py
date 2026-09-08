"""WakeWordListener — openWakeWord inference loop, async-native.

Fully implemented and unit-testable regardless of whether a trained model is
configured; main.py only starts it once `wake_word.enabled` is true and
`wake_word.model_path` points at a real .onnx file (see ROADMAP.md for how
to finish training "Hey Luna" and flip it on — this repo ships with it
disabled, see decision #2 in the Phase 1 report).

Runs as a plain `async def run()` coroutine: every blocking mic read and
every model.predict() call is wrapped in `await asyncio.to_thread(...)`, so
the loop can check `mic_gate.is_paused` between iterations without a second
thread. On resume from a pause it calls `model.reset()` before reading again
so a stray sound doesn't false-trigger on stale prediction state.

Barge-in: while `is_busy()` is True (EXECUTING/SPEAKING and, conservatively,
THINKING — see state_machine.StateMachine.is_busy), a detection calls
`on_interrupt()` instead of `on_wake()`, so Luna never stacks a second
listen cycle on top of a running one. A cooldown window after every
detection (longer after an interrupt) stops Luna's own "Yes?" / response
audio from self-triggering the listener.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np

from luna_core.audio.capture import MicGate
from luna_core.config import WakeWordConfig
from luna_core.logging_setup import get_logger

logger = get_logger("wake_word")

OnWake = Callable[[], Awaitable[None]]
IsBusy = Callable[[], bool]


class WakeWordListener:
    def __init__(
        self,
        config: WakeWordConfig,
        mic_gate: MicGate,
        on_wake: OnWake,
        on_interrupt: OnWake,
        is_busy: IsBusy,
    ) -> None:
        self._config = config
        self._gate = mic_gate
        self._on_wake = on_wake
        self._on_interrupt = on_interrupt
        self._is_busy = is_busy
        self._stop = False
        self._ignore_until = 0.0

    def request_stop(self) -> None:
        self._stop = True

    async def run(self) -> None:
        import pyaudio
        from openwakeword.model import Model

        if not self._config.model_path:
            raise RuntimeError("WakeWordListener.run() called with no model_path configured")

        model = Model(wakeword_models=[self._config.model_path], inference_framework="onnx")
        model_key = Path(self._config.model_path).stem

        audio = pyaudio.PyAudio()
        mic = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=self._config.chunk_size,
        )
        logger.info("wake word listener started (phrase=%r, model=%s)", self._config.phrase, model_key)
        try:
            while not self._stop:
                if self._gate.is_paused:
                    if mic.is_active():
                        mic.stop_stream()
                    await self._gate.wait_until_resumed()
                    if self._stop:
                        break
                    mic.start_stream()
                    model.reset()  # clear stale predictions after resume
                    logger.debug("mic resumed, wake model reset")
                    continue

                try:
                    raw = await asyncio.to_thread(mic.read, self._config.chunk_size, False)
                except Exception:
                    await asyncio.sleep(0.05)
                    continue

                pcm = np.frombuffer(raw, dtype=np.int16)
                predictions = await asyncio.to_thread(model.predict, pcm)
                score = float(predictions.get(model_key, 0.0))
                if score < self._config.threshold:
                    continue

                model.reset()
                now = time.monotonic()
                if now < self._ignore_until:
                    continue

                if self._is_busy():
                    self._ignore_until = now + self._config.interrupt_cooldown_seconds
                    logger.info("wake word during busy state -> barge-in interrupt")
                    await self._on_interrupt()
                else:
                    self._ignore_until = now + self._config.cooldown_seconds
                    await self._on_wake()
        finally:
            mic.stop_stream()
            mic.close()
            audio.terminate()
            logger.info("wake word listener stopped")
