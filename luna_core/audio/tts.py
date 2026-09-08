"""PiperTTS — text-to-speech, always invoked as a subprocess.

Piper's phonemizer dependency (espeak-ng) is GPL-3.0-or-later. Importing a
Piper Python binding in-process would pull GPL-licensed code into this
process; shelling out to the `piper` CLI via asyncio.create_subprocess_exec
keeps it on the far side of a process boundary instead — see
ARCHITECTURE.md for the full rationale and the per-voice license notes
(Piper voices are licensed individually, not under one blanket license).

Streams sentence-by-sentence so playback can start before the whole
response has finished synthesizing, and checks the shared CancellationToken
between sentences / during playback so barge-in cuts audio immediately.
"""
from __future__ import annotations

import asyncio
import re
import sys

import numpy as np

from luna_core.cancellation import CancellationToken, LunaCancelled
from luna_core.config import TtsConfig
from luna_core.logging_setup import get_logger

logger = get_logger("tts")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]


class PiperTTS:
    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._current_proc: asyncio.subprocess.Process | None = None

    def _piper_command(self) -> list[str]:
        if self._config.piper_executable:
            return [self._config.piper_executable]
        return [sys.executable, "-m", "piper"]

    async def synthesize(self, text: str) -> np.ndarray:
        """Run one Piper subprocess for `text`, return int16 PCM samples."""
        cmd = [*self._piper_command(), "--model", self._config.voice_model_path, "--output-raw"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._current_proc = proc
        try:
            stdout, stderr = await proc.communicate((text + "\n").encode("utf-8"))
        finally:
            if self._current_proc is proc:
                self._current_proc = None
        if proc.returncode != 0:
            logger.error("piper exited %s: %s", proc.returncode, stderr.decode(errors="replace")[-500:])
            return np.zeros(0, dtype=np.int16)
        return np.frombuffer(stdout, dtype=np.int16)

    async def speak(self, text: str, cancellation: CancellationToken) -> None:
        """Synthesize and play `text` sentence by sentence."""
        import sounddevice as sd

        sentences = split_sentences(text) or ([text.strip()] if text.strip() else [])
        try:
            for sentence in sentences:
                cancellation.check()
                pcm = await self.synthesize(sentence)
                if pcm.size == 0:
                    continue
                cancellation.check()
                audio = pcm.astype(np.float32) / 32767.0
                sd.play(audio, samplerate=self._config.sample_rate)
                await self._wait_or_cancel(cancellation)
        finally:
            sd.stop()

    async def _wait_or_cancel(self, cancellation: CancellationToken) -> None:
        import sounddevice as sd

        while True:
            stream = sd.get_stream()
            if stream is None or not stream.active:
                return
            if cancellation.is_cancelled:
                sd.stop()
                raise LunaCancelled()
            await asyncio.sleep(0.05)

    def stop(self) -> None:
        """Hard-stop playback and kill any in-flight Piper subprocess. Used
        for barge-in — must be safe to call from a non-async context too."""
        import sounddevice as sd

        sd.stop()
        proc = self._current_proc
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
