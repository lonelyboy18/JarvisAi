"""STTEngine (faster-whisper) and the async mic recorder that feeds it.

record_until_silence mirrors jarvis/stt.py's algorithm (wait for speech to
start, then stop after ~2.1s of silence) but is `async def`: it awaits
asyncio.to_thread() per 300ms chunk instead of blocking the event loop, and
calls the shared CancellationToken's .check() between chunks so a "Luna
stop" recognized elsewhere unwinds the recording loop via LunaCancelled
instead of a manually-polled threading.Event.
"""
from __future__ import annotations

import asyncio

import numpy as np

from luna_core.cancellation import CancellationToken
from luna_core.config import SttConfig
from luna_core.logging_setup import get_logger

logger = get_logger("stt")


class STTEngine:
    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._model: object | None = None
        self._model_device: str | None = None

    def _get_model(self, force_cpu: bool = False) -> object:
        if self._model is not None and not force_cpu:
            return self._model
        from faster_whisper import WhisperModel

        device = "cpu" if force_cpu else self._config.device
        compute_type = "int8" if force_cpu else self._config.compute_type
        try:
            self._model = WhisperModel(self._config.model, device=device, compute_type=compute_type)
            self._model_device = device
            logger.info("loaded whisper model=%s device=%s", self._config.model, device)
        except Exception as exc:
            if device != "cpu":
                logger.warning("%s failed to load (%s), falling back to cpu", device, exc)
                self._model = WhisperModel(self._config.model, device="cpu", compute_type="int8")
                self._model_device = "cpu"
            else:
                raise
        return self._model

    async def transcribe(self, audio: np.ndarray) -> str:
        def _run(force_cpu: bool = False) -> str:
            model = self._get_model(force_cpu=force_cpu)
            segments, _ = model.transcribe(audio, beam_size=5, language="en")  # type: ignore[attr-defined]
            return " ".join(seg.text.strip() for seg in segments).strip()

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            if self._model_device != "cpu":
                logger.warning("%s transcription failed (%s), retrying on cpu", self._model_device, exc)
                try:
                    return await asyncio.to_thread(_run, True)
                except Exception as exc2:
                    logger.error("cpu transcription also failed: %s", exc2)
                    return ""
            logger.error("transcription failed: %s", exc)
            return ""


async def record_until_silence(config: SttConfig, cancellation: CancellationToken) -> np.ndarray:
    """Record microphone audio until silence follows detected speech.
    Raises LunaCancelled if the token is set mid-recording."""
    import sounddevice as sd

    sample_rate = config.sample_rate
    chunk = int(sample_rate * 0.3)
    recording: list[np.ndarray] = []
    silent_chunks = 0
    heard_speech = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while True:
            cancellation.check()
            data, _ = await asyncio.to_thread(stream.read, chunk)
            flat = data.flatten()
            recording.append(flat)
            rms = float(np.sqrt(np.mean(flat**2)))

            if rms >= config.speech_threshold:
                if not heard_speech:
                    logger.info("speech detected (rms=%.4f)", rms)
                heard_speech = True
                silent_chunks = 0
            elif heard_speech and rms < config.silence_threshold:
                silent_chunks += 1

            if heard_speech and silent_chunks >= config.silent_chunks_needed:
                logger.info("silence detected, stopping recording")
                break
            if len(recording) * chunk >= config.max_seconds * sample_rate:
                logger.info("max recording time reached (%ss)", config.max_seconds)
                break

    if not recording:
        return np.zeros(chunk, dtype=np.float32)
    return np.concatenate(recording)
