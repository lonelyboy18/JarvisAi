from unittest.mock import MagicMock

import numpy as np
import pytest

from luna_core.audio.stt import STTEngine, record_until_silence
from luna_core.cancellation import CancellationToken, LunaCancelled
from luna_core.config import SttConfig


@pytest.mark.asyncio
async def test_transcribe_concatenates_segments(monkeypatch):
    seg1 = MagicMock()
    seg1.text = "hello"
    seg2 = MagicMock()
    seg2.text = " world"
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([seg1, seg2], None)
    monkeypatch.setattr("faster_whisper.WhisperModel", lambda *a, **kw: fake_model)

    engine = STTEngine(SttConfig(model="tiny", device="cpu", compute_type="int8"))
    result = await engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert result == "hello world"


@pytest.mark.asyncio
async def test_transcribe_falls_back_to_cpu_on_device_load_failure(monkeypatch):
    calls = {"n": 0}

    def fake_ctor(model_name, device, compute_type):
        calls["n"] += 1
        if device != "cpu":
            raise RuntimeError("cuda not available")
        m = MagicMock()
        seg = MagicMock()
        seg.text = "ok"
        m.transcribe.return_value = ([seg], None)
        return m

    monkeypatch.setattr("faster_whisper.WhisperModel", fake_ctor)

    engine = STTEngine(SttConfig(model="tiny", device="cuda", compute_type="float16"))
    result = await engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert result == "ok"
    assert calls["n"] == 2


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self._i = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        if self._i < len(self._chunks):
            chunk = self._chunks[self._i]
            self._i += 1
        else:
            chunk = np.zeros((n, 1), dtype=np.float32)
        return chunk, False


@pytest.mark.asyncio
async def test_record_until_silence_stops_after_speech_then_silence(monkeypatch):
    cfg = SttConfig(sample_rate=16000, silent_chunks_needed=2, max_seconds=5)
    chunk = int(cfg.sample_rate * 0.3)
    loud = np.full((chunk, 1), 0.5, dtype=np.float32)
    quiet = np.zeros((chunk, 1), dtype=np.float32)
    fake_stream = _FakeStream([loud, quiet, quiet])
    monkeypatch.setattr("sounddevice.InputStream", lambda **kw: fake_stream)

    audio = await record_until_silence(cfg, CancellationToken())
    assert audio.shape[0] == chunk * 3


@pytest.mark.asyncio
async def test_record_until_silence_raises_when_cancelled(monkeypatch):
    cfg = SttConfig(sample_rate=16000)
    chunk = int(cfg.sample_rate * 0.3)
    fake_stream = _FakeStream([np.zeros((chunk, 1), dtype=np.float32)] * 50)
    monkeypatch.setattr("sounddevice.InputStream", lambda **kw: fake_stream)

    token = CancellationToken()
    token.cancel()
    with pytest.raises(LunaCancelled):
        await record_until_silence(cfg, token)
