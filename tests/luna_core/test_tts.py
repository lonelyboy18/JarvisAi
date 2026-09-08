import asyncio
import sys

import numpy as np
import pytest

from luna_core.audio.tts import PiperTTS, split_sentences
from luna_core.config import TtsConfig


def test_split_sentences_splits_on_terminators():
    assert split_sentences("Hi there. How are you? Great!") == ["Hi there.", "How are you?", "Great!"]


def test_split_sentences_handles_empty_string():
    assert split_sentences("") == []


def test_piper_command_uses_module_invocation_by_default():
    tts = PiperTTS(TtsConfig(voice_model_path="fake.onnx"))
    assert tts._piper_command() == [sys.executable, "-m", "piper"]


def test_piper_command_uses_configured_executable_override():
    tts = PiperTTS(TtsConfig(voice_model_path="fake.onnx", piper_executable="/opt/piper/piper"))
    assert tts._piper_command() == ["/opt/piper/piper"]


class _FakeProcess:
    def __init__(self, stdout: bytes, returncode: int = 0):
        self._stdout = stdout
        self.returncode = returncode
        self.killed = False

    async def communicate(self, data=None):
        return self._stdout, b""

    def kill(self):
        self.killed = True


@pytest.mark.asyncio
async def test_synthesize_returns_pcm_from_stdout(monkeypatch):
    fake_pcm = np.array([1, 2, 3], dtype=np.int16).tobytes()
    fake_proc = _FakeProcess(fake_pcm)

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    tts = PiperTTS(TtsConfig(voice_model_path="fake.onnx"))
    pcm = await tts.synthesize("hello")
    assert pcm.tolist() == [1, 2, 3]


@pytest.mark.asyncio
async def test_synthesize_returns_empty_on_nonzero_exit(monkeypatch):
    fake_proc = _FakeProcess(b"", returncode=1)

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    tts = PiperTTS(TtsConfig(voice_model_path="fake.onnx"))
    pcm = await tts.synthesize("hello")
    assert pcm.size == 0
