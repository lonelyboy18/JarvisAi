"""WakeWordListener tests use fake pyaudio/openwakeword objects — no real
audio hardware or trained model needed to verify the cooldown, barge-in,
and mic-reset-on-resume logic that jarvis's implementation didn't have."""
import asyncio

import pytest

from luna_core.audio.capture import MicGate
from luna_core.audio.wake_word import WakeWordListener
from luna_core.config import WakeWordConfig


class _FakeStream:
    def __init__(self):
        self.active = True

    def is_active(self):
        return self.active

    def stop_stream(self):
        self.active = False

    def start_stream(self):
        self.active = True

    def read(self, chunk_size, exception_on_overflow=False):
        return b"\x00\x00" * chunk_size

    def close(self):
        pass


class _FakePyAudio:
    def __init__(self):
        self.stream = _FakeStream()

    def open(self, **kwargs):
        return self.stream

    def terminate(self):
        pass


class _FakeModel:
    def __init__(self, scores):
        self._scores = list(scores)
        self.reset_calls = 0

    def predict(self, pcm):
        score = self._scores.pop(0) if self._scores else 0.0
        return {"hey_luna": score}

    def reset(self):
        self.reset_calls += 1


def _config(**overrides):
    defaults = dict(
        model_path="models/hey_luna.onnx",
        threshold=0.5,
        cooldown_seconds=0.05,
        interrupt_cooldown_seconds=0.05,
    )
    defaults.update(overrides)
    return WakeWordConfig(**defaults)


@pytest.mark.asyncio
async def test_run_calls_on_wake_when_idle(monkeypatch):
    monkeypatch.setattr("pyaudio.PyAudio", lambda: _FakePyAudio())
    monkeypatch.setattr("openwakeword.model.Model", lambda **kw: _FakeModel(scores=[0.9]))

    gate = MicGate()
    woken = asyncio.Event()
    interrupted = asyncio.Event()

    listener = WakeWordListener(
        _config(), gate,
        on_wake=_setter(woken), on_interrupt=_setter(interrupted), is_busy=lambda: False,
    )
    task = asyncio.create_task(listener.run())
    await asyncio.wait_for(woken.wait(), timeout=2)
    listener.request_stop()
    gate.resume()
    await asyncio.wait_for(task, timeout=2)
    assert not interrupted.is_set()


@pytest.mark.asyncio
async def test_run_calls_on_interrupt_when_busy_instead_of_on_wake(monkeypatch):
    monkeypatch.setattr("pyaudio.PyAudio", lambda: _FakePyAudio())
    monkeypatch.setattr("openwakeword.model.Model", lambda **kw: _FakeModel(scores=[0.9]))

    gate = MicGate()
    woken = asyncio.Event()
    interrupted = asyncio.Event()

    listener = WakeWordListener(
        _config(), gate,
        on_wake=_setter(woken), on_interrupt=_setter(interrupted), is_busy=lambda: True,
    )
    task = asyncio.create_task(listener.run())
    await asyncio.wait_for(interrupted.wait(), timeout=2)
    listener.request_stop()
    gate.resume()
    await asyncio.wait_for(task, timeout=2)
    assert not woken.is_set()


@pytest.mark.asyncio
async def test_model_reset_after_mic_gate_resume(monkeypatch):
    monkeypatch.setattr("pyaudio.PyAudio", lambda: _FakePyAudio())
    fake_model = _FakeModel(scores=[])
    monkeypatch.setattr("openwakeword.model.Model", lambda **kw: fake_model)

    gate = MicGate()
    gate.pause()

    async def noop():
        pass

    listener = WakeWordListener(_config(), gate, on_wake=noop, on_interrupt=noop, is_busy=lambda: False)
    task = asyncio.create_task(listener.run())
    await asyncio.sleep(0.05)
    assert fake_model.reset_calls == 0

    gate.resume()
    await asyncio.sleep(0.1)
    assert fake_model.reset_calls >= 1

    listener.request_stop()
    gate.resume()
    await asyncio.wait_for(task, timeout=2)


def _setter(event: asyncio.Event):
    async def _set():
        event.set()

    return _set
