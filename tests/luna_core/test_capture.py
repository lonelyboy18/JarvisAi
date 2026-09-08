import asyncio

import pytest

from luna_core.audio.capture import MicGate


def test_not_paused_by_default():
    assert MicGate().is_paused is False


def test_pause_and_resume():
    gate = MicGate()
    gate.pause()
    assert gate.is_paused is True
    gate.resume()
    assert gate.is_paused is False


@pytest.mark.asyncio
async def test_wait_until_resumed_blocks_while_paused_then_returns():
    gate = MicGate()
    gate.pause()
    waiter = asyncio.create_task(gate.wait_until_resumed())
    await asyncio.sleep(0.05)
    assert not waiter.done()
    gate.resume()
    await asyncio.wait_for(waiter, timeout=1)
    assert waiter.done()
