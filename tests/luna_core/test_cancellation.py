import asyncio

import pytest

from luna_core.cancellation import CancellationToken, LunaCancelled


def test_check_is_a_no_op_before_cancel():
    token = CancellationToken()
    token.check()  # should not raise


def test_check_raises_after_cancel():
    token = CancellationToken()
    token.cancel()
    with pytest.raises(LunaCancelled):
        token.check()


def test_reset_clears_cancellation():
    token = CancellationToken()
    token.cancel()
    token.reset()
    assert token.is_cancelled is False
    token.check()  # should not raise


@pytest.mark.asyncio
async def test_wait_returns_after_cancel():
    token = CancellationToken()

    async def canceller():
        await asyncio.sleep(0.01)
        token.cancel()

    asyncio.create_task(canceller())
    await asyncio.wait_for(token.wait(), timeout=1)
    assert token.is_cancelled
