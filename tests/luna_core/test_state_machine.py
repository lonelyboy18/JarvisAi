import pytest

from luna_core.events import EventBus, EventType
from luna_core.state_machine import InvalidTransitionError, LunaState, StateMachine


@pytest.mark.asyncio
async def test_valid_transition_updates_state_and_publishes():
    bus = EventBus()
    q = bus.subscribe()
    sm = StateMachine(bus)
    await sm.transition_to(LunaState.WAKE, reason="wake_detected")
    assert sm.current is LunaState.WAKE
    event = q.get_nowait()
    assert event.type is EventType.STATE_CHANGED
    assert event.data == {"from": "idle", "to": "wake", "reason": "wake_detected"}


@pytest.mark.asyncio
async def test_invalid_transition_raises_and_does_not_change_state():
    sm = StateMachine(EventBus())
    with pytest.raises(InvalidTransitionError):
        await sm.transition_to(LunaState.SPEAKING)
    assert sm.current is LunaState.IDLE


@pytest.mark.asyncio
async def test_error_and_interrupted_reachable_from_any_state():
    sm = StateMachine(EventBus())
    await sm.transition_to(LunaState.WAKE)
    await sm.transition_to(LunaState.LISTENING)
    await sm.transition_to(LunaState.THINKING)
    await sm.transition_to(LunaState.EXECUTING)
    await sm.transition_to(LunaState.INTERRUPTED)
    assert sm.current is LunaState.INTERRUPTED

    await sm.transition_to(LunaState.IDLE)
    await sm.transition_to(LunaState.WAKE)
    await sm.transition_to(LunaState.ERROR)
    assert sm.current is LunaState.ERROR


@pytest.mark.asyncio
async def test_is_busy_reflects_busy_states():
    sm = StateMachine(EventBus())
    assert sm.is_busy is False
    await sm.transition_to(LunaState.WAKE)
    assert sm.is_busy is False
    await sm.transition_to(LunaState.LISTENING)
    assert sm.is_busy is True
    await sm.transition_to(LunaState.THINKING)
    assert sm.is_busy is True


@pytest.mark.asyncio
async def test_typed_input_can_skip_listening_via_wake_to_thinking():
    sm = StateMachine(EventBus())
    await sm.transition_to(LunaState.WAKE, reason="typed_input")
    await sm.transition_to(LunaState.THINKING, reason="calling_llm")
    assert sm.current is LunaState.THINKING
