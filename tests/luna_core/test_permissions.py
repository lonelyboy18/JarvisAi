import pytest

from luna_core.events import EventBus, EventType
from luna_core.permissions.engine import PermissionEngine
from luna_core.permissions.models import PermissionLevel, PermissionRequest


@pytest.mark.asyncio
async def test_unregistered_tool_denied_by_default():
    engine = PermissionEngine(default_level=PermissionLevel.DENY, tool_levels={})
    decision = await engine.authorize(PermissionRequest(tool_name="click_at"))
    assert decision.allowed is False
    assert decision.level is PermissionLevel.DENY


@pytest.mark.asyncio
async def test_explicitly_allowed_tool_is_allowed():
    engine = PermissionEngine(
        default_level=PermissionLevel.DENY,
        tool_levels={"get_weather": PermissionLevel.ALLOW},
    )
    decision = await engine.authorize(PermissionRequest(tool_name="get_weather"))
    assert decision.allowed is True


@pytest.mark.asyncio
async def test_ask_without_approver_denies_safely():
    engine = PermissionEngine(
        default_level=PermissionLevel.DENY,
        tool_levels={"delete_file": PermissionLevel.ASK},
    )
    decision = await engine.authorize(PermissionRequest(tool_name="delete_file"))
    assert decision.allowed is False
    assert "no approver" in decision.reason


@pytest.mark.asyncio
async def test_ask_with_approver_delegates_decision():
    async def approve(_request):
        return True

    engine = PermissionEngine(
        default_level=PermissionLevel.DENY,
        tool_levels={"delete_file": PermissionLevel.ASK},
    )
    engine.set_approver(approve)
    decision = await engine.authorize(PermissionRequest(tool_name="delete_file"))
    assert decision.allowed is True


@pytest.mark.asyncio
async def test_authorize_publishes_event():
    bus = EventBus()
    q = bus.subscribe()
    engine = PermissionEngine(default_level=PermissionLevel.DENY, event_bus=bus)
    await engine.authorize(PermissionRequest(tool_name="run_python"))
    event = q.get_nowait()
    assert event.type is EventType.TOOL_DENIED
    assert event.data["tool"] == "run_python"
    assert event.data["allowed"] is False
