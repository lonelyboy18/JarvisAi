import asyncio

import pytest

from luna_core.events import Event, EventBus, EventType


@pytest.mark.asyncio
async def test_publish_fans_out_to_all_subscribers():
    bus = EventBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    bus.publish_type(EventType.WAKE_DETECTED, source="voice")

    e1 = await asyncio.wait_for(q1.get(), timeout=1)
    e2 = await asyncio.wait_for(q2.get(), timeout=1)
    assert e1.type is EventType.WAKE_DETECTED
    assert e1.data == {"source": "voice"}
    assert e2.data == e1.data


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    bus = EventBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    bus.publish_type(EventType.ERROR, message="boom")
    assert q.empty()


def test_event_to_dict_serializes_enum_value():
    event = Event(type=EventType.STATE_CHANGED, data={"from": "idle", "to": "wake"})
    d = event.to_dict()
    assert d["type"] == "state_changed"
    assert d["data"] == {"from": "idle", "to": "wake"}
