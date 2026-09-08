import json

from fastapi.testclient import TestClient

from luna_core.cancellation import CancellationToken
from luna_core.config import LlmConfig, LunaConfig, ProviderConfig
from luna_core.events import EventBus
from luna_core.permissions.engine import PermissionEngine
from luna_core.permissions.models import PermissionLevel
from luna_core.providers import ProviderRouter
from luna_core.server import create_app
from luna_core.state_machine import StateMachine
from luna_core.voice_pipeline import VoicePipeline


class _StubStt:
    async def transcribe(self, audio):
        return ""


class _StubTts:
    async def speak(self, text, cancellation: CancellationToken):
        pass

    def stop(self):
        pass


class _StubConversation:
    async def handle_utterance(self, text, cancellation: CancellationToken):
        return "stub reply"

    def reset(self):
        pass


def _build_app():
    config = LunaConfig(
        llm=LlmConfig(
            active_provider="ollama",
            providers={"ollama": ProviderConfig(type="ollama", label="Ollama (Local)", model="qwen3:8b")},
        )
    )
    bus = EventBus()
    state = StateMachine(bus)
    router = ProviderRouter(config.llm)
    permissions = PermissionEngine(default_level=PermissionLevel.DENY, event_bus=bus)
    pipeline = VoicePipeline(config, bus, state, _StubStt(), _StubTts(), _StubConversation())
    app = create_app(config, bus, pipeline, router)
    del permissions  # constructed for parity with main.py's graph, unused directly here
    return app


def test_websocket_sends_connected_message_on_connect():
    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        data = json.loads(ws.receive_text())
        assert data["type"] == "connected"


def test_websocket_chat_message_drives_pipeline_to_idle():
    """Uses the `chat` (typed) path, not `activate` — activate() records
    real microphone audio via sounddevice, which has no place in a unit
    test. See ROADMAP.md / the Phase 1 report for why voice activation
    itself is exercised manually, not in the automated suite."""
    client = TestClient(_build_app())
    with client.websocket_connect("/ws") as ws:
        ws.receive_text()  # connected banner
        ws.send_text(json.dumps({"type": "chat", "text": "hello luna"}))

        seen_final_idle = False
        for _ in range(10):
            msg = json.loads(ws.receive_text())
            if msg["type"] == "state_changed" and msg["data"]["to"] == "idle":
                seen_final_idle = True
                break
        assert seen_final_idle


def test_api_state_endpoint_reports_idle_initially():
    client = TestClient(_build_app())
    resp = client.get("/api/state")
    assert resp.status_code == 200
    assert resp.json() == {"state": "idle"}


def test_api_providers_endpoint_lists_configured_providers():
    client = TestClient(_build_app())
    resp = client.get("/api/providers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] == "ollama"
    assert "ollama" in body["providers"]


def test_api_set_provider_rejects_unknown_key():
    client = TestClient(_build_app())
    resp = client.post("/api/providers/does-not-exist")
    assert resp.status_code == 400
