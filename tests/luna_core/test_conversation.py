import pytest

from luna_core.cancellation import CancellationToken, LunaCancelled
from luna_core.conversation import ConversationLoop, is_stop_command
from luna_core.events import EventBus
from luna_core.permissions.engine import PermissionEngine
from luna_core.permissions.models import PermissionLevel
from luna_core.providers import ChatResult, ToolCall


class _StubRouter:
    def __init__(self, result: ChatResult):
        self._result = result
        self.last_messages = None

    async def chat(self, messages, *, tools=None, temperature=None):
        self.last_messages = messages
        return self._result


@pytest.mark.parametrize("phrase", ["stop", "Stop.", "luna stop", "cancel", "never mind", "NEVERMIND!"])
def test_is_stop_command_recognizes_phrases(phrase):
    assert is_stop_command(phrase) is True


@pytest.mark.parametrize("phrase", ["stop the music please", "cancellation", "hello luna"])
def test_is_stop_command_rejects_non_exact_phrases(phrase):
    assert is_stop_command(phrase) is False


@pytest.mark.asyncio
async def test_handle_utterance_returns_llm_text_and_updates_history():
    router = _StubRouter(ChatResult(text="Hi! How can I help?"))
    permissions = PermissionEngine(default_level=PermissionLevel.DENY)
    loop = ConversationLoop(router, permissions)
    token = CancellationToken()

    reply = await loop.handle_utterance("hello luna", token)

    assert reply == "Hi! How can I help?"
    assert router.last_messages[-1] == {"role": "user", "content": "hello luna"}
    assert loop._history[-1] == {"role": "assistant", "content": "Hi! How can I help?"}


@pytest.mark.asyncio
async def test_hallucinated_tool_call_is_denied_not_executed():
    result = ChatResult(text="", tool_calls=[ToolCall(id="1", name="delete_file", arguments={"path": "x"})])
    router = _StubRouter(result)
    bus = EventBus()
    q = bus.subscribe()
    permissions = PermissionEngine(default_level=PermissionLevel.DENY, event_bus=bus)
    loop = ConversationLoop(router, permissions)
    token = CancellationToken()

    reply = await loop.handle_utterance("delete my file", token)

    assert "can't do that yet" in reply.lower()
    event = q.get_nowait()
    assert event.data["tool"] == "delete_file"
    assert event.data["allowed"] is False


@pytest.mark.asyncio
async def test_handle_utterance_checks_cancellation_before_calling_llm():
    router = _StubRouter(ChatResult(text="should not be reached"))
    permissions = PermissionEngine(default_level=PermissionLevel.DENY)
    loop = ConversationLoop(router, permissions)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(LunaCancelled):
        await loop.handle_utterance("hello", token)
    assert router.last_messages is None


@pytest.mark.asyncio
async def test_history_is_trimmed_to_max_messages():
    router = _StubRouter(ChatResult(text="ok"))
    permissions = PermissionEngine(default_level=PermissionLevel.DENY)
    loop = ConversationLoop(router, permissions, max_history=4)
    token = CancellationToken()

    for i in range(5):
        await loop.handle_utterance(f"message {i}", token)

    assert len(loop._history) == 4
