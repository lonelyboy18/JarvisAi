import pytest

from luna_core.config import LlmConfig, ProviderConfig
from luna_core.providers import ChatResult, ProviderRouter


class _FakeProvider:
    def __init__(self, text: str = "ok", raise_error: bool = False):
        self._text = text
        self._raise = raise_error
        self.calls = 0

    async def chat(self, messages, *, tools=None, temperature=0.7):
        self.calls += 1
        if self._raise:
            raise RuntimeError("boom")
        return ChatResult(text=self._text)


@pytest.mark.asyncio
async def test_router_delegates_to_active_provider():
    llm_cfg = LlmConfig(
        active_provider="ollama",
        providers={"ollama": ProviderConfig(type="ollama", model="qwen3:8b")},
    )
    router = ProviderRouter(llm_cfg)
    fake = _FakeProvider(text="hi there")
    router._providers["ollama"] = fake

    result = await router.chat([{"role": "user", "content": "hello"}])
    assert result.text == "hi there"
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_router_falls_back_to_ollama_on_error():
    llm_cfg = LlmConfig(
        active_provider="cloud",
        providers={
            "cloud": ProviderConfig(type="openai", model="gpt", base_url="http://x"),
            "ollama": ProviderConfig(type="ollama", model="qwen3:8b"),
        },
    )
    router = ProviderRouter(llm_cfg)
    failing = _FakeProvider(raise_error=True)
    fallback = _FakeProvider(text="fallback reply")
    router._providers["cloud"] = failing
    router._providers["ollama"] = fallback

    result = await router.chat([{"role": "user", "content": "hello"}])
    assert result.text == "fallback reply"
    assert failing.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_router_reraises_when_fallback_also_fails():
    llm_cfg = LlmConfig(
        active_provider="ollama",
        providers={"ollama": ProviderConfig(type="ollama", model="qwen3:8b")},
    )
    router = ProviderRouter(llm_cfg)
    router._providers["ollama"] = _FakeProvider(raise_error=True)

    with pytest.raises(RuntimeError):
        await router.chat([{"role": "user", "content": "hello"}])


def test_switch_active_rejects_unknown_provider():
    llm_cfg = LlmConfig(active_provider="ollama", providers={"ollama": ProviderConfig(type="ollama", model="x")})
    router = ProviderRouter(llm_cfg)
    assert router.switch_active("nonexistent") is False
    assert router.switch_active("ollama") is True
    assert router.active == "ollama"
