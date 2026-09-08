"""ProviderRouter — async LLM provider abstraction.

Replaces jarvis/main.py's _call_llm / _call_openai_provider /
_call_ollama_provider / module-level OpenAI client cache with a small class
hierarchy: an LLMProvider ABC, one implementation per backend, and a router
that owns the active-provider selection and the fall-back-to-Ollama-on-error
behaviour jarvis's _process_request had.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from luna_core.config import LlmConfig, ProviderConfig
from luna_core.logging_setup import get_logger

logger = get_logger("providers")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw) if raw else {}
    return dict(raw or {})


class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> ChatResult: ...


class OllamaProvider(LLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> ChatResult:
        import ollama

        client = ollama.AsyncClient(host=self._config.base_url or None)
        response = await client.chat(
            model=self._config.model,
            messages=messages,
            tools=tools or None,
            options={"temperature": temperature},
        )
        message = response.message
        tool_calls = [
            ToolCall(id=str(i), name=tc.function.name, arguments=_parse_args(tc.function.arguments))
            for i, tc in enumerate(message.tool_calls or [])
        ]
        return ChatResult(text=message.content or "", tool_calls=tool_calls)


class OpenAICompatibleProvider(LLMProvider):
    """LM Studio, NVIDIA NIM, OpenAI, or any other OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self._config.base_url,
                api_key=self._config.api_key or "not-needed",
                timeout=60.0,
            )
        return self._client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> ChatResult:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            tools=tools or None,
            temperature=temperature,
        )
        message = response.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_args(tc.function.arguments))
            for tc in (message.tool_calls or [])
        ]
        return ChatResult(text=message.content or "", tool_calls=tool_calls)


def _build_provider(config: ProviderConfig) -> LLMProvider:
    if config.type == "openai":
        return OpenAICompatibleProvider(config)
    return OllamaProvider(config)


class ProviderRouter:
    def __init__(self, llm_config: LlmConfig) -> None:
        self._config = llm_config
        self._providers: dict[str, LLMProvider] = {
            key: _build_provider(provider_config) for key, provider_config in llm_config.providers.items()
        }
        self._active = llm_config.active_provider

    @property
    def active(self) -> str:
        return self._active

    def providers(self) -> dict[str, ProviderConfig]:
        return dict(self._config.providers)

    def switch_active(self, key: str) -> bool:
        if key not in self._providers:
            return False
        self._active = key
        return True

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        temp = temperature if temperature is not None else self._config.temperature
        provider = self._providers.get(self._active)
        if provider is None:
            raise RuntimeError(f"Unknown active provider: {self._active}")
        try:
            return await provider.chat(messages, tools=tools, temperature=temp)
        except Exception as exc:
            logger.warning("provider %s failed (%s), falling back to ollama", self._active, exc)
            fallback = self._providers.get("ollama")
            if fallback is None or fallback is provider:
                raise
            return await fallback.chat(messages, tools=tools, temperature=temp)
