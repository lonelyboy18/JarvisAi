"""ConversationLoop — turns transcribed (or typed) text into a spoken reply.

Stop phrases are checked on the raw transcript before anything reaches the
LLM (see VoicePipeline._run_turn) — is_stop_command is exported here so both
the pipeline and the WebSocket typed-chat path share one definition.

Phase 1 registers zero tools (`tools=[]` on every ProviderRouter.chat call),
but if a provider hallucinates tool_calls anyway, every one is routed
through PermissionEngine.authorize() before anything happens — a real gated
path from day one, not a comment promising one later.
"""
from __future__ import annotations

from datetime import datetime

from luna_core.cancellation import CancellationToken
from luna_core.logging_setup import get_logger
from luna_core.permissions.engine import PermissionEngine
from luna_core.permissions.models import PermissionRequest
from luna_core.providers import ChatResult, ProviderRouter

logger = get_logger("conversation")

_STOP_PHRASES = {
    "stop", "luna stop", "hey luna stop",
    "cancel", "abort", "never mind", "nevermind", "shut up", "be quiet",
}


def is_stop_command(text: str) -> bool:
    return text.strip().lower().rstrip(".,!?") in _STOP_PHRASES


def _system_prompt() -> str:
    now = datetime.now().strftime("%A, %B %d %Y, %I:%M %p")
    return (
        "You are Luna, a local, private voice assistant. Be concise, warm, and direct — "
        "answer in 1-3 sentences unless more detail is clearly needed. "
        "You do not have any tools available yet; if asked to do something requiring "
        "one (controlling the computer, browsing, files), say so plainly. "
        f"Current date and time: {now}."
    )


class ConversationLoop:
    def __init__(
        self, provider_router: ProviderRouter, permission_engine: PermissionEngine, max_history: int = 10
    ) -> None:
        self._router = provider_router
        self._permissions = permission_engine
        self._history: list[dict[str, str]] = []
        self._max_history = max_history

    def reset(self) -> None:
        self._history.clear()

    async def handle_utterance(self, text: str, cancellation: CancellationToken) -> str:
        cancellation.check()
        messages = [
            {"role": "system", "content": _system_prompt()},
            *self._history,
            {"role": "user", "content": text},
        ]

        result: ChatResult = await self._router.chat(messages, tools=[])
        cancellation.check()

        reply_text = result.text or "I'm not sure how to respond to that."
        if result.tool_calls:
            reply_text = await self._reject_tool_calls(result)

        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": reply_text})
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        return reply_text

    async def _reject_tool_calls(self, result: ChatResult) -> str:
        for call in result.tool_calls:
            decision = await self._permissions.authorize(
                PermissionRequest(tool_name=call.name, args=call.arguments)
            )
            logger.info("hallucinated tool_call %s -> %s", call.name, decision.reason)
        return result.text or "I can't do that yet — I don't have any tools available."
