"""PermissionEngine — the single gate every tool call must go through.

jarvis/tools/router.py dispatched every LLM-requested tool call
unconditionally (`dispatch(tool_name, args)` straight from the model's
decision, zero gating). PermissionEngine.authorize() is the fix: it looks up
a per-tool policy (ALLOW / DENY / ASK), defaults unknown tools to DENY, logs
every decision, and publishes it to the EventBus for observability. Phase 1
registers zero tools, but ConversationLoop still routes any (unexpected)
tool_calls the LLM returns through this — see conversation.py.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from luna_core.events import Event, EventBus, EventType
from luna_core.logging_setup import get_logger
from luna_core.permissions.models import PermissionDecision, PermissionLevel, PermissionRequest

logger = get_logger("permissions")

Approver = Callable[[PermissionRequest], Awaitable[bool]]


class PermissionEngine:
    def __init__(
        self,
        default_level: PermissionLevel = PermissionLevel.DENY,
        tool_levels: dict[str, PermissionLevel] | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._default = default_level
        self._tool_levels = tool_levels or {}
        self._bus = event_bus
        self._approver: Approver | None = None

    def set_approver(self, approver: Approver | None) -> None:
        """Register an interactive approval callback for ASK-level tools.
        Nothing registers one in Phase 1 — ASK degrades to a safe deny."""
        self._approver = approver

    async def authorize(self, request: PermissionRequest) -> PermissionDecision:
        level = self._tool_levels.get(request.tool_name, self._default)
        decision = await self._decide(request, level)
        logger.info(
            "authorize tool=%s level=%s allowed=%s reason=%s",
            request.tool_name, level.value, decision.allowed, decision.reason,
        )
        if self._bus is not None:
            self._bus.publish(Event(
                type=EventType.TOOL_CALL if decision.allowed else EventType.TOOL_DENIED,
                data={"tool": request.tool_name, "allowed": decision.allowed, "reason": decision.reason},
            ))
        return decision

    async def _decide(self, request: PermissionRequest, level: PermissionLevel) -> PermissionDecision:
        if level is PermissionLevel.ALLOW:
            return PermissionDecision(True, level, "allowed by policy")
        if level is PermissionLevel.DENY:
            return PermissionDecision(False, level, "denied by policy")
        if self._approver is None:
            return PermissionDecision(False, level, "ask-level tool with no approver registered")
        approved = await self._approver(request)
        reason = "approved by user" if approved else "rejected by user"
        return PermissionDecision(approved, level, reason)
