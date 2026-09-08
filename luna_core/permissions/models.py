from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PermissionLevel(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionRequest:
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    context: str = ""


@dataclass
class PermissionDecision:
    allowed: bool
    level: PermissionLevel
    reason: str
