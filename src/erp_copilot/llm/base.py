"""Model-agnostic types shared by the agent loop and model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str = ""


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    name: str = ""  # tool name for role == "tool"


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    name: str

    def generate(self, system: str, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse: ...
