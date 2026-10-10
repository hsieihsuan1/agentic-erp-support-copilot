from __future__ import annotations

import os

from .base import LLMClient, LLMResponse, Message, ToolCall
from .gemini import GeminiClient
from .mock import MockLLM

__all__ = ["GeminiClient", "LLMClient", "LLMResponse", "Message", "MockLLM", "ToolCall", "make_client"]


def make_client(provider: str = "auto") -> LLMClient:
    """``auto`` uses Gemini when GEMINI_API_KEY is set, otherwise the offline mock."""
    if provider == "mock":
        return MockLLM()
    if provider == "gemini":
        return GeminiClient.from_env()
    if provider == "auto":
        return GeminiClient.from_env() if os.environ.get("GEMINI_API_KEY") else MockLLM()
    raise ValueError(f"unknown provider '{provider}'")
