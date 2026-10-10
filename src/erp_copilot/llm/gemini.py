"""Gemini adapter using the public REST API (generateContent with function calling).

Standard library only. The HTTP transport is injectable so the request/response
mapping can be unit tested without a key or network.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any

from .base import LLMResponse, Message, ToolCall

Transport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]
DEFAULT_MODEL = "gemini-2.5-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _urllib_transport(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https endpoint)
        data: dict[str, Any] = json.load(resp)
    return data


class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, transport: Transport | None = None) -> None:
        self.name = f"gemini:{model}"
        self.model = model
        self._key = api_key
        self._transport = transport or _urllib_transport

    @classmethod
    def from_env(cls) -> GeminiClient:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (use --provider mock to run offline)")
        return cls(key, os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))

    @staticmethod
    def build_request(system: str, messages: list[Message], tools: list[dict[str, Any]]) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
            elif m.role == "assistant":
                parts: list[dict[str, Any]] = [{"text": m.content}] if m.content else []
                parts += [{"functionCall": {"name": c.name, "args": c.args}} for c in m.tool_calls]
                contents.append({"role": "model", "parts": parts})
            elif m.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {"functionResponse": {"name": m.name, "response": {"result": json.loads(m.content)}}}
                        ],
                    }
                )
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "tools": [{"functionDeclarations": tools}],
            "generationConfig": {"temperature": 0},
        }

    @staticmethod
    def parse_response(data: dict[str, Any]) -> LLMResponse:
        parts = data["candidates"][0]["content"].get("parts", [])
        text = "".join(p["text"] for p in parts if "text" in p)
        calls = [
            ToolCall(p["functionCall"]["name"], dict(p["functionCall"].get("args", {})), id=f"call_{i}")
            for i, p in enumerate(x for x in parts if "functionCall" in x)
        ]
        return LLMResponse(text=text, tool_calls=calls)

    def generate(self, system: str, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        body = self.build_request(system, messages, tools)
        data = self._transport(ENDPOINT.format(model=self.model), {"x-goog-api-key": self._key}, body)
        return self.parse_response(data)
