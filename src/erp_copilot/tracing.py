"""Lightweight tracing: nested spans with timing and attributes, exportable as JSONL."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    start: float
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return 0.0 if self.end is None else round((self.end - self.start) * 1000, 3)

    def to_dict(self, trace_id: str) -> dict[str, Any]:
        return {
            "trace_id": trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "attrs": self.attrs,
        }


class Tracer:
    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex[:12]
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = self._stack[-1].span_id if self._stack else None
        span = Span(uuid.uuid4().hex[:8], parent, name, time.perf_counter(), attrs=dict(attrs))
        self.spans.append(span)
        self._stack.append(span)
        try:
            yield span
        except Exception as exc:
            span.attrs["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = time.perf_counter()
            self._stack.pop()

    def names(self) -> list[str]:
        return [s.name for s in self.spans]

    def export_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for s in self.spans:
                fh.write(json.dumps(s.to_dict(self.trace_id), default=str) + "\n")

    def render(self) -> str:
        depth: dict[str, int] = {}
        lines = [f"trace {self.trace_id}"]
        for s in self.spans:
            d = depth[s.parent_id] + 1 if s.parent_id else 0
            depth[s.span_id] = d
            shown = {k: v for k, v in s.attrs.items() if k not in {"result"}}
            extra = " ".join(f"{k}={_short(v)}" for k, v in shown.items())
            lines.append(f"{'  ' * (d + 1)}- {s.name} ({s.duration_ms:.2f} ms) {extra}".rstrip())
        return "\n".join(lines)


def _short(value: Any, limit: int = 70) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 1] + "…"
