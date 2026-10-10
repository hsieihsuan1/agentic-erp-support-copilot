"""Deterministic offline model.

It implements the same interface as a real LLM (messages in, tool calls or text out),
so the agent loop, guardrails, approvals, tracing and evals all run exactly as they
would with Gemini. It is a rule-based planner, NOT a language model: use it for tests,
CI and demos, and use a real model to judge language quality.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..tools import top_results
from .base import LLMResponse, Message, ToolCall

ID_RE = re.compile(r"\b(INV|ORD|TCK)-\d{4}\b", re.IGNORECASE)
ESCALATE_RE = re.compile(
    r"\b(human|real person|manager|escalat\w*|speak to|talk to|lawyer|legal|complain\w*|angry|furious|cancel (my|our) contract)\b",
    re.IGNORECASE,
)
CREDIT_RE = re.compile(r"\b(credit memo|credit note|refund|credit)\b", re.IGNORECASE)
KB_RE = re.compile(
    r"\b(how|what|why|policy|steps|process|explain|fee|fees|terms|sla|can i|where|when|does|do|can|is there)\b",
    re.IGNORECASE,
)
# With a record ID in the message, only clear "how/why/policy" wording also triggers a KB search.
KB_WITH_ID_RE = re.compile(
    r"\b(how|why|policy|steps|process|explain|fee|fees|terms|sla|can i|release|released)\b", re.IGNORECASE
)
AMOUNT_RE = re.compile(r"(?:\$\s?([\d,]+(?:\.\d+)?))|(?:([\d,]+(?:\.\d+)?)\s*(?:usd|dollars))", re.IGNORECASE)

_TOOL_FOR_PREFIX = {
    "INV": ("get_invoice_status", "invoice_id"),
    "ORD": ("get_order_status", "order_id"),
    "TCK": ("lookup_ticket", "ticket_id"),
}


def _key(name: str, args: dict[str, Any]) -> str:
    return name + json.dumps(args, sort_keys=True)


class MockLLM:
    name = "mock-planner"

    def generate(self, system: str, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        last_user = max(i for i, m in enumerate(messages) if m.role == "user")
        text = messages[last_user].content
        turn = messages[last_user + 1 :]
        done = {_key(c.name, c.args) for m in turn for c in m.tool_calls}
        results = [(m.name, json.loads(m.content)) for m in turn if m.role == "tool"]

        plan = self._plan(text)
        # Dynamic step: a knowledge-base miss means we cannot answer, so hand off to a human.
        kb_missed = any(n == "search_kb" and not r.get("found") for n, r in results)
        if kb_missed and not any(p[0] == "escalate_to_human" for p in plan):
            plan.append(self._escalation(text, "No knowledge base answer found"))
        for name, args in plan:
            if _key(name, args) not in done:
                return LLMResponse(tool_calls=[ToolCall(name, args, id=f"call_{len(done)}")])
        return LLMResponse(text=self._compose(text, results))

    # --- planning ------------------------------------------------------------

    def _escalation(self, text: str, reason: str) -> tuple[str, dict[str, Any]]:
        priority = "high" if re.search(r"\b(angry|furious|lawyer|legal|cancel)\b", text, re.I) else "medium"
        return "escalate_to_human", {"reason": reason, "summary": text.strip()[:300], "priority": priority}

    def _plan(self, text: str) -> list[tuple[str, dict[str, Any]]]:
        plan: list[tuple[str, dict[str, Any]]] = []
        ids = []
        for m in ID_RE.finditer(text):
            ident = m.group(0).upper()
            if ident not in ids:
                ids.append(ident)
        for ident in ids:
            tool, arg = _TOOL_FOR_PREFIX[ident[:3]]
            plan.append((tool, {arg: ident}))
        invoices = [i for i in ids if i.startswith("INV")]
        if CREDIT_RE.search(text) and invoices:
            amount = self._amount(text)
            if amount is not None:
                plan.append(
                    (
                        "request_credit_memo",
                        {"invoice_id": invoices[0], "amount": amount, "reason": "Customer-requested credit"},
                    )
                )
        stripped = ID_RE.sub("", text)
        wants_escalation = bool(ESCALATE_RE.search(text))
        wants_kb = (
            bool(KB_WITH_ID_RE.search(stripped)) if ids else (bool(KB_RE.search(stripped)) or not wants_escalation)
        )
        if wants_kb:
            plan.append(("search_kb", {"query": stripped.strip()}))
        if wants_escalation:
            plan.append(self._escalation(text, "Customer asked for a human or issue needs review"))
        return plan

    @staticmethod
    def _amount(text: str) -> float | None:
        m = AMOUNT_RE.search(ID_RE.sub("", text))
        if not m:
            return None
        return float((m.group(1) or m.group(2)).replace(",", ""))

    # --- composing the final answer -------------------------------------------

    def _compose(self, text: str, results: list[tuple[str, dict[str, Any]]]) -> str:
        lines: list[str] = []
        for name, r in results:
            if r.get("error") == "not_found":
                lines.append(f"I couldn't find {r['id']} on your account, please double-check the number.")
            elif r.get("error"):
                lines.append(f"That request was rejected: {r.get('detail', r['error'])}.")
            elif name == "get_invoice_status":
                s = f"Invoice {r['id']} is {r['status'].replace('_', ' ')}: {r['amount']:,.2f} {r['currency']}, due {r['due_date']}"
                if r["days_overdue"]:
                    s += f" ({r['days_overdue']} days overdue)"
                if r["hold_reason"]:
                    s += f". Reason: {r['hold_reason']}"
                lines.append(s + ".")
            elif name == "get_order_status":
                s = f"Order {r['id']} is {r['status']}"
                if r["eta"]:
                    s += f", ETA {r['eta']}"
                if r["carrier"]:
                    s += f" via {r['carrier']} (tracking {r['tracking']})"
                lines.append(s + ".")
            elif name == "lookup_ticket":
                lines.append(
                    f"Ticket {r['id']} ({r['priority']} priority) is {r['status']}: {r['subject']}. "
                    f"Latest note: {r['notes'][-1]}"
                )
            elif name == "search_kb":
                for top in top_results(r):
                    lines.append(f"{top['text']} [source: {top['ref']}]")
            elif name == "request_credit_memo":
                if r.get("status") == "denied_by_human":
                    lines.append("A reviewer declined the credit memo request, so nothing was issued.")
                else:
                    lines.append(
                        f"Credit memo {r['id']} for {r['amount']:,.2f} on {r['invoice_id']} "
                        f"was approved by a human reviewer (status: {r['status'].replace('_', ' ')})."
                    )
            elif name == "escalate_to_human":
                lines.append(
                    f"I've handed this to a human agent ({r['id']}, {r['priority']} priority). "
                    "They will follow up with you."
                )
        if CREDIT_RE.search(text) and ID_RE.search(text) and self._amount(text) is None:
            lines.append("How much would you like credited?")
        if not results:
            return "Could you share an invoice, order or ticket number, or describe the issue in more detail?"
        return " ".join(lines)
