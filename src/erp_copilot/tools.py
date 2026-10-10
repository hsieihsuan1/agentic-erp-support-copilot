"""Agent tools. Each tool has a JSON schema, is scoped to the signed-in customer, and
write tools can require human approval before they run."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from .db import AS_OF
from .rag import Retriever

# Below this BM25 score the knowledge base is treated as "no answer" and the agent escalates.
KB_MIN_SCORE = 2.0
# Heuristic thresholds, tuned on this small synthetic corpus (see README "Honest scope notes").
KB_MIN_COVERAGE = 0.35


def top_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunks to quote: the best hit plus a close runner-up (within 80 percent of its score)."""
    rows: list[dict[str, Any]] = result.get("results", [])
    if not result.get("found") or not rows:
        return []
    return [r for r in rows[:2] if r["score"] >= 0.8 * rows[0]["score"]]


ToolFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    requires_approval: bool = False

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


class ToolRegistry:
    def __init__(self, conn: sqlite3.Connection, customer_id: str, retriever: Retriever) -> None:
        self.conn = conn
        self.customer_id = customer_id
        self.retriever = retriever
        self._tools: dict[str, Tool] = {}
        for tool in self._build():
            self._tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def validate(self, name: str, args: dict[str, Any]) -> str | None:
        """Return an error string if the call is invalid, else None."""
        tool = self._tools.get(name)
        if tool is None:
            return f"unknown tool '{name}'"
        for key in tool.parameters.get("required", []):
            if key not in args or args[key] in (None, ""):
                return f"missing required argument '{key}'"
        for key in args:
            if key not in tool.parameters["properties"]:
                return f"unexpected argument '{key}'"
        return None

    # --- tool implementations -------------------------------------------------

    def _build(self) -> list[Tool]:
        return [
            Tool(
                "search_kb",
                "Search the ERP knowledge base for policies and how-to guidance.",
                _obj({"query": {"type": "string"}}, ["query"]),
                self._search_kb,
            ),
            Tool(
                "get_invoice_status",
                "Look up an invoice (status, amount, due date, hold reason) by ID like INV-1001.",
                _obj({"invoice_id": {"type": "string"}}, ["invoice_id"]),
                self._get_invoice,
            ),
            Tool(
                "get_order_status",
                "Look up an order (status, ETA, tracking) by ID like ORD-2001.",
                _obj({"order_id": {"type": "string"}}, ["order_id"]),
                self._get_order,
            ),
            Tool(
                "lookup_ticket",
                "Look up a support ticket by ID like TCK-3001.",
                _obj({"ticket_id": {"type": "string"}}, ["ticket_id"]),
                self._lookup_ticket,
            ),
            Tool(
                "request_credit_memo",
                "Request a credit memo against an invoice. Requires human approval.",
                _obj(
                    {
                        "invoice_id": {"type": "string"},
                        "amount": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    ["invoice_id", "amount", "reason"],
                ),
                self._credit_memo,
                requires_approval=True,
            ),
            Tool(
                "escalate_to_human",
                "Hand the conversation to a human agent with a short summary.",
                _obj(
                    {
                        "reason": {"type": "string"},
                        "summary": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    },
                    ["reason", "summary"],
                ),
                self._escalate,
            ),
        ]

    def _search_kb(self, args: dict[str, Any]) -> dict[str, Any]:
        hits = self.retriever.search(str(args["query"]), k=3)
        found = bool(hits) and hits[0].score >= KB_MIN_SCORE and hits[0].coverage >= KB_MIN_COVERAGE
        return {
            "found": found,
            "results": [
                {
                    "ref": h.chunk.ref,
                    "title": h.chunk.title,
                    "section": h.chunk.section,
                    "text": h.chunk.text,
                    "score": h.score,
                    "coverage": h.coverage,
                }
                for h in hits
            ],
        }

    def _row(self, table: str, key: str) -> sqlite3.Row | None:
        # Tenant isolation: rows of other customers look exactly like missing rows.
        cur = self.conn.execute(
            f"SELECT * FROM {table} WHERE id = ? AND customer_id = ?",  # noqa: S608 (fixed names)
            (key.upper(), self.customer_id),
        )
        row: sqlite3.Row | None = cur.fetchone()
        return row

    def _get_invoice(self, args: dict[str, Any]) -> dict[str, Any]:
        row = self._row("invoices", str(args["invoice_id"]))
        if row is None:
            return {"error": "not_found", "id": str(args["invoice_id"]).upper()}
        days_overdue = 0
        if row["status"] == "overdue":
            days_overdue = (date.fromisoformat(AS_OF) - date.fromisoformat(row["due_date"])).days
        return {
            "id": row["id"],
            "status": row["status"],
            "amount": row["amount"],
            "currency": row["currency"],
            "due_date": row["due_date"],
            "days_overdue": days_overdue,
            "hold_reason": row["hold_reason"],
            "po_number": row["po_number"],
        }

    def _get_order(self, args: dict[str, Any]) -> dict[str, Any]:
        row = self._row("orders", str(args["order_id"]))
        if row is None:
            return {"error": "not_found", "id": str(args["order_id"]).upper()}
        return {k: row[k] for k in ("id", "status", "eta", "carrier", "tracking", "invoice_id")}

    def _lookup_ticket(self, args: dict[str, Any]) -> dict[str, Any]:
        row = self._row("tickets", str(args["ticket_id"]))
        if row is None:
            return {"error": "not_found", "id": str(args["ticket_id"]).upper()}
        return {
            "id": row["id"],
            "subject": row["subject"],
            "status": row["status"],
            "priority": row["priority"],
            "notes": json.loads(row["notes"]),
        }

    def _credit_memo(self, args: dict[str, Any]) -> dict[str, Any]:
        invoice = self._row("invoices", str(args["invoice_id"]))
        if invoice is None:
            return {"error": "not_found", "id": str(args["invoice_id"]).upper()}
        amount = float(args["amount"])
        if amount <= 0 or amount > invoice["amount"]:
            return {
                "error": "invalid_amount",
                "detail": f"amount must be > 0 and <= invoice total {invoice['amount']:.2f}",
            }
        memo_id = f"CM-{self.conn.execute('SELECT COUNT(*) FROM credit_memos').fetchone()[0] + 1:04d}"
        needs_manager = amount > 1000
        self.conn.execute(
            "INSERT INTO credit_memos VALUES (?,?,?,?,?)",
            (memo_id, invoice["id"], amount, str(args["reason"]), "pending_manager" if needs_manager else "approved"),
        )
        self.conn.commit()
        return {
            "id": memo_id,
            "invoice_id": invoice["id"],
            "amount": amount,
            "status": "pending_manager" if needs_manager else "approved",
        }

    def _escalate(self, args: dict[str, Any]) -> dict[str, Any]:
        esc_id = f"ESC-{self.conn.execute('SELECT COUNT(*) FROM escalations').fetchone()[0] + 1:04d}"
        priority = str(args.get("priority", "medium"))
        self.conn.execute(
            "INSERT INTO escalations VALUES (?,?,?,?,?,?)",
            (esc_id, self.customer_id, str(args["reason"]), str(args["summary"]), priority, "open"),
        )
        self.conn.commit()
        return {"id": esc_id, "priority": priority, "status": "open"}
