import json

from erp_copilot import db
from erp_copilot.rag import Retriever
from erp_copilot.tools import ToolRegistry


def registry(customer: str = "C-1001") -> ToolRegistry:
    return ToolRegistry(db.connect(), customer, Retriever.from_dir())


def test_invoice_overdue_days_use_fixed_date() -> None:
    out = registry().get("get_invoice_status").fn({"invoice_id": "inv-1002"})  # type: ignore[union-attr]
    assert out["status"] == "overdue" and out["days_overdue"] == 37


def test_other_tenant_looks_like_not_found() -> None:
    out = registry("C-1001").get("get_invoice_status").fn({"invoice_id": "INV-1004"})  # type: ignore[union-attr]
    assert out == {"error": "not_found", "id": "INV-1004"}


def test_validation_catches_bad_calls() -> None:
    r = registry()
    assert r.validate("nope", {}) == "unknown tool 'nope'"
    assert "missing" in (r.validate("get_order_status", {}) or "")
    assert "unexpected" in (r.validate("get_order_status", {"order_id": "ORD-2001", "x": 1}) or "")
    assert r.validate("get_order_status", {"order_id": "ORD-2001"}) is None


def test_credit_memo_rules() -> None:
    r = registry()
    memo = r.get("request_credit_memo")
    assert memo is not None and memo.requires_approval
    too_big = memo.fn({"invoice_id": "INV-1002", "amount": 99999, "reason": "x"})
    assert too_big["error"] == "invalid_amount"
    small = memo.fn({"invoice_id": "INV-1002", "amount": 500, "reason": "x"})
    assert small["status"] == "approved"
    large = memo.fn({"invoice_id": "INV-1002", "amount": 1500, "reason": "x"})
    assert large["status"] == "pending_manager"


def test_escalation_is_persisted() -> None:
    r = registry()
    out = r.get("escalate_to_human").fn({"reason": "r", "summary": "s"})  # type: ignore[union-attr]
    row = r.conn.execute("SELECT status FROM escalations WHERE id = ?", (out["id"],)).fetchone()
    assert row["status"] == "open"


def test_tool_specs_are_json_serialisable() -> None:
    json.dumps(registry().specs())
