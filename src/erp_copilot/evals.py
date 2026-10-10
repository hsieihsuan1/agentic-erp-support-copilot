"""Evaluation harness: behavioural regression cases for the agent plus retrieval metrics.

Each case checks observable behaviour (which tools ran, whether a human was involved,
what the answer contains) rather than exact wording, so it also works with a real model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import Agent, AgentResult, approve_all, deny_all
from .llm import LLMClient
from .rag import Retriever

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "evals" / "cases.json"
DEFAULT_RETRIEVAL = Path(__file__).resolve().parents[2] / "evals" / "retrieval.json"


@dataclass
class CaseOutcome:
    id: str
    category: str
    passed: bool
    failures: list[str] = field(default_factory=list)


def check_case(case: dict[str, Any], res: AgentResult) -> list[str]:
    """Return a list of failure descriptions (empty means the case passed)."""
    fails: list[str] = []
    answer = res.answer.lower()
    for name in case.get("expect_tools", []):
        if name not in res.tool_names:
            fails.append(f"expected tool '{name}' to run, got {res.tool_names}")
    for name in case.get("forbid_tools", []):
        if name in res.tool_names:
            fails.append(f"forbidden tool '{name}' ran")
    for text in case.get("answer_contains", []):
        # "a|b" means any one alternative is acceptable (real models word things differently)
        if not any(alt.strip().lower() in answer for alt in text.split("|")):
            fails.append(f"answer missing '{text}'")
    for text in case.get("answer_excludes", []):
        if text.lower() in answer:
            fails.append(f"answer must not contain '{text}'")
    if "expect_escalation" in case and res.escalated != case["expect_escalation"]:
        fails.append(f"escalated={res.escalated}, expected {case['expect_escalation']}")
    if "expect_blocked" in case and res.blocked_reason != case["expect_blocked"]:
        fails.append(f"blocked_reason={res.blocked_reason}, expected {case['expect_blocked']}")
    if "expect_approval" in case and res.approval_requested != case["expect_approval"]:
        fails.append(f"approval_requested={res.approval_requested}, expected {case['expect_approval']}")
    if "expect_citation" in case and case["expect_citation"] not in ",".join(res.citations):
        fails.append(f"missing citation '{case['expect_citation']}', got {res.citations}")
    if case.get("expect_injection_neutralized") and not res.injection_neutralized:
        fails.append("tool output injection was not neutralised")
    if case.get("max_steps") and res.steps > case["max_steps"]:
        fails.append(f"took {res.steps} steps, max {case['max_steps']}")
    return fails


def run_cases(llm: LLMClient, cases: list[dict[str, Any]]) -> list[CaseOutcome]:
    outcomes: list[CaseOutcome] = []
    for case in cases:
        approver = approve_all if case.get("approval", "deny") == "approve" else deny_all
        agent = Agent(llm, customer_id=case.get("customer", "C-1001"), approver=approver)
        res = agent.run(case["input"])
        fails = check_case(case, res)
        outcomes.append(CaseOutcome(case["id"], case.get("category", "general"), not fails, fails))
    return outcomes


def retrieval_metrics(retriever: Retriever, pairs: list[dict[str, str]], k: int = 3) -> dict[str, float]:
    """recall@k and MRR of the expected document for each question."""
    hits_at_k, rr = 0, 0.0
    for p in pairs:
        ranked = [h.chunk.doc_id for h in retriever.search(p["question"], k=k)]
        if p["doc"] in ranked:
            hits_at_k += 1
            rr += 1 / (ranked.index(p["doc"]) + 1)
    n = len(pairs) or 1
    return {f"recall@{k}": round(hits_at_k / n, 3), "mrr": round(rr / n, 3), "questions": float(len(pairs))}


def summarize(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    by_cat: dict[str, list[bool]] = {}
    for o in outcomes:
        by_cat.setdefault(o.category, []).append(o.passed)
    total = len(outcomes) or 1
    return {
        "cases": len(outcomes),
        "passed": sum(o.passed for o in outcomes),
        "pass_rate": round(sum(o.passed for o in outcomes) / total, 3),
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())},
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
