from erp_copilot.cli import main
from erp_copilot.evals import (
    DEFAULT_CASES,
    DEFAULT_RETRIEVAL,
    load_json,
    retrieval_metrics,
    run_cases,
    summarize,
)
from erp_copilot.llm import MockLLM
from erp_copilot.rag import Retriever


def test_all_behaviour_cases_pass_offline() -> None:
    outcomes = run_cases(MockLLM(), load_json(DEFAULT_CASES))
    failed = {o.id: o.failures for o in outcomes if not o.passed}
    assert not failed, failed
    assert summarize(outcomes)["cases"] >= 20


def test_retrieval_quality_gate() -> None:
    m = retrieval_metrics(Retriever.from_dir(), load_json(DEFAULT_RETRIEVAL))
    assert m["recall@3"] >= 0.9 and m["mrr"] >= 0.8


def test_eval_gate_catches_regressions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "cases.json"
    bad.write_text('[{"id": "x", "input": "Where is ORD-2001?", "answer_contains": ["teleported"]}]')
    assert main(["eval", "--cases", str(bad)]) == 1


def test_kb_misses_do_not_get_an_answer() -> None:
    from erp_copilot.agent import Agent

    res = Agent(MockLLM()).run("Does the system support single sign on with Okta?")
    assert res.escalated and res.citations == []
