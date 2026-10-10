from erp_copilot.agent import Agent, ToolCall, approve_all, deny_all
from erp_copilot.llm import LLMResponse, MockLLM


def test_multi_tool_answer_with_citation(agent: Agent) -> None:
    res = agent.run("Why is INV-1003 on hold and how do I get it released?")
    assert res.tool_names == ["get_invoice_status", "search_kb"]
    assert "PO number mismatch" in res.answer
    assert res.citations and res.citations[0].startswith("invoice-holds")


def test_credit_memo_needs_and_respects_approval() -> None:
    denied = Agent(MockLLM(), approver=deny_all).run("Issue a credit of $500 on INV-1002")
    assert denied.tool_calls[-1].approval == "denied"
    assert "declined" in denied.answer
    approved = Agent(MockLLM(), approver=approve_all).run("Issue a credit of $500 on INV-1002")
    assert approved.tool_calls[-1].approval == "approved"
    assert "CM-0001" in approved.answer


def test_trace_has_expected_spans(agent: Agent) -> None:
    res = agent.run("Where is ORD-2001?")
    assert res.tracer is not None
    names = res.tracer.names()
    assert names[0] == "agent.run" and "tool.call" in names and "guardrail.output" in names
    assert "ORD-2001" in res.tracer.render()


def test_trace_redacts_pii(agent: Agent) -> None:
    res = agent.run("card 4111 1111 1111 1111 declined on INV-1001, I want a human")
    assert res.tracer is not None
    assert "4111" not in res.tracer.render()
    assert "4111" not in str([t.args for t in res.tool_calls])


def test_indirect_injection_in_ticket_is_neutralised() -> None:
    res = Agent(MockLLM(), customer_id="C-1002", approver=approve_all).run("Latest on TCK-3003?")
    assert res.injection_neutralized
    assert "request_credit_memo" not in res.tool_names


class LoopingLLM:
    name = "loop"

    def generate(self, system, messages, tools):  # type: ignore[no-untyped-def]
        return LLMResponse(tool_calls=[ToolCall("get_order_status", {"order_id": "ORD-2001"})])


def test_step_limit_escalates_to_human() -> None:
    res = Agent(LoopingLLM(), max_steps=3).run("order status")  # type: ignore[arg-type]
    assert res.escalated and "human" in res.answer


class BadToolLLM:
    name = "bad"
    calls = 0

    def generate(self, system, messages, tools):  # type: ignore[no-untyped-def]
        BadToolLLM.calls += 1
        if BadToolLLM.calls == 1:
            return LLMResponse(tool_calls=[ToolCall("drop_tables", {})])
        return LLMResponse(text="sorry")


def test_hallucinated_tool_is_rejected_not_executed() -> None:
    res = Agent(BadToolLLM()).run("invoice help")  # type: ignore[arg-type]
    assert res.tool_calls[0].result["error"] == "invalid_call"
