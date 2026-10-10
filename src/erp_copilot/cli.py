"""Command line interface: ask, chat, eval, kb-search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent, ToolCall, approve_all, deny_all
from .evals import (
    DEFAULT_CASES,
    DEFAULT_RETRIEVAL,
    load_json,
    retrieval_metrics,
    run_cases,
    summarize,
)
from .llm import make_client
from .rag import Retriever


def _interactive_approver(call: ToolCall, summary: str) -> bool:
    print(f"\n  [human approval needed] {summary}")
    return input("  approve? [y/N] ").strip().lower() in {"y", "yes"}


def _approver(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    if args.auto_approve:
        return approve_all
    if args.auto_deny or not sys.stdin.isatty():
        return deny_all
    return _interactive_approver


def _print_result(res, show_trace: bool) -> None:  # type: ignore[no-untyped-def]
    print(f"\n{res.answer}\n")
    if res.tool_calls:
        print("tools: " + ", ".join(res.tool_names))
    if res.citations:
        print("citations: " + ", ".join(res.citations))
    if res.blocked_reason:
        print(f"blocked: {res.blocked_reason}")
    if show_trace and res.tracer:
        print("\n" + res.tracer.render())


def cmd_ask(args: argparse.Namespace) -> int:
    agent = Agent(make_client(args.provider), customer_id=args.customer, approver=_approver(args))
    res = agent.run(args.question)
    _print_result(res, args.trace)
    if args.trace_file and res.tracer:
        res.tracer.export_jsonl(Path(args.trace_file))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    agent = Agent(make_client(args.provider), customer_id=args.customer, approver=_approver(args))
    print(f"ERP support copilot ({agent.llm.name}), customer {args.customer}. Ctrl-D to exit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            return 0
        if text:
            _print_result(agent.run(text), args.trace)


def cmd_eval(args: argparse.Namespace) -> int:
    llm = make_client(args.provider)
    cases = load_json(Path(args.cases))
    outcomes = run_cases(llm, cases)
    for o in outcomes:
        print(f"{'PASS' if o.passed else 'FAIL'}  [{o.category}] {o.id}")
        for f in o.failures:
            print(f"        - {f}")
    summary = summarize(outcomes)
    retrieval = retrieval_metrics(Retriever.from_dir(), load_json(Path(args.retrieval)))
    print(f"\nmodel: {llm.name}")
    print(f"cases: {summary['passed']}/{summary['cases']} passed (pass rate {summary['pass_rate']})")
    for cat, score in summary["by_category"].items():
        print(f"  {cat:<18} {score}")
    print(
        f"retrieval: recall@3={retrieval['recall@3']} mrr={retrieval['mrr']} ({int(retrieval['questions'])} questions)"
    )
    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "model": llm.name,
                    "summary": summary,
                    "retrieval": retrieval,
                    "failures": {o.id: o.failures for o in outcomes if not o.passed},
                },
                indent=2,
            )
        )
    ok = summary["pass_rate"] >= args.min_pass and retrieval["recall@3"] >= args.min_recall
    if not ok:
        print("\nEVAL GATE FAILED")
    return 0 if ok else 1


def cmd_kb_search(args: argparse.Namespace) -> int:
    for h in Retriever.from_dir().search(args.query, k=args.k):
        print(f"{h.score:7.3f}  {h.chunk.ref}\n         {h.chunk.text[:110]}...")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="erp-copilot", description="Agentic ERP support copilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--provider", choices=["auto", "mock", "gemini"], default="auto")

    def session(sp: argparse.ArgumentParser) -> None:
        common(sp)
        sp.add_argument("--customer", default="C-1001", help="signed-in customer (C-1001..C-1003)")
        sp.add_argument("--trace", action="store_true", help="print the span tree")
        sp.add_argument("--auto-approve", action="store_true")
        sp.add_argument("--auto-deny", action="store_true")

    a = sub.add_parser("ask", help="ask one question")
    session(a)
    a.add_argument("question")
    a.add_argument("--trace-file", help="append spans as JSONL")
    a.set_defaults(func=cmd_ask)

    c = sub.add_parser("chat", help="interactive session")
    session(c)
    c.set_defaults(func=cmd_chat)

    e = sub.add_parser("eval", help="run behaviour + retrieval evals")
    e.add_argument("--provider", choices=["auto", "mock", "gemini"], default="mock")
    e.add_argument("--cases", default=str(DEFAULT_CASES))
    e.add_argument("--retrieval", default=str(DEFAULT_RETRIEVAL))
    e.add_argument("--min-pass", type=float, default=1.0)
    e.add_argument("--min-recall", type=float, default=0.9)
    e.add_argument("--report", help="write a JSON report")
    e.set_defaults(func=cmd_eval)

    k = sub.add_parser("kb-search", help="debug the retriever")
    k.add_argument("query")
    k.add_argument("-k", type=int, default=3)
    k.set_defaults(func=cmd_kb_search)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
