import json

from erp_copilot.llm import GeminiClient, Message, ToolCall


def test_request_mapping() -> None:
    msgs = [
        Message("user", "status of INV-1001"),
        Message("assistant", "", [ToolCall("get_invoice_status", {"invoice_id": "INV-1001"})]),
        Message("tool", json.dumps({"status": "open"}), name="get_invoice_status"),
    ]
    body = GeminiClient.build_request("sys", msgs, [{"name": "get_invoice_status"}])
    roles = [c["role"] for c in body["contents"]]
    assert roles == ["user", "model", "user"]
    assert body["contents"][1]["parts"][0]["functionCall"]["name"] == "get_invoice_status"
    assert body["contents"][2]["parts"][0]["functionResponse"]["response"] == {"result": {"status": "open"}}
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "get_invoice_status"


def test_response_mapping_and_injected_transport() -> None:
    seen: dict = {}

    def transport(url, headers, body):  # type: ignore[no-untyped-def]
        seen.update(url=url, headers=headers)
        return {
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "search_kb", "args": {"query": "late fee"}}}]}}
            ]
        }

    out = GeminiClient("k", "m", transport=transport).generate("s", [Message("user", "hi")], [])
    assert out.tool_calls[0].name == "search_kb" and out.tool_calls[0].args == {"query": "late fee"}
    assert seen["headers"] == {"x-goog-api-key": "k"} and "models/m:generateContent" in seen["url"]
