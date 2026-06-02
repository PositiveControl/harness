from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from harness.model.adapter import ChatMessage, ModelAdapter
from harness.model.vllm import VllmAdapter
from harness.tools.base import (
    ModelReply,
    StreamComplete,
    StreamText,
    ToolCall,
    ToolSpec,
)

# ---------- mock plumbing -------------------------------------------------


def _make_factory(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., httpx.Client]:
    """Construct a real httpx.Client wired to a MockTransport. We patch
    `httpx.Client` to this factory so the adapter's `with httpx.Client(...)`
    contexts route every request through `handler`.

    Capture the unpatched `httpx.Client` in the closure so the factory
    can instantiate the real class without recursing through its own
    patched binding."""
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    return factory


def _json_response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def _sse_response(frames: list[dict[str, Any] | str]) -> httpx.Response:
    """Build an SSE body. Each frame becomes `data: <body>\\n\\n`.
    String frames pass through verbatim (used for `[DONE]`)."""
    lines: list[str] = []
    for frame in frames:
        body = frame if isinstance(frame, str) else json.dumps(frame)
        lines.append(f"data: {body}\n\n")
    return httpx.Response(
        200,
        content="".join(lines).encode("utf-8"),
        headers={"content-type": "text/event-stream"},
    )


# ---------- construction / protocol --------------------------------------


def test_vllm_adapter_satisfies_protocol() -> None:
    adapter = VllmAdapter(model="Qwen/Qwen2.5-32B-Instruct-AWQ")
    assert isinstance(adapter, ModelAdapter)
    assert adapter.id == "vllm:Qwen/Qwen2.5-32B-Instruct-AWQ"
    assert adapter.context_window > 0


def test_vllm_adapter_strips_trailing_slash() -> None:
    adapter = VllmAdapter(base_url="http://localhost:8000/v1/")
    assert adapter.base_url == "http://localhost:8000/v1"


def test_vllm_adapter_does_no_network_on_construction() -> None:
    with patch("httpx.Client") as mock_client:
        VllmAdapter(model="some-model", base_url="http://nowhere:8000/v1")
    mock_client.assert_not_called()


def test_vllm_adapter_id_reflects_base_url_when_model_unknown() -> None:
    adapter = VllmAdapter(base_url="http://gx10-1.tailnet:8000/v1")
    assert adapter.id == "vllm:http://gx10-1.tailnet:8000/v1"


# ---------- model discovery ----------------------------------------------


def test_vllm_discovers_model_via_v1_models_when_unspecified() -> None:
    served = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return _json_response({"object": "list", "data": [{"id": served, "object": "model"}]})

    adapter = VllmAdapter()
    with patch("httpx.Client", _make_factory(handler)):
        assert adapter.model == served
    assert captured["url"] == "http://localhost:8000/v1/models"
    assert captured["method"] == "GET"
    assert adapter.id == f"vllm:{served}"


def test_vllm_raises_when_v1_models_returns_empty_list() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response({"object": "list", "data": []})

    adapter = VllmAdapter()
    with (
        patch("httpx.Client", _make_factory(handler)),
        pytest.raises(RuntimeError, match="no models loaded"),
    ):
        _ = adapter.model


# ---------- complete (non-streaming, no tools) ---------------------------


def test_vllm_complete_posts_expected_payload() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi there"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete(
            [ChatMessage(role="user", content="hi")],
            max_tokens=32,
            temperature=0.2,
        )

    assert reply == "hi there"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["method"] == "POST"
    body = captured["body"]
    assert body["model"] == "m"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 32
    assert body["temperature"] == pytest.approx(0.2)


def test_vllm_complete_raises_on_unreachable_server() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    adapter = VllmAdapter(model="m", base_url="http://127.0.0.1:1/v1")
    with (
        patch("httpx.Client", _make_factory(handler)),
        pytest.raises(RuntimeError, match="Cannot reach vLLM"),
    ):
        adapter.complete([ChatMessage(role="user", content="hi")])


def test_vllm_complete_raises_on_http_error_with_detail() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "OOM"})

    adapter = VllmAdapter(model="m")
    with (
        patch("httpx.Client", _make_factory(handler)),
        pytest.raises(RuntimeError, match="HTTP 500"),
    ):
        adapter.complete([ChatMessage(role="user", content="hi")])


# ---------- complete_with_tools ------------------------------------------


def _weather_spec() -> ToolSpec:
    return ToolSpec(
        name="weather",
        description="get weather",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        tier="read",
    )


def test_vllm_complete_with_tools_round_trip() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_0",
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": json.dumps({"city": "SF"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="weather in SF?")],
            tools=[_weather_spec()],
        )

    assert isinstance(reply, ModelReply)
    assert reply.content == ""
    assert reply.tool_calls == (ToolCall(name="weather", arguments={"city": "SF"}),)
    assert reply.was_truncated is False
    # Tool spec was carried through unchanged.
    sent_tools = captured["body"]["tools"]
    assert sent_tools[0]["function"]["name"] == "weather"


def test_vllm_complete_with_tools_marks_truncation_on_length() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "half a thoug"},
                        "finish_reason": "length",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools([ChatMessage(role="user", content="hi")])

    assert reply.was_truncated is True
    assert reply.content == "half a thoug"


def test_vllm_renders_assistant_tool_calls_and_tool_role_pairing() -> None:
    """A second-round request that echoes a prior assistant tool_calls
    turn + the tool-role result must produce OpenAI-shaped messages
    with matching synthetic tool_call_id pairs so vLLM's chat template
    can attribute them."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    history = [
        ChatMessage(role="user", content="weather in SF?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(ToolCall(name="weather", arguments={"city": "SF"}),),
        ),
        ChatMessage(
            role="tool",
            content='{"temp":62}',
            name="weather",
            tool_call_id="call_0_weather",
        ),
    ]
    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        adapter.complete_with_tools(history)

    msgs = captured["body"]["messages"]
    # Assistant turn carries tool_calls + content=None (no text).
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] is None
    assert msgs[1]["tool_calls"][0]["id"] == "call_0_weather"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "weather"
    # Arguments serialized as JSON STRING, not dict (OpenAI spec).
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == '{"city":"SF"}'
    # Tool role carries the matching id.
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "call_0_weather"
    assert msgs[2]["name"] == "weather"


# ---------- streaming ----------------------------------------------------


def test_vllm_stream_yields_content_deltas() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
                {"choices": [{"index": 0, "delta": {"content": "lo!"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        text = "".join(adapter.stream([ChatMessage(role="user", content="say hi")]))
    assert text == "Hello!"


def test_vllm_stream_surfaces_http_error_body_not_response_not_read() -> None:
    """harness-gtd0: when vLLM rejects a streaming request (e.g. the
    context-length validator returns 400), the adapter must surface the
    server's error body in the RuntimeError — not the misleading
    httpx.ResponseNotRead that you get from touching .text on an
    un-consumed streaming response.

    The historical failure surfaced on the client as `ResponseNotRead`
    instead of the server's error body. (Context-length 4xx rejections
    take the dedicated PromptBudgetError path — covered separately; this
    test uses a generic 400 to pin the body-surfacing fix itself.)
    """
    detail = (
        '{"error": {"message": "malformed sampling parameters: temperature '
        'must be >= 0", "type": "validation_error", "code": 400}}'
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            content=detail.encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    adapter = VllmAdapter(model="m")
    with (
        patch("httpx.Client", _make_factory(handler)),
        pytest.raises(RuntimeError) as excinfo,
    ):
        # Streaming endpoint — consume the iterator to drive the request.
        list(adapter.stream([ChatMessage(role="user", content="x" * 10_000)]))
    msg = str(excinfo.value)
    assert "HTTP 400" in msg
    assert "malformed sampling parameters" in msg
    # The original symptom — make sure the new path doesn't regress to it.
    assert "ResponseNotRead" not in msg


def test_vllm_stream_with_tools_accumulates_tool_call_deltas() -> None:
    """Tool calls arrive across multiple SSE frames in streaming mode:
    name lands on the first frame for an index; arguments build up
    across subsequent frames as a JSON-string fragment stream."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                # Tool call partial 1: name + opening of args.
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "type": "function",
                                        "function": {
                                            "name": "weather",
                                            "arguments": '{"city":',
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                },
                # Partial 2: remainder of args.
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": '"SF"}'}}]
                            },
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        chunks = list(
            adapter.stream_with_tools(
                [ChatMessage(role="user", content="weather in SF?")],
                tools=[_weather_spec()],
            )
        )

    # Last chunk is a StreamComplete; preceding chunks (if any) are
    # StreamText for the empty role frame they pass on as content="".
    assert isinstance(chunks[-1], StreamComplete)
    # The role frame had content="" — adapter shouldn't yield empty text.
    text_chunks = [c for c in chunks if isinstance(c, StreamText)]
    assert all(c.text for c in text_chunks)

    reply = chunks[-1].reply
    assert reply.content == ""
    assert reply.tool_calls == (ToolCall(name="weather", arguments={"city": "SF"}),)
    assert reply.was_truncated is False


def test_vllm_stream_with_tools_yields_text_for_normal_reply() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"index": 0, "delta": {"content": "no tool"}}]},
                {"choices": [{"index": 0, "delta": {"content": " needed"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        chunks = list(
            adapter.stream_with_tools(
                [ChatMessage(role="user", content="hi")],
                tools=[_weather_spec()],
            )
        )

    text_chunks = [c for c in chunks if isinstance(c, StreamText)]
    assert "".join(c.text for c in text_chunks) == "no tool needed"
    assert isinstance(chunks[-1], StreamComplete)
    assert chunks[-1].reply.tool_calls == ()
    assert chunks[-1].reply.content == "no tool needed"


# ---------- trace logger (harness-97mq) ----------------------------------


def test_vllm_trace_no_env_no_writes(tmp_path: Any) -> None:
    """When HARNESS_VLLM_TRACE is unset, no file is written. The trace
    logger must be opt-in — a misconfigured developer machine should
    never accumulate trace records by default."""
    trace_target = tmp_path / "trace.jsonl"

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)), patch.dict("os.environ", {}, clear=False):
        # Ensure the env var really is unset for this branch.
        import os

        os.environ.pop("HARNESS_VLLM_TRACE", None)
        adapter.complete_with_tools([ChatMessage(role="user", content="hi")])

    assert not trace_target.exists()


def test_vllm_trace_complete_with_tools_writes_jsonl(tmp_path: Any) -> None:
    """complete_with_tools must append a JSONL record with the request,
    raw response, and parsed_calls when HARNESS_VLLM_TRACE is set."""
    trace_target = tmp_path / "trace.jsonl"

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_0",
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": json.dumps({"city": "SF"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with (
        patch("httpx.Client", _make_factory(handler)),
        patch.dict("os.environ", {"HARNESS_VLLM_TRACE": str(trace_target)}),
    ):
        adapter.complete_with_tools(
            [ChatMessage(role="user", content="weather in SF?")],
            tools=[_weather_spec()],
        )

    assert trace_target.exists()
    lines = trace_target.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["mode"] == "complete_with_tools"
    assert record["model"] == "m"
    assert record["request"]["tools"][0]["function"]["name"] == "weather"
    assert (
        record["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "weather"
    )
    assert record["parsed_calls"] == [{"name": "weather", "arguments": {"city": "SF"}}]
    assert "ts" in record


def test_vllm_trace_complete_writes_jsonl(tmp_path: Any) -> None:
    """harness-5t0a: the non-tool complete() path must trace too — this is
    what the driver critic + grounding-verify gate call, and it was
    invisible before."""
    trace_target = tmp_path / "trace.jsonl"

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "REFUTED: line is fine"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with (
        patch("httpx.Client", _make_factory(handler)),
        patch.dict("os.environ", {"HARNESS_VLLM_TRACE": str(trace_target)}),
    ):
        out = adapter.complete([ChatMessage(role="user", content="is this grounded?")])

    assert out == "REFUTED: line is fine"
    lines = trace_target.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["mode"] == "complete"
    assert record["model"] == "m"
    assert record["request"]["messages"][0]["content"] == "is this grounded?"
    assert record["response"]["choices"][0]["message"]["content"] == "REFUTED: line is fine"
    assert "ts" in record


def test_vllm_trace_stream_writes_jsonl(tmp_path: Any) -> None:
    """harness-5t0a: the non-tool stream() path traces the reassembled
    content once the generator is fully consumed."""
    trace_target = tmp_path / "trace.jsonl"

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {"choices": [{"index": 0, "delta": {"content": "hello "}}]},
                {"choices": [{"index": 0, "delta": {"content": "world"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with (
        patch("httpx.Client", _make_factory(handler)),
        patch.dict("os.environ", {"HARNESS_VLLM_TRACE": str(trace_target)}),
    ):
        chunks = list(adapter.stream([ChatMessage(role="user", content="hi")]))

    assert "".join(chunks) == "hello world"
    lines = trace_target.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["mode"] == "stream"
    assert record["model"] == "m"
    assert record["reassembled_message"]["content"] == "hello world"


def test_vllm_trace_stream_with_tools_writes_jsonl(tmp_path: Any) -> None:
    """Streaming path must also write a trace record — using
    reassembled_message + per-index tool_call slots, which is what the
    operator needs to inspect when a drive turn returns empty args."""
    trace_target = tmp_path / "trace.jsonl"

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"name": "weather", "arguments": ""},
                                    }
                                ]
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '{"city": "SF"}'},
                                    }
                                ]
                            },
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with (
        patch("httpx.Client", _make_factory(handler)),
        patch.dict("os.environ", {"HARNESS_VLLM_TRACE": str(trace_target)}),
    ):
        list(
            adapter.stream_with_tools(
                [ChatMessage(role="user", content="hi")],
                tools=[_weather_spec()],
            )
        )

    lines = trace_target.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["mode"] == "stream_with_tools"
    assert record["finish_reason"] == "tool_calls"
    assembled = record["reassembled_message"]["tool_calls"][0]
    assert assembled["function"]["name"] == "weather"
    assert assembled["function"]["arguments"] == '{"city": "SF"}'
    assert record["parsed_calls"] == [{"name": "weather", "arguments": {"city": "SF"}}]


def test_vllm_trace_bad_path_does_not_crash(tmp_path: Any) -> None:
    """If HARNESS_VLLM_TRACE points to an unwritable path, the trace
    logger must silently no-op rather than break the live turn. This
    is the contract — diagnostic, not load-bearing."""
    # Point trace at a path whose parent is a file (mkdir will fail).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    bad_path = blocker / "trace.jsonl"

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "fine"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with (
        patch("httpx.Client", _make_factory(handler)),
        patch.dict("os.environ", {"HARNESS_VLLM_TRACE": str(bad_path)}),
    ):
        reply = adapter.complete_with_tools([ChatMessage(role="user", content="hi")])

    # Turn succeeded despite trace failure.
    assert reply.content == "fine"


# ---------- factory + cli plumbing ---------------------------------------


def test_vllm_factory_make_adapter_returns_vllm() -> None:
    from harness.model.factory import make_adapter

    adapter = make_adapter("vllm")
    assert isinstance(adapter, VllmAdapter)
    assert adapter.base_url == "http://localhost:8000/v1"


def test_vllm_resolver_routes_model_repo_to_base_url() -> None:
    """`--model vllm --model-repo http://gx10-1.tailnet:8000/v1` is the
    intended Phase 1 invocation. Confirm the resolver maps model_repo
    to base_url (NOT to a model id) for the vllm adapter.

    The resolver invokes `.load()` to fail fast on an unreachable
    endpoint, so we stub /v1/models to return a fake served model and
    keep the test offline."""
    from harness.cli import _resolve_adapter

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response({"data": [{"id": "fake-served-model"}]})

    with patch("httpx.Client", _make_factory(handler)):
        adapter = _resolve_adapter("vllm", model_repo="http://gx10-1.tailnet:8000/v1")
    assert isinstance(adapter, VllmAdapter)
    assert adapter.base_url == "http://gx10-1.tailnet:8000/v1"
    assert adapter.id == "vllm:fake-served-model"


# ---------- count_tokens / load ------------------------------------------


def test_vllm_count_tokens_uses_char_heuristic() -> None:
    adapter = VllmAdapter(model="m")
    n = adapter.count_tokens([ChatMessage(role="user", content="hello world")])
    # Heuristic: len(content)//4 + 4 per message → 11//4 + 4 == 6.
    assert n == 6


def test_vllm_load_pings_v1_models() -> None:
    """load() is a no-op for vLLM (server already pre-loaded), but it
    should at least resolve the served model — making it the natural
    place to fail fast on an unreachable endpoint at chat startup."""

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return _json_response({"data": [{"id": "Q"}]})

    adapter = VllmAdapter()
    with patch("httpx.Client", _make_factory(handler)):
        adapter.load()
    assert seen == ["http://localhost:8000/v1/models"]


# ---------- qwen-tag fallback (vLLM parser-config independence) ----------


def test_vllm_falls_back_to_qwen_tools_tag_when_tool_calls_empty() -> None:
    """vLLM 0.21 + Qwen2.5-Coder-32B-Instruct-AWQ + --tool-call-parser hermes
    leaks `<tools>{...}</tools>` through as content with tool_calls=[].
    The adapter must reparse content as a fallback so the orchestrator
    still sees a structured call (live-smoke diagnostic 2026-05-22)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<tools>\n"
                                '{"name": "list_dir", "arguments": {"path": "."}}'
                                "\n</tools>"
                            ),
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="ls")],
            tools=[_weather_spec()],
        )

    assert reply.tool_calls == (ToolCall(name="list_dir", arguments={"path": "."}),)
    assert "<tools>" not in reply.content
    assert "list_dir" not in reply.content  # the JSON body should be gone too


def test_vllm_falls_back_to_qwen_tool_call_tag_when_tool_calls_empty() -> None:
    """Same fallback, but the Qwen2.5 standard wrapper this time."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<tool_call>"
                                '{"name":"weather","arguments":{"city":"SF"}}'
                                "</tool_call>"
                            ),
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="weather?")],
            tools=[_weather_spec()],
        )

    assert reply.tool_calls == (ToolCall(name="weather", arguments={"city": "SF"}),)


def test_vllm_does_not_double_parse_when_tool_calls_populated() -> None:
    """When vLLM extracts tool_calls structurally, trust them — don't
    also fallback-parse the content, even if it happens to contain
    JSON-shaped text. Prevents double-emission when both paths land."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": 'The result is: {"some": "json"}',
                            "tool_calls": [
                                {
                                    "id": "call_0",
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": json.dumps({"city": "SF"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="weather in SF?")],
            tools=[_weather_spec()],
        )

    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "weather"
    # Content is preserved verbatim — we didn't run the fallback parser.
    assert "The result is" in reply.content


def test_vllm_stream_with_tools_masks_tools_tag_and_recovers_call() -> None:
    """Live-stream variant: tag spans must not leak to visible
    StreamText deltas, and the terminal StreamComplete must carry the
    parsed call even though vLLM sent tool_calls=[]."""

    # Build the SSE deltas in pieces so the JSON body inside a single
    # quoted Python string doesn't confuse the parser. Each fragment
    # gets its own delta frame — simulates how vLLM streams a tool
    # call across multiple SSE messages.
    tools_open = "<tools>\n"
    json_name = '{"name": "list_dir", '
    json_args = '"arguments": {"path": "."}}'
    tools_close = "\n</tools>"

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"index": 0, "delta": {"content": tools_open}}]},
                {"choices": [{"index": 0, "delta": {"content": json_name}}]},
                {"choices": [{"index": 0, "delta": {"content": json_args}}]},
                {"choices": [{"index": 0, "delta": {"content": tools_close}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        chunks = list(
            adapter.stream_with_tools(
                [ChatMessage(role="user", content="ls")],
                tools=[_weather_spec()],
            )
        )

    text_chunks = [c for c in chunks if isinstance(c, StreamText)]
    visible = "".join(c.text for c in text_chunks)
    # The masker hides everything from `<tools>` through `</tools>`,
    # so the user never sees raw tool-call JSON mid-stream.
    assert "<tools>" not in visible
    assert "list_dir" not in visible
    assert "</tools>" not in visible

    final = chunks[-1]
    assert isinstance(final, StreamComplete)
    assert final.reply.tool_calls == (ToolCall(name="list_dir", arguments={"path": "."}),)


def test_vllm_falls_back_to_bare_json_when_no_wrapper_and_no_tool_calls() -> None:
    """Observed live (2026-05-22): vLLM 0.21 + Qwen2.5-Coder against
    the harness's persona+tools prompt sometimes drops the `<tools>`
    wrapper entirely and emits bare `{"name":…,"arguments":…}` in
    content. With tool_calls=[], the orchestrator sees raw JSON as
    prose and the tool never runs. The bare-JSON fallback handles
    this case, conditioned on `tools` being present in the request."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"name": "list_dir", "arguments": {}}',
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="ls")],
            tools=[_weather_spec()],
        )

    assert reply.tool_calls == (ToolCall(name="list_dir", arguments={}),)
    # Content emptied — the bare JSON WAS the tool call, nothing else
    # to surface as prose.
    assert reply.content == ""


def test_vllm_bare_json_fallback_does_not_fire_without_tools_in_request() -> None:
    """Safety guard: a plain chat reply that happens to contain only a
    JSON object should NOT be reinterpreted as a tool call. Only fire
    the bare-JSON fallback when tools were actually requested."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"name": "Alice", "arguments": {}}',
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        # tools=None → fallback must not fire.
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="give me a JSON object")],
            tools=None,
        )

    assert reply.tool_calls == ()
    assert reply.content == '{"name": "Alice", "arguments": {}}'


def test_vllm_bare_json_fallback_rejects_prose_with_json_inside() -> None:
    """A response like `Here's an example: {"foo": 1}` must not be
    reinterpreted — content doesn't trim to a bare JSON object."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": 'Here is the result: {"foo": 1}',
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="what's the result?")],
            tools=[_weather_spec()],
        )

    assert reply.tool_calls == ()
    assert "Here is the result" in reply.content


def test_vllm_bare_json_fallback_rejects_json_without_name_field() -> None:
    """A bare JSON object missing the `name` key isn't a tool call,
    even structurally — pass through as content."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"foo": "bar"}',
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="say something")],
            tools=[_weather_spec()],
        )

    assert reply.tool_calls == ()
    assert reply.content == '{"foo": "bar"}'


def test_vllm_extracts_bare_json_with_prose_preamble() -> None:
    """Observed live (2026-05-22): Qwen2.5-Coder emits a tool-intent
    sentence ("Let me check the directory.\\n") BEFORE the bare-JSON
    call. The renderer's _is_suppressible drops the preamble per-
    sentence, but without a permissive fallback the trailing JSON
    falls through to the user. The fallback must find the JSON
    anywhere in content, not just when it's the whole content."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                'Let me check the directory.\n{"name": "list_dir", "arguments": {}}'
                            ),
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="ls")],
            tools=[_weather_spec()],
        )

    assert reply.tool_calls == (ToolCall(name="list_dir", arguments={}),)
    # JSON stripped; preamble preserved so the renderer's per-sentence
    # filter can still drop "Let me check…" (it matches _TOOL_INTENT_RE
    # in the renderer's _is_suppressible).
    assert "{" not in reply.content
    assert "Let me check" in reply.content


def test_vllm_extracts_bare_json_with_trailing_special_token_leak() -> None:
    """Observed live: Coder-32B emits `<|im_start|>` runaway tokens AFTER
    the JSON tool call. The stop-token default truncates this server-
    side, but if the truncation doesn't fire the parser must still
    find the JSON in the middle of the leak."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<|im_start|>\n"
                                '{"name": "list_dir", "arguments": {"path": "scratch"}}\n'
                                "<|im_start|>"
                            ),
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="ls scratch")],
            tools=[_weather_spec()],
        )

    assert reply.tool_calls == (ToolCall(name="list_dir", arguments={"path": "scratch"}),)


def test_vllm_bare_json_lenient_recovers_unclosed_envelope() -> None:
    """harness-bwmd: on the harness-3jo1 drive halt (2026-05-23, loop
    3823cd0b, bail records 05:21:12 + 05:22:21), Qwen2.5-Coder 32B
    emitted a 2.2 KB edit_file call with multi-line code in
    `old_string`/`new_string`. The nested `{` and `}` inside those
    strings confused the model — it dropped the envelope's outer `}`
    before the trailing ``` fence. Strict json.raw_decode rejected;
    the lenient pad-and-retry path must extract the call.

    Repro shape: prose preamble, ```json fence, payload missing one
    closing `}`, ``` fence end, trailing narrative + teaser.
    """
    truncated_call = (
        '{"name": "edit_file", "arguments": '
        '{"path": "game.js", '
        '"old_string": "const x = 1;\\n}", '
        '"new_string": "const y = 2;\\n}"'
        "}"  # closes arguments dict — but envelope's outer } is MISSING
    )
    content = (
        "Here's the updated edit with more context:\n\n"
        "```json\n"
        f"{truncated_call}\n"
        "```\n\n"
        "This should correctly rename the variable.\n\n"
        "Let's apply this edit."
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="fix the conflict")],
            tools=[_weather_spec()],
        )

    assert reply.tool_calls == (
        ToolCall(
            name="edit_file",
            arguments={
                "path": "game.js",
                "old_string": "const x = 1;\n}",
                "new_string": "const y = 2;\n}",
            },
        ),
    )


def test_vllm_bare_json_lenient_skips_when_real_json_error() -> None:
    """harness-bwmd: the lenient pad-and-retry path only kicks in when
    the envelope was unclosed. A JSON whose braces are balanced but
    fails to parse for OTHER reasons (bad escape, control char in
    string) must NOT trigger pad-and-retry — that path is reserved for
    the specific 'model lost count of nested braces' shape."""

    # Balanced braces but invalid escape inside string.
    content = '{"name": "edit_file", "arguments": {"path": "x", "old_string": "bad\\zescape"}}'

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="x")],
            tools=[_weather_spec()],
        )

    # No tool extracted — lenient path correctly held back on a
    # balanced-but-malformed envelope. Reply content stays unchanged.
    assert reply.tool_calls == ()


def test_vllm_stream_suppresses_bare_json_after_prose_preamble() -> None:
    """Streamed variant of the prose-preamble case. The first text
    delta is prose ("Let me check.\\n") — shape detection cannot
    engage on `startswith("{")`. The new mid-content regex detects
    `{"name":` once it arrives across chunks and starts suppressing
    from that point."""
    preamble = "Let me check the directory.\n"
    json_open = '{"name": '
    json_close = '"list_dir", "arguments": {}}'

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"index": 0, "delta": {"content": preamble}}]},
                {"choices": [{"index": 0, "delta": {"content": json_open}}]},
                {"choices": [{"index": 0, "delta": {"content": json_close}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        chunks = list(
            adapter.stream_with_tools(
                [ChatMessage(role="user", content="ls")],
                tools=[_weather_spec()],
            )
        )

    text_chunks = [c for c in chunks if isinstance(c, StreamText)]
    visible = "".join(c.text for c in text_chunks)
    # The preamble may or may not be in visible depending on masker
    # timing, but the JSON must not leak.
    assert '"name"' not in visible
    assert "list_dir" not in visible

    final = chunks[-1]
    assert isinstance(final, StreamComplete)
    assert final.reply.tool_calls == (ToolCall(name="list_dir", arguments={}),)


def test_vllm_stream_with_tools_recovers_bare_json_call() -> None:
    """Streamed variant of the bare-JSON fallback."""
    bare_json = '{"name": "list_dir", "arguments": {}}'

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"index": 0, "delta": {"content": bare_json}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        chunks = list(
            adapter.stream_with_tools(
                [ChatMessage(role="user", content="ls")],
                tools=[_weather_spec()],
            )
        )

    final = chunks[-1]
    assert isinstance(final, StreamComplete)
    assert final.reply.tool_calls == (ToolCall(name="list_dir", arguments={}),)
    assert final.reply.content == ""


def test_vllm_includes_stop_tokens_in_payload_by_default() -> None:
    """`<|im_start|>` should be in the stop list — Qwen models hallucinate
    multi-turn conversations in a single completion under tool-use mode,
    and stopping at the role boundary truncates that cleanly. Observed
    live (2026-05-22): Coder-32B emitting `<|im_start|>` in content."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        adapter.complete([ChatMessage(role="user", content="hi")])

    assert "stop" in captured["body"]
    assert "<|im_start|>" in captured["body"]["stop"]


def test_vllm_stop_tokens_overridable() -> None:
    """A caller that knows better — e.g. a non-Qwen model — can pass
    an explicit stop tuple to override the default."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {"choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    adapter = VllmAdapter(model="m", stop=("[END]",))
    with patch("httpx.Client", _make_factory(handler)):
        adapter.complete([ChatMessage(role="user", content="hi")])

    assert captured["body"]["stop"] == ["[END]"]


def test_vllm_stop_empty_tuple_omits_field() -> None:
    """Passing an empty tuple is the way to disable stop sequences
    entirely — the field shouldn't land in the payload at all."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {"choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    adapter = VllmAdapter(model="m", stop=())
    with patch("httpx.Client", _make_factory(handler)):
        adapter.complete([ChatMessage(role="user", content="hi")])

    assert "stop" not in captured["body"]


def test_vllm_stream_with_tools_suppresses_bare_json_visible_output() -> None:
    """When the model's first non-whitespace char is `{`, the adapter
    treats subsequent content as a candidate tool call and withholds
    visible StreamText emissions. The user shouldn't see raw JSON
    flicker across the chat while the orchestrator routes the call."""
    bare_json = '{"name": "list_dir", "arguments": {}}'

    def handler(_request: httpx.Request) -> httpx.Response:
        # Split the JSON across multiple chunks to exercise the
        # cross-frame suppression — once the first `{` lands, all
        # subsequent deltas should be withheld too.
        return _sse_response(
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"index": 0, "delta": {"content": '{"name": '}}]},
                {"choices": [{"index": 0, "delta": {"content": '"list_dir", '}}]},
                {"choices": [{"index": 0, "delta": {"content": '"arguments": {}}'}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        chunks = list(
            adapter.stream_with_tools(
                [ChatMessage(role="user", content="ls")],
                tools=[_weather_spec()],
            )
        )

    text_chunks = [c for c in chunks if isinstance(c, StreamText)]
    visible = "".join(c.text for c in text_chunks)
    # No raw JSON should leak to the visible stream.
    assert "{" not in visible
    assert "list_dir" not in visible
    # Tool call still recovered via the fallback parser on stream end.
    final = chunks[-1]
    assert isinstance(final, StreamComplete)
    assert final.reply.tool_calls == (ToolCall(name="list_dir", arguments={}),)
    # bare_json was the entire content, so final reply text is empty.
    assert final.reply.content == ""
    _ = bare_json  # documentation aid; same as what handler builds


def test_vllm_stream_with_tools_releases_buffer_when_not_a_tool_call() -> None:
    """If content STARTS with `{` (triggering suppression) but turns
    out NOT to be a parseable tool call, the buffered text must be
    released as a StreamText before StreamComplete — otherwise the
    user sees an empty turn while the transcript records the text."""

    def handler(_request: httpx.Request) -> httpx.Response:
        # Bare JSON without `name`+`arguments` shape — fallback rejects.
        return _sse_response(
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"index": 0, "delta": {"content": '{"foo": '}}]},
                {"choices": [{"index": 0, "delta": {"content": '"bar"}'}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        chunks = list(
            adapter.stream_with_tools(
                [ChatMessage(role="user", content="give me json")],
                tools=[_weather_spec()],
            )
        )

    text_chunks = [c for c in chunks if isinstance(c, StreamText)]
    visible = "".join(c.text for c in text_chunks)
    # The full buffered content surfaces in a single StreamText (after
    # the loop, before StreamComplete) so the UI shows what was hidden.
    assert '{"foo": "bar"}' in visible
    final = chunks[-1]
    assert isinstance(final, StreamComplete)
    assert final.reply.tool_calls == ()


def test_vllm_stream_with_tools_streams_normally_when_content_starts_with_prose() -> None:
    """Negative control: if the first non-whitespace char isn't `{`,
    the suppression logic stays disengaged and live streaming works
    as before. Prose replies under --tools should not pay any latency
    cost from the bare-JSON guard."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"index": 0, "delta": {"content": "The answer "}}]},
                {"choices": [{"index": 0, "delta": {"content": "is 42."}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        chunks = list(
            adapter.stream_with_tools(
                [ChatMessage(role="user", content="what's the answer?")],
                tools=[_weather_spec()],
            )
        )

    # Two visible StreamText deltas (one per non-empty content chunk),
    # not buffered into a single emission at the end.
    text_chunks = [c for c in chunks if isinstance(c, StreamText)]
    assert len(text_chunks) >= 2
    assert "".join(c.text for c in text_chunks) == "The answer is 42."


def test_qwen_parse_recognizes_tools_wrapper_form() -> None:
    """Unit-test the qwen_parse extension directly. The MLX path
    consumes this same helper, so a regression in the <tools> form
    would also affect MLX adapters running against Coder-family
    fine-tunes that emit the plural wrapper."""
    from harness.model.qwen_parse import _parse_qwen_tool_calls

    raw = 'before <tools>{"name": "x", "arguments": {"a": 1}}</tools> after'
    content, calls = _parse_qwen_tool_calls(raw)
    assert calls == [ToolCall(name="x", arguments={"a": 1})]
    assert "<tools>" not in content
    assert "before" in content
    assert "after" in content


# ---------- guard against accidental re-imports --------------------------


# ---------- tool-bail diagnostic logging ---------------------------------


def test_vllm_logs_tool_bail_when_prose_only_reply_with_tools_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """harness-p4ht: when the model emits prose-only content after a tool
    result (no JSON, no tag wrapper), the orchestrator silently ends the
    turn — there's no signal to debug from. Wire `_log_tool_bail` so the
    adapter writes a JSONL row with raw_tail + tag flags whenever tools
    were sent but no call was parsed."""
    captured: list[tuple[str, str]] = []

    def fake_log(raw: str, parsed_content: str) -> None:
        captured.append((raw, parsed_content))

    monkeypatch.setattr("harness.model.vllm._log_tool_bail", fake_log)

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "It seems we should pass the path as arguments.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="grep keydown")],
            tools=[_weather_spec()],
        )

    assert reply.tool_calls == ()
    assert len(captured) == 1
    raw, parsed = captured[0]
    assert "as arguments" in raw
    assert parsed == "It seems we should pass the path as arguments."


def test_vllm_does_not_log_tool_bail_when_no_tools_in_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain chat reply that contains no tool call must NOT trip the
    bail diagnostic — `tool_bail.jsonl` is specifically for the
    'tools-offered-but-not-called' failure mode."""
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "harness.model.vllm._log_tool_bail",
        lambda r, p: captured.append((r, p)),
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello world.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        adapter.complete_with_tools(
            [ChatMessage(role="user", content="hi")],
            tools=None,
        )

    assert captured == []


def test_vllm_does_not_log_tool_bail_when_call_was_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the fallback ladder DOES recover a tool call from bare JSON,
    the bail diagnostic must stay silent — this isn't a bail."""
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "harness.model.vllm._log_tool_bail",
        lambda r, p: captured.append((r, p)),
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"name": "list_dir", "arguments": {}}',
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="ls")],
            tools=[_weather_spec()],
        )

    assert reply.tool_calls == (ToolCall(name="list_dir", arguments={}),)
    assert captured == []


def test_vllm_stream_logs_tool_bail_when_prose_only_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streamed variant of the bail diagnostic — the drive loop uses
    stream_with_tools, so the bail path must fire there too."""
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "harness.model.vllm._log_tool_bail",
        lambda r, p: captured.append((r, p)),
    )

    frames: list[dict[str, Any] | str] = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Let's grep"},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " the game.js file."},
                    "finish_reason": "stop",
                }
            ]
        },
        "[DONE]",
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(frames)

    adapter = VllmAdapter(model="m")
    with patch("httpx.Client", _make_factory(handler)):
        chunks = list(
            adapter.stream_with_tools(
                [ChatMessage(role="user", content="find keydown listeners")],
                tools=[_weather_spec()],
            )
        )

    complete = chunks[-1]
    assert isinstance(complete, StreamComplete)
    assert complete.reply.tool_calls == ()
    assert len(captured) == 1
    raw, _ = captured[0]
    assert "game.js" in raw


# ---------- output-budget clamping (harness-2epb) ------------------------


def test_vllm_complete_clamps_max_tokens_to_window() -> None:
    """When prompt + requested max_tokens would overflow the served
    window, the adapter clamps max_tokens before posting — vLLM never
    sees an over-budget request."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {"choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    # Small window + a prompt whose heuristic count (len//4 + 4) is ~100.
    adapter = VllmAdapter(model="m", context_window=200)
    prompt = ChatMessage(role="user", content="x" * 384)  # 384//4 + 4 = 100 tokens
    with patch("httpx.Client", _make_factory(handler)):
        adapter.complete([prompt], max_tokens=512)
    # available = 200 - 100 - DEFAULT_OUTPUT_SAFETY_MARGIN(32) = 68
    assert captured["body"]["max_tokens"] == 68


def test_vllm_complete_with_tools_clamps_max_tokens() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {"choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    adapter = VllmAdapter(model="m", context_window=200)
    prompt = ChatMessage(role="user", content="x" * 384)
    with patch("httpx.Client", _make_factory(handler)):
        adapter.complete_with_tools([prompt], max_tokens=512)
    assert captured["body"]["max_tokens"] == 68


def test_vllm_complete_does_not_clamp_when_request_fits() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {"choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    adapter = VllmAdapter(model="m", context_window=32768)
    with patch("httpx.Client", _make_factory(handler)):
        adapter.complete([ChatMessage(role="user", content="hi")], max_tokens=2048)
    assert captured["body"]["max_tokens"] == 2048


def test_vllm_422_context_overflow_raises_prompt_budget_error() -> None:
    """If a request still lands over-budget (heuristic undercount), the
    vLLM 422 is translated into a clear PromptBudgetError rather than an
    opaque RuntimeError."""
    from harness.model.adapter import PromptBudgetError

    detail = (
        "This model's maximum context length is 32768 tokens. However, you "
        "requested 2048 output tokens and your prompt contains at least 30721 "
        "input tokens, for a total of at least 32769 tokens."
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"object": "error", "message": detail})

    adapter = VllmAdapter(model="m", context_window=32768)
    with (
        patch("httpx.Client", _make_factory(handler)),
        pytest.raises(PromptBudgetError, match="over-budget"),
    ):
        adapter.complete([ChatMessage(role="user", content="hi")], max_tokens=2048)


def test_vllm_other_4xx_still_raises_runtime_error() -> None:
    """A non-overflow 4xx must NOT be misclassified as a budget error."""
    from harness.model.adapter import PromptBudgetError

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"object": "error", "message": "malformed request"})

    adapter = VllmAdapter(model="m", context_window=32768)
    with (
        patch("httpx.Client", _make_factory(handler)),
        pytest.raises(RuntimeError) as excinfo,
    ):
        adapter.complete([ChatMessage(role="user", content="hi")], max_tokens=2048)
    assert not isinstance(excinfo.value, PromptBudgetError)


def test_vllm_adapter_does_not_import_vllm_sdk() -> None:
    """Adapter-boundary check: this module must talk to vLLM over HTTP
    only. Importing the `vllm` Python SDK would pull CUDA wheels onto
    Apple Silicon and break `uv sync` on the Mac."""
    import harness.model.vllm as mod

    source = mod.__file__
    assert source is not None
    with open(source) as fh:
        text = fh.read()
    assert "import vllm" not in text
    assert "from vllm" not in text


# ---------- unused-import suppression (pytest only needs the names) ------

# `Iterator` is imported above only to make the SSE response builder's
# return type readable in test setup; pytest doesn't actually need it.
_ = Iterator
