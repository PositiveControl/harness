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


# ---------- guard against accidental re-imports --------------------------


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
