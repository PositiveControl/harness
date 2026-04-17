from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from harness.model.adapter import ChatMessage, ModelAdapter
from harness.model.ollama import OllamaAdapter
from harness.tools.base import ToolCall, ToolSpec


def test_ollama_adapter_satisfies_protocol() -> None:
    adapter = OllamaAdapter()
    assert isinstance(adapter, ModelAdapter)
    assert adapter.id == "ollama:gemma4:latest"
    assert adapter.context_window > 0


def test_ollama_adapter_id_reflects_model() -> None:
    adapter = OllamaAdapter(model="llama3.2:1b")
    assert adapter.id == "ollama:llama3.2:1b"


def test_ollama_adapter_strips_trailing_slash() -> None:
    adapter = OllamaAdapter(base_url="http://localhost:11434/")
    assert adapter.base_url == "http://localhost:11434"


def test_ollama_adapter_does_no_network_on_construction() -> None:
    with patch("harness.model.ollama.urlopen") as mock_urlopen:
        OllamaAdapter()
    mock_urlopen.assert_not_called()


class _FakeResponse:
    """Minimal context-manager stand-in for urlopen's return value."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_ollama_complete_posts_expected_payload() -> None:
    adapter = OllamaAdapter(model="gemma4:latest")
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"message": {"role": "assistant", "content": "hi there"}})

    with patch("harness.model.ollama.urlopen", side_effect=fake_urlopen):
        reply = adapter.complete(
            [ChatMessage(role="user", content="hi")],
            max_tokens=32,
            temperature=0.2,
        )

    assert reply == "hi there"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["method"] == "POST"
    assert captured["body"]["model"] == "gemma4:latest"
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["body"]["options"]["num_predict"] == 32
    assert captured["body"]["options"]["temperature"] == pytest.approx(0.2)


def test_ollama_complete_raises_on_unreachable_daemon() -> None:
    from urllib.error import URLError

    adapter = OllamaAdapter(base_url="http://127.0.0.1:1")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise URLError("connection refused")

    with (
        patch("harness.model.ollama.urlopen", side_effect=boom),
        pytest.raises(RuntimeError, match="Cannot reach Ollama"),
    ):
        adapter.complete([ChatMessage(role="user", content="hi")])


def test_ollama_load_hits_generate_endpoint_with_empty_prompt() -> None:
    adapter = OllamaAdapter(model="gemma4:latest")
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"done": True})

    with patch("harness.model.ollama.urlopen", side_effect=fake_urlopen):
        adapter.load()

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["body"] == {"model": "gemma4:latest", "prompt": "", "stream": False}


def test_ollama_load_raises_on_unreachable_daemon() -> None:
    from urllib.error import URLError

    adapter = OllamaAdapter()

    def boom(*_args: object, **_kwargs: object) -> None:
        raise URLError("connection refused")

    with (
        patch("harness.model.ollama.urlopen", side_effect=boom),
        pytest.raises(RuntimeError, match="Cannot reach Ollama"),
    ):
        adapter.load()


def test_ollama_complete_with_tools_parses_native_tool_calls() -> None:
    adapter = OllamaAdapter(model="gemma4:latest")
    captured: dict[str, Any] = {}
    fake_payload = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "README.md"}}}],
        }
    }

    def fake_urlopen(req: Any, timeout: float = 0) -> _FakeResponse:
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(fake_payload)

    spec = ToolSpec(
        name="read_file",
        description="read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        tier="read",
    )
    with patch("harness.model.ollama.urlopen", side_effect=fake_urlopen):
        reply = adapter.complete_with_tools(
            [ChatMessage(role="user", content="show README")],
            tools=[spec],
        )

    assert reply.content == ""
    assert reply.wants_tools
    assert reply.tool_calls == (ToolCall(name="read_file", arguments={"path": "README.md"}),)
    assert captured["body"]["tools"][0]["function"]["name"] == "read_file"


def test_ollama_complete_with_tools_handles_string_arguments() -> None:
    """Some model templates emit arguments as a JSON string instead of a
    dict. Adapter should decode the string before surfacing ToolCall."""
    adapter = OllamaAdapter()
    fake_payload = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "shell",
                        "arguments": '{"cmd": "ls"}',
                    }
                }
            ],
        }
    }

    def fake_urlopen(_req: Any, timeout: float = 0) -> _FakeResponse:
        return _FakeResponse(fake_payload)

    with patch("harness.model.ollama.urlopen", side_effect=fake_urlopen):
        reply = adapter.complete_with_tools([ChatMessage(role="user", content="list")])

    assert reply.tool_calls == (ToolCall(name="shell", arguments={"cmd": "ls"}),)


def test_ollama_complete_with_tools_text_only_reply() -> None:
    adapter = OllamaAdapter()
    fake_payload = {"message": {"role": "assistant", "content": "done", "tool_calls": None}}

    def fake_urlopen(_req: Any, timeout: float = 0) -> _FakeResponse:
        return _FakeResponse(fake_payload)

    with patch("harness.model.ollama.urlopen", side_effect=fake_urlopen):
        reply = adapter.complete_with_tools([ChatMessage(role="user", content="hi")])

    assert reply.content == "done"
    assert reply.tool_calls == ()
    assert not reply.wants_tools


def test_ollama_messages_renders_tool_and_assistant_turns() -> None:
    """Assistant turns with tool_calls expose them; tool-role turns
    carry name."""
    from harness.model.ollama import _messages_for_ollama

    msgs = [
        ChatMessage(role="user", content="go"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(ToolCall(name="shell", arguments={"cmd": "ls"}),),
        ),
        ChatMessage(role="tool", content="a b c", name="shell"),
    ]
    out = _messages_for_ollama(msgs)
    assert out[0] == {"role": "user", "content": "go"}
    assert out[1]["tool_calls"][0]["function"]["name"] == "shell"
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"cmd": "ls"}
    assert out[2] == {"role": "tool", "content": "a b c", "name": "shell"}


@pytest.mark.skipif(
    "os.environ.get('HARNESS_TEST_OLLAMA') != '1'",
    reason="set HARNESS_TEST_OLLAMA=1 to run against a live Ollama daemon",
)
def test_ollama_adapter_smoke() -> None:
    adapter = OllamaAdapter()
    reply = adapter.complete(
        [ChatMessage(role="user", content="Say 'ok' and nothing else.")],
        max_tokens=16,
        temperature=0.0,
    )
    assert reply
