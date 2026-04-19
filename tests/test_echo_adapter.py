"""Minimal coverage for EchoAdapter — the wiring-test adapter. Tests
the full surface including the tool-loop stub so `--tools` paths
don't crash on AttributeError when exercised with --model echo.
"""

from __future__ import annotations

from harness.model.adapter import ChatMessage
from harness.model.echo import EchoAdapter
from harness.tools.base import ModelReply


def test_echo_complete_returns_last_user_message() -> None:
    adapter = EchoAdapter()
    out = adapter.complete(
        [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="prior"),
            ChatMessage(role="user", content="latest"),
        ]
    )
    assert out == "[echo] latest"


def test_echo_stream_yields_single_chunk() -> None:
    adapter = EchoAdapter()
    chunks = list(adapter.stream([ChatMessage(role="user", content="stream me")]))
    assert chunks == ["[echo] stream me"]


def test_echo_complete_with_tools_returns_modelreply_no_calls() -> None:
    """Tool-loop stub: echo can't decide when to call a tool, so it
    returns the echo text with an empty tool_calls tuple. Keeps the
    --tools path runnable for wiring tests without a real model."""
    adapter = EchoAdapter()
    reply = adapter.complete_with_tools(
        [ChatMessage(role="user", content="try tools")]
    )
    assert isinstance(reply, ModelReply)
    assert reply.content == "[echo] try tools"
    assert reply.tool_calls == ()
    assert reply.wants_tools is False
