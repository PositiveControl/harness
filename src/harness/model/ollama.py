from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from harness.model.adapter import ChatMessage, approx_token_count
from harness.tools.base import (
    ModelReply,
    StreamChunk,
    StreamComplete,
    StreamText,
    ToolCall,
    ToolSpec,
)


def _messages_for_ollama(messages: Iterable[ChatMessage]) -> list[dict[str, Any]]:
    """Render ChatMessage records into Ollama's /api/chat payload shape.
    Assistant turns with tool_calls expose them; tool-role turns carry
    `name` so Ollama's template can attribute the result."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in m.tool_calls
            ]
        if m.role == "tool" and m.name is not None:
            d["name"] = m.name
        out.append(d)
    return out


def _tool_spec_for_ollama(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _parse_ollama_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Extract ToolCall records from Ollama's tool_calls array. Ollama
    normally returns arguments as a dict; some model templates emit a
    JSON-encoded string — tolerate both."""
    parsed: list[ToolCall] = []
    for raw in raw_calls:
        fn = raw.get("function", {}) or {}
        name = fn.get("name")
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if isinstance(name, str) and isinstance(arguments, dict):
            parsed.append(ToolCall(name=name, arguments=arguments))
    return parsed


class OllamaAdapter:
    """Adapter that talks to a local Ollama daemon via /api/chat.

    Only this module knows the wire format — downstream callers still
    speak `ChatMessage` + `.complete()`. Using Ollama lets us reuse
    model blobs already pulled via `ollama pull` instead of
    re-downloading them via `mlx-community`. Cost: an HTTP round-trip
    per call plus Ollama's own scheduler — a little slower than native
    MLX, but identical from the harness' perspective.

    Construction is cheap — no network I/O until `.complete()`. Missing
    model or an unreachable daemon surfaces as a `RuntimeError` on the
    first call rather than at import time.
    """

    def __init__(
        self,
        model: str = "gemma4:latest",
        *,
        base_url: str = "http://localhost:11434",
        context_window: int = 32_768,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.context_window = context_window
        self.timeout = timeout
        self.id = f"ollama:{model}"

    def load(self) -> None:
        """Prime Ollama to pull the weights into memory right now.

        Ollama lazy-loads a model on its first /api/chat call, which
        can add 5-30s of latency to the first user turn. Hitting
        /api/generate with an empty prompt triggers the load without
        producing output. The harness CLI calls this at startup so the
        first prompt feels instant. Safe to call repeatedly — Ollama
        just bumps the keep-alive timer on an already-loaded model."""
        body = json.dumps(
            {"model": self.model, "prompt": "", "stream": False},
        ).encode("utf-8")
        req = Request(  # noqa: S310
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                resp.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} preloading {self.model!r}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url} — is `ollama serve` running?"
            ) from exc

    def count_tokens(self, messages: Iterable[ChatMessage]) -> int:
        """Ollama has no synchronous tokenize endpoint — we'd have to
        issue a /api/chat call with num_predict=0 and parse `prompt_eval_count`
        to get an exact number, which is too expensive per turn. Fall
        back to the char heuristic; good enough for a fill meter."""
        return approx_token_count(messages)

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = self._post_chat(payload)
        message = data.get("message") or {}
        content = message.get("content", "")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response shape: {data!r}")
        return content

    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Token streaming via Ollama's /api/chat with stream=true.
        Ollama emits NDJSON — one line per token batch; the final line
        carries `done: true`. Yields each message.content delta; tool
        calls are ignored on this path (see stream_with_tools)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        for chunk in self._post_chat_stream(payload):
            message = chunk.get("message") or {}
            delta = message.get("content", "")
            if isinstance(delta, str) and delta:
                yield delta
            if chunk.get("done"):
                break

    def stream_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> Iterator[StreamChunk]:
        """Token streaming with tool-call awareness. Ollama returns
        parsed `tool_calls` structurally on the final `done: true`
        message (not inline tags), so unlike the MLX path we don't need
        to mask anything from the visible stream — we just forward
        content deltas and accumulate metadata for the terminal
        StreamComplete."""
        materialized = list(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_for_ollama(materialized),
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = [_tool_spec_for_ollama(t) for t in tools]

        content_parts: list[str] = []
        final_message: dict[str, Any] = {}
        done_reason: str | None = None
        for chunk in self._post_chat_stream(payload):
            message = chunk.get("message") or {}
            delta = message.get("content", "")
            if isinstance(delta, str) and delta:
                content_parts.append(delta)
                yield StreamText(text=delta)
            if chunk.get("done"):
                final_message = message
                reason = chunk.get("done_reason")
                if isinstance(reason, str):
                    done_reason = reason
                break

        # Tool calls may arrive on the final message even with streaming.
        raw_calls = final_message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raw_calls = []
        parsed = _parse_ollama_tool_calls(raw_calls)
        content = "".join(content_parts)
        # Some Ollama builds only emit tool_calls when stream=False. If
        # content is empty but no tool_calls landed, we're likely in that
        # case — fall back to a blocking retry so tool use still works.
        if not content and not parsed and tools:
            fallback = self.complete_with_tools(
                materialized, tools=tools, max_tokens=max_tokens, temperature=temperature
            )
            yield StreamComplete(reply=fallback)
            return
        yield StreamComplete(
            reply=ModelReply(
                content=content,
                tool_calls=tuple(parsed),
                was_truncated=done_reason == "length",
                had_unparseable_call=False,
            )
        )

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        """Blocking tool-aware variant. Kept separate from
        `stream_with_tools` because some Ollama builds only surface
        `tool_calls` on non-streaming responses — the tool loop can use
        this as a fallback when the streaming path returns empty."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_for_ollama(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = [_tool_spec_for_ollama(t) for t in tools]

        data = self._post_chat(payload)
        message = data.get("message") or {}
        content = message.get("content", "")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response shape: {data!r}")
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raw_calls = []
        parsed = _parse_ollama_tool_calls(raw_calls)
        return ModelReply(
            content=content,
            tool_calls=tuple(parsed),
            was_truncated=data.get("done_reason") == "length",
            had_unparseable_call=False,
        )

    def _post_chat_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """POST /api/chat with `"stream": true` and yield each NDJSON
        line as a parsed dict. Connection stays open until the server
        sends `done: true` or the response ends."""
        body = json.dumps(payload).encode("utf-8")
        req = Request(  # noqa: S310
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                for line in resp:
                    if not line.strip():
                        continue
                    parsed = json.loads(line.decode("utf-8"))
                    if not isinstance(parsed, dict):
                        continue
                    yield parsed
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} for model {self.model!r}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url} — is `ollama serve` running?"
            ) from exc

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(  # noqa: S310
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                parsed = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} for model {self.model!r}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url} — is `ollama serve` running?"
            ) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Unexpected Ollama response shape: {parsed!r}")
        return parsed
