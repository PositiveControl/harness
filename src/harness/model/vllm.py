"""vLLM HTTP adapter — talks to an OpenAI-compatible /v1/chat/completions
endpoint served by vLLM.

vLLM exposes the OpenAI Chat Completions API natively when launched via
its OpenAI server entrypoint (`vllm serve` / `vllm/vllm-openai` container).
Format compatibility is broad — messages, tools, tool_calls, streaming
deltas — so the adapter stays shallow: shape the request, parse the
response, hand off to the harness.

Adapter-boundary check: only `httpx` (already a hard dep) is imported.
No `vllm` Python SDK, which would pull CUDA wheels onto the Mac.

vLLM serves exactly one model per process. The adapter discovers the
served model id lazily via GET /v1/models so callers don't have to
mirror what the server was launched with — useful when a daisy-chained
cluster swaps the model behind a fixed URL."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

import httpx

from harness.model.adapter import ChatMessage, approx_token_count
from harness.tools.base import (
    ModelReply,
    StreamChunk,
    StreamComplete,
    StreamText,
    ToolCall,
    ToolSpec,
)


def _synth_tool_call_id(idx: int, name: str) -> str:
    """Produce a deterministic per-turn id for tool_calls. vLLM doesn't
    strictly enforce uniqueness across the conversation, but the model
    template requires SOMETHING to pair assistant.tool_calls[i] with
    the matching tool-role tool_call_id. We synthesize from index +
    name; that's enough for single-turn pairing."""
    return f"call_{idx}_{name}"


def _messages_for_openai(messages: Iterable[ChatMessage]) -> list[dict[str, Any]]:
    """Render ChatMessage records into the OpenAI /v1/chat/completions
    message shape. Assistant turns expose tool_calls; tool-role turns
    carry tool_call_id so the chat template can attribute the result.

    `arguments` MUST be a JSON-encoded string per the OpenAI spec —
    not a dict. vLLM's stricter parsers reject the dict form."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": _synth_tool_call_id(i, tc.name),
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, separators=(",", ":")),
                    },
                }
                for i, tc in enumerate(m.tool_calls)
            ]
            # OpenAI spec: assistant with tool_calls may have null content.
            if not m.content:
                d["content"] = None
        if m.role == "tool":
            d["tool_call_id"] = m.tool_call_id or _synth_tool_call_id(0, m.name or "")
            if m.name:
                d["name"] = m.name
        out.append(d)
    return out


def _tool_spec_for_openai(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _parse_openai_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Extract ToolCall records from OpenAI/vLLM tool_calls arrays.
    OpenAI/vLLM serializes arguments as a JSON-encoded STRING (per spec).
    Some Qwen finetunes occasionally emit a dict — tolerate both."""
    parsed: list[ToolCall] = []
    for raw in raw_calls:
        fn = raw.get("function", {}) or {}
        name = fn.get("name")
        arguments: Any = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                arguments = {}
        if isinstance(name, str) and isinstance(arguments, dict):
            parsed.append(ToolCall(name=name, arguments=arguments))
    return parsed


class VllmAdapter:
    """Adapter that talks to a vLLM server via its OpenAI-compatible
    HTTP API.

    Construction is cheap — no network I/O until the first call.
    Missing server / model surfaces as `RuntimeError` on first use
    rather than at import time.

    `model` is optional: when None, the adapter discovers vLLM's
    currently-served model via GET /v1/models on first use. This lets
    the cluster swap the served model behind a fixed URL without the
    chat client needing to know.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str = "http://localhost:8000/v1",
        context_window: int = 32_768,
        timeout: float = 300.0,
        api_key: str | None = None,
    ) -> None:
        self._model: str | None = model
        self.base_url = base_url.rstrip("/")
        self.context_window = context_window
        self.timeout = timeout
        self._api_key = api_key
        self.id = f"vllm:{model}" if model else f"vllm:{self.base_url}"

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self._discover_model()
            self.id = f"vllm:{self._model}"
        return self._model

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _discover_model(self) -> str:
        """GET /v1/models — pick the first served id. vLLM hosts exactly
        one model per process, so this is unambiguous."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(f"{self.base_url}/models", headers=self._headers())
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot reach vLLM at {self.base_url} — is `vllm serve` running? ({exc})"
            ) from exc
        items = data.get("data") or []
        if not items:
            raise RuntimeError(f"vLLM at {self.base_url} reports no models loaded.")
        first = items[0]
        served_id = first.get("id") if isinstance(first, dict) else None
        if not isinstance(served_id, str):
            raise RuntimeError(f"Unexpected /v1/models payload from vLLM: {data!r}")
        return served_id

    def load(self) -> None:
        """vLLM pre-loads the model on serve startup; nothing to do here
        beyond a reachability ping (which also resolves self.model)."""
        _ = self.model

    def count_tokens(self, messages: Iterable[ChatMessage]) -> int:
        """vLLM ships POST /tokenize but it costs a network RTT per call;
        for a live fill meter we'd issue dozens per session. Fall back
        to the char heuristic — good enough for a UI gauge."""
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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._post("/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM returned no choices: {data!r}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            return ""
        return content

    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for chunk in self._post_stream("/chat/completions", payload):
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                yield text
            if choices[0].get("finish_reason"):
                break

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_for_openai(messages),
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [_tool_spec_for_openai(t) for t in tools]
        data = self._post("/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM returned no choices: {data!r}")
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = ""
        raw_calls = msg.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raw_calls = []
        parsed = _parse_openai_tool_calls(raw_calls)
        return ModelReply(
            content=content,
            tool_calls=tuple(parsed),
            was_truncated=choices[0].get("finish_reason") == "length",
            had_unparseable_call=False,
        )

    def stream_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> Iterator[StreamChunk]:
        """Streaming with tool-call awareness over SSE.

        vLLM emits tool_calls as deltas with the call index, name (on
        the first frame for a given index), and an arguments STRING
        built up across frames. We accumulate per-index, then decode
        the arguments JSON on the terminal StreamComplete."""
        materialized = list(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_for_openai(materialized),
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [_tool_spec_for_openai(t) for t in tools]

        content_parts: list[str] = []
        tc_acc: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        for chunk in self._post_stream("/chat/completions", payload):
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                content_parts.append(text)
                yield StreamText(text=text)
            tc_delta = delta.get("tool_calls") or []
            if isinstance(tc_delta, list):
                for tcd in tc_delta:
                    if not isinstance(tcd, dict):
                        continue
                    idx_raw = tcd.get("index", 0)
                    idx = int(idx_raw) if isinstance(idx_raw, int) else 0
                    fn = tcd.get("function") or {}
                    slot = tc_acc.setdefault(idx, {"name": "", "arguments": ""})
                    fn_name = fn.get("name")
                    if isinstance(fn_name, str) and fn_name:
                        slot["name"] = fn_name
                    fn_args = fn.get("arguments")
                    if isinstance(fn_args, str):
                        slot["arguments"] += fn_args
            fr = choice.get("finish_reason")
            if isinstance(fr, str):
                finish_reason = fr

        parsed: list[ToolCall] = []
        for _, slot in sorted(tc_acc.items()):
            name = slot["name"]
            args_raw = slot["arguments"]
            try:
                args = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                args = {}
            if name and isinstance(args, dict):
                parsed.append(ToolCall(name=name, arguments=args))

        yield StreamComplete(
            reply=ModelReply(
                content="".join(content_parts),
                tool_calls=tuple(parsed),
                was_truncated=finish_reason == "length",
                had_unparseable_call=False,
            )
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(
                    f"{self.base_url}{path}",
                    headers=self._headers(),
                    json=payload,
                )
                r.raise_for_status()
                data: Any = r.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise RuntimeError(
                f"vLLM returned HTTP {exc.response.status_code} for {self._model!r}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot reach vLLM at {self.base_url} — is `vllm serve` running? ({exc})"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected vLLM response shape: {data!r}")
        return data

    def _post_stream(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """SSE stream parser. vLLM emits `data: <json>\\n\\n` frames,
        terminated by `data: [DONE]`. Blank lines are keepalives."""
        try:
            with (
                httpx.Client(timeout=self.timeout) as client,
                client.stream(
                    "POST",
                    f"{self.base_url}{path}",
                    headers=self._headers(),
                    json=payload,
                ) as r,
            ):
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        return
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(chunk, dict):
                        yield chunk
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise RuntimeError(
                f"vLLM returned HTTP {exc.response.status_code} for {self._model!r}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot reach vLLM at {self.base_url} — is `vllm serve` running? ({exc})"
            ) from exc
