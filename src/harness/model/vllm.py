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
import os
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from harness.model.adapter import ChatMessage, approx_token_count
from harness.model.qwen_parse import _log_tool_bail, _parse_qwen_tool_calls, _TagMasker
from harness.tools.base import (
    ModelReply,
    StreamChunk,
    StreamComplete,
    StreamText,
    ToolCall,
    ToolSpec,
)

# Trace-log env var (harness-97mq). When set to a writable path, every
# complete_with_tools / stream_with_tools call appends one JSONL record
# containing the request payload (sans api key) and either the raw
# response body (non-streaming) or the reassembled message + per-index
# tool_call slots (streaming). Used to capture failing drive turns
# where the tool-call args parse as empty — gives ground truth on what
# the model emitted vs what we parsed.
#
# Opt-in: unset env = no-op. Failures are silent (diagnostic, not a
# contract — same policy as data/tool_bail.jsonl).
_TRACE_ENV: str = "HARNESS_VLLM_TRACE"


def _vllm_trace(record: dict[str, Any]) -> None:
    """Append one JSONL record to ``$HARNESS_VLLM_TRACE`` if set.

    Adds an ISO-8601 ``ts`` field automatically. Errors swallowed —
    a misconfigured trace path must never break a live turn.
    """
    path_str = os.environ.get(_TRACE_ENV)
    if not path_str:
        return
    try:
        record = {"ts": datetime.now(UTC).isoformat(), **record}
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: S110 — diagnostic only; must not break a turn
        pass


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


# Maximum number of `}` chars the lenient fallback will append when
# trying to reconstruct an unclosed JSON envelope. Qwen2.5-Coder's
# observed failure mode is dropping one outer `}` on heavily-nested
# replies; allow up to 3 to cover the "model forgot 2 of 3 nested
# closes" tail of the distribution without trying every conceivable
# pad. harness-bwmd.
_LENIENT_MAX_BRACE_PAD: int = 3


def _scan_json_envelope(content: str, start: int) -> tuple[int, int]:
    """Walk `content` from `content[start]` (must be `{`) tracking
    brace depth and JSON-string state. Used by the lenient bare-JSON
    fallback to find a truncation point when strict raw_decode fails.

    Returns `(end_idx, missing_closes)`:
      - On a natural close (depth hits 0): `(i + 1, 0)` where `i` is
        the matching `}`.
      - On an unclosed envelope: `(last_close_at_depth_gt_0, depth)`
        — the position just past the last `}` we saw while depth was
        still > 0, paired with the remaining open depth. The caller
        can slice `content[start:last_close]` and pad with `}` * depth
        to recover the model's intended JSON.

    Stops early at `\\n```` (markdown fence end) when depth > 0 — that's
    the strongest signal the JSON region has ended and the trailing
    prose isn't part of it. Without this guard, stray `{` / `}` in the
    trailing English (e.g. `{name}` template syntax in a hint) could
    fool the depth counter.

    String-aware: braces inside JSON strings (between `"`s, respecting
    `\\` escapes) don't change depth. That's load-bearing — the failing
    Qwen2.5-Coder shape on harness-3jo1 had multi-line code in
    `old_string` / `new_string` full of `{` and `}` chars."""
    depth = 0
    in_string = False
    last_close_at_depth_gt_0: int | None = None
    i = start
    while i < len(content):
        ch = content[i]
        if in_string:
            if ch == "\\" and i + 1 < len(content):
                i += 2  # skip escape sequence as one unit
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return (i + 1, 0)
                if depth > 0:
                    last_close_at_depth_gt_0 = i + 1
            elif ch == "\n" and depth > 0 and content[i + 1 : i + 4] == "```":
                # Markdown fence ending the JSON region; trailing
                # text is prose, not JSON.
                break
        i += 1

    if last_close_at_depth_gt_0 is not None and depth > 0:
        return (last_close_at_depth_gt_0, depth)
    return (len(content), depth)


def _parse_bare_json_tool_call(content: str) -> tuple[str, list[ToolCall]]:
    """Last-ditch fallback: when vLLM's parser didn't extract anything
    AND the model didn't wrap its call in any Qwen-family tag, the
    response can still be a bare `{"name": …, "arguments": …}` JSON
    object — observed live with vLLM 0.21 + Qwen2.5-Coder against the
    harness's tool-augmented system prompt.

    Locates the JSON anywhere in content (`{"name":` signature regex,
    then `json.raw_decode` to find the closing brace by parsing). This
    covers two shapes:
      - content trims to a single JSON object (clean tool-only reply)
      - content has a tool-intent preamble ("Let me check.\\n") +
        trailing `<|im_start|>` leak + the JSON in the middle (real
        Qwen2.5-Coder shape under --enable-auto-tool-choice)

    Constrained so prose containing JSON doesn't get mis-parsed:
      - the object must have a string `name` AND a dict `arguments`
        (or coerce-able to one — accept missing/null arguments as {})

    Lenient pad-and-retry path (harness-bwmd): when strict raw_decode
    fails, walk the candidate with `_scan_json_envelope` to detect an
    unclosed envelope and pad up to `_LENIENT_MAX_BRACE_PAD` missing
    `}` chars. Catches the Qwen2.5-Coder 32B failure shape where the
    model loses count of nested braces inside multi-line code strings
    and drops the outer envelope's close (drive halt 2026-05-23 on
    harness-3jo1: a 2.2 KB JSON missing one final `}` bailed via
    teaser, model retried with the same shape, context exhausted).

    Returns (stripped_content, calls). On miss, returns the input
    content unchanged with an empty call list."""
    match = _BARE_JSON_CALL_RE.search(content)
    if match is None:
        return content, []
    start = match.start()
    decoder = json.JSONDecoder()
    end_idx: int
    data: Any
    try:
        data, end_idx = decoder.raw_decode(content, start)
    except json.JSONDecodeError:
        lenient = _lenient_parse_unclosed_envelope(content, start)
        if lenient is None:
            return content, []
        data, end_idx = lenient
    if not isinstance(data, dict):
        return content, []
    name = data.get("name")
    arguments = data.get("arguments")
    if not isinstance(name, str) or not name:
        return content, []
    if arguments is None:
        arguments = {}
    elif isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        return content, []
    # Strip the JSON span from content; whatever prose / leaked tokens
    # bracketed it stay so the orchestrator + renderer can still
    # surface meaningful text (the renderer's per-sentence
    # _is_suppressible filter will drop "Let me check…" etc).
    stripped_content = (content[:start] + content[end_idx:]).strip()
    return stripped_content, [ToolCall(name=name, arguments=arguments)]


def _lenient_parse_unclosed_envelope(content: str, start: int) -> tuple[Any, int] | None:
    """Run after strict raw_decode failed. Use `_scan_json_envelope`
    to find a truncation point and pad with up to
    `_LENIENT_MAX_BRACE_PAD` closing braces. Returns `(data, end_idx)`
    on the first pad that parses + carries a `name` key; None on
    miss. harness-bwmd."""
    candidate_end, missing = _scan_json_envelope(content, start)
    if missing <= 0:
        # Envelope closed naturally — strict parse must have failed
        # on a different shape (bad escape, control char in string).
        # Don't try to fix; let the caller fall through.
        return None
    if missing > _LENIENT_MAX_BRACE_PAD:
        return None
    candidate = content[start:candidate_end].rstrip()
    for pad in range(1, missing + 1):
        try:
            data = json.loads(candidate + "}" * pad)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("name"):
            return data, candidate_end
    return None


# Bare-JSON tool-call signature. Matches `{"name":` with optional
# whitespace inside the wrapper. Used by stream_with_tools as a
# mid-stream detector when the model emits a tool-intent preamble
# ("Let me check…") BEFORE the actual JSON — the preamble masks the
# start-of-content signal that the simpler `startswith("{")` check
# relied on, but this regex picks the call out wherever it lands.
_BARE_JSON_CALL_RE = re.compile(r'\{\s*"name"\s*:')


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

    # `<|im_start|>` is Qwen's role-boundary token. It should never
    # appear in user-visible content; when it does, the model is
    # hallucinating a multi-turn conversation in a single completion
    # (observed live with Qwen2.5-Coder + --enable-auto-tool-choice).
    # Passing it as an explicit stop sequence makes vLLM truncate at
    # the first occurrence — cleaner than masking after the fact.
    _DEFAULT_STOP: tuple[str, ...] = ("<|im_start|>",)

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str = "http://localhost:8000/v1",
        context_window: int = 32_768,
        timeout: float = 300.0,
        api_key: str | None = None,
        stop: tuple[str, ...] | None = None,
    ) -> None:
        self._model: str | None = model
        self.base_url = base_url.rstrip("/")
        self.context_window = context_window
        self.timeout = timeout
        self._api_key = api_key
        self.stop: tuple[str, ...] = self._DEFAULT_STOP if stop is None else stop
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
        if self.stop:
            payload["stop"] = list(self.stop)
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
        if self.stop:
            payload["stop"] = list(self.stop)
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
        if self.stop:
            payload["stop"] = list(self.stop)
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
        # Fallback ladder: vLLM's --tool-call-parser is advisory and
        # sometimes the model bypasses the wrapping format entirely.
        # When `tools` was sent in the request and the response carries
        # no structured tool_calls, try (1) Qwen-family tag wrappers
        # (<tool_call>/<tools>/<function=>), then (2) bare-JSON
        # `{"name":…,"arguments":…}`. Order matters — tagged form is
        # the strict signal, bare form is permissive and only kicks
        # in when the strict path returned nothing.
        raw_for_bail = content
        if tools and not parsed and content:
            stripped, fallback_calls = _parse_qwen_tool_calls(content)
            if fallback_calls:
                content = stripped
                parsed = fallback_calls
            else:
                stripped, fallback_calls = _parse_bare_json_tool_call(content)
                if fallback_calls:
                    content = stripped
                    parsed = fallback_calls
        if tools and not parsed and raw_for_bail.strip():
            _log_tool_bail(raw_for_bail, content)
        _vllm_trace(
            {
                "mode": "complete_with_tools",
                "model": self.model,
                "request": payload,
                "response": data,
                "parsed_calls": [{"name": c.name, "arguments": c.arguments} for c in parsed],
            }
        )
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
        if self.stop:
            payload["stop"] = list(self.stop)

        # Three collectors:
        # - raw_parts keeps every delta (tags + bare JSON included) so
        #   the post-stream fallback ladder can reparse content if vLLM
        #   didn't extract tool_calls structurally.
        # - masker hides tag spans (<tool_call> / <tools> / <function=>)
        #   from the live user-visible stream.
        # - suppressing_bare_json kicks in when the accumulated content
        #   contains a bare-JSON tool-call signature anywhere (matches
        #   `{"name":` with optional whitespace). Detects two shapes:
        #     a) content starts with `{` immediately
        #     b) a tool-intent preamble ("Let me check.\n") precedes
        #        the JSON — the renderer's _is_suppressible filter eats
        #        the preamble per-sentence, but without this regex the
        #        trailing JSON falls through to the screen.
        #   Once detected, all subsequent text is withheld; final reply
        #   lands via StreamComplete after fallback parse.
        raw_parts: list[str] = []
        masker = _TagMasker()
        suppressing_bare_json = False
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
                raw_parts.append(text)
                if tools and not suppressing_bare_json:
                    joined = "".join(raw_parts)
                    if _BARE_JSON_CALL_RE.search(joined):
                        suppressing_bare_json = True
                if not suppressing_bare_json:
                    visible = masker.feed(text)
                    if visible:
                        yield StreamText(text=visible)
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

        # Drain anything the masker was holding (visible mode only —
        # mid-tag truncation drops the buffer, matching MLX behavior).
        tail = masker.flush()
        if tail:
            yield StreamText(text=tail)

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

        full_content = "".join(raw_parts)
        raw_for_bail = full_content
        # Same fallback ladder as complete_with_tools: tagged form
        # first (strict), then bare-JSON (permissive, only when tools
        # were sent). Keeps the orchestrator's tool path working
        # regardless of which --tool-call-parser the server was
        # launched with and regardless of whether the model wrapped
        # its call.
        if tools and not parsed and full_content:
            stripped, fallback_calls = _parse_qwen_tool_calls(full_content)
            if fallback_calls:
                full_content = stripped
                parsed = fallback_calls
            else:
                stripped, fallback_calls = _parse_bare_json_tool_call(full_content)
                if fallback_calls:
                    full_content = stripped
                    parsed = fallback_calls
        if tools and not parsed and raw_for_bail.strip():
            _log_tool_bail(raw_for_bail, full_content)

        # If we suppressed StreamText because content started with `{`
        # but the bare-JSON didn't actually parse as a tool call (e.g.
        # the model emitted a JSON object as legitimate prose), release
        # the buffered content as a single StreamText so the UI shows
        # it. Without this the user sees an empty turn while the
        # transcript records the text — confusing.
        if suppressing_bare_json and not parsed and full_content:
            yield StreamText(text=full_content)

        _vllm_trace(
            {
                "mode": "stream_with_tools",
                "model": self.model,
                "request": payload,
                "reassembled_message": {
                    "content": full_content,
                    "tool_calls": [
                        {
                            "index": idx,
                            "function": {
                                "name": tc_acc[idx]["name"],
                                "arguments": tc_acc[idx]["arguments"],
                            },
                        }
                        for idx in sorted(tc_acc.keys())
                    ],
                },
                "finish_reason": finish_reason,
                "parsed_calls": [{"name": c.name, "arguments": c.arguments} for c in parsed],
            }
        )

        yield StreamComplete(
            reply=ModelReply(
                content=full_content,
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
                # raise_for_status on a streaming response leaves the body
                # un-read; touching .text outside the `with` then raises
                # httpx.ResponseNotRead instead of surfacing the server's
                # actual error message (harness-gtd0). Materialize the body
                # inline so the outer except clause can format it cleanly.
                if r.status_code >= 400:
                    r.read()
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
