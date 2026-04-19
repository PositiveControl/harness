"""Qwen tool-call parsing + tag-masking helpers.

Extracted from model/mlx.py (harness-782y). Owns the
Qwen-family-specific formats that the MLX adapter needs to produce
and consume:

- `_messages_to_dicts` / `_messages_to_dicts_with_tools` — render
  ChatMessage records into the dict shape Qwen's chat template
  expects.
- `_tool_spec_to_schema` — OpenAI function-calling shape.
- `_parse_qwen_tool_calls` — extract `<tool_call>…</tool_call>`
  (JSON + XML variants) from a raw completion string.
- `_TagMasker` — stream filter that hides tool-call tag spans from
  live token output so raw payloads never surface to the user.
- `_log_tool_bail` — JSONL diagnostic for 0-tool-call model replies.

All private underscore-prefixed — consumed only by the adapter and
its tests (which import via `harness.model.mlx` re-exports).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from harness.model.adapter import ChatMessage
from harness.tools.base import ToolCall, ToolSpec


def _messages_to_dicts(messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
    """Project ChatMessage records down to {role, content} dicts.
    Tool-role messages and the `name` field are ignored here —
    non-tool-aware path."""
    return [{"role": m.role, "content": m.content} for m in messages]


def _messages_to_dicts_with_tools(
    messages: Iterable[ChatMessage],
) -> list[dict[str, Any]]:
    """Render messages for tool-aware chat. Assistant turns with
    tool_calls expose them in the dict; tool-role turns carry their
    tool_call_id.

    `arguments` is rendered as the raw dict (not JSON-serialized) —
    Qwen3-Coder's chat template iterates it with `|items`, and
    Qwen2.5's template feeds it through `|tojson` internally, so
    the dict form works for both families."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        if m.role == "tool" and m.name is not None:
            d["name"] = m.name
        out.append(d)
    return out


def _tool_spec_to_schema(spec: ToolSpec) -> dict[str, Any]:
    """Render a ToolSpec as the OpenAI-style function-calling schema
    that modern chat templates (Qwen, Hermes, etc.) understand."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


# Qwen2.5 / Hermes: <tool_call>{...JSON...}</tool_call>
_TOOL_CALL_JSON_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Qwen3-Coder: <tool_call><function=NAME><parameter=KEY>VAL</parameter>...</function></tool_call>
_TOOL_CALL_XML_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_PATTERN = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
_PARAMETER_PATTERN = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)


def _parse_qwen_tool_calls(raw: str) -> tuple[str, list[ToolCall]]:
    """Extract tool-call blocks from a Qwen model's raw output.

    Supports two formats:
      - Qwen2.5 / Hermes JSON: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
      - Qwen3-Coder XML: a `<tool_call>` block containing
        `<function=NAME><parameter=KEY>VAL</parameter>…</function>`

    Returns (content_with_blocks_stripped, list_of_calls). Malformed
    blocks are dropped rather than raising — the model will retry on
    the next round if it cared."""
    calls: list[ToolCall] = []

    for match in _TOOL_CALL_JSON_PATTERN.finditer(raw):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(name, str):
            continue
        if not isinstance(arguments, dict):
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = {}
        calls.append(ToolCall(name=name, arguments=arguments))

    if not calls:
        for fn_match in _FUNCTION_PATTERN.finditer(raw):
            name = fn_match.group(1)
            fn_body = fn_match.group(2)
            args: dict[str, Any] = {}
            for pm in _PARAMETER_PATTERN.finditer(fn_body):
                args[pm.group(1)] = pm.group(2).strip()
            calls.append(ToolCall(name=name, arguments=args))

    content = _TOOL_CALL_JSON_PATTERN.sub("", raw)
    content = _FUNCTION_PATTERN.sub("", content)
    content = _TOOL_CALL_XML_PATTERN.sub("", content)
    content = re.sub(r"</?tool_call>", "", content).strip()
    return content, calls


def _log_tool_bail(raw: str, parsed_content: str) -> None:
    """Diagnostic: append a JSONL row to data/tool_bail.jsonl whenever
    the model was offered tools but parsed 0 calls and still emitted
    text. Three failure modes look identical from the loop: truncation
    mid-call, malformed-JSON parse drop, and model bail-mid-thought.
    Capturing raw_tail + tag-presence flags lets us tell them apart.

    Errors are swallowed — never break a real turn for a diagnostic."""
    try:
        from harness.config import settings

        path = settings.data_path / "tool_bail.jsonl"
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "raw_len": len(raw),
            "raw_tail": raw[-400:],
            "parsed_content_tail": parsed_content[-200:],
            "has_open_tag": "<tool_call>" in raw,
            "has_close_tag": "</tool_call>" in raw,
            "has_function_open": "<function=" in raw,
        }
        with path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:  # noqa: S110 — diagnostic-only; never break a real turn
        pass


class _TagMasker:
    """Stream filter that elides Qwen tool-call tag spans from a live
    text stream. Hides everything between `<tool_call>` / `</tool_call>`
    and between `<function=...>` / `</function>` so the user never sees
    raw JSON or XML tool-call payloads mid-generation.

    Keeps a small rolling tail in visible mode so a tag opening that
    straddles a delta boundary ("...hello<to" + "ol_call>...") isn't
    leaked before the masker can recognize it. `_MAX_TAIL` must be at
    least the length of the longest recognized opening tag."""

    _OPEN_TAGS = ("<tool_call>", "<function=")
    _CLOSE_TAGS = ("</tool_call>", "</function>")
    _MAX_TAIL = 15

    def __init__(self) -> None:
        self._buf = ""
        self._hidden = False

    def feed(self, delta: str) -> str:
        self._buf += delta
        out: list[str] = []
        while True:
            if not self._hidden:
                earliest = -1
                for tag in self._OPEN_TAGS:
                    idx = self._buf.find(tag)
                    if idx != -1 and (earliest == -1 or idx < earliest):
                        earliest = idx
                if earliest == -1:
                    safe_cut = len(self._buf) - self._MAX_TAIL
                    if safe_cut > 0:
                        out.append(self._buf[:safe_cut])
                        self._buf = self._buf[safe_cut:]
                    break
                out.append(self._buf[:earliest])
                self._buf = self._buf[earliest:]
                self._hidden = True
            else:
                end = -1
                end_tag_len = 0
                for close in self._CLOSE_TAGS:
                    idx = self._buf.find(close)
                    if idx != -1 and (end == -1 or idx < end):
                        end = idx
                        end_tag_len = len(close)
                if end == -1:
                    break
                self._buf = self._buf[end + end_tag_len :]
                self._hidden = False
        return "".join(out)

    def flush(self) -> str:
        """Called at stream end. If still inside a hidden span the
        buffer is dropped (mid-tag truncation — nothing safe to show)."""
        if self._hidden:
            self._buf = ""
            return ""
        out = self._buf
        self._buf = ""
        return out
