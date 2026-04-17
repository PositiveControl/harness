from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

from harness.model.adapter import ChatMessage, approx_token_count
from harness.tools.base import (
    ModelReply,
    StreamChunk,
    StreamComplete,
    StreamText,
    ToolCall,
    ToolSpec,
)


def _messages_to_dicts(messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
    """Project ChatMessage records down to the {role, content} dicts that
    tokenizer.apply_chat_template expects. Tool-role messages and the
    `name` field are ignored here — non-tool-aware path."""
    return [{"role": m.role, "content": m.content} for m in messages]


def _messages_to_dicts_with_tools(
    messages: Iterable[ChatMessage],
) -> list[dict[str, Any]]:
    """Render messages for tool-aware chat. Assistant turns with
    tool_calls expose them in the dict; tool-role turns carry their
    tool_call_id. Qwen's chat template reads these correctly.

    `arguments` is rendered as the raw dict (not JSON-serialized) —
    Qwen3-Coder's chat template iterates it with `|items`, and Qwen2.5's
    template feeds it through `|tojson` internally, so the dict form
    works for both families. Pre-dumping to a string breaks Qwen3-Coder."""
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

    # JSON first — if the inner body is a JSON object, this regex catches
    # it. XML bodies won't match because they don't start with `{`.
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
            # Qwen sometimes emits arguments as a JSON-encoded string; try to recover.
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = {}
        calls.append(ToolCall(name=name, arguments=arguments))

    # If nothing matched as JSON, try the XML function format (Qwen3-Coder).
    # The outer `<tool_call>` wrapper is optional because the chat template
    # often primes the opening tag as part of the generation prompt — the
    # model only emits everything from `<function=...>` onward.
    if not calls:
        for fn_match in _FUNCTION_PATTERN.finditer(raw):
            name = fn_match.group(1)
            fn_body = fn_match.group(2)
            args: dict[str, Any] = {}
            for pm in _PARAMETER_PATTERN.finditer(fn_body):
                args[pm.group(1)] = pm.group(2).strip()
            calls.append(ToolCall(name=name, arguments=args))

    # Strip every tool-call sigil from the surfaced content: JSON blocks,
    # XML `<function>` blocks, and any orphan `<tool_call>` / `</tool_call>`
    # tags left behind by prompt-primed openings.
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


class MLXAdapter:
    """MLX-backed adapter. Load is deferred until the first `complete()`
    call, or until `.load()` is called explicitly. This keeps construction
    cheap (useful for the CLI factory and for unit tests that only need
    to assert protocol compliance).

    Only this module imports `mlx_lm`. The adapter boundary must stay clean
    so other runtimes (llama.cpp, Ollama, OpenAI-compatible) can swap in
    without any other file having to change."""

    def __init__(
        self,
        repo: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
        *,
        adapter_path: str | None = None,
        context_window: int = 131_072,
    ) -> None:
        self.repo = repo
        self.adapter_path = adapter_path
        # Suffix the id with the LoRA adapter name so evals and logs can tell
        # a LoRA-adapted run from the base model.
        if adapter_path:
            from pathlib import Path

            lora_name = Path(adapter_path).stem or "lora"
            self.id = f"mlx:{repo.split('/')[-1]}+lora:{lora_name}"
        else:
            self.id = f"mlx:{repo.split('/')[-1]}"
        self.context_window = context_window
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        """Load model + tokenizer from the HF cache (downloading if
        missing). Safe to call more than once; no-op after the first.

        When `adapter_path` is set, mlx_lm applies the LoRA weights on
        top of the base model at load time. `adapter_path` must be a
        DIRECTORY produced by `mlx_lm.lora` training — it should
        contain `adapter_config.json` plus the weight files. The
        result behaves like any other adapter from our perspective —
        no changes downstream."""
        if self._model is not None:
            return
        from mlx_lm import load as _load

        # mlx_lm.load() returns a 2- or 3-tuple depending on `return_config`;
        # we only need model + tokenizer. Index instead of unpacking so the
        # Union type-checks cleanly.
        if self.adapter_path:
            loaded = _load(self.repo, adapter_path=self.adapter_path)
        else:
            loaded = _load(self.repo)
        self._model = loaded[0]
        self._tokenizer = loaded[1]

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def count_tokens(self, messages: Iterable[ChatMessage]) -> int:
        """Exact token count via the model's own tokenizer when loaded;
        char-heuristic fallback otherwise. We don't trigger `_ensure_loaded`
        here — the CLI renders the meter every turn including before the
        first generation, and forcing a load just for a display value
        would add tens of seconds of startup latency."""
        if self._tokenizer is None:
            return approx_token_count(messages)
        dicts = _messages_to_dicts(messages)
        ids = self._tokenizer.apply_chat_template(dicts, tokenize=True, add_generation_prompt=True)
        return len(ids)

    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Token-by-token streaming for plain chat. Each yielded string
        is the incremental text for that generation step (as reported by
        mlx_lm.stream_generate)."""
        self._ensure_loaded()
        from mlx_lm import stream_generate as _stream_generate
        from mlx_lm.sample_utils import make_sampler

        dicts = _messages_to_dicts(messages)
        assert self._tokenizer is not None
        prompt = self._tokenizer.apply_chat_template(
            dicts, tokenize=False, add_generation_prompt=True
        )
        sampler = make_sampler(temp=temperature, top_p=0.9)
        for resp in _stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            sampler=sampler,
            max_tokens=max_tokens,
        ):
            if resp.text:
                yield resp.text

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        return "".join(self.stream(messages, max_tokens=max_tokens, temperature=temperature))

    def stream_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> Iterator[StreamChunk]:
        """Token streaming with tool-call awareness. Yields:

        - zero or more `StreamText` chunks carrying the visible text
          (with `<tool_call>` / `<function=...>` spans masked out via
          `_TagMasker` so raw JSON/XML never surfaces to the user).
        - exactly one final `StreamComplete` chunk carrying the parsed
          `ModelReply` (content, tool_calls, truncation + unparseable
          flags). The tool loop consumes the StreamComplete to dispatch.

        Temperature defaults to 0.5 — lower than chat default because
        tool use wants deliberate, parseable output, not creativity."""
        self._ensure_loaded()
        from mlx_lm import stream_generate as _stream_generate
        from mlx_lm.sample_utils import make_sampler

        dicts = _messages_to_dicts_with_tools(messages)
        tool_schemas = [_tool_spec_to_schema(t) for t in tools] if tools else None
        assert self._tokenizer is not None
        prompt = self._tokenizer.apply_chat_template(
            dicts,
            tokenize=False,
            add_generation_prompt=True,
            tools=tool_schemas,
        )
        sampler = make_sampler(temp=temperature, top_p=0.9)

        raw_parts: list[str] = []
        masker = _TagMasker()
        for resp in _stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            sampler=sampler,
            max_tokens=max_tokens,
        ):
            delta = resp.text
            if not delta:
                continue
            raw_parts.append(delta)
            visible = masker.feed(delta)
            if visible:
                yield StreamText(text=visible)
        tail = masker.flush()
        if tail:
            yield StreamText(text=tail)

        raw = "".join(raw_parts)
        content, tool_calls = _parse_qwen_tool_calls(raw)
        try:
            out_tokens = self._tokenizer.encode(raw)
            was_truncated = len(out_tokens) >= max_tokens - 1
        except Exception:
            # Heuristic only — never break a real turn over a tokenizer hiccup.
            was_truncated = False
        had_unparseable_call = not tool_calls and ("<tool_call>" in raw or "<function=" in raw)
        if tools and not tool_calls and raw.strip():
            _log_tool_bail(raw, content)
        yield StreamComplete(
            reply=ModelReply(
                content=content,
                tool_calls=tuple(tool_calls),
                was_truncated=was_truncated,
                had_unparseable_call=had_unparseable_call,
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
        for chunk in self.stream_with_tools(
            messages, tools=tools, max_tokens=max_tokens, temperature=temperature
        ):
            if isinstance(chunk, StreamComplete):
                return chunk.reply
        raise RuntimeError("stream_with_tools exhausted without StreamComplete")
