from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from harness.model.adapter import ChatMessage
from harness.tools.base import ModelReply, ToolCall, ToolSpec


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
    tool_call_id. Qwen's chat template reads these correctly."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
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


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_qwen_tool_calls(raw: str) -> tuple[str, list[ToolCall]]:
    """Extract <tool_call>JSON</tool_call> blocks from Qwen's output.
    Returns (content_with_blocks_stripped, list_of_calls). Malformed
    blocks are dropped rather than raising — the model will retry on
    the next round if it cared."""
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_PATTERN.finditer(raw):
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
    content = _TOOL_CALL_PATTERN.sub("", raw).strip()
    return content, calls


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
        repo: str = "mlx-community/Qwen2.5-32B-Instruct-4bit",
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

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self._ensure_loaded()
        from mlx_lm import generate as _generate
        from mlx_lm.sample_utils import make_sampler

        dicts = _messages_to_dicts(messages)
        assert self._tokenizer is not None  # for type narrowing; _ensure_loaded guarantees
        prompt = self._tokenizer.apply_chat_template(
            dicts, tokenize=False, add_generation_prompt=True
        )
        sampler = make_sampler(temp=temperature, top_p=0.9)
        output: str = _generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            sampler=sampler,
            max_tokens=max_tokens,
        )
        return output

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        """Generate with tool-calling awareness. If `tools` is given,
        the model can emit <tool_call> blocks; the returned ModelReply
        carries the parsed calls and the text content (stripped of the
        call blocks). The orchestrator executes calls and loops back.

        Temperature defaults to 0.5 — lower than chat default because
        tool use wants deliberate, parseable output, not creativity."""
        self._ensure_loaded()
        from mlx_lm import generate as _generate
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
        raw: str = _generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            sampler=sampler,
            max_tokens=max_tokens,
        )
        content, tool_calls = _parse_qwen_tool_calls(raw)
        return ModelReply(content=content, tool_calls=tuple(tool_calls))
