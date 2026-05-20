from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from harness.model.adapter import ChatMessage, approx_token_count
from harness.model.qwen_parse import (
    _log_tool_bail,
    _messages_to_dicts,
    _messages_to_dicts_with_tools,
    _parse_qwen_tool_calls,
    _TagMasker,
    _tool_spec_to_schema,
)
from harness.tools.base import (
    ModelReply,
    StreamChunk,
    StreamComplete,
    StreamText,
    ToolSpec,
)

# Re-export Qwen parsers for tests and external callers that used to
# import them from here (harness-782y moved the impl into
# model/qwen_parse.py but kept the import path stable).
__all__ = [
    "MLXAdapter",
    "_TagMasker",
    "_log_tool_bail",
    "_messages_to_dicts",
    "_messages_to_dicts_with_tools",
    "_parse_qwen_tool_calls",
    "_tool_spec_to_schema",
]


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
        # harness-gt0m: 131_072 → 32_768. Qwen 2.5's native attention
        # is solid at 32K; the 128K nominal is YaRN-extended and
        # suffers lost-in-the-middle past ~64K. Lower default keeps
        # `--compact-at` (default 0.8 * context_window) firing at a
        # useful threshold (~26K vs ~105K, where it effectively never
        # fired interactively) and trims KV-cache memory pressure.
        # Callers who genuinely need long-doc work can override via
        # the constructor.
        context_window: int = 32_768,
        cache_limit_mb: int | None = None,
        draft_repo: str | None = None,
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
        # Falls back to Settings.mlx_cache_limit_mb (HARNESS_MLX_CACHE_
        # LIMIT_MB env) when caller passes None. Zero caps the cache at
        # 0 bytes — every dealloc returns to the system allocator, which
        # trades peak RSS for per-turn re-alloc latency. See Settings
        # for the full rationale.
        from harness.config import settings as _settings

        self.cache_limit_mb = (
            cache_limit_mb if cache_limit_mb is not None else _settings.mlx_cache_limit_mb
        )
        # Speculative-decoding draft model. A smaller same-vocab model
        # (e.g. Qwen2.5-0.5B-Instruct-4bit paired with 7B/32B target)
        # drafts candidate tokens that the target model verifies in
        # parallel — distribution-preserving, so output quality is
        # mathematically identical to non-speculative decoding. Falls
        # back to Settings.mlx_draft_model_repo (HARNESS_MLX_DRAFT_
        # MODEL_REPO env) when caller passes None. None = disabled.
        self.draft_repo = draft_repo if draft_repo is not None else _settings.mlx_draft_model_repo
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._draft_model: Any | None = None
        self._cache_limit_applied: bool = False

    def load(self) -> None:
        """Load model + tokenizer from the HF cache (downloading if
        missing). Safe to call more than once; no-op after the first.

        When `adapter_path` is set, mlx_lm applies the LoRA weights on
        top of the base model at load time. `adapter_path` must be a
        DIRECTORY produced by `mlx_lm.lora` training — it should
        contain `adapter_config.json` plus the weight files.

        When `draft_repo` is set, also loads a smaller draft model for
        speculative decoding. The draft must share the main model's
        tokenizer vocab; mismatched drafts are dropped with a warning
        rather than raising, so a misconfigured env var doesn't kill
        the session."""
        if self._model is not None:
            return
        self._apply_cache_limit()
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
        if self.draft_repo:
            self._load_draft(_load)

    def _load_draft(self, loader: Any) -> None:
        """Load the draft model + verify vocab compatibility. On any
        failure (download error, vocab mismatch) we warn and continue
        non-speculative — speculative decoding is a throughput
        optimization, never a correctness requirement."""
        import warnings

        try:
            loaded = loader(self.draft_repo)
        except Exception as exc:
            warnings.warn(
                f"draft model {self.draft_repo!r} failed to load "
                f"({type(exc).__name__}: {exc}); running without "
                "speculative decoding",
                stacklevel=3,
            )
            return
        draft_model = loaded[0]
        draft_tokenizer = loaded[1]
        # mlx_lm's speculative_generate_step requires the draft and main
        # to share a tokenizer vocab. Compare vocab sizes as a cheap
        # proxy — tokenizers that actually agree on ids will have the
        # same size. A deep equality check would catch more cases but
        # is overkill for the Qwen-family happy path this targets.
        main_vocab = getattr(self._tokenizer, "vocab_size", None)
        draft_vocab = getattr(draft_tokenizer, "vocab_size", None)
        if main_vocab is not None and draft_vocab is not None and main_vocab != draft_vocab:
            warnings.warn(
                f"draft model {self.draft_repo!r} vocab size {draft_vocab} "
                f"!= main vocab {main_vocab}; running without speculative "
                "decoding",
                stacklevel=3,
            )
            return
        self._draft_model = draft_model

    def _apply_cache_limit(self) -> None:
        """Apply `self.cache_limit_mb` to MLX's free-cache cap. No-op if
        the limit is None or has already been applied in this process.
        Errors from the MLX API are swallowed — a missing cap should
        not block model loading."""
        if self.cache_limit_mb is None or self._cache_limit_applied:
            return
        try:
            import mlx.core as mx

            mx.set_cache_limit(self.cache_limit_mb * 1024 * 1024)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"MLX cache-limit cap not applied ({type(exc).__name__}: {exc})",
                stacklevel=2,
            )
        self._cache_limit_applied = True

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def count_tokens(self, messages: Iterable[ChatMessage]) -> int:
        """Exact token count via the model's own tokenizer when loaded;
        char-heuristic fallback otherwise. Never triggers `_ensure_loaded`
        — the CLI renders the meter every turn including before the
        first generation, and forcing a load just for a display value
        would add tens of seconds of startup latency."""
        if self._tokenizer is None:
            return approx_token_count(messages)
        dicts = _messages_to_dicts(messages)
        ids = self._tokenizer.apply_chat_template(dicts, tokenize=True, add_generation_prompt=True)
        return len(ids)

    def _build_prompt(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> str:
        """Shared prompt-assembly path for every generation entry point.
        Loads the model lazily, renders the messages as dicts, and runs
        `apply_chat_template` with tool schemas attached when provided."""
        self._ensure_loaded()
        assert self._tokenizer is not None
        if tools is None:
            dicts: list[dict[str, Any]] = list(_messages_to_dicts(messages))
            return str(
                self._tokenizer.apply_chat_template(
                    dicts, tokenize=False, add_generation_prompt=True
                )
            )
        dicts_with_tools = _messages_to_dicts_with_tools(messages)
        tool_schemas = [_tool_spec_to_schema(t) for t in tools]
        return str(
            self._tokenizer.apply_chat_template(
                dicts_with_tools,
                tokenize=False,
                add_generation_prompt=True,
                tools=tool_schemas,
            )
        )

    def _make_sampler(self, temperature: float) -> Any:
        """Unified sampler config (top_p=0.9 across all entry points).
        Isolated so the temperature policy per entry point stays
        explicit at the call site."""
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(temp=temperature, top_p=0.9)

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
        from mlx_lm import stream_generate as _stream_generate

        prompt = self._build_prompt(messages)
        assert self._tokenizer is not None  # narrows for mypy; _build_prompt loaded it
        sampler = self._make_sampler(temperature)
        for resp in _stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            draft_model=self._draft_model,
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

    def complete_grammar(
        self,
        messages: Iterable[ChatMessage],
        schema: dict[str, object],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate JSON output constrained to `schema` via outlines.
        Requires the `grammar` extra. Shares the already-loaded mlx
        model + tokenizer with the rest of this adapter so the router
        doesn't cost a second copy of weights.

        Returns the raw JSON string (callers parse + coerce). Any
        outlines / runtime failure bubbles as an exception — the
        caller (GrammarRouter) catches and returns None to preserve
        the advisory contract."""
        try:
            # outlines ships without py.typed; mypy sees `json` as
            # not-explicitly-exported even though it's the public API.
            from outlines.generate import json as outlines_json  # type: ignore[attr-defined]
            from outlines.models.mlxlm import MLXLM
            from outlines.samplers import greedy, multinomial
        except ImportError as exc:
            raise RuntimeError(
                "complete_grammar requires the `grammar` extra. "
                "Install with: uv sync --extra grammar"
            ) from exc

        prompt = self._build_prompt(messages)
        assert self._tokenizer is not None  # narrows for mypy; _build_prompt loaded it
        # MLXLM wraps our pre-loaded model + tokenizer; no extra load.
        wrapped = MLXLM(model=self._model, tokenizer=self._tokenizer)
        # outlines 0.2.x does not forward `temperature` to MLXLM.generate
        # (it silently lost its **kwargs path somewhere in the 0.1→0.2
        # rewrite). Temperature must be set at generator-construction
        # time via the sampler instead. Greedy when temp == 0 matches
        # our router's deterministic-decode contract.
        sampler = (
            greedy()  # type: ignore[no-untyped-call]
            if temperature <= 0
            else multinomial(temperature=temperature)
        )
        # outlines 0.1.x expects the schema as a JSON string (not a dict) —
        # its docstring says it accepts a Pydantic class, a function, or
        # a string containing the JSON Schema spec. Dump the dict here so
        # the caller (GrammarRouter) can keep working with dicts.
        generator = outlines_json(wrapped, json.dumps(schema), sampler=sampler)  # type: ignore[arg-type]
        raw = generator(prompt, max_tokens=max_tokens)
        # outlines can return either a Python object (dict) or the raw
        # JSON string depending on version. Normalize to a string so the
        # caller's parser path stays identical to the free-form path.
        if isinstance(raw, str):
            return raw
        return json.dumps(raw)

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
        from mlx_lm import stream_generate as _stream_generate

        prompt = self._build_prompt(messages, tools=tools)
        assert self._tokenizer is not None  # narrows for mypy; _build_prompt loaded it
        sampler = self._make_sampler(temperature)

        raw_parts: list[str] = []
        masker = _TagMasker()
        for resp in _stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            draft_model=self._draft_model,
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
        assert self._tokenizer is not None
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
