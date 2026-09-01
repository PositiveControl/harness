from __future__ import annotations

import base64
import math
import mimetypes
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from harness.tools.base import ToolCall

Role = Literal["system", "user", "assistant", "tool"]

ImageDetail = Literal["auto", "low", "high"]


@dataclass(frozen=True)
class ImageRef:
    """A single image attached to a ChatMessage (vision spike, harness
    vision boundary).

    `url` is either a remote http(s) URL or a `data:` URI carrying inline
    base64 bytes. Either form is accepted by the OpenAI-compatible
    image_url content-part shape that vLLM serves for VLMs.

    This is the universal, runtime-neutral representation: a vision-capable
    adapter renders it into its wire format; text-only adapters (echo, the
    text path of any adapter) ignore `ChatMessage.images` entirely. Keeping
    the bytes behind a `data:` URI here means the adapter boundary never has
    to know whether an image came from disk, a URL, or a frame grab.

    `detail` maps to OpenAI's image fidelity hint; servers that don't honor
    it ignore it harmlessly."""

    url: str
    detail: ImageDetail = "auto"


def image_from_path(path: str | Path, *, detail: ImageDetail = "auto") -> ImageRef:
    """Read a local image file into a base64 `data:` URI ImageRef.

    Mime type is sniffed from the suffix; falls back to image/png when the
    suffix is unknown (vLLM's image loader keys off the decoded bytes, not
    the declared mime, so a wrong-but-plausible type is harmless)."""
    p = Path(path)
    mime, _ = mimetypes.guess_type(p.name)
    if mime is None or not mime.startswith("image/"):
        mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return ImageRef(url=f"data:{mime};base64,{b64}", detail=detail)


def image_from_url(url: str, *, detail: ImageDetail = "auto") -> ImageRef:
    """Wrap a remote image URL. vLLM fetches it server-side at inference
    time, so the URL must be reachable from the serving host, not the
    client."""
    return ImageRef(url=url, detail=detail)


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None
    # Populated on assistant messages that issued tool calls; preserved
    # across rounds so the model's chat template can reconstruct the turn.
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    # Populated on tool-role messages (the result of a specific call).
    tool_call_id: str | None = None
    # Vision spike: images attached to this message. Empty on text turns.
    # A vision-capable adapter renders these alongside `content` as
    # multimodal content parts; text-only adapters ignore them.
    images: tuple[ImageRef, ...] = field(default_factory=tuple)


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal adapter contract. Implementations wrap a specific runtime
    (MLX, llama.cpp, Ollama, OpenAI-compatible endpoints, etc.).

    Phase 0 is sync + non-streaming. Streaming and async arrive in Phase 1
    when the orchestrator becomes asyncio-native."""

    id: str

    # Read-only on the protocol so an adapter may resolve its window
    # lazily (harness-chzp2): VllmAdapter reads the served model's
    # max_model_len on first use and the persona wrappers forward it.
    # A plain `context_window: int` attribute still satisfies this.
    @property
    def context_window(self) -> int: ...

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str: ...


@runtime_checkable
class GrammarCapableAdapter(Protocol):
    """Optional extension for adapters that support grammar-constrained
    decoding — generation where every sampled token must continue a
    valid parse of the supplied JSON schema.

    Used by the GrammarRouter to guarantee valid JSON + valid
    `tool_name` choice by construction. Adapters that don't support
    this don't need to implement it; GrammarRouter checks for the
    method structurally and falls through if it isn't present."""

    id: str

    # Read-only on the protocol so an adapter may resolve its window
    # lazily (harness-chzp2): VllmAdapter reads the served model's
    # max_model_len on first use and the persona wrappers forward it.
    # A plain `context_window: int` attribute still satisfies this.
    @property
    def context_window(self) -> int: ...

    def complete_grammar(
        self,
        messages: Iterable[ChatMessage],
        schema: dict[str, object],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str: ...


# Output-budget reservation (harness-2epb). A request to a fixed-window
# model is sized `prompt_tokens + max_tokens`; if that sum exceeds the
# window the server rejects the whole turn (vLLM returns HTTP 422). We
# reserve a small margin on top of the generation budget so heuristic
# token-count drift can't push the real request one token over the edge.
DEFAULT_OUTPUT_SAFETY_MARGIN = 32
# Proportional component of that margin (harness-ccksu). A flat 32 tokens
# cannot absorb the char heuristic's error, because that error scales with
# the prompt: `approx_token_count` assumes ~4 chars/token, which is close
# on prose (measured +2% on a 51-token prompt against gx10's tokenizer)
# and much worse on the punctuation-dense content a tool loop accumulates
# — the funky_chicken turn that exposed this drifted 6% once ASCII art was
# in the message list, 33 tokens over a 32-token margin, and vLLM rejected
# the whole request. 10% is ~2x the observed worst case.
#
# This only ever costs anything when the request is already at the window
# boundary; a request that fits passes through untouched.
DRIFT_MARGIN_FRACTION = 0.10
# Floor below which a clamped generation budget is useless — better to
# surface a clear error than emit a request that can only dribble out a
# few tokens before truncating.
MIN_OUTPUT_TOKENS = 16


class PromptBudgetError(RuntimeError):
    """The prompt is too large to leave room for generation within the
    model's context window. Raised instead of letting the runtime reject
    the request with an opaque 422 / KV-cache overflow (harness-2epb)."""


def budget_max_tokens(
    *,
    context_window: int,
    prompt_tokens: int,
    requested_max: int,
    safety_margin: int = DEFAULT_OUTPUT_SAFETY_MARGIN,
    min_output: int = MIN_OUTPUT_TOKENS,
) -> int:
    """Clamp a generation budget so `prompt_tokens + result <=
    context_window - safety_margin`. Returns `requested_max` untouched
    when it already fits; otherwise the largest budget that fits.

    `safety_margin` is a floor, not the whole reservation: the margin
    preferred is `max(safety_margin, prompt_tokens *
    DRIFT_MARGIN_FRACTION)`, because heuristic token-count error scales
    with prompt size and a flat cushion stops covering it (harness-ccksu).

    The proportional part is a cushion, not a reservation — when it
    would leave less than `min_output` of room, it shrinks back to
    `safety_margin` rather than refusing the turn. Otherwise a big
    prompt near the window would start raising where it used to
    generate, turning a working (if tight) turn into a hard error.
    Take the cushion when it's affordable; never let it starve the
    budget on its own.

    Raises `PromptBudgetError` only when the prompt leaves less than
    `min_output` tokens of room even at the floor margin — at that point
    compaction (or a shorter prompt) is the only remedy, and a clear
    error beats a silent overflow or a 1-token reply. `context_window <=
    0` means the adapter doesn't advertise a window (e.g. a test stub);
    the budget passes through unclamped."""
    if context_window <= 0:
        return requested_max
    margin = max(safety_margin, math.ceil(prompt_tokens * DRIFT_MARGIN_FRACTION))
    available = context_window - prompt_tokens - margin
    if available < min_output:
        margin = safety_margin
        available = context_window - prompt_tokens - margin
    if available < min_output:
        raise PromptBudgetError(
            f"prompt is {prompt_tokens} tokens; with a {margin}-token "
            f"safety margin only {max(available, 0)} of the {context_window}-token "
            f"window remain for generation (need >= {min_output}). Compact the "
            "history or shorten the prompt."
        )
    return min(requested_max, available)


def approx_token_count(messages: Iterable[ChatMessage]) -> int:
    """Char-heuristic token count: ~4 chars per token plus ~4 tokens of
    role/delimiter overhead per message. Shared fallback for adapters
    that don't have a tokenizer on hand."""
    total = 0
    for m in messages:
        total += len(m.content) // 4 + 4
    return total


def count_tokens(adapter: object, messages: Iterable[ChatMessage]) -> int:
    """Ask the adapter for an exact token count when it can produce one;
    otherwise fall back to `approx_token_count`. Kept off the core
    `ModelAdapter` protocol so test stubs and alternative adapters don't
    all have to implement it."""
    materialized = list(messages)
    fn = getattr(adapter, "count_tokens", None)
    if callable(fn):
        try:
            return int(fn(materialized))
        except Exception:
            return approx_token_count(materialized)
    return approx_token_count(materialized)
