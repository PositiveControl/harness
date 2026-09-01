"""CavemanRewriter — ab's voice layer.

Wraps a base ModelAdapter in a two-pass generation pipeline: pass 1
emits substance, pass 2 rewrites the draft into caveman-compressed
prose at the configured intensity. Mirrors the architecture of
`persona.rewriter.PersonaAdapter` so the orchestrator can swap between
Airton's voice layer and ab's without restructuring the chat loop.

Intensity levels (`lite` | `full` | `ultra`) progress in drop-rate:

- `lite`: filler, hedging, and pleasantries dropped. Articles mostly
  kept. Complete sentences preferred.
- `full`: articles dropped. Fragments OK. Classic caveman.
- `ultra`: maximum compression. Telegraph style.

Three invariants hold across every intensity:

1. Substance is preserved. Every piece of advice, option, refusal, or
   fact in the draft must still be present in the rewrite.
2. Reason-attachment survives. `because` clauses, "blocks X", "blocks-
   others", deadline markers, and other justification fragments stay
   in the output; the *why* on a priority line never compresses out.
3. Code blocks, file paths, error quotes, numbers, dates, and IDs are
   preserved exactly.

The register_map (loaded from `character/airton_b/register_map.yaml`)
lets the orchestrator pass a per-surface intensity hint via
`pick_intensity(surface)`. Known auto-clarity surfaces
(destructive-confirm, error-retraction, clarifying-response) resolve
to `normal`, which bypasses the rewrite entirely and returns the
base-adapter draft unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from harness.model.adapter import ChatMessage
from harness.model.adapter import count_tokens as _count_tokens

if TYPE_CHECKING:
    from harness.model.adapter import ModelAdapter

ALLOWED_INTENSITIES = ("lite", "full", "ultra", "normal")


_BASE_PRESERVATION_RULES = """\
PRESERVE across every intensity:

  - Every piece of advice, option, refusal, fact, or number in the
    draft must still be present in the rewrite.
  - Every `because` / reason clause. If a line in the draft has a
    justification ("blocks executor", "date-locked, tomorrow",
    "deadline Mon"), that justification MUST survive verbatim.
  - Code blocks, file paths, error strings, dates, deadlines, IDs,
    and proper nouns — reproduce exactly.
  - Any quoted user text — leave the quote alone.
  - Tool call intent / structure — if the draft names a command or
    file, the rewrite names the same command or file.

Return ONLY the rewritten reply. No preamble, no explanation of what
you changed."""


_LITE_INSTRUCTIONS = (
    """\
Compress the DRAFT into caveman-lite register. Keep it readable; this
is the default for ordered lists and capture dialogues where fragment
order matters.

DROP:
  - Filler: just, really, basically, actually, simply, quite, rather.
  - Hedging: perhaps, maybe, might want to, could consider.
  - Pleasantries: sure, certainly, of course, happy to, I'd be glad to.
  - Generic openers: "That's a solid approach", "Great question",
    "Let me break it down", "Here's how", "I cannot comply".

KEEP:
  - Articles (a / an / the) where dropping them would mislead.
  - Complete sentences when clarity is at stake.
  - Short synonyms: big > extensive, fix > solve, use > utilize,
    start > commence.

"""
    + _BASE_PRESERVATION_RULES
)


_FULL_INSTRUCTIONS = (
    """\
Compress the DRAFT into caveman-full register. Classic caveman:
fragments OK, minimal syntax, maximum substance density.

DROP:
  - Articles (a / an / the) when meaning is unambiguous.
  - All filler (just, really, basically, actually, simply, quite).
  - All pleasantries and hedging.
  - Generic openers.
  - Linking words ("in order to", "so that") when the context makes
    them redundant.

REWRITE:
  - Pattern: "[thing] [action] [reason]. [next step]."
    Example: "Bug in auth middleware. Token check use `<` not `<=`. Fix:"
  - Use short synonyms aggressively.
  - Fragments are OK and often right.

"""
    + _BASE_PRESERVATION_RULES
)


_ULTRA_INSTRUCTIONS = (
    """\
Compress the DRAFT into caveman-ultra register. Telegraph style;
maximum compression. Use only when the caller explicitly asks for it.

DROP:
  - Everything `full` drops.
  - Auxiliary verbs when tense is unambiguous (is / are / was / were).
  - Subjects when obvious from context.
  - Punctuation inside fragments where line breaks carry the grouping.

REWRITE:
  - Single-clause fragments, line-per-fragment where possible.
  - Verb-noun-qualifier ordering: "Ship PR. Blocks executor. Mon."

"""
    + _BASE_PRESERVATION_RULES
)


_INSTRUCTIONS_BY_INTENSITY = {
    "lite": _LITE_INSTRUCTIONS,
    "full": _FULL_INSTRUCTIONS,
    "ultra": _ULTRA_INSTRUCTIONS,
}


def build_caveman_messages(draft: str, *, intensity: str) -> list[ChatMessage]:
    """Compose the messages for a caveman rewrite pass. The system
    prompt holds the intensity-specific rules and the preservation
    invariants; the user message is the draft to be rewritten."""
    if intensity not in _INSTRUCTIONS_BY_INTENSITY:
        raise ValueError(
            f"intensity must be one of {tuple(_INSTRUCTIONS_BY_INTENSITY)!r}, got {intensity!r}"
        )
    system = ChatMessage(role="system", content=_INSTRUCTIONS_BY_INTENSITY[intensity])
    user = ChatMessage(
        role="user",
        content=f"Draft to rewrite in caveman-{intensity}:\n\n{draft}\n\nRewrite:",
    )
    return [system, user]


def load_register_map(path: Path) -> dict[str, object]:
    """Parse a register_map.yaml file. Missing / empty file → empty
    map (caller falls back to `default_intensity`). Unknown intensity
    values raise so typos surface before they ship."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"register_map must be a mapping; got {type(data).__name__}")
    default = data.get("default")
    if default is not None and default not in ALLOWED_INTENSITIES:
        raise ValueError(
            f"register_map default must be one of {ALLOWED_INTENSITIES}; got {default!r}"
        )
    surfaces = data.get("surfaces") or {}
    if not isinstance(surfaces, dict):
        raise ValueError("register_map.surfaces must be a mapping")
    for surface, value in surfaces.items():
        if value not in ALLOWED_INTENSITIES:
            raise ValueError(
                f"register_map surface {surface!r} has invalid intensity {value!r}; "
                f"must be one of {ALLOWED_INTENSITIES}"
            )
    return data


class CavemanRewriter:
    """Voice-rewrite wrapper for ab. Compose on top of any base
    ModelAdapter — the chat loop, voice evals, and tool loop all call
    this the same way Airton's PersonaAdapter is called.

    Behaviour knobs:

    - `intensity`: default rewrite intensity when no surface is named.
      Overridden per-turn by `pick_intensity(surface)`.
    - `register_map`: dict loaded from register_map.yaml; the
      orchestrator passes surface hints and `pick_intensity` resolves
      them to an intensity.
    - `rewrite_on_tools`: when False (default), tool-loop replies skip
      the caveman pass — rewriter compresses, which is wrong for
      investigate/summarize turns where the user needs full prose.
    """

    def __init__(
        self,
        base: ModelAdapter,
        *,
        intensity: str = "lite",
        register_map: Mapping[str, object] | None = None,
        rewrite_on_tools: bool = False,
        rewriter_temperature: float = 0.2,
        rewriter_max_tokens: int | None = None,
    ) -> None:
        if intensity not in ALLOWED_INTENSITIES:
            raise ValueError(f"intensity must be one of {ALLOWED_INTENSITIES}; got {intensity!r}")
        self.base = base
        self.intensity = intensity
        self.register_map: Mapping[str, object] = register_map or {}
        self.rewrite_on_tools = rewrite_on_tools
        self.rewriter_temperature = rewriter_temperature
        self.rewriter_max_tokens = rewriter_max_tokens
        self.id = f"caveman[{base.id}]"

    @property
    def context_window(self) -> int:
        """Forward, don't snapshot (harness-chzp2). The wrapped adapter
        may resolve its window lazily — VllmAdapter reads the served
        model's max_model_len on first use — so copying the value at
        construction would freeze every persona-wrapped session on the
        pre-discovery default."""
        return self.base.context_window

    def load(self) -> None:
        loader = getattr(self.base, "load", None)
        if callable(loader):
            loader()

    def count_tokens(self, messages: Iterable[ChatMessage]) -> int:
        return _count_tokens(self.base, messages)

    def pick_intensity(self, surface: str | None) -> str:
        """Resolve a per-turn intensity from a surface hint. Falls back
        to the register_map's `default`, then the rewriter's configured
        intensity. Unknown surfaces resolve to the default — no silent
        exceptions for off-map surfaces."""
        if surface is None:
            return self._map_default()
        surfaces = self.register_map.get("surfaces") or {}
        if isinstance(surfaces, dict) and surface in surfaces:
            value = surfaces[surface]
            if isinstance(value, str):
                return value
        return self._map_default()

    def _map_default(self) -> str:
        default = self.register_map.get("default")
        if isinstance(default, str) and default in ALLOWED_INTENSITIES:
            return default
        return self.intensity

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        surface: str | None = None,
    ) -> str:
        draft = self.base.complete(messages, max_tokens=max_tokens, temperature=temperature)
        return self._rewrite(draft, max_tokens=max_tokens, surface=surface)

    def _rewrite(
        self,
        draft: str,
        *,
        max_tokens: int,
        surface: str | None,
    ) -> str:
        chosen = self.pick_intensity(surface)
        if chosen == "normal":
            return draft
        rewrite_cap = self.rewriter_max_tokens if self.rewriter_max_tokens else max_tokens
        rewrite_msgs = build_caveman_messages(draft, intensity=chosen)
        return self.base.complete(
            rewrite_msgs,
            max_tokens=rewrite_cap,
            temperature=self.rewriter_temperature,
        )

    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        surface: str | None = None,
    ) -> Iterator[str]:
        """Stream the substance pass, then stream the caveman rewrite.
        When `surface` resolves to `normal` (auto-clarity bypass), the
        rewrite pass is skipped and the draft streams uninterrupted."""
        base_stream = getattr(self.base, "stream", None)
        if not callable(base_stream):
            yield self.complete(
                messages, max_tokens=max_tokens, temperature=temperature, surface=surface
            )
            return

        draft_parts: list[str] = []
        for delta in base_stream(messages, max_tokens=max_tokens, temperature=temperature):
            draft_parts.append(delta)
            yield delta

        chosen = self.pick_intensity(surface)
        if chosen == "normal":
            return

        draft = "".join(draft_parts)
        rewrite_cap = self.rewriter_max_tokens if self.rewriter_max_tokens else max_tokens
        yield "\n\n*— caveman pass —*\n\n"
        rewrite_msgs = build_caveman_messages(draft, intensity=chosen)
        yield from base_stream(
            rewrite_msgs,
            max_tokens=rewrite_cap,
            temperature=self.rewriter_temperature,
        )

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: object = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        surface: str | None = None,
    ) -> object:
        """Pass tool-use turns through to the base adapter. When
        `rewrite_on_tools` is False (default), the rewriter stays out
        of the loop entirely — tool results stream unmodified to the
        caller. When True, the final string reply from the tool loop
        is rewritten.

        `tools` is keyword-only to match the _ToolCapableAdapter
        protocol used by the orchestrator; previous positional-arg
        wiring silently broke every real tool turn because MLX /
        Ollama / Echo all declare tools keyword-only."""
        # The base adapter owns the tool-loop contract; re-exporting it
        # without touching the intermediate steps keeps the rewriter
        # from interfering with model ↔ tool message shaping.
        adapter_with_tools = getattr(self.base, "complete_with_tools", None)
        if not callable(adapter_with_tools):
            raise AttributeError(
                f"base adapter {type(self.base).__name__} has no complete_with_tools"
            )
        result = adapter_with_tools(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not self.rewrite_on_tools:
            return result
        if isinstance(result, str):
            return self._rewrite(result, max_tokens=max_tokens, surface=surface)
        return result
