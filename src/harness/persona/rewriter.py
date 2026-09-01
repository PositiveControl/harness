from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

from harness.citation import CitationGrammar
from harness.model.adapter import ChatMessage
from harness.model.adapter import count_tokens as _count_tokens

if TYPE_CHECKING:
    from harness.character import Character, VoiceSample
    from harness.model.adapter import ModelAdapter


# Citation extraction is grammar-driven (harness-jaqe). The
# `_CITATION_PATTERNS` constant that used to live here (FAA: AIM /
# CFR / JO 7110.65 / AC) moved to `character/airton_c{,1}/core.yaml`
# under the `citation_grammar` block — every citation function below
# now takes a `grammar: CitationGrammar | None` argument and
# short-circuits when None. Characters without citation discipline
# (Airton, ab, echo) leave grammar None and the rewriter / preserve-
# citations / lead-with-citation passes silently no-op.


def _normalise_cite(cite: str) -> str:
    """Fold case, whitespace, and unicode-minus so
    'AIM 4-4-7' == 'aim 4−4−7' == 'AIM  4-4-7' for dedup purposes."""
    return cite.lower().replace("−", "-").replace(" ", "")


def extract_citations(text: str, grammar: CitationGrammar | None = None) -> list[str]:
    """Return every citation-shaped substring in `text` in appearance
    order, preserving surface form (model's actual casing, dash
    variant). Duplicates preserved — callers dedup via `_normalise_cite`
    as needed. Narrow patterns: won't match 'chapter 5' or '91' alone
    or 'airman 4-7'.

    `grammar` is the active character's citation grammar (loaded from
    core.yaml). When None, returns []  — characters without a
    citation grammar can't have citations to extract."""
    if grammar is None:
        return []
    out: list[str] = []
    for pattern in grammar.surface_patterns:
        out.extend(match.group(0) for match in pattern.finditer(text))
    return out


# How many characters from the start of the text count as "the
# opening" for the lead_with_citation check. 20 chars accommodates
# `Per ` / `Per the ` / `According to ` prefixes that voice samples
# sometimes use ahead of the citation, but rejects citations buried
# mid-sentence (e.g. position 22 in `Issue missed approach. JO ...`).
_LEAD_OPENING_BUDGET = 20


def _first_citation_match(text: str, grammar: CitationGrammar | None) -> re.Match[str] | None:
    """Earliest citation match in `text` across all surface patterns,
    sorted by text-position (NOT pattern order — `extract_citations`
    iterates patterns first which doesn't preserve textual ordering).
    Returns None when grammar is None — no patterns to match."""
    if grammar is None:
        return None
    earliest: re.Match[str] | None = None
    for pattern in grammar.surface_patterns:
        match = pattern.search(text)
        if match is None:
            continue
        if earliest is None or match.start() < earliest.start():
            earliest = match
    return earliest


def lead_with_citation(text: str, grammar: CitationGrammar | None = None) -> str:
    """Forward citation-discipline pass (mirror of `preserve_citations`).

    Scans `text` for citation-shaped substrings (the same patterns
    `preserve_citations` re-injects) and ensures the reply opens with
    one. Three cases:

      1. `text` already opens with a citation (citation start index is
         within the first ~80 chars): return unchanged.
      2. `text` contains at least one citation, but not at the opening:
         hoist the EARLIEST-by-position citation to the front as
         `<cite> — <body>`. Matches the airton_c1 voice-sample pattern
         (e.g. `JO 7110.65 §10-1-1 — an emergency is...`).
      3. `text` contains no citation: return unchanged. Nothing to
         hoist; this is `MissingCitationHook` /
         `UngroundedCitationHook`'s domain.

    The post-rewrite fixup `preserve_citations` runs LATER on the
    rewriter's output. The forward step ensures the rewriter sees a
    citation-first draft so the rewriter prompt's "preserve citations
    verbatim" instruction has the right material to preserve. If the
    rewriter strips the citation anyway, `preserve_citations` re-appends
    it as a trailing em-dash line — belt-and-suspenders.

    Character-gated: callers should only invoke this when
    `character.lead_with_citation` is True (airton_c1's directive
    "Name the chapter and section before answering"). General-purpose
    characters leave behaviour unchanged.

    Position-based opening detection avoids the `7110.65` false-
    positive a sentence-end heuristic would hit (the period in the
    decimal would cut the head at char 7 and miss the actual
    citation that starts at char 0)."""
    if not text:
        return text
    first = _first_citation_match(text, grammar)
    if first is None:
        return text
    if first.start() < _LEAD_OPENING_BUDGET:
        return text
    # Hoist the earliest citation to the front. Strip any leading
    # whitespace from the body so the result reads cleanly.
    body = text.lstrip()
    return f"{first.group(0)} — {body}"


def preserve_citations(draft: str, rewritten: str, grammar: CitationGrammar | None = None) -> str:
    """Append citations that survived pass-1 but disappeared in the
    rewrite, so `ppl_readback_basics`-style failures where the style
    pass compresses away `AIM 4-4-7` stop happening.

    Returns `rewritten` unchanged when every draft citation is already
    present — keeps rewriter output pristine in the common case. When
    citations went missing, appends them as a terminal em-dash line
    so a reader can still find the anchor without the body being
    restructured.

    First occurrence's surface form from the draft wins — preserves
    the model's actual phrasing rather than reconstructing it.

    `grammar` drives extraction. None → no citations to track →
    `rewritten` returned untouched."""
    draft_cites = extract_citations(draft, grammar)
    if not draft_cites:
        return rewritten
    rewritten_lower = _normalise_cite(rewritten)
    seen: set[str] = set()
    missing: list[str] = []
    for cite in draft_cites:
        key = _normalise_cite(cite)
        if key in seen:
            continue
        seen.add(key)
        if key not in rewritten_lower:
            missing.append(cite)
    if not missing:
        return rewritten
    return f"{rewritten.rstrip()}\n\n— {', '.join(missing)}"


_STYLE_REWRITER_INSTRUCTIONS = """\
You are the voice editor for {name}. Above you see how {name} talks.

Below is a DRAFT reply someone wrote. Rewrite it in {name}'s voice.

PRESERVE the substance: every piece of advice, option, refusal, or
fact in the draft must remain in the rewrite. Citations and section
references are SUBSTANCE, not style — preserve them verbatim.
Patterns like `AIM 4-4-7`, `14 CFR §91.155`, `§ 91.103`,
`JO 7110.65 §2-6-4`, `AC 90-66B` are the anchor that lets the reader
verify the rule; dropping them or paraphrasing them into "the AIM
covers this" strips the value out of the reply. Keep the exact form,
even if it reads as a parenthetical. CHANGE only the style:

  - Match the LENGTH of the examples above. If they're 1-4 sentences
    and the draft is eight paragraphs, cut the draft to 1-4 sentences.
    Brevity is not a loss of substance — paraphrase, collapse, drop
    filler. The draft is almost always too long.
  - Cut generic-assistant filler. Do not begin with "That's a solid
    approach", "Here are a few tips", "Certainly", "Great question",
    "Let me break it down", "Here's how", or "I cannot comply". If
    the draft does, replace the opener with a direct statement.
  - Numbered lists ("1.", "2.") are almost always wrong for {name};
    {name} uses dashes instead.
  - Dashed bullets are fine — often right — when each item is ONE
    CLAUSE and there are 2-5 items. If a bullet runs multiple
    sentences, it's a tutorial in disguise: collapse it into a
    neighboring bullet, tighten each to one clause, or drop the list
    structure and write prose.
  - Do NOT explode prose into bullets. If the draft's substance is
    three sentences, keep it as three sentences.
  - Cut mid-sentence filler: "ensure that", "make sure to",
    "comprehensive", "maintains robustness", "various scenarios",
    "given the complexity", "feel free to", "let me know". These are
    assistant tells. Replace with direct statements or cut.
  - When the draft admits not knowing, say "Don't know" plainly and
    list the paths to try.
  - When the draft refuses, state the concrete reason and the right
    alternative in the same breath.
  - Never close with "Would you like to discuss further" or similar.
    If a question genuinely needs to be asked, ask it tersely.
  - First person, "I". Never hide that you are software; when asked,
    say so directly.

Return only the rewritten reply. No preamble, no explanation of
what you changed."""


_CONCRETE_REWRITER_INSTRUCTIONS = """\
You are sharpening a reply that's already in {name}'s voice. Make it
MORE CONCRETE. Substance stays; abstraction goes. Citations and
section references (`AIM N-N-N`, `14 CFR §N.N`, `§ N.N`,
`JO 7110.65 §N-N-N`, `AC N-N`) stay verbatim — even if the surrounding
prose tightens, the exact section identifier must remain.

Specific substitutions to apply wherever they fit:

  - "ensure X is done" → "do X" or just the imperative.
  - "consider doing Y" → "do Y" when Y is the recommendation.
  - "maintains robustness / correctness / maintainability" → cut; these
    are assumed, not claims.
  - "various scenarios" or "various conditions" → name two or three
    specific scenarios, or cut.
  - "given the complexity" / "given the importance" → cut; if the
    complexity matters, name the specific complexity.
  - If the reply QUOTES a rule ("never X without asking"), replace the
    quote with the concrete action {name} would take instead ("I'll
    rebase locally and open a PR").
  - If the reply gives generic advice, swap to a first-person plan:
    "I'd start with...", "I checked X — ...", "Give me a minute with Y".
  - If a question is still needed, make it one short question.

Do not add new advice. Do not lengthen. Return only the sharpened
reply. No preamble."""


# Backward-compatible alias; callers use the style instructions by default.
_REWRITER_INSTRUCTIONS = _STYLE_REWRITER_INSTRUCTIONS


def build_rewriter_messages(
    character: Character,
    draft: str,
    *,
    exclude_example_ids: frozenset[str] | None = None,
    include_samples: Sequence[VoiceSample] | None = None,
    focus: str = "style",
) -> list[ChatMessage]:
    """Compose the messages for a voice-rewrite pass. The system prompt
    reuses the character sheet (voice examples + style rules) and
    appends rewriter-specific instructions. The user message is the
    draft to be rewritten.

    `include_samples` and `exclude_example_ids` forward to
    `Character.system_prompt`. With retrieval active the caller passes
    the retrieved sample set via `include_samples` for both passes so
    pass 1 and pass 2 see the same anchors.

    `focus` selects which instruction block the rewriter receives:

    - "style" (default): length, openers, bullets, filler — the
      first-pass voice rewrite.
    - "concrete": substitute abstract advice with specific actions and
      first-person moves — the optional second rewrite pass."""
    base_system = character.system_prompt(
        exclude_example_ids=exclude_example_ids,
        include_samples=include_samples,
    )
    if focus == "concrete":
        instructions = _CONCRETE_REWRITER_INSTRUCTIONS.format(name=character.name)
        user_framing = f"Reply to sharpen in {character.name}'s voice"
    elif focus == "style":
        instructions = _STYLE_REWRITER_INSTRUCTIONS.format(name=character.name)
        user_framing = f"Draft reply to rewrite in {character.name}'s voice"
    else:
        raise ValueError(f"Unknown rewriter focus: {focus!r}")

    system = ChatMessage(
        role="system",
        content=f"{base_system}\n\n{instructions}",
    )
    user = ChatMessage(
        role="user",
        content=f"{user_framing}.\n\nDraft:\n{draft}\n\nRewrite:",
    )
    return [system, user]


class PersonaAdapter:
    """Compose a base ModelAdapter with a second-pass voice rewriter.

    Pass 1 (base adapter + caller-supplied system prompt): substance.
    Pass 2 (base adapter + rewriter system prompt + draft): style.

    For chat, this is transparent: `complete()` returns the rewritten
    reply. For the voice eval, leave-one-out honesty is maintained by
    `run_voice_eval` — it drives both passes directly rather than
    going through this adapter, so exclusion stays consistent.

    The same underlying model handles both passes here. A future
    optimization is a smaller/cheaper rewriter (Qwen 7B, say) while
    keeping the 32B for substance."""

    def __init__(
        self,
        base: ModelAdapter,
        character: Character,
        *,
        rewriter_temperature: float = 0.2,
        rewriter_max_tokens: int | None = None,
        chain_rewrites: bool = False,
    ) -> None:
        self.base = base
        self.character = character
        self.rewriter_temperature = rewriter_temperature
        self.rewriter_max_tokens = rewriter_max_tokens
        self.chain_rewrites = chain_rewrites
        self.id = f"persona[{base.id}]"

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

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        draft = self.base.complete(messages, max_tokens=max_tokens, temperature=temperature)
        # Forward citation-discipline pass (plan #7). Mirror of the
        # post-rewrite preserve_citations fixup: hoists the first
        # citation to the opening of the draft when the character's
        # directive is to lead with the section anchor (airton_c1's
        # "Name the chapter and section before answering"). The
        # rewriter then preserves it; preserve_citations re-appends
        # if the rewriter still drops it.
        if getattr(self.character, "lead_with_citation", False):
            draft = lead_with_citation(draft, self.character.citation_grammar)
        rewrite_cap = self.rewriter_max_tokens if self.rewriter_max_tokens else max_tokens

        style_msgs = build_rewriter_messages(self.character, draft, focus="style")
        styled = self.base.complete(
            style_msgs,
            max_tokens=rewrite_cap,
            temperature=self.rewriter_temperature,
        )
        if not self.chain_rewrites:
            # Re-inject any citations the rewriter dropped (harness-cco).
            # Pass-1 draft is the source of truth for what citations
            # should be in the reply; the rewriter is only supposed to
            # change style.
            return preserve_citations(draft, styled, self.character.citation_grammar)

        concrete_msgs = build_rewriter_messages(self.character, styled, focus="concrete")
        concrete = self.base.complete(
            concrete_msgs,
            max_tokens=rewrite_cap,
            temperature=self.rewriter_temperature,
        )
        # Same fixup after pass-3 — measure against the original draft
        # so a citation dropped in pass-2 AND pass-3 still gets back.
        return preserve_citations(draft, concrete, self.character.citation_grammar)

    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream the two (or three) persona passes back-to-back.

        Pass 1 streams the base-adapter draft. A dim separator then
        marks the handoff to pass 2, which streams the voice rewrite.
        When `chain_rewrites` is on, pass 3 streams the concrete-
        substitution rewrite after a second separator. Every token the
        model generates reaches the caller; the caller decides what to
        render on-screen vs keep only for transcript."""
        base_stream = getattr(self.base, "stream", None)
        if not callable(base_stream):
            # Base adapter has no streaming — fall back to one-shot output
            # of the fully-composed complete(). Streams as one chunk.
            yield self.complete(messages, max_tokens=max_tokens, temperature=temperature)
            return

        rewrite_cap = self.rewriter_max_tokens if self.rewriter_max_tokens else max_tokens

        draft_parts: list[str] = []
        for delta in base_stream(messages, max_tokens=max_tokens, temperature=temperature):
            draft_parts.append(delta)
            yield delta
        draft = "".join(draft_parts)
        # Forward citation-discipline pass (plan #7) — see complete()
        # for rationale. Streamed deltas have already reached the user
        # so we don't re-emit the hoisted draft to the caller; we only
        # use it as the rewriter's input. This means the streamed
        # draft view may not be citation-first, but the rewriter sees
        # the canonical form and the final reply (post-rewriter +
        # preserve_citations) is the citation-first one.
        if getattr(self.character, "lead_with_citation", False):
            draft = lead_with_citation(draft, self.character.citation_grammar)

        yield "\n\n*— voice pass —*\n\n"

        style_msgs = build_rewriter_messages(self.character, draft, focus="style")
        styled_parts: list[str] = []
        for delta in base_stream(
            style_msgs, max_tokens=rewrite_cap, temperature=self.rewriter_temperature
        ):
            styled_parts.append(delta)
            yield delta
        if not self.chain_rewrites:
            return

        styled = "".join(styled_parts)
        yield "\n\n*— concrete pass —*\n\n"
        concrete_msgs = build_rewriter_messages(self.character, styled, focus="concrete")
        yield from base_stream(
            concrete_msgs, max_tokens=rewrite_cap, temperature=self.rewriter_temperature
        )
