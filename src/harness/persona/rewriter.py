from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from harness.model.adapter import ChatMessage

if TYPE_CHECKING:
    from harness.character import Character, VoiceSample
    from harness.model.adapter import ModelAdapter


_STYLE_REWRITER_INSTRUCTIONS = """\
You are the voice editor for {name}. Above you see how {name} talks.

Below is a DRAFT reply someone wrote. Rewrite it in {name}'s voice.

PRESERVE the substance: every piece of advice, option, refusal, or
fact in the draft must remain in the rewrite. CHANGE only the style:

  - Match the LENGTH of the examples above. If they're 1-4 sentences
    and the draft is eight paragraphs, cut the draft to 1-4 sentences.
    Brevity is not a loss of substance — paraphrase, collapse, drop
    filler. The draft is almost always too long.
  - Cut generic-assistant filler. Do not begin with "That's a solid
    approach", "Here are a few tips", "Certainly", "Great question",
    "Let me break it down", "Here's how", or "I cannot comply". If
    the draft does, replace the opener with a direct statement.
  - Prose by default, not numbered lists. Numbered lists ("1.", "2.")
    are almost always wrong for {name}. Use dashes for two or three
    concrete alternatives only.
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
MORE CONCRETE. Substance stays; abstraction goes.

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
        self.context_window = base.context_window

    def load(self) -> None:
        loader = getattr(self.base, "load", None)
        if callable(loader):
            loader()

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        draft = self.base.complete(messages, max_tokens=max_tokens, temperature=temperature)
        rewrite_cap = self.rewriter_max_tokens if self.rewriter_max_tokens else max_tokens

        style_msgs = build_rewriter_messages(self.character, draft, focus="style")
        styled = self.base.complete(
            style_msgs,
            max_tokens=rewrite_cap,
            temperature=self.rewriter_temperature,
        )
        if not self.chain_rewrites:
            return styled

        concrete_msgs = build_rewriter_messages(self.character, styled, focus="concrete")
        return self.base.complete(
            concrete_msgs,
            max_tokens=rewrite_cap,
            temperature=self.rewriter_temperature,
        )
