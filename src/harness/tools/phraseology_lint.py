"""Phraseology lint tool — cite-grounded ATC transmission verifier (harness-q35t).

Phase-1 of the cite-grounded ATC transmission verifier (epic harness-0pte).
Takes a controller utterance, retrieves the top-K JO 7110.65 chunks via
airton_c1's hardened hybrid stack, prompts the model to compare the
utterance to the candidate PHRASEOLOGY blocks, and returns a structured
verdict + JO 7110.65 § citation.

Cite-or-silent is enforced at two gates:

1. Pre-model: empty hybrid retrieval ⇒ short-circuit `out_of_scope`
   without invoking the model. No candidate sections, no answer.
2. Post-model: the cite the model picked must appear among the
   candidate-section anchors derived from the question's top-K
   retrieval (the same rule the cite-grounding catcher applies in
   `harness/persona/cite_grounding.py`). Ungrounded cite ⇒ verdict
   downgraded to `out_of_scope` (refuse rather than guess).

The pure function `lint_utterance()` is what the CLI subcommand and the
batch eval harness call. `PhraseologyLintTool` wraps it as a chat-tier
Tool so a session can call it via the standard tool loop.

Contract for downstream consumers (eval / CLI / chat-side rendering):

  PhraseologyVerdict(
      verdict          = "ok" | "wrong" | "incomplete" | "out_of_scope",
      expected_section = "<chapter>-<section>-<para>" | None,
      expected_phraseology = "<canonical template>" | None,
      mismatch         = "<one-line reason>" | None,   # wrong/incomplete only
      citation_quote   = "<verbatim chunk excerpt>" | None,
  )
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from harness.persona.cite_grounding import _extract_anchor
from harness.tools.base import ToolHit, ToolResult, ToolSpec

if TYPE_CHECKING:
    from harness.model.adapter import ChatMessage, ModelAdapter
    from harness.store.episodic import EpisodicRecord, EpisodicStore


Verdict = Literal["ok", "wrong", "incomplete", "out_of_scope"]
_VALID_VERDICTS: frozenset[str] = frozenset({"ok", "wrong", "incomplete", "out_of_scope"})


# Body excerpt cap per candidate. Sized to keep the lint prompt under
# ~3 candidates x 800 chars + scaffolding ~= 3 KB so the cost stays
# tolerable on 7B/32B Qwen with the persona system prompt off (the
# lint pipeline runs the model bare, not through PersonaAdapter — see
# `lint_utterance` doc).
_CANDIDATE_BODY_CAP = 800

# Default top-K for hybrid retrieval. Matches `SearchMemoryTool._DEFAULT_K`
# so the candidate-anchor set the lint pipeline sees mirrors what the
# chat path would surface for the same utterance.
_DEFAULT_K = 8

# Number of candidates rendered into the lint prompt. Smaller than the
# retrieval K so the model gets the strongest few rather than the long
# tail; the cite-grounding gate still uses the full top-K anchor set
# so the model can't pick a section we never showed it.
_PROMPT_CANDIDATES = 3


@dataclass(frozen=True)
class PhraseologyVerdict:
    """Structured output of one lint call. Mirrors the
    `phraseology_eval.yaml` row shape so eval scoring is a direct
    field-by-field comparison."""

    verdict: Verdict
    expected_section: str | None
    expected_phraseology: str | None
    mismatch: str | None
    citation_quote: str | None


def lint_utterance(
    utterance: str,
    *,
    adapter: ModelAdapter,
    episodic_store: EpisodicStore,
    scenario_hint: str | None = None,
    user_id: str | None = None,
    k: int = _DEFAULT_K,
    temperature: float = 0.0,
) -> PhraseologyVerdict:
    """Lint one ATC utterance against JO 7110.65.

    Returns a structured verdict. Cite-or-silent on empty retrieval +
    on ungrounded model citations (see module docstring).

    The model is invoked bare (no PersonaAdapter wrap) — the lint
    output is structured JSON, not voice-shaped prose. Persona
    rewriting would compress / paraphrase the canonical phraseology
    string and break the eval comparison.

    `temperature=0.0` is the eval default — verdict stability matters
    more than sampling variance. Raise it only when probing for
    consensus across rolls.
    """

    query = utterance if not scenario_hint else f"{utterance}\n[scenario: {scenario_hint}]"
    hits = episodic_store.search(query, k=k, mode="hybrid", user_id=user_id)
    if not hits:
        # Pre-model gate: nothing in the corpus matched. Don't ask the
        # model — refuse outright. Distinguishes a phraseology gap in
        # the rulebook from a phraseology violation.
        return PhraseologyVerdict(
            verdict="out_of_scope",
            expected_section=None,
            expected_phraseology=None,
            mismatch=None,
            citation_quote=None,
        )

    candidate_anchors: list[str] = []
    seen: set[str] = set()
    for rec, _score in hits:
        anchor = _extract_anchor(rec.principle or "")
        if anchor and anchor not in seen:
            seen.add(anchor)
            candidate_anchors.append(anchor)

    prompt_messages = _build_lint_messages(utterance, scenario_hint, hits)
    raw = adapter.complete(prompt_messages, temperature=temperature, max_tokens=512)
    parsed = _parse_verdict_json(raw)

    if parsed is None:
        # Model failed to emit parseable JSON — refuse rather than
        # invent a verdict. Same cite-or-silent discipline.
        return PhraseologyVerdict(
            verdict="out_of_scope",
            expected_section=None,
            expected_phraseology=None,
            mismatch="model output unparseable",
            citation_quote=None,
        )

    # Cite-grounding gate. If the model picked a section that isn't in
    # the candidate-anchor set, that's a real-but-wrong-section fab on
    # this query — even if the section exists somewhere in the corpus.
    # Downgrade to out_of_scope.
    section = parsed.expected_section
    if section is not None and section not in candidate_anchors:
        return PhraseologyVerdict(
            verdict="out_of_scope",
            expected_section=None,
            expected_phraseology=None,
            mismatch=f"cite §{section} not grounded in question retrieval",
            citation_quote=None,
        )

    # Out-of-scope verdicts must null the citation fields. The model
    # sometimes parrots a section even when calling the utterance OOS;
    # normalize so the eval-side comparison is unambiguous.
    if parsed.verdict == "out_of_scope":
        return PhraseologyVerdict(
            verdict="out_of_scope",
            expected_section=None,
            expected_phraseology=None,
            mismatch=parsed.mismatch,
            citation_quote=None,
        )

    return parsed


def _build_lint_messages(
    utterance: str,
    scenario_hint: str | None,
    hits: Sequence[tuple[EpisodicRecord, float]],
) -> list[ChatMessage]:
    """Render the lint prompt. System message states the rules and the
    JSON schema; user message carries the utterance + candidate
    sections."""
    from harness.model.adapter import ChatMessage

    system = (
        "You are a JO 7110.65 phraseology linter for U.S. air-traffic "
        "controllers. Compare a controller utterance to the canonical "
        "phraseology in the candidate sections below and return a "
        "structured verdict.\n\n"
        "Note on slot values: ATC speaks digits phonetically ('TWO "
        "SEVEN' = runway 27, 'ONE TWO THREE POINT FOUR' = frequency "
        "123.4) and uses NATO phonetics for letters ('JULIETT' = "
        "taxiway J). These spelled forms ARE filled slot values.\n\n"
        "Verdicts:\n"
        "- ok: utterance matches the canonical phraseology for the "
        "governing section (slot values like runway number, call sign, "
        "and frequency may differ from the example).\n"
        "- wrong: utterance has a substantive mismatch against canonical "
        "(wrong verb, wrong code, wrong unit, wrong noun-class).\n"
        "- incomplete: utterance is missing a JO 7110.65-required "
        "element for that section (e.g. takeoff clearance without "
        "runway number).\n"
        "- out_of_scope: utterance is not governed by any candidate "
        "section. Pilot transmissions (Mayday, request clearance, "
        "position reports) are out_of_scope — JO 7110.65 prescribes "
        "controller phraseology only.\n\n"
        "Rules:\n"
        "1. Cite ONLY sections from the candidate list. Do not invent §s.\n"
        "2. Return JSON ONLY — no preamble, no markdown fence.\n"
        "3. For out_of_scope, set expected_section / "
        "expected_phraseology / citation_quote to null.\n\n"
        "Schema:\n"
        '{"verdict": "ok"|"wrong"|"incomplete"|"out_of_scope", '
        '"expected_section": "<chapter>-<section>-<para>"|null, '
        '"expected_phraseology": "<canonical template>"|null, '
        '"mismatch": "<one-line reason>"|null, '
        '"citation_quote": "<verbatim excerpt from candidate body>"|null}'
    )

    candidates: list[str] = []
    for idx, (rec, _score) in enumerate(hits[:_PROMPT_CANDIDATES], start=1):
        anchor = _extract_anchor(getattr(rec, "principle", "") or "")
        title = getattr(rec, "title", "")
        body = getattr(rec, "body", "") or ""
        excerpt = body[:_CANDIDATE_BODY_CAP]
        ellipsis = "…" if len(body) > _CANDIDATE_BODY_CAP else ""
        candidates.append(f"[{idx}] §{anchor} {title}\n{excerpt}{ellipsis}".rstrip())

    hint_line = f"\nSCENARIO HINT: {scenario_hint}" if scenario_hint else ""
    user = (
        f"UTTERANCE: {utterance}{hint_line}\n\n"
        f"CANDIDATE SECTIONS FROM JO 7110.65:\n\n"
        + "\n\n".join(candidates)
        + "\n\nReturn the JSON verdict now."
    )

    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]


# Tolerant JSON extractor — grabs the first balanced `{...}` block
# in case the model wraps its output in stray prose despite the
# instruction. Same pattern the model router uses for tolerant decode.
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict_json(raw: str) -> PhraseologyVerdict | None:
    """Parse the model's verdict JSON. Returns None on any failure
    (no parseable block, missing fields, invalid verdict enum). The
    caller treats None as cite-or-silent ⇒ out_of_scope."""
    if not raw:
        return None
    match = _JSON_BLOCK_RE.search(raw)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    verdict_raw = data.get("verdict")
    if not isinstance(verdict_raw, str) or verdict_raw not in _VALID_VERDICTS:
        return None

    def _opt_str(key: str) -> str | None:
        val = data.get(key)
        if val is None:
            return None
        if isinstance(val, str):
            stripped = val.strip()
            return stripped or None
        return None

    section = _opt_str("expected_section")
    if section is not None:
        # Normalize unicode minus → hyphen so matches against the
        # candidate-anchor set are byte-stable.
        section = section.replace("−", "-").lstrip("§").strip()  # noqa: RUF001

    return PhraseologyVerdict(
        verdict=verdict_raw,  # type: ignore[arg-type]  # checked against frozenset above
        expected_section=section,
        expected_phraseology=_opt_str("expected_phraseology"),
        mismatch=_opt_str("mismatch"),
        citation_quote=_opt_str("citation_quote"),
    )


@dataclass
class PhraseologyLintTool:
    """Chat-tier wrapper around `lint_utterance()`. Lets a chat session
    verify a controller utterance against JO 7110.65 mid-conversation.

    Output is the JSON-serialised PhraseologyVerdict so the calling
    model can quote individual fields back to the user. Structured
    `hits` populated for the per-turn audit log and the
    citations_grounded set declares the candidate-anchor section as
    grounded (when the verdict carries one)."""

    adapter: ModelAdapter
    store: EpisodicStore
    user_id: str | None = None
    k: int = _DEFAULT_K

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="phraseology_lint",
            description=(
                "Verify a controller-side ATC utterance against the JO "
                "7110.65 rulebook. Returns verdict (ok | wrong | "
                "incomplete | out_of_scope), the canonical phraseology, "
                "the governing section (e.g. '3-9-10'), and a one-line "
                "mismatch reason when the utterance violates canonical. "
                "Pilot transmissions and non-rule chitchat return "
                "out_of_scope — this tool lints controller phraseology "
                "only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "utterance": {
                        "type": "string",
                        "description": "The ATC controller transmission to lint.",
                    },
                    "scenario_hint": {
                        "type": "string",
                        "description": (
                            "Optional operational context "
                            "(departure | arrival | handoff | emergency). "
                            "Biases retrieval toward the right chapter; "
                            "omit when unsure."
                        ),
                    },
                },
                "required": ["utterance"],
            },
            tier="read",
            display_name="Lint phraseology",
        )

    def call(self, *, utterance: str, scenario_hint: str | None = None) -> ToolResult:
        verdict = lint_utterance(
            utterance,
            adapter=self.adapter,
            episodic_store=self.store,
            scenario_hint=scenario_hint,
            user_id=self.user_id,
            k=self.k,
        )
        payload = {
            "verdict": verdict.verdict,
            "expected_section": verdict.expected_section,
            "expected_phraseology": verdict.expected_phraseology,
            "mismatch": verdict.mismatch,
            "citation_quote": verdict.citation_quote,
        }
        grounded: frozenset[str] = (
            frozenset({verdict.expected_section}) if verdict.expected_section else frozenset()
        )
        # Surface the verdict's section as the single ToolHit so the
        # audit log and any downstream low-confidence hook can read it
        # without re-parsing the JSON.
        hits: tuple[ToolHit, ...] = ()
        if verdict.expected_section is not None:
            hits = (
                ToolHit(
                    source="episodic",
                    external_id=None,
                    title=f"§{verdict.expected_section}",
                    score=1.0,
                    principle=verdict.expected_phraseology,
                ),
            )
        return ToolResult(
            tool_name=self.spec.name,
            output=json.dumps(payload, ensure_ascii=False),
            hits=hits,
            citations_grounded=grounded,
        )
