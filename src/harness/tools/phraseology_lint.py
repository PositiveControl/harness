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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from harness.persona.cite_grounding import _extract_anchor
from harness.tools.base import ToolHit, ToolResult, ToolSpec

if TYPE_CHECKING:
    from harness.citation import CitationGrammar
    from harness.model.adapter import ChatMessage, ModelAdapter
    from harness.store.episodic import EpisodicRecord, EpisodicStore

VerbAnchorMap = Mapping[str, tuple[str, ...]]


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


def default_verb_anchors_path(character_path: Path) -> Path:
    """Conventional location under `character/<name>/corpus/`. Mirrors
    `default_synonyms_path` in retrieval.query_expander — keeps every
    consumer (CLI, tool, eval) reading from the same place."""
    return character_path / "corpus" / "verb_anchors.yaml"


def load_verb_anchors(path: Path) -> VerbAnchorMap:
    """Load a section → distinctive-verb-anchor map (harness-ptya).

    File format::

        "10-2-6":          # JO § anchor — section number string
          - SQUAWK         # one verb / code phrase per list entry
          - "7500"
        "5-7-2":
          - REDUCE SPEED
          - INCREASE SPEED

    Returns an empty mapping when the file is missing or malformed —
    the lint pipeline still works, just without verb-anchor re-rank.
    Anchors are uppercased on load so match-time comparison is case
    insensitive without per-call string ops.
    """
    if not path.is_file():
        return {}
    try:
        import yaml

        with path.open(encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for raw_section, raw_verbs in data.items():
        if not isinstance(raw_section, str) or not isinstance(raw_verbs, list):
            continue
        section = raw_section.strip().lstrip("§").strip()
        if not section:
            continue
        verbs = tuple(
            str(v).strip().upper() for v in raw_verbs if isinstance(v, str) and str(v).strip()
        )
        if verbs:
            out[section] = verbs
    return out


def _matched_anchor_sections(utterance: str, verb_anchors: VerbAnchorMap) -> set[str]:
    """Sections whose verb-anchor map fires on this utterance. Word-
    boundary match against the uppercased utterance so 'CONTACT' matches
    'CONTACT DEPARTURE' but not 'TRANSCATHETER'. Multi-word anchors
    ('LINE UP AND WAIT') also work — we re.escape the literal."""
    if not verb_anchors:
        return set()
    upper = utterance.upper()
    matched: set[str] = set()
    for section, verbs in verb_anchors.items():
        for verb in verbs:
            if re.search(rf"\b{re.escape(verb)}\b", upper):
                matched.add(section)
                break
    return matched


def _verb_anchor_rerank(
    hits: Sequence[tuple[EpisodicRecord, float]],
    matched_sections: set[str],
    grammar: CitationGrammar | None,
) -> list[tuple[EpisodicRecord, float]]:
    """Stable partition: hits whose §-anchor is in `matched_sections`
    move to the front, retaining their relative retrieval order. Hits
    in unmatched sections keep their relative order in the back. When
    `matched_sections` is empty (no verb fired), pass through unchanged.

    The cite-grounding gate already filters ungrounded picks, so
    promoting matched-§ hits can only help — but if the matched § isn't
    in the slate at all, rerank is a no-op. Pair with
    `_inject_missing_anchored_sections` to fix that case.

    `grammar` (harness-jaqe) drives anchor extraction. When None,
    no rows have extractable anchors so the partition is a no-op."""
    if not matched_sections:
        return list(hits)
    front: list[tuple[EpisodicRecord, float]] = []
    back: list[tuple[EpisodicRecord, float]] = []
    for rec, score in hits:
        anchor = _extract_anchor(rec.principle or "", grammar)
        if anchor and anchor in matched_sections:
            front.append((rec, score))
        else:
            back.append((rec, score))
    return front + back


# Sentinel score for virtually-injected hits. Below the BGE-small dense
# cosine floor so it doesn't disrupt other rank ordering, but non-zero
# so log lines can distinguish "absent" from "present-but-low".
_INJECTED_HIT_SCORE = 0.001


def _principle_source(principle: str) -> str:
    """First whitespace-bounded token of a principle string. The atc
    ingest convention is `<SOURCE> §<section> (<title>)` for JO and
    `<SOURCE> §<section>` for AIM/CFR/PCG/PHAK, so the leading token
    is the publication identifier. Returns '' on empty input."""
    return principle.split(maxsplit=1)[0] if principle else ""


def _filter_by_source(
    hits: Sequence[tuple[EpisodicRecord, float]],
    source_filter: str | None,
) -> list[tuple[EpisodicRecord, float]]:
    """Drop hits whose principle prefix doesn't match `source_filter`.
    No-op when `source_filter` is None. Used to keep the JO-only lint
    contract intact when the active character has a multi-source corpus
    (e.g. airton_c carries JO + AIM + 14 CFR + PCG + PHAK; airton_c1 is
    JO-only and the filter is a no-op there)."""
    if source_filter is None:
        return list(hits)
    return [
        (rec, score)
        for rec, score in hits
        if _principle_source(rec.principle or "") == source_filter
    ]


def _inject_missing_anchored_sections(
    hits: Sequence[tuple[EpisodicRecord, float]],
    matched_sections: set[str],
    episodic_store: EpisodicStore,
    user_id: str | None,
    grammar: CitationGrammar | None,
    source_filter: str | None = None,
) -> list[tuple[EpisodicRecord, float]]:
    """Virtual-hit injection. When a verb anchor fires for a section
    that hybrid retrieval didn't surface in the slate, do a text-mode
    FTS5 probe for that section number and prepend one matching row.
    The follow-up rerank then promotes it to rank 0.

    Why this is safe:

    - The cite-grounding gate downstream still requires the model's
      pick to land on a section actually present in the slate; injection
      makes the right § *available* but doesn't fabricate a verdict.
    - Text-mode search for a literal section anchor (e.g. ``10-2-6``)
      lands deterministic FTS5 hits because the anchor lives in
      `principle` which is in the FTS5 sidecar.
    - When the section has no rows in the corpus at all, injection is
      silently a no-op (the FTS5 probe returns empty).

    `user_id` flows through so injected rows respect the same
    user-scoping as the primary retrieval call.
    """
    if not matched_sections:
        return list(hits)
    sections_in_slate: set[str] = set()
    for rec, _ in hits:
        anchor = _extract_anchor(rec.principle or "", grammar)
        if anchor:
            sections_in_slate.add(anchor)
    missing = matched_sections - sections_in_slate
    if not missing:
        return list(hits)

    extended: list[tuple[EpisodicRecord, float]] = list(hits)
    seen_external: set[str | None] = {rec.external_id for rec, _ in hits}
    # Sort for deterministic injection order — lets tests pin a stable
    # output and keeps multi-anchor utterances reproducible. k=5 because
    # text-mode FTS5 on a bare section number can return cross-section
    # hits ahead of the target — e.g. '10-2-6' as a query also matches
    # §5-2-5 ("HIJACK/UNLAWFUL INTERFERENCE", string match on the term)
    # and §13-2-6 (lexicographically similar number). The anchor-equality
    # filter below picks the right row from a slightly wider FTS slate.
    for section in sorted(missing):
        # Widen the FTS slate when source-filtering: the §-anchor lives
        # in the principle of every source's row, so a bare section number
        # query matches across publications. We need enough headroom to
        # find the JO-shaped row past any AIM/CFR rows that share the
        # numeric section id (collision keys: 10-1-1, 2-1-6, 5-4-5).
        fetch_k = 5 if source_filter is None else 12
        section_hits = episodic_store.search(section, k=fetch_k, mode="text", user_id=user_id)
        section_hits = _filter_by_source(section_hits, source_filter)
        for rec, _score in section_hits:
            anchor = _extract_anchor(rec.principle or "", grammar)
            if anchor != section:
                continue
            if rec.external_id in seen_external:
                continue
            extended.append((rec, _INJECTED_HIT_SCORE))
            seen_external.add(rec.external_id)
            break
    return extended


@dataclass(frozen=True)
class PhraseologyVerdict:
    """Structured output of one lint call. Mirrors the
    `phraseology_eval.yaml` row shape so eval scoring is a direct
    field-by-field comparison.

    Field names ('expected_section', 'expected_phraseology') are
    ATC-flavoured but the SHAPE is generic (cited anchor + canonical
    template). Future corpus-conformance lints (legal citation, RFC
    compliance, ...) reuse this dataclass — the names map naturally:
    section ⇒ document anchor, phraseology ⇒ canonical-template
    excerpt. A consumer that needs different domain-language can
    define a parallel dataclass with the same shape and adapt.
    """

    verdict: Verdict
    expected_section: str | None
    expected_phraseology: str | None
    mismatch: str | None
    citation_quote: str | None


@dataclass(frozen=True)
class CorpusLintConfig:
    """Domain config for the generic corpus-conformance lint pipeline
    (harness-d2k5).

    The pipeline structure — retrieve top-K candidates, prompt the
    model with a JSON-shaped verdict schema, cite-check the picked
    anchor against the question's top-K — is universal. The text
    that gets baked into the prompt (corpus name, audience role,
    slot-value norms, out-of-scope examples, candidate-block
    formatting) is per-corpus. This dataclass holds those bits so
    `lint_against_corpus()` can drive the loop without knowing the
    domain.

    Phraseology / JO 7110.65 ships as `PHRASEOLOGY_CONFIG`. New
    consumers (legal citation, RFC compliance, ...) instantiate
    their own.
    """

    # Full system message for the lint pass. Should declare:
    #   - The audience / domain ("You are a JO 7110.65 linter for ...").
    #   - The 4 verdicts (ok / wrong / incomplete / out_of_scope) with
    #     domain-specific examples.
    #   - The JSON schema the model must emit (verbatim string —
    #     downstream parsing keys on this shape).
    #   - Any normalisation notes (e.g. ATC phonetic digits).
    system_prompt: str
    # Format string for one candidate-section block. Placeholders:
    #   {idx}      — 1-based candidate index
    #   {anchor}   — section anchor extracted via grammar
    #   {title}    — record title
    #   {excerpt}  — body excerpt capped at _CANDIDATE_BODY_CAP
    #   {ellipsis} — "…" when body was truncated, "" otherwise
    candidate_block_format: str
    # Format string for the user message body. Placeholders:
    #   {utterance}         — the utterance under lint
    #   {scenario_hint_line}— "\nSCENARIO HINT: <hint>" or ""
    #   {candidates}        — "\n\n"-joined candidate blocks
    user_prompt_template: str


def lint_against_corpus(
    utterance: str,
    *,
    adapter: ModelAdapter,
    episodic_store: EpisodicStore,
    grammar: CitationGrammar | None,
    config: CorpusLintConfig,
    scenario_hint: str | None = None,
    user_id: str | None = None,
    k: int = _DEFAULT_K,
    temperature: float = 0.0,
    verb_anchors: VerbAnchorMap | None = None,
    source_filter: str | None = None,
) -> PhraseologyVerdict:
    """Generic corpus-conformance lint pipeline (harness-d2k5).

    Retrieves top-K corpus chunks for the utterance, prompts the
    model to compare it against the candidate canonical templates,
    and cite-grounds the verdict against the question's top-K
    anchors. Domain wording (corpus name, verdict examples,
    normalisation notes) comes from `config`.

    Returns a structured PhraseologyVerdict. Cite-or-silent on:
      - empty hybrid retrieval ⇒ out_of_scope, model never invoked.
      - unparseable model output ⇒ out_of_scope with diagnostic.
      - model picked an anchor not in the question's top-K
        candidate set ⇒ downgrade to out_of_scope.

    The model is invoked bare (no PersonaAdapter wrap) — the lint
    output is structured JSON, not voice-shaped prose. Persona
    rewriting would compress / paraphrase the canonical template
    and break the eval comparison.

    `temperature=0.0` is the eval default — verdict stability matters
    more than sampling variance. Raise it only when probing for
    consensus across rolls.

    `source_filter` keeps the JO-only lint contract intact when the
    active character has a multi-source corpus. Set to "JO_7110.65"
    for any character that ships the `phraseology` profile in its
    `tool_descriptions.yaml` (the data-driven equivalent of the old
    `_ATC_FAMILY_NAMES` membership check, harness-a2sa). None disables
    the filter (correct for JO-only corpora like airton_c1 where the
    filter would be a no-op anyway). When set, hybrid retrieval is
    requested at 4x the target K to give the post-filter enough
    headroom on corpora where JO is a minority share (airton_c is ~23%
    JO; controller-shaped queries skew higher but a wider initial slate
    keeps the post-filter from starving the prompt).
    """

    query = utterance if not scenario_hint else f"{utterance}\n[scenario: {scenario_hint}]"
    fetch_k = k if source_filter is None else k * 4
    raw_hits = episodic_store.search(query, k=fetch_k, mode="hybrid", user_id=user_id)
    raw_hits = _filter_by_source(raw_hits, source_filter)[:k]
    # Verb-anchor pipeline (harness-ptya). Two stages:
    #   1. Virtual-hit injection: when a verb fires for a § the embedding
    #      didn't surface, text-mode-fetch one row so it enters the slate.
    #      Cluster #3 (SQUAWK / hijack) needs this — §10-2-6 is dominated
    #      by §2-4-17 (Numbers Usage) on the dense embedding and never
    #      enters the K=8 slate without help.
    #   2. Rerank: matched-§ hits move to the front (stable partition).
    # When no verb fires, both stages are no-ops.
    matched_sections = _matched_anchor_sections(utterance, verb_anchors) if verb_anchors else set()
    if matched_sections:
        injected = _inject_missing_anchored_sections(
            raw_hits,
            matched_sections,
            episodic_store,
            user_id,
            grammar,
            source_filter=source_filter,
        )
        hits = _verb_anchor_rerank(injected, matched_sections, grammar)
    else:
        hits = list(raw_hits)
    if not hits:
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
        anchor = _extract_anchor(rec.principle or "", grammar)
        if anchor and anchor not in seen:
            seen.add(anchor)
            candidate_anchors.append(anchor)

    prompt_messages = _build_lint_messages(utterance, scenario_hint, hits, grammar, config)
    raw = adapter.complete(prompt_messages, temperature=temperature, max_tokens=512)
    parsed = _parse_verdict_json(raw)

    if parsed is None:
        return PhraseologyVerdict(
            verdict="out_of_scope",
            expected_section=None,
            expected_phraseology=None,
            mismatch="model output unparseable",
            citation_quote=None,
        )

    section = parsed.expected_section
    if section is not None and section not in candidate_anchors:
        return PhraseologyVerdict(
            verdict="out_of_scope",
            expected_section=None,
            expected_phraseology=None,
            mismatch=f"cite §{section} not grounded in question retrieval",
            citation_quote=None,
        )

    if parsed.verdict == "out_of_scope":
        return PhraseologyVerdict(
            verdict="out_of_scope",
            expected_section=None,
            expected_phraseology=None,
            mismatch=parsed.mismatch,
            citation_quote=None,
        )

    return parsed


# ATC / JO 7110.65 phraseology config — the original lint domain.
# Lifted verbatim from `_build_lint_messages`'s pre-d2k5 prompt
# strings; verdict definitions, slot-value note, schema, and
# candidate / user-prompt shapes are preserved byte-for-byte so
# fixture YAMLs and eval baselines stay green.
PHRASEOLOGY_CONFIG = CorpusLintConfig(
    system_prompt=(
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
    ),
    candidate_block_format="[{idx}] §{anchor} {title}\n{excerpt}{ellipsis}",
    user_prompt_template=(
        "UTTERANCE: {utterance}{scenario_hint_line}\n\n"
        "CANDIDATE SECTIONS FROM JO 7110.65:\n\n"
        "{candidates}\n\n"
        "Return the JSON verdict now."
    ),
)


def lint_utterance(
    utterance: str,
    *,
    adapter: ModelAdapter,
    episodic_store: EpisodicStore,
    grammar: CitationGrammar | None,
    scenario_hint: str | None = None,
    user_id: str | None = None,
    k: int = _DEFAULT_K,
    temperature: float = 0.0,
    verb_anchors: VerbAnchorMap | None = None,
    source_filter: str | None = None,
) -> PhraseologyVerdict:
    """Lint one ATC utterance against JO 7110.65 — thin wrapper around
    `lint_against_corpus()` carrying the FAA-shaped config.

    Kept as a stable name so the eval / CLI / chat-tool surfaces don't
    re-import. New domain lints should call `lint_against_corpus()`
    directly with their own `CorpusLintConfig`.

    `verb_anchors` (harness-ptya) and `source_filter` (multi-source
    corpus support) pass straight through to the generic pipeline.
    """
    return lint_against_corpus(
        utterance,
        adapter=adapter,
        episodic_store=episodic_store,
        grammar=grammar,
        config=PHRASEOLOGY_CONFIG,
        scenario_hint=scenario_hint,
        user_id=user_id,
        k=k,
        temperature=temperature,
        verb_anchors=verb_anchors,
        source_filter=source_filter,
    )


def _build_lint_messages(
    utterance: str,
    scenario_hint: str | None,
    hits: Sequence[tuple[EpisodicRecord, float]],
    grammar: CitationGrammar | None,
    config: CorpusLintConfig,
) -> list[ChatMessage]:
    """Render the lint prompt from `config`. System message comes
    verbatim from config.system_prompt; user message templates the
    utterance + candidate-block list into config.user_prompt_template;
    each candidate block uses config.candidate_block_format."""
    from harness.model.adapter import ChatMessage

    candidates: list[str] = []
    for idx, (rec, _score) in enumerate(hits[:_PROMPT_CANDIDATES], start=1):
        anchor = _extract_anchor(getattr(rec, "principle", "") or "", grammar)
        title = getattr(rec, "title", "")
        body = getattr(rec, "body", "") or ""
        excerpt = body[:_CANDIDATE_BODY_CAP]
        ellipsis = "…" if len(body) > _CANDIDATE_BODY_CAP else ""
        candidates.append(
            config.candidate_block_format.format(
                idx=idx,
                anchor=anchor,
                title=title,
                excerpt=excerpt,
                ellipsis=ellipsis,
            ).rstrip()
        )

    scenario_hint_line = f"\nSCENARIO HINT: {scenario_hint}" if scenario_hint else ""
    user = config.user_prompt_template.format(
        utterance=utterance,
        scenario_hint_line=scenario_hint_line,
        candidates="\n\n".join(candidates),
    )

    return [
        ChatMessage(role="system", content=config.system_prompt),
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
    grammar: CitationGrammar | None = None
    user_id: str | None = None
    k: int = _DEFAULT_K
    verb_anchors: VerbAnchorMap | None = None
    source_filter: str | None = None

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
            grammar=self.grammar,
            scenario_hint=scenario_hint,
            user_id=self.user_id,
            k=self.k,
            verb_anchors=self.verb_anchors,
            source_filter=self.source_filter,
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
