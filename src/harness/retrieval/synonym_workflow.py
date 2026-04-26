"""Phase 1 of the synonym management plan (harness-rup2). Helpers
shared by `scripts/suggest_synonyms.py` and `scripts/test_synonyms.py`.

The conviction (see `character/airton_c1/synonyms_policy.md`):
synonyms are configuration, not code. YAML stays the source of truth.
This module gives the human reviewer the two missing tools:

  - **Suggester support**: read the retrieval-eval baseline, find
    cases that miss or land at weak ranks, build prompts that ask
    a small model for candidate lay-form phrases per case.
  - **Tester support**: take a proposed YAML, run the retrieval eval
    with the proposal merged into the live expander, diff per-case
    ranks against the saved baseline, and classify each proposed
    section as LIFT / MIXED / DEAD / HARMFUL.

Neither piece auto-commits. The verdict is advisory; humans paste the
proposed YAML into the right file (default `query_synonyms.yaml` per
policy) once the verdict is LIFT.

This module never imports a model adapter directly — adapters are
threaded in via the `_LLMAdapterProto` Protocol from
`harness.retrieval.query_expander`, so test stubs only have to
implement `complete()`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from harness.retrieval.query_expander import (
    QueryExpander,
    _parse_rewrites,
    _parse_sections_file,
)

if TYPE_CHECKING:
    from harness.evals.atc import AtcFixtureRow
    from harness.evals.atc_retrieval import RetrievalCase
    from harness.retrieval.query_expander import _LLMAdapterProto


# ---------- proposal parsing ----------


def load_proposed_yaml(path: Path) -> dict[str, tuple[str, ...]]:
    """Read a proposed-synonyms YAML (same shape as
    `query_synonyms.yaml`) into a section→variants dict. Reuses the
    query_expander's parser so accepted shapes stay in lockstep."""
    return _parse_sections_file(path)


def merge_proposed(
    base_sections: Mapping[str, tuple[str, ...]],
    proposed: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Union of base + proposed; proposed entries APPEND to base
    entries on shared sections (deduped, preserving base order). The
    tester uses this to build the under-test expander; the suggester
    uses it to scope what's already covered when proposing."""
    merged: dict[str, tuple[str, ...]] = {}
    for section in list(base_sections) + [s for s in proposed if s not in base_sections]:
        seen: dict[str, None] = {}
        for source in (base_sections.get(section, ()), proposed.get(section, ())):
            for variant in source:
                seen.setdefault(variant, None)
        merged[section] = tuple(seen)
    return merged


def expander_with(
    *,
    shared_path: Path | None,
    query_only_path: Path | None,
    proposed: Mapping[str, tuple[str, ...]] | None = None,
) -> QueryExpander:
    """Build a `QueryExpander` with optional proposed entries layered
    on top of the on-disk shared + query-only files. Mirrors
    `load_query_expander`'s 2-file merge but adds a third tier (the
    proposal) on top so the tester can run an A/B against the live
    expander without mutating either file on disk."""
    shared = _parse_sections_file(shared_path) if shared_path is not None else {}
    query_only = _parse_sections_file(query_only_path) if query_only_path is not None else {}
    layered = merge_proposed(shared, query_only)
    if proposed:
        layered = merge_proposed(layered, proposed)
    return QueryExpander(layered)


# ---------- classification ----------


class Verdict(StrEnum):
    """Tester output per proposed section. `LIFT` is the only verdict
    that justifies committing the entry; everything else asks the
    reviewer to refine, prune, or abandon."""

    LIFT = "LIFT"  # target case rank improved; no other case regressed.
    MIXED = "MIXED"  # target case improved, BUT some other case regressed.
    DEAD = "DEAD"  # no measurable case-rank change vs baseline.
    HARMFUL = "HARMFUL"  # target case unchanged or worse, OR aggregate worse.


@dataclass(frozen=True)
class CaseRankChange:
    """Per-case before/after rank tuple. None means hard miss."""

    case_id: str
    before: int | None
    after: int | None

    @property
    def improved(self) -> bool:
        if self.before is None and self.after is not None:
            return True
        if self.before is None or self.after is None:
            return False
        return self.after < self.before

    @property
    def regressed(self) -> bool:
        if self.before is not None and self.after is None:
            return True
        if self.before is None or self.after is None:
            return False
        return self.after > self.before


@dataclass(frozen=True)
class SectionVerdict:
    """One verdict per proposed section. `target_cases` is the subset
    of fixture cases whose `expected_citations` include the proposed
    section — those are the cases the entry is *supposed* to help.
    `collateral_changes` lists other cases whose rank moved (either
    direction), surfaced so reviewers can spot accidental neighbour
    bleed."""

    section: str
    proposed_variants: tuple[str, ...]
    target_changes: tuple[CaseRankChange, ...]
    collateral_changes: tuple[CaseRankChange, ...]
    verdict: Verdict
    rationale: str


def _cases_for_section(fixture: Iterable[AtcFixtureRow], section: str) -> tuple[str, ...]:
    """IDs of fixture rows whose `expected_citations` mention the
    given section anchor. Section is the bare anchor form (e.g.
    `5-10-11`); the fixture's expected_citations carries the same
    form (canonical) so direct string equality is the right test."""
    out: list[str] = []
    for row in fixture:
        for entry in row.expected_citations:
            if section in entry:
                out.append(row.id)
                break
    return tuple(out)


def classify_section(
    section: str,
    proposed_variants: tuple[str, ...],
    target_case_ids: tuple[str, ...],
    before_ranks: Mapping[str, int | None],
    after_ranks: Mapping[str, int | None],
) -> SectionVerdict:
    """Compare a proposed section's effect on its target cases vs all
    other cases. Returns a single SectionVerdict. The rules:

      - DEAD: every target case had identical before/after rank AND
        no other case moved.
      - HARMFUL: aggregate regression — any case (target OR other)
        moved to a worse rank without any compensating target lift.
      - MIXED: at least one target case improved AND at least one
        other case regressed.
      - LIFT: at least one target case improved; no case regressed
        (or only target cases regressed in a controlled way — but
        we treat ANY regression as MIXED-or-worse, never LIFT).
    """
    target_changes: list[CaseRankChange] = []
    collateral_changes: list[CaseRankChange] = []
    for case_id in sorted(set(before_ranks) | set(after_ranks)):
        before = before_ranks.get(case_id)
        after = after_ranks.get(case_id)
        if before == after:
            continue
        change = CaseRankChange(case_id=case_id, before=before, after=after)
        if case_id in target_case_ids:
            target_changes.append(change)
        else:
            collateral_changes.append(change)

    any_improvement = any(c.improved for c in target_changes)
    any_target_regression = any(c.regressed for c in target_changes)
    any_collateral_regression = any(c.regressed for c in collateral_changes)

    if not target_changes and not collateral_changes:
        verdict = Verdict.DEAD
        rationale = (
            f"No case moved. Section §{section} variants don't "
            "trigger on any fixture query, or trigger but don't shift "
            "ranking. Refine the entry or prune."
        )
    elif any_target_regression and not any_improvement:
        verdict = Verdict.HARMFUL
        rationale = (
            f"Target case(s) for §{section} regressed. Entry hurts "
            "the cases it's supposed to help. Refine phrasing or "
            "abandon."
        )
    elif any_collateral_regression and not any_improvement:
        verdict = Verdict.HARMFUL
        rationale = (
            "No target lift, but other case(s) regressed. Entry "
            "leaks into neighbour sections. Tighten phrasing."
        )
    elif any_improvement and any_collateral_regression:
        verdict = Verdict.MIXED
        rationale = (
            f"Target case(s) for §{section} improved BUT one or more "
            "other cases regressed. Decide if the trade is worth it; "
            "usually means the entry is too generic."
        )
    elif any_improvement and not any_collateral_regression:
        verdict = Verdict.LIFT
        rationale = (
            f"Target case(s) for §{section} improved with no "
            "collateral regression. Safe to commit per policy."
        )
    else:
        # Catch-all: target unchanged + collateral mixed.
        verdict = Verdict.DEAD
        rationale = (
            "No target lift; only collateral movement. Entry isn't doing its job — refine or prune."
        )

    return SectionVerdict(
        section=section,
        proposed_variants=proposed_variants,
        target_changes=tuple(target_changes),
        collateral_changes=tuple(collateral_changes),
        verdict=verdict,
        rationale=rationale,
    )


# ---------- baseline helpers ----------


def case_ranks(cases: Iterable[RetrievalCase]) -> dict[str, int | None]:
    """Pull `{case_id: rank_of_first_expected}` out of a
    RetrievalResult.cases sequence. Handles None ranks (hard misses)
    by passing them through."""
    return {c.id: c.rank_of_first_expected for c in cases}


def baseline_ranks(baseline: Mapping[str, object]) -> dict[str, int | None]:
    """Same shape as `case_ranks` but reads from a saved baseline
    JSON dict (the on-disk form the comparator gate uses)."""
    out: dict[str, int | None] = {}
    raw_cases = baseline.get("cases", []) or []
    if not isinstance(raw_cases, list):
        return out
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            continue
        case_id = raw.get("id")
        if not isinstance(case_id, str):
            continue
        rank = raw.get("rank_of_first_expected")
        out[case_id] = int(rank) if isinstance(rank, int) else None
    return out


# ---------- suggester support ----------


_DEFAULT_SUGGESTION_PROMPT = """\
A retrieval system over {context_hint} failed to find the section
the user was asking about. The expected section's stored body is
shown below. Propose 3-5 short lay-form phrases (2-6 words each)
that bridge the gap between the user's wording and the section's
wording.

Rules:
- Output ONLY phrases, one per line. No numbering, no bullets,
  no quotes, no commentary, no examples.
- Use the user's lay vocabulary on one side and the document's
  jargon on the other — phrases that contain BOTH are best.
- Each phrase must be specific to this section. Generic ATC
  jargon ("aircraft separation", "approach phraseology") leaks
  onto neighbour sections and hurts retrieval precision.
- Do not paraphrase the user's exact words verbatim — those are
  already in the query.

User query: {query}

Expected section: §{section}
Expected section body:
{body}

Lay-form phrases (one per line, no numbering):"""


def build_suggestion_prompt(
    *,
    query: str,
    section: str,
    body: str,
    context_hint: str = "FAA Order JO 7110.65 (Air Traffic Control)",
    body_cap: int = 800,
) -> str:
    """Format the suggester's prompt. Body is truncated to `body_cap`
    chars to stay under the 3B router's 4-8k context budget — the
    first ~800 chars of a JO chunk carry the substance, the rest is
    NOTE/REFERENCE/PHRASEOLOGY scaffolding."""
    truncated = body[:body_cap].rstrip()
    if len(body) > body_cap:
        truncated += "..."
    return _DEFAULT_SUGGESTION_PROMPT.format(
        context_hint=context_hint,
        query=query,
        section=section,
        body=truncated,
    )


def suggest_for_case(
    *,
    adapter: _LLMAdapterProto,
    query: str,
    section: str,
    body: str,
    max_rewrites: int = 5,
    max_tokens: int = 192,
    temperature: float = 0.0,
    context_hint: str = "FAA Order JO 7110.65 (Air Traffic Control)",
) -> tuple[str, ...]:
    """Run the suggester end-to-end for one case. Returns the cleaned
    tuple of variants. Empty tuple on adapter failure — graceful
    degradation, the caller decides whether to skip or retry."""
    from harness.model.adapter import ChatMessage

    prompt = build_suggestion_prompt(
        query=query, section=section, body=body, context_hint=context_hint
    )
    messages = [ChatMessage(role="user", content=prompt)]
    try:
        raw = adapter.complete(messages, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        return ()
    return _parse_rewrites(raw, max_rewrites=max_rewrites)


def emit_suggestion_yaml_block(
    *,
    section: str,
    variants: tuple[str, ...],
    case_id: str,
    current_rank: int | None,
    indent: int = 2,
) -> str:
    """Format one section's suggestion as a YAML block ready to paste
    into `query_synonyms.yaml`. Includes the audit-trail comment per
    the policy."""
    pad = " " * indent
    rank_repr = "hard-miss" if current_rank is None else f"rank {current_rank}"
    lines: list[str] = [
        f'{pad}"{section}":',
        f"{pad}  # Suggested for case `{case_id}` ({rank_repr} in current",
        f"{pad}  # baseline). REVIEW + EDIT before pasting into",
        f"{pad}  # query_synonyms.yaml. Update the audit-trail comment",
        f"{pad}  # with the measured rank delta after running",
        f"{pad}  # scripts/test_synonyms.py.",
    ]
    if not variants:
        lines.append(f"{pad}  # (suggester returned no variants — refine the prompt)")
        lines.append(f"{pad}  []")
    else:
        for v in variants:
            lines.append(f"{pad}  - {v}")
    lines.append("")
    return "\n".join(lines)


# ---------- selection helpers ----------


@dataclass(frozen=True)
class CandidateMiss:
    """One fixture case the suggester wants to hand to the model.
    `expected_section` is the first anchor in `expected_citations` —
    the suggester emits one section block per candidate; if a case
    has alternates, the reviewer can manually attach its phrases to
    other sections too."""

    case_id: str
    query: str
    expected_section: str
    current_rank: int | None
    chunk_body: str = field(default="")


def select_candidates(
    *,
    fixture: Iterable[AtcFixtureRow],
    baseline: Mapping[str, object],
    rank_threshold: int = 3,
) -> tuple[CandidateMiss, ...]:
    """Walk the fixture + baseline; emit a CandidateMiss per case
    whose rank is None (hard miss) OR > `rank_threshold`. Each
    candidate carries the case's first expected anchor, leaving
    body lookup to the caller (which has the EpisodicStore handle).
    Cases with no expected_citations are skipped."""
    ranks = baseline_ranks(baseline)
    out: list[CandidateMiss] = []
    for row in fixture:
        first_anchor = next(
            (alt for entry in row.expected_citations for alt in entry if alt),
            None,
        )
        if not first_anchor:
            continue
        rank = ranks.get(row.id)
        if rank is None or rank > rank_threshold:
            out.append(
                CandidateMiss(
                    case_id=row.id,
                    query=row.question,
                    expected_section=first_anchor,
                    current_rank=rank,
                )
            )
    return tuple(out)


# ---------- formatted reporting ----------


def format_verdict_report(verdicts: Iterable[SectionVerdict]) -> str:
    """Human-readable single-string report for stdout. One block per
    section with verdict label, target case deltas, collateral
    deltas, and the rationale text."""
    blocks: list[str] = []
    for v in verdicts:
        lines = [f"=== §{v.section}: {v.verdict.value} ==="]
        if v.target_changes:
            lines.append("  target cases:")
            for change in v.target_changes:
                arrow = f"rank {_render_rank(change.before)} → {_render_rank(change.after)}"
                lines.append(f"    {change.case_id:<48} {arrow}")
        else:
            lines.append("  target cases: (no movement)")
        if v.collateral_changes:
            lines.append("  collateral cases:")
            for change in v.collateral_changes:
                arrow = f"rank {_render_rank(change.before)} → {_render_rank(change.after)}"
                lines.append(f"    {change.case_id:<48} {arrow}")
        lines.append(f"  rationale: {v.rationale}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_rank(rank: int | None) -> str:
    return "—" if rank is None else str(rank)


# ---------- yaml output for the suggester driver ----------


def emit_suggestion_doc_header() -> str:
    """Top-of-document banner for `scripts/suggest_synonyms.py`'s
    stdout. Reminds the reviewer this is a draft, not an automatic
    commit."""
    return (
        "# Suggested synonym entries — REVIEW BEFORE PASTING.\n"
        "# Generated by scripts/suggest_synonyms.py.\n"
        "# Per the synonym policy (character/airton_c1/synonyms_policy.md):\n"
        "#   1. Default destination is corpus/query_synonyms.yaml.\n"
        "#   2. Run scripts/test_synonyms.py against this draft before\n"
        "#      pasting — only LIFT verdicts justify a commit.\n"
        "#   3. Update each block's audit-trail comment with the measured\n"
        "#      rank-with vs rank-without delta after testing.\n"
        "#\n"
        "version: 1\n"
        "sections:\n"
    )


def emit_suggestion_doc(
    blocks: Iterable[str],
) -> str:
    """Compose the full suggester output: header + section blocks."""
    return emit_suggestion_doc_header() + "".join(blocks)


__all__ = [
    "CandidateMiss",
    "CaseRankChange",
    "SectionVerdict",
    "Verdict",
    "baseline_ranks",
    "build_suggestion_prompt",
    "case_ranks",
    "cases_for_section",
    "classify_section",
    "emit_suggestion_doc",
    "emit_suggestion_doc_header",
    "emit_suggestion_yaml_block",
    "expander_with",
    "format_verdict_report",
    "load_proposed_yaml",
    "merge_proposed",
    "select_candidates",
    "suggest_for_case",
]


# Public re-export so callers don't have to dip into private name
def cases_for_section(fixture: Iterable[AtcFixtureRow], section: str) -> tuple[str, ...]:
    """IDs of fixture rows whose `expected_citations` mention the
    given section anchor. Section is the bare anchor form (e.g.
    `5-10-11`)."""
    return _cases_for_section(fixture, section)
