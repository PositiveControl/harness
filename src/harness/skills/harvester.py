"""Harvest ab's bd thought-graph into episodic-tier='procedural'
records so the main retrieval path can surface them on relevant user
turns (sota punch #7, harness-vu3).

Design invariants:

- **Idempotent.** `external_id = bead id`, so re-running the harvester
  is safe and cheap. EpisodicStore.ingest short-circuits on duplicate
  external_id without touching the embedder.
- **Closed-only by default.** An in-flight thought is still a
  hypothesis. Only closed `thought:decision` / `thought:observation`
  beads earned the "this is how we did it" stamp worth re-injecting.
- **Shared, not per-user.** Ab's thoughts aren't Mark's — they live
  at `user_id IS NULL` so every speaker sees them. If future personas
  need per-user skill silos, the call site can override `user_id`.
- **Body carries the thought type in prose.** We prefix the body with
  `DECISION:` / `OBSERVATION:` so even dense-cosine (pre contextual-
  chunking) matches on the type token. The contextual-chunking header
  (harness-2am) adds a `principle: decision` tag separately, which
  helps queries like "what have we decided about X" resolve.
- **No retrieval changes.** Procedural rows participate in the
  existing `EpisodicStore.search()` via tier-agnostic filtering;
  `cli.py` already injects episodic hits into the system prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from harness.store._bd_types import BeadsIssue
from harness.store.bd_adapter import BeadsAdapter
from harness.store.episodic import EpisodicStore

# Default filter: closed intentional choices + post-mortem learnings.
# Skipped: thought:hypothesis / thought:question — speculative until
# they resolve into a decision or observation.
DEFAULT_HARVEST_LABELS: tuple[str, ...] = (
    "thought:decision",
    "thought:observation",
)


@dataclass(frozen=True)
class HarvestReport:
    """Outcome of a single harvester run.

    `scanned` counts every bead that matched the label filter,
    regardless of whether we chose to ingest it (closed-only gating
    trims further). `newly_ingested` and `already_present` partition
    the rows we considered; they sum to `scanned` minus any rows
    dropped for status mismatch."""

    scanned: int
    newly_ingested: int
    already_present: int
    status_filter: str
    labels_filtered: tuple[str, ...]
    ingested_ids: tuple[str, ...]


def _principle_from_labels(labels: Sequence[str]) -> str | None:
    """Pick the first `thought:<type>` label and return `<type>`.
    Used for the episodic record's `principle` field so queries about
    decisions / observations resolve via the contextual-chunking tag."""
    for label in labels:
        if label.startswith("thought:"):
            return label.split(":", 1)[1]
    return None


def _body_from_issue(issue: BeadsIssue) -> str:
    """Build the embedded body. Prefix with the thought type so even
    non-contextually-chunked callers get a lexical signal; concatenate
    the bead's description (when present on `raw`) for semantic
    content."""
    principle = _principle_from_labels(issue.labels) or "thought"
    prefix = principle.upper()
    description = issue.raw.get("description") if isinstance(issue.raw, dict) else None
    description_text = str(description).strip() if description else ""
    if description_text:
        return f"{prefix}: {issue.title}\n\n{description_text}"
    return f"{prefix}: {issue.title}"


def _matches_filter(issue: BeadsIssue, labels: Sequence[str]) -> bool:
    """Any-of match against the allowed-label tuple."""
    wanted = set(labels)
    return any(label in wanted for label in issue.labels)


def harvest_bd_skills(
    *,
    ab_adapter: BeadsAdapter,
    episodic: EpisodicStore,
    labels: Sequence[str] = DEFAULT_HARVEST_LABELS,
    status: str = "closed",
) -> HarvestReport:
    """Pull matching beads from ab's bd store, ingest each as a
    procedural-tier episodic record.

    Default filter: `status='closed'` AND label ∈
    `DEFAULT_HARVEST_LABELS`. Override `status` to `'all'` to include
    open thoughts (useful for debug / dry-run; not recommended for
    production — open thoughts are speculative).

    Ingestion is idempotent on `external_id = bead.id`, so the second
    call is a no-op for already-harvested beads. New beads go through
    the embedder the first time only.
    """
    # bd lists don't filter by label client-side; we list with the
    # status filter, then narrow to thought-labels in Python.
    # assignee=None lets the call see both ab-owned and user-owned
    # beads, matching the bd adapter's default exclude behavior.
    issues = ab_adapter.list_issues() if status == "all" else ab_adapter.list_issues(status=status)

    matching = [issue for issue in issues if _matches_filter(issue, labels)]
    newly: list[str] = []
    existing: list[str] = []
    for issue in matching:
        if episodic.has(issue.id):
            existing.append(issue.id)
            continue
        principle = _principle_from_labels(issue.labels)
        episodic.ingest(
            external_id=issue.id,
            title=issue.title,
            body=_body_from_issue(issue),
            principle=principle,
            tags=list(issue.labels),
            tier="procedural",
            source="bd",
            # Shared (character-level). Ab's thoughts aren't tied to
            # any single speaker; they describe how the agent itself
            # has acted.
            user_id=None,
        )
        newly.append(issue.id)

    return HarvestReport(
        scanned=len(matching),
        newly_ingested=len(newly),
        already_present=len(existing),
        status_filter=status,
        labels_filtered=tuple(labels),
        ingested_ids=tuple(newly),
    )
