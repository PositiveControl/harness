"""Harvest bd's persistent memories (`bd remember` / `retro record`)
into episodic-tier='procedural' records so the main retrieval path
can surface them on relevant user turns (harness-9yd).

Parallel to `skills/harvester.py`, which ingests thought-graph beads
(`thought:decision` / `thought:observation`). Shares the same
destination (episodic, procedural tier, shared scope) and the same
idempotency model (external_id), but the source substrate is
different — bd's key/body memory store rather than the bead graph.

Design invariants:

- **Idempotent.** `external_id = f"bd-mem:{key}"` — re-running is a
  no-op on existing rows, EpisodicStore.ingest short-circuits on
  duplicate external_id without touching the embedder.
- **Namespaced external_id.** The `bd-mem:` prefix structurally
  prevents collision with bead ids (`harness-xyz`) even though they
  don't collide today.
- **Shared, not per-user.** bd memories describe the character and
  the relationship — they live at `user_id IS NULL` so every speaker
  sees them, matching the skill harvester's reasoning.
- **principle='bd_memory'.** Contextual-chunking header becomes
  `[tier: procedural; principle: bd_memory; date: ...]`, so queries
  like "what do you remember about me" resolve via the principle tag.
- **Body verbatim.** No type-prefix — bd memories are assertions, not
  agent decisions/observations, so the `DECISION:` / `OBSERVATION:`
  pattern from the skill harvester doesn't apply.
- **Delete reconciliation is out of scope here.** `bd forget` will
  leave the mirrored row live; a follow-up bead handles the tombstone
  plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.store.bd_adapter import BeadsAdapter
from harness.store.episodic import EpisodicStore

EXTERNAL_ID_PREFIX = "bd-mem:"
MEMORY_PRINCIPLE = "bd_memory"
MEMORY_SOURCE = "bd_memory"
MEMORY_TAGS: tuple[str, ...] = ("bd", "memory")


@dataclass(frozen=True)
class MemoryHarvestReport:
    """Outcome of a single bd-memory harvester run.

    `scanned` is the total key count returned by `bd memories --json`.
    `newly_ingested` + `already_present` partition the same set;
    their sum equals `scanned`."""

    scanned: int
    newly_ingested: int
    already_present: int
    ingested_keys: tuple[str, ...]


def _external_id_for(key: str) -> str:
    return f"{EXTERNAL_ID_PREFIX}{key}"


def harvest_bd_memories(
    *,
    ab_adapter: BeadsAdapter,
    episodic: EpisodicStore,
) -> MemoryHarvestReport:
    """Pull every bd memory via `memories_json()` and ingest each as a
    procedural-tier episodic record.

    Re-running is cheap: idempotent on `external_id = bd-mem:<key>`,
    so only brand-new memories go through the embedder. Empty stores
    report `scanned=0` with no inserts."""
    raw = ab_adapter.memories_json()
    newly: list[str] = []
    existing: list[str] = []
    for key, body in raw.items():
        external_id = _external_id_for(key)
        if episodic.has(external_id):
            existing.append(key)
            continue
        episodic.ingest(
            external_id=external_id,
            title=key,
            body=body,
            principle=MEMORY_PRINCIPLE,
            tags=list(MEMORY_TAGS),
            tier="procedural",
            source=MEMORY_SOURCE,
            # Shared: bd memories describe the character and the
            # relationship — every speaker sees them.
            user_id=None,
        )
        newly.append(key)

    return MemoryHarvestReport(
        scanned=len(raw),
        newly_ingested=len(newly),
        already_present=len(existing),
        ingested_keys=tuple(newly),
    )
