"""Retrieval Context Package (harness-s3f7 / Phase 3).

A first-class envelope around shape-aware retrieval. Replaces the
opportunistic per-turn pattern (call each store independently, glue
the results into the system prompt) with a single object the agent
receives, the orchestrator builds, and any caller can audit.

Four fields, named after the user's data-retrieval-primitives notes:

  - **Intent** — what task is being served (a string the orchestrator
    sees in the contract definition). Goes into logs + audit trails;
    not consumed by the model directly.
  - **Access** — who can see this. user_id scoping is mandatory on
    every per-user retrieval call; the package carries the policy
    explicitly so the orchestrator can pass it down to each store.
  - **Proof** — per-hit provenance: which store returned it, which
    record_id, which method, which score. Auditable, attributable,
    rebuildable.
  - **Budget** — soft token cap. The orchestrator packs hits greedily
    in priority order (required slots first), stops when adding the
    next hit would overflow. Hits that didn't fit are tracked
    separately so the agent can ask for more if needed.

The package itself is pure data. Construction lives in
`harness.retrieval.contract.assemble_package`; the two modules are
split because the value types want to be import-cheap (the orchestrator
imports several shaped stores).

Token estimation here is char-count / 4 — a known-imprecise but
zero-dependency proxy. For the budget-enforcement contract this is
good enough: we're not trying to hit an exact token count, we're
trying to avoid runaway context bloat. A real tokenizer can be
plugged in later via the `token_estimator` parameter on `TokenBudget`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    """Cheap zero-dependency token estimate. Char-count / 4 is the
    rule-of-thumb proxy that most agent harnesses use when an exact
    tokenizer isn't worth the load. Off by maybe 20% on either side
    for English; closer for code, looser for non-Latin scripts. Use
    `TokenBudget.token_estimator` to swap in a real tokenizer when
    that precision actually matters."""
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class AccessPolicy:
    """Who can see this package. The orchestrator passes `user_id`
    down to every store's `search()` call — same contract the chat
    pipeline already enforces. `role` is an opaque label (e.g.
    "returns-handler") that contracts can dispatch on; not currently
    used for any access check, but reserved so contracts can later
    grow role-based slot filtering without breaking the package shape.
    """

    user_id: str | None
    role: str | None = None


@dataclass(frozen=True)
class TokenBudget:
    """Soft cap on the package's total token cost. `max_tokens` is the
    target; the orchestrator packs hits until the next one would
    overflow, then stops. `token_estimator` defaults to the cheap
    char-count proxy; callers with a real tokenizer pass it in.

    Soft, not hard: a single oversized hit is still included if it's
    the only thing in a required slot — the alternative would be to
    silently drop required content, which violates the contract.
    Overflows are reported in `RetrievalContextPackage.overflow_hits`
    so the caller knows the budget was exceeded and can decide what
    to do (truncate, drop, ask for more).
    """

    max_tokens: int
    token_estimator: Callable[[str], int] = estimate_tokens

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")


@dataclass(frozen=True)
class Provenance:
    """Per-hit attribution. Every hit in a `RetrievalContextPackage`
    carries one of these so downstream callers can:
      - trace any model-generated claim back to its source row,
      - re-run the same retrieval deterministically,
      - audit which store fired for which slot."""

    store: str  # "episodic" | "tree" | "tabular"
    record_id: str  # external_id / path / row id
    method: str  # "hybrid" / "dense" / "text" / "sql"
    score: float


@dataclass(frozen=True)
class PackagedHit:
    """One retrieved item, packaged for the agent. `body` is the text
    the model will see; `provenance` is the audit trail; `est_tokens`
    is what was charged against the budget. The `slot_name` is which
    contract slot this hit answers — lets the model (or a renderer)
    group related hits."""

    slot_name: str
    body: str
    provenance: Provenance
    est_tokens: int


@dataclass(frozen=True)
class RetrievalContextPackage:
    """The first-class envelope. Built by `assemble_package`, consumed
    by the agent (via a system-prompt renderer in a follow-up) or by
    any caller that needs to audit what retrieval surfaced.

    `missing_required_slots` is the load-bearing failure signal —
    when a required slot came up empty, the agent gets told *what's
    missing* rather than silently working with incomplete context.
    Per the data-retrieval-primitives note: 'what happens when
    something is missing.'

    `overflow_hits` carry hits that were retrieved but didn't fit
    in the budget. Kept attached (not discarded) because a caller
    might still want to fall back to them if the budget allowed.
    """

    intent: str
    access: AccessPolicy
    budget: TokenBudget
    hits: tuple[PackagedHit, ...]
    missing_required_slots: tuple[str, ...]
    overflow_hits: tuple[PackagedHit, ...] = field(default_factory=tuple)

    @property
    def tokens_used(self) -> int:
        return sum(h.est_tokens for h in self.hits)

    @property
    def tokens_overflow(self) -> int:
        return sum(h.est_tokens for h in self.overflow_hits)

    @property
    def is_complete(self) -> bool:
        """True when every required slot got at least its minimum
        cardinality. Convenience wrapper around `missing_required_slots`
        for callers that only need the yes/no."""
        return not self.missing_required_slots

    def hits_for_slot(self, slot_name: str) -> tuple[PackagedHit, ...]:
        """Hits attributable to one contract slot. Renderers use this
        to group output ('Customer history:\\n  ...\\nRefund policy:\\n
        ...') instead of one flat ranked list."""
        return tuple(h for h in self.hits if h.slot_name == slot_name)

    def provenance_for_slot(self, slot_name: str) -> tuple[Provenance, ...]:
        """Audit-friendly view: every provenance record attached to
        this slot's hits, in rank order."""
        return tuple(h.provenance for h in self.hits if h.slot_name == slot_name)
