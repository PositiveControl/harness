"""Contract Bundle + retrieval orchestrator (harness-s3f7 / Phase 3).

A `ContractBundle` is a YAML-declared list of *slots* — what an agent
in a given role needs to receive to do its job. Per the data-
retrieval-primitives notes:

    Agent job (what work is being done)
        ↓
    Data Contract (shape of data, what's required)
        ↓
    Database (only after the job has been defined and the requisite
    data shape is confirmed)

This module owns the middle layer. The contract says what to fetch
and from where; the orchestrator walks the contract, calls each
shaped store, packages the results into a `RetrievalContextPackage`.

Why contracts instead of opportunistic per-turn retrieval: the
session-start retrieval pipeline today calls each store independently
and glues the results into the system prompt. That works for a
generalist persona. For a role-specialized agent (returns handler,
support triager, compliance reviewer), the question 'did I get
everything I need to do my job?' has a yes/no answer — and the
contract is what makes it explicit and auditable.

Slot shape (per slot in the YAML):

  - `name`            — slot id, surfaces in package output + missing
                        list. Required.
  - `store`           — 'episodic' | 'tree' | 'tabular'. Required.
  - `query_template`  — Python-style {var} substitution against
                        `assemble_package`'s `variables` dict. For
                        episodic + tree slots.
  - `sql_template`    — same substitution for tabular slots. Replaces
                        `query_template` when present.
  - `table_name`      — required when store='tabular'.
  - `required`        — bool, default False.
  - `min_cardinality` — int, default 1. Required slot misses when its
                        hit count is below this.
  - `max_hits`        — int, default 5. Pre-budget cap.

The orchestrator is intentionally pure: takes a contract, variables,
access, and a `StoreBundle` (so a test can pass mocks and a CLI
caller can pass real stores). No model dependency. No I/O beyond the
stores it's given.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from harness.retrieval.context_package import (
    AccessPolicy,
    PackagedHit,
    Provenance,
    RetrievalContextPackage,
    TokenBudget,
)
from harness.store.document_tree import DocumentTreeStore
from harness.store.episodic import EpisodicStore
from harness.store.tabular import TabularStore

_VALID_STORES = frozenset({"episodic", "tree", "tabular"})


@dataclass(frozen=True)
class SlotSpec:
    """One slot in a contract. Validation happens at load time so the
    orchestrator can assume well-formed slots."""

    name: str
    store: str
    required: bool = False
    min_cardinality: int = 1
    max_hits: int = 5
    query_template: str | None = None
    sql_template: str | None = None
    table_name: str | None = None

    def __post_init__(self) -> None:
        if self.store not in _VALID_STORES:
            raise ValueError(
                f"slot {self.name!r}: store must be one of {sorted(_VALID_STORES)}, "
                f"got {self.store!r}"
            )
        if self.store == "tabular":
            if not self.sql_template:
                raise ValueError(f"slot {self.name!r}: tabular slot needs `sql_template`")
            if not self.table_name:
                raise ValueError(f"slot {self.name!r}: tabular slot needs `table_name`")
        else:
            if not self.query_template:
                raise ValueError(f"slot {self.name!r}: {self.store} slot needs `query_template`")
        if self.max_hits < 1:
            raise ValueError(f"slot {self.name!r}: max_hits must be >= 1")
        if self.min_cardinality < 0:
            raise ValueError(f"slot {self.name!r}: min_cardinality must be >= 0")


@dataclass(frozen=True)
class ContractBundle:
    """A complete role contract. `intent` is the human-readable label
    shown in audit / debug output; `slots` is the ordered list of
    what to fetch. Order matters: the orchestrator processes slots in
    declaration order, required slots get their full max_hits before
    optional slots even start consuming budget."""

    role: str
    intent: str
    budget_tokens: int
    slots: tuple[SlotSpec, ...]


@dataclass
class StoreBundle:
    """Container for the concrete stores the orchestrator dispatches
    to. Each is optional so tests can pass only what they need;
    callers that wire a slot for a store they didn't supply get an
    explicit error rather than silent skip."""

    episodic: EpisodicStore | None = None
    tree: DocumentTreeStore | None = None
    tabular: TabularStore | None = None


@dataclass(frozen=True)
class _RawHit:
    """Internal: a hit before budget enforcement. Body, score, and the
    provenance partially constructed (store + record_id + method;
    score normalized to float)."""

    slot_name: str
    body: str
    provenance: Provenance


# ---------- loading ----------


def load_contract(path: Path) -> ContractBundle:
    """Parse a contract YAML into a `ContractBundle`. Raises ValueError
    with a clear message on malformed slots so fixture authors see
    what's wrong without diving into stack traces."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: top-level must be a mapping")
    role = _required_str(raw, "role", path)
    intent = _required_str(raw, "intent", path)
    budget = raw.get("budget_tokens")
    if not isinstance(budget, int) or budget < 1:
        raise ValueError(f"{path}: `budget_tokens` must be a positive int")
    slots_raw = raw.get("slots")
    if not isinstance(slots_raw, list) or not slots_raw:
        raise ValueError(f"{path}: `slots` must be a non-empty list")
    slots = tuple(_parse_slot(entry, path, idx) for idx, entry in enumerate(slots_raw))
    return ContractBundle(role=role, intent=intent, budget_tokens=budget, slots=slots)


def _parse_slot(entry: object, path: Path, idx: int) -> SlotSpec:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{path}: slots[{idx}] must be a mapping")
    return SlotSpec(
        name=_required_str(entry, "name", path, idx=idx),
        store=_required_str(entry, "store", path, idx=idx),
        required=bool(entry.get("required", False)),
        min_cardinality=int(entry.get("min_cardinality", 1)),
        max_hits=int(entry.get("max_hits", 5)),
        query_template=_optional_str(entry, "query_template"),
        sql_template=_optional_str(entry, "sql_template"),
        table_name=_optional_str(entry, "table_name"),
    )


def _required_str(entry: Mapping[str, Any], key: str, path: Path, *, idx: int | None = None) -> str:
    val = entry.get(key)
    if not isinstance(val, str) or not val.strip():
        loc = f"slots[{idx}]" if idx is not None else "top-level"
        raise ValueError(f"{path}: {loc} missing or empty {key!r}")
    return val.strip()


def _optional_str(entry: Mapping[str, Any], key: str) -> str | None:
    val = entry.get(key)
    if not isinstance(val, str) or not val.strip():
        return None
    return val.strip()


# ---------- orchestration ----------


def assemble_package(
    contract: ContractBundle,
    *,
    variables: Mapping[str, Any],
    access: AccessPolicy,
    stores: StoreBundle,
    budget: TokenBudget | None = None,
) -> RetrievalContextPackage:
    """Walk a contract, call each shaped store, package the results
    into a `RetrievalContextPackage`.

    Required slots get their full `max_hits` first; optional slots
    fill any remaining budget. A required slot that returns fewer
    than `min_cardinality` hits surfaces in `missing_required_slots`
    — the caller decides whether to abort or surface a 'what's
    missing' message to the user.

    Pure function. No I/O beyond the stores in `stores`. No model
    dependency. Determinism inherits from the underlying stores'
    `search()` semantics.
    """
    effective_budget = budget or TokenBudget(max_tokens=contract.budget_tokens)
    missing: list[str] = []
    required_raw: list[_RawHit] = []
    optional_raw: list[_RawHit] = []

    for slot in contract.slots:
        hits = _fetch_slot(slot, variables=variables, access=access, stores=stores)
        if slot.required and len(hits) < slot.min_cardinality:
            missing.append(slot.name)
        if slot.required:
            required_raw.extend(hits)
        else:
            optional_raw.extend(hits)

    packed, overflow = _pack_within_budget(
        required_raw=required_raw,
        optional_raw=optional_raw,
        budget=effective_budget,
    )
    return RetrievalContextPackage(
        intent=contract.intent,
        access=access,
        budget=effective_budget,
        hits=tuple(packed),
        missing_required_slots=tuple(missing),
        overflow_hits=tuple(overflow),
    )


def _fetch_slot(
    slot: SlotSpec,
    *,
    variables: Mapping[str, Any],
    access: AccessPolicy,
    stores: StoreBundle,
) -> list[_RawHit]:
    """Dispatch one slot to its store, return raw hits in rank order.
    Failures (missing store, render errors) raise with a clear name so
    the contract author can fix the slot rather than getting a silent
    no-hit result."""
    if slot.store == "episodic":
        if stores.episodic is None:
            raise ValueError(
                f"slot {slot.name!r}: episodic slot needs an episodic store in StoreBundle"
            )
        query = _render_template(slot.query_template or "", variables, slot=slot)
        ep_results = stores.episodic.search(
            query, k=slot.max_hits, mode="hybrid", user_id=access.user_id
        )
        return [
            _RawHit(
                slot_name=slot.name,
                body=_format_episodic_body(rec.title, rec.body, rec.principle),
                provenance=Provenance(
                    store="episodic",
                    record_id=str(rec.external_id) if rec.external_id else f"id:{rec.id}",
                    method="hybrid",
                    score=float(score),
                ),
            )
            for rec, score in ep_results
        ]

    if slot.store == "tree":
        if stores.tree is None:
            raise ValueError(f"slot {slot.name!r}: tree slot needs a tree store in StoreBundle")
        query = _render_template(slot.query_template or "", variables, slot=slot)
        tree_results = stores.tree.search(query, k=slot.max_hits, mode="hybrid")
        return [
            _RawHit(
                slot_name=slot.name,
                body=_format_tree_body(node.heading, node.body, node.path),
                provenance=Provenance(
                    store="tree",
                    record_id=node.path,
                    method="hybrid",
                    score=float(score),
                ),
            )
            for node, score in tree_results
        ]

    if slot.store == "tabular":
        if stores.tabular is None:
            raise ValueError(
                f"slot {slot.name!r}: tabular slot needs a tabular store in StoreBundle"
            )
        sql = _render_template(slot.sql_template or "", variables, slot=slot)
        table_name = slot.table_name or ""
        result = stores.tabular.query_sql(table_name, sql)
        # Each returned row becomes one hit; the body is a markdown-ish
        # rendering of (column: value) pairs so the model can grok it.
        hits: list[_RawHit] = []
        for rank, row in enumerate(result.rows[: slot.max_hits]):
            row_id = _row_id_for_provenance(result.columns, row)
            hits.append(
                _RawHit(
                    slot_name=slot.name,
                    body=_format_tabular_body(result.columns, row),
                    provenance=Provenance(
                        store="tabular",
                        record_id=row_id,
                        method="sql",
                        # Score-by-rank: earlier rows ranked higher.
                        # SQL imposes its own ORDER BY; we preserve it.
                        score=1.0 - rank * 0.01,
                    ),
                )
            )
        return hits

    raise ValueError(f"slot {slot.name!r}: unknown store {slot.store!r}")


def _render_template(template: str, variables: Mapping[str, Any], *, slot: SlotSpec) -> str:
    try:
        return template.format(**variables)
    except KeyError as exc:
        raise ValueError(
            f"slot {slot.name!r}: template references variable {exc.args[0]!r} "
            f"that wasn't provided in `variables`"
        ) from exc


def _format_episodic_body(title: str, body: str, principle: str | None) -> str:
    parts = [title.strip()] if title.strip() else []
    if principle and principle.strip():
        parts.append(f"({principle.strip()})")
    if body and body.strip():
        parts.append(body.strip())
    return " — ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")


def _format_tree_body(heading: str, body: str, path: str) -> str:
    head = f"§{path} {heading.strip()}"
    return f"{head}\n{body.strip()}" if body.strip() else head


def _format_tabular_body(columns: Iterable[str], row: tuple[Any, ...]) -> str:
    return ", ".join(f"{c}={v}" for c, v in zip(columns, row, strict=False))


def _row_id_for_provenance(columns: tuple[str, ...], row: tuple[Any, ...]) -> str:
    """Best-effort row id for tabular hits. If the projection includes
    a column named like an id, use that; else fall back to a coarse
    concatenation. Provenance never claims more than what was in the
    projection — the caller picks columns."""
    for candidate in ("id", "order_id", "row_id"):
        if candidate in columns:
            return f"{candidate}={row[columns.index(candidate)]}"
    return "row=" + ",".join(str(v) for v in row[:3])


def _pack_within_budget(
    *,
    required_raw: list[_RawHit],
    optional_raw: list[_RawHit],
    budget: TokenBudget,
) -> tuple[list[PackagedHit], list[PackagedHit]]:
    """Greedy budget enforcement. Required hits go first — if a
    required hit doesn't fit, it's still included (overflow is
    preferable to silently dropping content the contract said was
    required). Optional hits then fill remaining budget; ones that
    would overflow are tracked in `overflow_hits` so the caller can
    still see what was retrieved."""
    estimator = budget.token_estimator
    used = 0
    packed: list[PackagedHit] = []
    overflow: list[PackagedHit] = []

    for raw in required_raw:
        tokens = estimator(raw.body)
        hit = PackagedHit(
            slot_name=raw.slot_name,
            body=raw.body,
            provenance=raw.provenance,
            est_tokens=tokens,
        )
        # Required hits always included; if they overflow the budget,
        # the caller learns via tokens_used vs budget.max_tokens.
        packed.append(hit)
        used += tokens

    for raw in optional_raw:
        tokens = estimator(raw.body)
        hit = PackagedHit(
            slot_name=raw.slot_name,
            body=raw.body,
            provenance=raw.provenance,
            est_tokens=tokens,
        )
        if used + tokens > budget.max_tokens:
            overflow.append(hit)
            continue
        packed.append(hit)
        used += tokens

    return packed, overflow


__all__ = [
    "ContractBundle",
    "SlotSpec",
    "StoreBundle",
    "assemble_package",
    "load_contract",
]
