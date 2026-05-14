"""Generic ingest pipeline for `DocumentTreeStore` (harness-2zf4).

Two input shapes feed one driver:

  - **Markdown** — heading levels (`# / ## / ###`) become depth 1, 2, 3
    nodes. Path is positional (e.g. `1-2-3`). Body is the text between
    a heading and the next heading of same-or-higher level. Body-bearing
    nodes embed; structural-only nodes (heading + no body) skip embed.

  - **JSONL** — pre-chunked rows grouped by one or more `depth_fields`,
    where each field's value at depth `i` is that node's path. Used by
    the ATC corpus today (`chapter` / `parent_section` / `section`).
    Intermediate-level nodes are structural-only (no body); the leaf
    level concatenates grouped chunk bodies and embeds.

Both adapters yield `NodeBlueprint` streams. The single driver
`ingest_blueprints()` walks the stream in order, looks up `parent_id`
from a path→id cache, and calls `DocumentTreeStore.ingest_node`. The
adapter doesn't talk to the store; the driver doesn't know the source
shape. This keeps adapters easy to add (a markdown.zip adapter, a PDF
adapter) without re-implementing parent-id bookkeeping each time.

PATH SCHEME (locked decision, see harness-2zf4 notes):

Positional slugs (`1`, `1-2`, `1-2-3`). Chosen for consistency with
the ATC corpus's section identifiers (`2-4-3` is JO 7110.65 §2-4-3).
The trade-off is that inserting a new `##` between existing siblings
shifts the positional id of every following sibling, invalidating any
external pointer that named the old path (audit-log record_ids, voice
samples that cite a section, contract record_ids in eval fixtures).

Mitigation: treat structural edits to ingested sources as additive
where possible. Renumbers are explicit and require a refresh of any
external references. A future option is a parallel `stable_id` column
on `tree_nodes` (slug-of-heading hash) that survives reorders, but
adding it is deferred until a renumber actually bites.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.store.document_tree import DocumentTreeStore


# ATX-style markdown heading: 1-6 leading `#` chars, then space, then
# the heading text. We deliberately ignore setext-style headings (---
# underlines) because they're rare in modern markdown and would
# complicate the line-by-line scanner.
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

_DEFAULT_NODE_TYPES = ("chapter", "section_group", "section", "section")
"""Default node_type per depth (1-indexed). Picked to match the ATC
corpus's vocabulary (`chapter` → `section_group` → `section`). Depths
beyond the tuple length reuse the last entry."""


@dataclass(frozen=True)
class NodeBlueprint:
    """One node about to be ingested. Adapters yield these in
    pre-order (parent before child); the driver maps `parent_path` to
    the parent's row id via a cache built during ingest."""

    parent_path: str | None
    path: str
    ordinal: int
    depth: int
    node_type: str
    heading: str
    body: str
    embed: bool


# ---------- markdown adapter ----------


def iter_markdown_nodes(
    source: Path,
    *,
    node_type_by_depth: tuple[str, ...] = _DEFAULT_NODE_TYPES,
) -> Iterator[NodeBlueprint]:
    """Stream a markdown file into `NodeBlueprint`s.

    The parser is intentionally simple: ATX-style headings only,
    positional path slugs, body = text between this heading and the
    next heading at same-or-higher depth (trimmed of trailing/leading
    whitespace). A heading with no body becomes a structural-only
    node (`embed=False`); a heading with body embeds, regardless of
    whether it also has children. This auto-adapts to document shape
    — a flat doc embeds every node; a deeply-nested doc embeds only
    the leaves that actually have prose.

    Skipped depths (e.g. `#` directly followed by `###`) are tolerated:
    the missing intermediate level is silently filled with `1` so the
    path stays well-formed (`1-1-1` instead of `1-?-1`). The skip is
    invisible — no synthetic node is yielded — because we don't have
    a heading to give it. Callers who need intermediate placeholders
    should add explicit `##` headings to the source.
    """
    text = source.read_text(encoding="utf-8")
    headings = _parse_markdown_headings(text)
    if not headings:
        return

    counters: list[int] = []
    for depth, heading, body in headings:
        # Reshape counters to the current depth. Three cases:
        #   1. depth > len(counters) — going deeper. Pad synthetic
        #      ancestors with ordinal=1 and append a fresh `1`.
        #   2. depth == len(counters) — sibling. Increment last.
        #   3. depth < len(counters) — popping back up. Truncate
        #      then increment.
        if depth > len(counters):
            while len(counters) < depth - 1:
                counters.append(1)
            counters.append(1)
        elif depth == len(counters):
            counters[-1] += 1
        else:
            counters = counters[:depth]
            counters[-1] += 1

        path = "-".join(str(c) for c in counters)
        parent_path = "-".join(str(c) for c in counters[:-1]) if depth > 1 else None
        ordinal = counters[-1]
        node_type = _node_type_for_depth(depth, node_type_by_depth)

        yield NodeBlueprint(
            parent_path=parent_path,
            path=path,
            ordinal=ordinal,
            depth=depth,
            node_type=node_type,
            heading=heading,
            body=body,
            embed=bool(body),
        )


def _parse_markdown_headings(text: str) -> list[tuple[int, str, str]]:
    """First-pass line scanner. Returns `(depth, heading, body)`
    tuples in source order. Body is everything from this heading's
    next line up to (but not including) the next heading line."""
    parsed: list[tuple[int, str, list[str]]] = []
    body_lines: list[str] = []
    for line in text.splitlines():
        match = _MD_HEADING_RE.match(line)
        if match is None:
            if parsed:
                body_lines.append(line)
            # Lines before the first heading (e.g., a frontmatter
            # block or a doc-level intro) are dropped — the markdown
            # adapter ingests only what lives under a heading. Doc-
            # level prose should sit under a top-level `# Title`.
            continue
        if parsed:
            parsed[-1] = (parsed[-1][0], parsed[-1][1], body_lines)
            body_lines = []
        depth = len(match.group(1))
        heading = match.group(2).strip()
        parsed.append((depth, heading, []))
    if parsed:
        parsed[-1] = (parsed[-1][0], parsed[-1][1], body_lines)

    return [(d, h, "\n".join(b).strip()) for d, h, b in parsed]


# ---------- jsonl adapter ----------


def iter_jsonl_nodes(
    source: Path,
    *,
    depth_fields: tuple[str, ...],
    heading_prefixes: tuple[str, ...] = (),
    leaf_heading_field: str | None = None,
    body_field: str = "body",
    chunk_index_field: str | None = "chunk_index",
    node_type_by_depth: tuple[str, ...] = _DEFAULT_NODE_TYPES,
) -> Iterator[NodeBlueprint]:
    """Stream a pre-chunked JSONL into `NodeBlueprint`s.

    `depth_fields` names the JSONL fields whose values define the
    hierarchy. For the ATC corpus that's `("chapter", "parent_section",
    "section")` — so a row with `chapter="2"`, `parent_section="2-4"`,
    `section="2-4-3"` produces three nodes with paths `"2"`, `"2-4"`,
    `"2-4-3"`. The leaf level (depth = len(depth_fields)) concatenates
    bodies from rows in the same group, ordered by `chunk_index_field`
    if provided.

    Intermediate-level nodes (above the leaf) are structural-only:
    no body, no embed. The JSONL doesn't carry intermediate prose by
    design — chapters in ATC are containers, not content. `heading_prefixes`
    lets the caller decorate intermediate headings ("Chapter 2", "§2-4")
    if the bare path value is too terse; default (empty tuple) uses
    the path value as the heading.

    `leaf_heading_field` names a JSONL field that holds the leaf's
    title (`"title"` for ATC). When None, the leaf heading is the path
    value too.

    Yields in stable order: nodes appear in the order their containing
    group is first encountered in the JSONL. Intermediate nodes emit
    once per (chapter-tuple-prefix) so the driver doesn't see
    duplicates. Leaf nodes always emit (one per terminal group).
    """
    if not depth_fields:
        raise ValueError("iter_jsonl_nodes: depth_fields must be non-empty")

    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as fp:
        for raw_line in fp:
            stripped = raw_line.strip()
            if stripped:
                rows.append(json.loads(stripped))

    # Group rows by the full depth-field tuple — that's the leaf-level
    # group. Insertion order is preserved (Python 3.7+ dict semantics)
    # so leaves emit in JSONL appearance order.
    groups: OrderedDict[tuple[str, ...], list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        key = tuple(str(row.get(f, "")) for f in depth_fields)
        if not all(key):
            continue  # skip rows missing any hierarchy field
        groups.setdefault(key, []).append(row)

    # Track which intermediate-level path tuples we've already yielded
    # so each chapter / section_group emits exactly once.
    seen_intermediates: set[tuple[str, ...]] = set()
    # Ordinals are per-(parent-path, depth) so siblings under the same
    # parent get sequential ordinals starting at 1, matching the
    # markdown adapter's convention.
    ordinal_at: dict[tuple[str | None, int], int] = defaultdict(int)

    leaf_depth = len(depth_fields)

    for key, group_rows in groups.items():
        # Emit any not-yet-seen ancestors of this leaf, in depth order.
        for d in range(1, leaf_depth):
            prefix = key[:d]
            if prefix in seen_intermediates:
                continue
            seen_intermediates.add(prefix)
            path = prefix[-1]
            parent_path = prefix[-2] if d > 1 else None
            parent_key = (parent_path, d)
            ordinal_at[parent_key] += 1
            heading = _heading_for_intermediate(
                depth=d, path_value=path, heading_prefixes=heading_prefixes
            )
            yield NodeBlueprint(
                parent_path=parent_path,
                path=path,
                ordinal=ordinal_at[parent_key],
                depth=d,
                node_type=_node_type_for_depth(d, node_type_by_depth),
                heading=heading,
                body="",
                embed=False,
            )

        # Now the leaf.
        leaf_path = key[-1]
        leaf_parent = key[-2] if leaf_depth > 1 else None
        leaf_key = (leaf_parent, leaf_depth)
        ordinal_at[leaf_key] += 1
        leaf_heading = _heading_for_leaf(
            rows=group_rows,
            path_value=leaf_path,
            leaf_heading_field=leaf_heading_field,
            depth=leaf_depth,
            heading_prefixes=heading_prefixes,
        )
        body = _join_bodies(group_rows, body_field=body_field, sort_field=chunk_index_field)
        yield NodeBlueprint(
            parent_path=leaf_parent,
            path=leaf_path,
            ordinal=ordinal_at[leaf_key],
            depth=leaf_depth,
            node_type=_node_type_for_depth(leaf_depth, node_type_by_depth),
            heading=leaf_heading,
            body=body,
            embed=bool(body),
        )


def _heading_for_intermediate(
    *, depth: int, path_value: str, heading_prefixes: tuple[str, ...]
) -> str:
    prefix = heading_prefixes[depth - 1] if depth - 1 < len(heading_prefixes) else ""
    return f"{prefix}{path_value}"


def _heading_for_leaf(
    *,
    rows: list[dict[str, Any]],
    path_value: str,
    leaf_heading_field: str | None,
    depth: int,
    heading_prefixes: tuple[str, ...],
) -> str:
    if leaf_heading_field:
        candidate = str(rows[0].get(leaf_heading_field, "")).strip()
        if candidate:
            return candidate
    return _heading_for_intermediate(
        depth=depth, path_value=path_value, heading_prefixes=heading_prefixes
    )


def _join_bodies(
    rows: list[dict[str, Any]],
    *,
    body_field: str,
    sort_field: str | None,
) -> str:
    ordered = sorted(rows, key=lambda r: int(r.get(sort_field, 0))) if sort_field else rows
    pieces = [str(r.get(body_field, "")).strip() for r in ordered]
    return "\n\n".join(p for p in pieces if p)


# ---------- driver ----------


def ingest_blueprints(
    store: DocumentTreeStore,
    *,
    document_name: str,
    source_uri: str | None,
    blueprints: Iterable[NodeBlueprint],
) -> dict[str, int]:
    """Drive a NodeBlueprint stream into the store. Returns row counts
    keyed by node_type so callers can log / assert without scanning the
    store.

    Idempotent on `(document_id, path)`: re-running over the same
    source produces the same row count because `DocumentTreeStore.ingest_node`
    returns the existing row when path collides. That matches the
    write-mostly-once posture of every other store in the codebase.

    Order assumption: the blueprint stream emits parents before
    children. The driver looks up `parent_id` from a path-keyed cache
    built as nodes write; encountering a child whose `parent_path`
    hasn't been seen yet raises a clear error so the adapter (not the
    driver) is the one that gets blamed for the ordering bug.
    """
    document = store.upsert_document(name=document_name, source_uri=source_uri)
    path_to_id: dict[str, int] = {}
    counts: dict[str, int] = defaultdict(int)

    for blueprint in blueprints:
        parent_id: int | None = None
        if blueprint.parent_path is not None:
            parent_id = path_to_id.get(blueprint.parent_path)
            if parent_id is None:
                raise ValueError(
                    f"ingest_blueprints: node {blueprint.path!r} references parent "
                    f"{blueprint.parent_path!r} which hasn't been emitted yet — "
                    "adapter must yield parents before children"
                )

        node = store.ingest_node(
            document_id=document.id,
            parent_id=parent_id,
            path=blueprint.path,
            ordinal=blueprint.ordinal,
            depth=blueprint.depth,
            node_type=blueprint.node_type,
            heading=blueprint.heading,
            body=blueprint.body,
            embed=blueprint.embed,
        )
        path_to_id[blueprint.path] = node.id
        counts[blueprint.node_type] += 1

    return dict(counts)


# ---------- helpers ----------


def _node_type_for_depth(depth: int, node_type_by_depth: tuple[str, ...]) -> str:
    if not node_type_by_depth:
        return "section"
    idx = min(depth - 1, len(node_type_by_depth) - 1)
    return node_type_by_depth[idx]


__all__ = [
    "NodeBlueprint",
    "ingest_blueprints",
    "iter_jsonl_nodes",
    "iter_markdown_nodes",
]
