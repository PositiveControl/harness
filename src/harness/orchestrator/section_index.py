"""Structural index of §-anchors that exist in a character's corpus,
used by `FabricatedSectionHook` (harness-aise) to refuse replies that
cite a section the corpus doesn't contain.

Different from the runtime retrieval path: retrieval gives the model
the chunk content; this index gives the orchestrator a yes/no answer
to 'is `§3-99-3` even a real section?'. A reply that cites a section
no row in the corpus carries is a fabrication regardless of how the
text reads, and we'd rather refuse than ship it.

Source of truth: `character/<name>/corpus/chunks/*.jsonl` — each row
already carries a `section` field stamped by the chunker. Reading the
JSONL is faster than waking the embedded store (no embedder load) and
is idempotent: same input → same frozenset, no per-process state.

The hook treats an empty set as 'silent' rather than 'block everything',
so this module is safe to call on characters without an enumerable
corpus — it returns frozenset() and the hook is a no-op."""

from __future__ import annotations

import json
from pathlib import Path


def collect_valid_anchors(chunks_dir: Path) -> frozenset[str]:
    """Walk every `*.jsonl` file under `chunks_dir`, harvest the
    `section` field on each row, and return the canonical §-anchor
    set augmented with §N-N parents.

    Parents widen acceptance: a corpus row stamped section=`3-10-3`
    contributes both `§3-10-3` (the exact paragraph anchor) and
    `§3-10` (the parent section). A reply that cites the parent
    without a paragraph is structurally valid — the parent has at
    least one paragraph in the corpus.

    Missing or empty `chunks_dir` returns frozenset() — used as the
    'silent' signal by FabricatedSectionHook."""
    if not chunks_dir.exists() or not chunks_dir.is_dir():
        return frozenset()
    out: set[str] = set()
    for jsonl in sorted(chunks_dir.glob("*.jsonl")):
        try:
            with jsonl.open(encoding="utf-8") as fp:
                for line in fp:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError:
                        # Skip malformed lines rather than failing the
                        # whole index — a single bad row shouldn't
                        # silently disable the existence check.
                        continue
                    section = row.get("section")
                    if not isinstance(section, str) or not section:
                        continue
                    out.add(f"§{section}")
                    parts = section.split("-")
                    if len(parts) == 3:
                        out.add(f"§{parts[0]}-{parts[1]}")
        except OSError:
            # Unreadable file (permissions / vanished mid-walk) — skip
            # to keep the index resilient. The hook will still work on
            # whatever rows DID load.
            continue
    return frozenset(out)
