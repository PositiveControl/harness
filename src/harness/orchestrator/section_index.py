"""Structural index of §-anchors that exist in a character's corpus,
used by `FabricatedSectionHook` (harness-aise) to refuse replies that
cite a section the corpus doesn't contain.

Different from the runtime retrieval path: retrieval gives the model
the chunk content; this index gives the orchestrator a yes/no answer
to 'is `§3-99-3` even a real section?'. A reply that cites a section
no row in the corpus carries is a fabrication regardless of how the
text reads, and we'd rather refuse than ship it.

Two sources, called separately and unioned by the CLI:

  * `collect_valid_anchors(chunks_dir)` — FAA characters
    (airton_c{,1,_tfr}). Reads `corpus/chunks/*.jsonl`; each row
    carries a `section` field stamped by the chunker.

  * `collect_valid_anchors_from_markdown(character_path, trees)` —
    markdown-corpus characters (airton_f). Walks each markdown source
    referenced by the character's `document_trees:` config, extracts
    numeric section headings (`## 1. Scope`, `### 2.1 Greeter`) and
    surfaces them as `§N` / `§N.N` anchors.

Reading the source directly is faster than waking the embedded store
(no embedder load) and is idempotent: same input → same frozenset,
no per-process state. The hook treats an empty set as 'silent' rather
than 'block everything', so this module is safe to call on characters
without an enumerable corpus — it returns frozenset() and the hook
is a no-op."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.character import DocumentTreeSpec


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


# Matches numeric section headings in markdown:
#   ## 1. Scope
#   ### 2.1 Greeter
#   ## 12.3.4 Subsection
# The number portion can be dot-separated (`2.1`, `12.3.4`) and is
# captured. Trailing `.` or `)` after the number is optional —
# different markdown corpora use different conventions. Requires
# whitespace after the number so prose containing "1. foo" doesn't
# false-positive as a heading.
_MD_NUMERIC_HEADING_RE = re.compile(r"^#+\s+(\d+(?:\.\d+)*)[.)]?\s+\S")


def collect_valid_anchors_from_markdown(
    document_trees: tuple[DocumentTreeSpec, ...],
) -> frozenset[str]:
    """Walk each markdown source referenced by the character's
    `document_trees:` config and harvest numeric section headings.

    For each `## <N>[. ]<title>` or `### <N.N>[. ]<title>` line, the
    leading `N` / `N.N` / `N.N.N` token contributes `§<N>` to the
    valid-anchor set. Non-markdown trees and missing source files
    are silently skipped — same "empty set = hook silent" contract
    as `collect_valid_anchors`.

    Used by markdown-corpus characters (airton_f, future legal /
    paper-reading personas). Pairs with the chunks-JSONL function
    above; the CLI unions the two so multi-corpus characters get
    both anchor shapes.

    `DocumentTreeSpec.source_path` is already resolved to an absolute
    Path by `load_character`, so no character-root join needed."""
    out: set[str] = set()
    for tree in document_trees:
        if getattr(tree, "source_format", None) != "markdown":
            continue
        source_path = getattr(tree, "source_path", None)
        if source_path is None or not isinstance(source_path, Path):
            continue
        if not source_path.exists() or not source_path.is_file():
            continue
        try:
            with source_path.open(encoding="utf-8") as fp:
                for line in fp:
                    match = _MD_NUMERIC_HEADING_RE.match(line.rstrip("\n"))
                    if match:
                        out.add(f"§{match.group(1)}")
        except OSError:
            # Unreadable file — same resilience contract as the
            # chunks walker above.
            continue
    return frozenset(out)
