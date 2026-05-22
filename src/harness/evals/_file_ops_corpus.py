"""Shared corpus for the harness-bw27 file-manipulation tool bench.

Both the wall-clock bench (``scripts/bench_file_ops.py``) and the
model-in-loop eval (``harness.evals.file_ops``) need:

  * the FileOpsTask records — deterministic fixture generators,
    Python oracles, three prompt phrasings each, and
  * a uniform way to construct each candidate tool against a
    sandbox-pinned workspace, with the output cap bumped past the
    model-protection default so digest checks see the full transform.

Phase 3 of harness-bw27 extracted this from
``scripts/bench_file_ops.py`` so the eval (Phase 3 part 2) can import
it via ``harness.evals._file_ops_corpus`` without the awkward
``importlib.util`` trick the tests had to use for the bench script.

The leading underscore signals "harness-bw27 internal" — the corpus
isn't meant for general consumption. Promote (or rename) when the
winner ships to a real profile in Phase 4.
"""

from __future__ import annotations

import hashlib
import json
import random
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class FileOpsTask:
    """A single benchmark task: prompt set + fixture builder + oracle."""

    id: str
    name: str
    description: str
    prompts: tuple[str, str, str]
    fixture_fn: Callable[[Path], None]
    oracle_fn: Callable[[Path], str]


# --- shared helpers ---------------------------------------------------------


def _rng(seed: int) -> random.Random:
    """Seeded RNG. The bench fixtures are not security-sensitive — the
    Random determinism guarantee is exactly what we want, and ruff's S311
    blanket warning doesn't apply (per the same precedent as
    tests/test_banter.py)."""
    return random.Random(seed)  # noqa: S311 — deterministic fixtures, not crypto


# --- task 1: extract column 3 from a 100k-line log --------------------------


def fixture_extract_col3(root: Path) -> None:
    """Synthesize logs/sample.log: 100k lines of the form
    ``2026-05-22T12:00:SS service host event-N user-K key=val-V``.
    Field order is intentional — col 3 (1-indexed) is the host so the
    prompts that say "hostname (column 3)" match reality. Col 5 is the
    ``user-K`` token, the target for the distinct-count task."""
    rng = _rng(0xC0FFEE_01)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    services = ("nginx", "redis", "postgres", "kafka", "harness")
    with (logs_dir / "sample.log").open("w") as fh:
        for i in range(100_000):
            host = f"ip-10-{rng.randint(0, 255)}-{rng.randint(0, 255)}-{rng.randint(0, 255)}"
            service = rng.choice(services)
            user_k = rng.randint(0, 999)
            key_v = rng.randint(0, 99)
            fh.write(
                f"2026-05-22T12:00:{i % 60:02d} {service} {host} "
                f"event-{i % 13} user-{user_k} key=val-{key_v}\n"
            )


def oracle_extract_col3(root: Path) -> str:
    """Whitespace-split each line, take index 2 (the host). Lines with
    fewer than 3 fields drop out."""
    text = (root / "logs" / "sample.log").read_text()
    cols: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            cols.append(parts[2])
    return "\n".join(cols) + "\n"


# --- task 2: replace foo->bar across 60 .py files ---------------------------


def fixture_multi_file_replace(root: Path) -> None:
    """60 small .py files under tests_ws/pkg_{0..5}/. Each file contains
    a mix of `foo_*` and `x_*` identifiers; the count of `foo` per file
    varies so the model can't shortcut by replacing a fixed number."""
    rng = _rng(0xC0FFEE_02)
    ws = root / "tests_ws"
    ws.mkdir(parents=True, exist_ok=True)
    for i in range(60):
        sub = ws / f"pkg_{i // 10}"
        sub.mkdir(parents=True, exist_ok=True)
        n_foos = rng.randint(0, 5)
        n_lines = rng.randint(3, 12)
        body_lines: list[str] = []
        for j in range(n_lines):
            if j < n_foos:
                body_lines.append(f"    foo_{j} = {rng.randint(0, 999)}")
            else:
                body_lines.append(f"    x_{j} = {rng.randint(0, 999)}")
        tail = "foo_0" if n_foos > 0 else "0"
        body = "\n".join(body_lines)
        (sub / f"mod_{i}.py").write_text(
            f"def f():\n{body}\n    return {tail}\n",
        )


def oracle_multi_file_replace(root: Path) -> str:
    """Deterministic manifest: one ``<relpath> <sha256[:16]>`` line per
    .py file after a global ``foo`` -> ``bar`` substitution. Sorted by
    path so the order of the rewriter doesn't matter."""
    ws = root / "tests_ws"
    lines: list[str] = []
    for path in sorted(ws.rglob("*.py")):
        rewritten = path.read_text().replace("foo", "bar")
        digest = hashlib.sha256(rewritten.encode()).hexdigest()[:16]
        rel = path.relative_to(root).as_posix()
        lines.append(f"{rel} {digest}")
    return "\n".join(lines) + "\n"


# --- task 3: filter JSONL by user, project two fields -----------------------


def fixture_filter_jsonl(root: Path) -> None:
    """10k JSONL events, three users in equal-ish proportions, four
    action types, a synthetic ts and a synthetic amount."""
    rng = _rng(0xC0FFEE_03)
    out = root / "data" / "events.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    users = ("alice", "bob", "carol")
    actions = ("login", "click", "purchase", "logout")
    with out.open("w") as fh:
        for i in range(10_000):
            event = {
                "id": i,
                "user": rng.choice(users),
                "action": rng.choice(actions),
                "ts": 1_700_000_000 + i,
                "amount": rng.randint(0, 10_000) / 100,
            }
            fh.write(json.dumps(event) + "\n")


def oracle_filter_jsonl(root: Path) -> str:
    """Keep records where user == 'alice', project (id, amount), preserve
    original input order, emit JSONL."""
    src = root / "data" / "events.jsonl"
    out: list[str] = []
    for line in src.read_text().splitlines():
        rec = json.loads(line)
        if rec["user"] == "alice":
            out.append(json.dumps({"id": rec["id"], "amount": rec["amount"]}))
    return "\n".join(out) + "\n"


# --- task 4: rewrite a 5-line function body ---------------------------------


_CALC_BEFORE = textwrap.dedent(
    '''\
    """Calc module — pre-edit state."""


    def compute_total(items):
        total = 0
        for x in items:
            total += x
        # legacy comment
        return total


    def other():
        return 42
    '''
)


_CALC_AFTER = textwrap.dedent(
    '''\
    """Calc module — pre-edit state."""


    def compute_total(items):
        return sum(items)


    def other():
        return 42
    '''
)


def fixture_function_body_rewrite(root: Path) -> None:
    src = root / "src" / "calc.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(_CALC_BEFORE)


def oracle_function_body_rewrite(_root: Path) -> str:
    """Expected post-edit file content. The model must rewrite the body
    of compute_total exactly; other() and the module docstring stay
    untouched. ``_root`` is unused — fixture is whole-file content, not
    derived from disk — but the signature matches the dispatch table."""
    return _CALC_AFTER


# --- task 5: distinct count of column 5 -------------------------------------


def fixture_distinct_count(root: Path) -> None:
    """Reuse the col3 fixture verbatim — same shape, same seed, so col 5
    (``user-K``) has a stable cardinality bench-to-bench."""
    fixture_extract_col3(root)


def oracle_distinct_count(root: Path) -> str:
    text = (root / "logs" / "sample.log").read_text()
    distinct: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            distinct.add(parts[4])
    return f"{len(distinct)}\n"


# --- task table -------------------------------------------------------------


BENCH_TASKS: tuple[FileOpsTask, ...] = (
    FileOpsTask(
        id="extract-col3",
        name="Extract column 3 from a 100k-line log",
        description=(
            "Pull the third whitespace-separated field from every line "
            "of logs/sample.log. Output: one value per line, in original "
            "input order."
        ),
        prompts=(
            "Extract the third whitespace-separated column from logs/sample.log.",
            "Pull out just the hostname (column 3) from each line in logs/sample.log.",
            "I need column 3 of logs/sample.log — one value per line, in order.",
        ),
        fixture_fn=fixture_extract_col3,
        oracle_fn=oracle_extract_col3,
    ),
    FileOpsTask(
        id="multi-file-replace",
        name="Replace foo->bar across 60 .py files",
        description=(
            "Rewrite every `foo` substring to `bar` in all .py files "
            "under tests_ws/. Scoring artifact: one ``<relpath> "
            "<sha256[:16]>`` line per file, sorted by path."
        ),
        prompts=(
            "Rename all occurrences of foo to bar across the tests_ws directory.",
            "Replace foo with bar in every .py file under tests_ws/.",
            "Bulk-rename foo->bar in tests_ws/**/*.py.",
        ),
        fixture_fn=fixture_multi_file_replace,
        oracle_fn=oracle_multi_file_replace,
    ),
    FileOpsTask(
        id="filter-jsonl",
        name="Filter JSONL by user, project two fields",
        description=(
            "From data/events.jsonl (10k lines), emit one JSONL line per "
            "record where user == 'alice', containing exactly {id, "
            "amount}, in original input order."
        ),
        prompts=(
            "From data/events.jsonl, keep only alice's events and output id + amount as JSONL.",
            "Filter events.jsonl to user=alice and project id and amount.",
            "I need alice's id and amount fields from data/events.jsonl, one JSON object per line.",
        ),
        fixture_fn=fixture_filter_jsonl,
        oracle_fn=oracle_filter_jsonl,
    ),
    FileOpsTask(
        id="function-body-rewrite",
        name="Rewrite a 5-line function body in one file",
        description=(
            "In src/calc.py, replace the body of compute_total() with "
            "the single line ``    return sum(items)``. Other functions "
            "and the module docstring stay untouched."
        ),
        prompts=(
            "Replace the body of compute_total() in src/calc.py with: return sum(items).",
            "Rewrite compute_total in src/calc.py to just return sum(items).",
            "Refactor src/calc.py: compute_total should be a one-liner that returns sum(items).",
        ),
        fixture_fn=fixture_function_body_rewrite,
        oracle_fn=oracle_function_body_rewrite,
    ),
    FileOpsTask(
        id="distinct-count",
        name="Count distinct values in column 5",
        description=(
            "Report the number of distinct whitespace-separated values "
            "in column 5 (1-indexed) of logs/sample.log. Output: a "
            "single integer on its own line."
        ),
        prompts=(
            "How many distinct values are in column 5 of logs/sample.log?",
            "Count distinct users (col 5) in logs/sample.log.",
            "Run sort -u on column 5 of logs/sample.log and tell me how many lines.",
        ),
        fixture_fn=fixture_distinct_count,
        oracle_fn=oracle_distinct_count,
    ),
)


# --- candidate factory ------------------------------------------------------


CandidateKind = Literal["stream_edit", "pyp_stream", "python_stream"]
ALL_CANDIDATES: tuple[CandidateKind, ...] = ("stream_edit", "pyp_stream", "python_stream")


def make_tool(kind: CandidateKind, root: Path) -> Any:
    """Construct the candidate tool for ``root``. Lazy-imports the
    pyp_stream module so a slim install (no `stream` extra) can still
    use stream_edit + python_stream.

    The output cap is bumped to 16 MB — the bench and model-in-loop
    eval bypass the live model boundary, so the production 512 KB cap
    (which protects the model from huge tool results) would distort
    digest checks on tasks like extract-col3 that produce 1.6 MB."""
    bench_output_cap = 16 * 1024 * 1024
    if kind == "stream_edit":
        from harness.tools.stream_edit import StreamEditTool

        return StreamEditTool(
            root=root,
            timeout_seconds=60.0,
            max_output_bytes=bench_output_cap,
        )
    if kind == "python_stream":
        from harness.tools.python_stream import PythonStreamTool

        return PythonStreamTool(
            root=root,
            timeout_seconds=60.0,
            max_output_bytes=bench_output_cap,
        )
    if kind == "pyp_stream":
        from harness.tools.pyp_stream import PypStreamTool

        return PypStreamTool(
            root=root,
            timeout_seconds=60.0,
            max_output_bytes=bench_output_cap,
        )
    raise ValueError(f"unknown candidate {kind!r}")


__all__ = [
    "ALL_CANDIDATES",
    "BENCH_TASKS",
    "CandidateKind",
    "FileOpsTask",
    "fixture_distinct_count",
    "fixture_extract_col3",
    "fixture_filter_jsonl",
    "fixture_function_body_rewrite",
    "fixture_multi_file_replace",
    "make_tool",
    "oracle_distinct_count",
    "oracle_extract_col3",
    "oracle_filter_jsonl",
    "oracle_function_body_rewrite",
    "oracle_multi_file_replace",
]
