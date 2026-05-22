"""Phase 1 of harness-bw27 — locked benchmark spec for file-manipulation tools.

Defines five concrete tasks the candidate tools (stream_edit, pyp_stream,
python_stream) will compete on. Each task ships with:

  * a deterministic fixture generator (seeded random.Random) that
    materializes input files into a per-task directory,
  * a Python oracle that produces the gold output for that fixture, and
  * three natural-language phrasings of the request so the model-in-loop
    eval (Phase 3) can measure first-try syntax success across plausible
    user prompts.

This module ships no candidate tools and no timing loop. Those land in
Phase 2/3 once we know what we're measuring against. The CLI here is a
self-check: generate the fixtures, run the oracles, and confirm the
digests match the pinned values in `tests/test_bench_file_ops.py`.

Usage:
    uv run python scripts/bench_file_ops.py list-tasks
    uv run python scripts/bench_file_ops.py generate-fixtures /tmp/file_ops
    uv run python scripts/bench_file_ops.py oracle /tmp/file_ops
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


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


# --- task 2: replace foo→bar across 60 .py files ----------------------------


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
    .py file after a global ``foo`` → ``bar`` substitution. Sorted by
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
    action types, a synthetic ts and a synthetic amount (cents → divide
    by 100 so it serialises cleanly)."""
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
        name="Replace foo→bar across 60 .py files",
        description=(
            "Rewrite every `foo` substring to `bar` in all .py files "
            "under tests_ws/. Scoring artifact: one ``<relpath> "
            "<sha256[:16]>`` line per file, sorted by path."
        ),
        prompts=(
            "Rename all occurrences of foo to bar across the tests_ws directory.",
            "Replace foo with bar in every .py file under tests_ws/.",
            "Bulk-rename foo→bar in tests_ws/**/*.py.",
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


# --- CLI --------------------------------------------------------------------


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cmd_generate(args: argparse.Namespace) -> None:
    target = Path(args.target).resolve()
    if target.exists():
        if not args.force:
            raise SystemExit(f"{target} already exists. Pass --force to wipe it first.")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for task in BENCH_TASKS:
        task_dir = target / task.id
        task_dir.mkdir()
        task.fixture_fn(task_dir)
        print(f"{task.id:25s} → {task_dir}")


def cmd_oracle(args: argparse.Namespace) -> None:
    target = Path(args.target).resolve()
    for task in BENCH_TASKS:
        task_dir = target / task.id
        if not task_dir.exists():
            print(f"{task.id:25s} (missing — run generate-fixtures first)")
            continue
        gold = task.oracle_fn(task_dir)
        digest = _digest(gold)
        print(f"{task.id:25s} sha256={digest[:16]}  size={len(gold)}B  lines={gold.count(chr(10))}")


def cmd_list(_args: argparse.Namespace) -> None:
    for task in BENCH_TASKS:
        print(f"=== {task.id} ===")
        print(f"  {task.name}")
        wrapped = textwrap.wrap(
            task.description,
            width=78,
            initial_indent="  ",
            subsequent_indent="  ",
        )
        for line in wrapped:
            print(line)
        print("  prompts:")
        for prompt in task.prompts:
            print(f"    - {prompt}")
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 of harness-bw27 — fixture + oracle definitions for "
            "the file-manipulation tool bench. Generate fixtures into a "
            "scratch dir, run oracles to confirm digests, or list the "
            "tasks and their prompts."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate-fixtures", help="Materialize input fixtures.")
    gen.add_argument("target", help="Directory to write fixtures into.")
    gen.add_argument("--force", action="store_true", help="Wipe target if it exists.")
    gen.set_defaults(func=cmd_generate)

    oracle = sub.add_parser("oracle", help="Run oracles against an existing fixture dir.")
    oracle.add_argument("target", help="Directory containing materialized fixtures.")
    oracle.set_defaults(func=cmd_oracle)

    ls = sub.add_parser("list-tasks", help="Print the task table.")
    ls.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
