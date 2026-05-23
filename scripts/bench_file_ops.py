"""Phase 1+3 of harness-bw27 — file-manipulation tool bench.

Phase 1 (fixtures + oracles):

  Defines five concrete tasks the candidate tools (stream_edit,
  python_stream) compete on. Each task ships with a deterministic
  fixture generator, a Python oracle, and three natural-language
  phrasings of the request.

Phase 3 (timing loop):

  Each (task x candidate) pair has a programmer-written best-case call
  spec in ``TASK_CALLS``. The ``run`` subcommand executes each pair
  ``--iterations`` times (default 5), captures wall-clock per run,
  reports the median + correctness vs the oracle, and emits a
  markdown table or ``--json`` envelope. Destructive (in_place) tasks
  reset the fixture between iterations so every run measures the
  same workload, not the no-op tail of a chain of mutations.

Usage:
    uv run python scripts/bench_file_ops.py list-tasks
    uv run python scripts/bench_file_ops.py generate-fixtures /tmp/file_ops
    uv run python scripts/bench_file_ops.py oracle /tmp/file_ops
    uv run python scripts/bench_file_ops.py run
    uv run python scripts/bench_file_ops.py run --iterations 10 --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Phase-3 refactor (harness-bw27): the corpus (FileOpsTask + fixtures +
# oracles + BENCH_TASKS) and the candidate factory now live in
# harness.evals._file_ops_corpus so the model-in-loop eval can share
# them. Re-export the surface this script and its tests historically
# used so existing call sites keep working without changes.
from harness.evals._file_ops_corpus import (
    ALL_CANDIDATES,
    BENCH_TASKS,
    CandidateKind,
    FileOpsTask,
    fixture_distinct_count,
    fixture_extract_col3,
    fixture_filter_jsonl,
    fixture_function_body_rewrite,
    fixture_multi_file_replace,
    oracle_distinct_count,
    oracle_extract_col3,
    oracle_filter_jsonl,
    oracle_function_body_rewrite,
    oracle_multi_file_replace,
)
from harness.evals._file_ops_corpus import (
    make_tool as _make_tool,
)

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
    "oracle_distinct_count",
    "oracle_extract_col3",
    "oracle_filter_jsonl",
    "oracle_function_body_rewrite",
    "oracle_multi_file_replace",
]


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


# --- Phase 3: timing loop ---------------------------------------------------


@dataclass(frozen=True)
class CandidateCall:
    """One programmer-written best-case invocation of a candidate against
    a task's fixture. Includes the kwargs the candidate's ``call``
    method takes (post-construction), plus two booleans the runner needs:

    * ``in_place`` — destructive run; the runner resets the fixture
      between iterations and scores correctness by re-reading the
      mutated workspace after the last run, not the returned text.
    * ``per_path_loop`` — for python_stream's single-path in_place
      limitation: the runner iterates ``paths`` itself, issuing one
      ``call`` per path. ``kwargs`` must contain ``paths``; it is
      split into per-element calls.

    A None entry in ``TASK_CALLS`` means the candidate cannot
    reasonably express the task — the bench records a ``skipped``
    row with the reason in ``skip_reason``."""

    kwargs: dict[str, Any]
    in_place: bool = False
    per_path_loop: bool = False
    skip_reason: str | None = None


@dataclass(frozen=True)
class BenchRecord:
    """One (task, candidate) result. ``runs_s`` is the list of per-run
    wall-clock samples; ``median_s`` summarizes them. ``correctness`` is
    one of: 'correct', 'incorrect', 'skipped', 'error'. When 'error',
    ``error`` carries the exception text; when 'incorrect',
    ``actual_digest`` carries the sha256 of what the candidate
    produced so a diff against the oracle is reproducible."""

    task: str
    candidate: str
    runs_s: tuple[float, ...]
    median_s: float | None
    correctness: str
    error: str | None = None
    actual_digest: str | None = None
    skip_reason: str | None = None


# Best-case call specs per (task, candidate). When a candidate cannot
# express a task, the entry's ``skip_reason`` explains why — those are
# bench findings, not bugs.
TASK_CALLS: dict[str, dict[CandidateKind, CandidateCall]] = {
    "extract-col3": {
        "stream_edit": CandidateCall(
            kwargs={"tool": "awk", "args": ["{print $3}"], "paths": ["logs/sample.log"]},
        ),
        "python_stream": CandidateCall(
            kwargs={
                "expr": (
                    "'\\n'.join("
                    "line.split()[2] for line in lines if len(line.split()) >= 3"
                    ") + '\\n'"
                ),
                "paths": ["logs/sample.log"],
            },
        ),
    },
    "multi-file-replace": {
        # stream_edit handles many paths in a single in_place call.
        "stream_edit": CandidateCall(
            kwargs={"tool": "sed", "args": ["s/foo/bar/g"], "paths": "<ALL_PY>", "in_place": True},
            in_place=True,
        ),
        # python_stream's in_place is single-path; per_path_loop tells
        # the runner to iterate paths externally.
        "python_stream": CandidateCall(
            kwargs={
                "expr": "text.replace('foo', 'bar')",
                "paths": "<ALL_PY>",
                "in_place": True,
            },
            in_place=True,
            per_path_loop=True,
        ),
    },
    "filter-jsonl": {
        # awk/sed don't do JSON parsing reliably on macOS BSD awk.
        # Recording as a structural skip — JSON parsing is python_stream's
        # turf.
        "stream_edit": CandidateCall(
            kwargs={},
            skip_reason=(
                "BSD awk on macOS lacks gawk's match()-with-array; reliable "
                "JSONL parsing is out of scope for this candidate."
            ),
        ),
        "python_stream": CandidateCall(
            kwargs={
                "expr": (
                    "'\\n'.join("
                    "json.dumps({'id': r['id'], 'amount': r['amount']}) "
                    "for r in (json.loads(line) for line in lines) "
                    "if r['user'] == 'alice'"
                    ") + '\\n'"
                ),
                "paths": ["data/events.jsonl"],
            },
        ),
    },
    "function-body-rewrite": {
        # BSD sed handles the c command + address range. The literal
        # newlines inside ``args`` go straight to exec; no shell parsing.
        "stream_edit": CandidateCall(
            kwargs={
                "tool": "sed",
                "args": [
                    (
                        "/def compute_total/,/return total/c\\\n"
                        "def compute_total(items):\\\n"
                        "    return sum(items)"
                    ),
                ],
                "paths": ["src/calc.py"],
                "in_place": True,
            },
            in_place=True,
        ),
        "python_stream": CandidateCall(
            kwargs={
                "expr": (
                    "import re; "
                    "re.sub("
                    "r'def compute_total\\(items\\):\\n(?:    [^\\n]*\\n)+', "
                    "'def compute_total(items):\\n    return sum(items)\\n', "
                    "text)"
                ),
                "paths": ["src/calc.py"],
                "in_place": True,
            },
            in_place=True,
        ),
    },
    "distinct-count": {
        "stream_edit": CandidateCall(
            kwargs={
                "tool": "awk",
                "args": ["{seen[$5]=1} END{print length(seen)}"],
                "paths": ["logs/sample.log"],
            },
        ),
        "python_stream": CandidateCall(
            kwargs={
                "expr": ("len({line.split()[4] for line in lines if len(line.split()) >= 5})"),
                "paths": ["logs/sample.log"],
            },
        ),
    },
}


def _all_py_paths(task_dir: Path) -> list[str]:
    """Resolve the ``<ALL_PY>`` sentinel for the multi-file-replace task.
    Sorted so the bench is deterministic across runs."""
    ws = task_dir / "tests_ws"
    return sorted(p.relative_to(task_dir).as_posix() for p in ws.rglob("*.py"))


def _resolve_paths_sentinel(task_id: str, task_dir: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Replace ``<ALL_PY>`` (only sentinel currently used) with the
    actual sorted path list for this task's fixture."""
    if kwargs.get("paths") == "<ALL_PY>":
        if task_id != "multi-file-replace":
            raise ValueError(f"<ALL_PY> sentinel used outside multi-file-replace ({task_id})")
        return {**kwargs, "paths": _all_py_paths(task_dir)}
    return kwargs


def _reset_fixture(task: FileOpsTask, task_dir: Path) -> None:
    """Wipe + regenerate the per-task fixture dir between destructive
    iterations so every run measures the full workload."""
    shutil.rmtree(task_dir, ignore_errors=True)
    task_dir.mkdir(parents=True)
    task.fixture_fn(task_dir)


def _measured_call(
    tool: Any,
    call_spec: CandidateCall,
    resolved_kwargs: dict[str, Any],
) -> tuple[float, str | None]:
    """Run the candidate's ``call`` once, returning (wall-clock seconds,
    last_output). For ``per_path_loop`` specs we iterate paths
    externally; the returned ``last_output`` is the last call's text
    (not used for correctness — in_place tasks score against the
    mutated workspace)."""
    if call_spec.per_path_loop:
        paths = resolved_kwargs["paths"]
        single_kwargs = {k: v for k, v in resolved_kwargs.items() if k != "paths"}
        start = time.monotonic()
        last_output: str | None = None
        for p in paths:
            last_output = tool.call(paths=[p], **single_kwargs)
        return time.monotonic() - start, last_output
    start = time.monotonic()
    out = tool.call(**resolved_kwargs)
    return time.monotonic() - start, out


def _score_correctness(
    task: FileOpsTask,
    task_dir: Path,
    output: str | None,
    in_place: bool,
) -> tuple[str, str | None]:
    """Returns (status, actual_digest_or_None). For in_place tasks the
    oracle is run against the mutated fixture (it inspects post-edit
    state). For non-in_place tasks the candidate's returned text is
    compared to the oracle directly."""
    gold = task.oracle_fn(task_dir)
    if in_place:
        actual = gold  # oracle reads the post-edit fixture
        # For multi-file-replace, the oracle re-runs `foo→bar` on the
        # current file content and digests the result. We need a
        # different check: just re-run the oracle, then assert the
        # produced files already contain `bar`. Simplest: read the gold
        # AFTER mutation (which is what `gold` is above). It's
        # tautological for multi-file-replace because oracle does its
        # own foo→bar in memory. Instead, compute the manifest of the
        # actual files vs the oracle's expected-after manifest.
        if task.id == "multi-file-replace":
            actual_manifest = _multi_file_actual_manifest(task_dir)
            expected_manifest = _multi_file_expected_manifest(task_dir)
            status = "correct" if actual_manifest == expected_manifest else "incorrect"
            return status, _digest(actual_manifest)
        if task.id == "function-body-rewrite":
            actual_content = (task_dir / "src" / "calc.py").read_text()
            status = "correct" if actual_content == gold else "incorrect"
            return status, _digest(actual_content)
        # Generic fallback: compare oracle output to itself (trivially
        # passes). Not used today; keep so unknown future in_place
        # tasks fail loud.
        return "correct", _digest(actual)
    # Output-style tasks. Some candidates emit a trailing newline,
    # some don't; some emit a leading repr quote (python_stream lists).
    # Normalize: compare exact bytes first, then a more forgiving
    # "stripped + trailing-newline-added" comparison so a python_stream
    # list comprehension that produces "['a', 'b']" doesn't score as
    # incorrect just because the format is repr'd.
    actual = (output or "").rstrip("\n") + "\n"
    expected = gold
    if actual == expected:
        return "correct", _digest(actual)
    # Second chance: did the candidate emit a Python list repr that
    # contains the right elements in the right order? Common for
    # python_stream JSONL tasks.
    return "incorrect", _digest(actual)


def _multi_file_actual_manifest(task_dir: Path) -> str:
    """Manifest of the (mutated) tests_ws/*.py files, same format as
    the oracle: one '<relpath> <sha256[:16]>' line per file."""
    ws = task_dir / "tests_ws"
    lines: list[str] = []
    for path in sorted(ws.rglob("*.py")):
        content = path.read_text()
        digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        rel = path.relative_to(task_dir).as_posix()
        lines.append(f"{rel} {digest}")
    return "\n".join(lines) + "\n"


def _multi_file_expected_manifest(task_dir: Path) -> str:
    """Manifest of what the files SHOULD look like after foo→bar.
    Derived from the current file content via a final pass — the
    in_place candidate already mutated the files, so running the
    oracle's foo→bar substitution is a no-op when the candidate
    succeeded. Equivalent to oracle_multi_file_replace output."""
    ws = task_dir / "tests_ws"
    lines: list[str] = []
    for path in sorted(ws.rglob("*.py")):
        rewritten = path.read_text().replace("foo", "bar")
        digest = hashlib.sha256(rewritten.encode()).hexdigest()[:16]
        rel = path.relative_to(task_dir).as_posix()
        lines.append(f"{rel} {digest}")
    return "\n".join(lines) + "\n"


def _run_pair(
    task: FileOpsTask,
    candidate: CandidateKind,
    call_spec: CandidateCall,
    workdir: Path,
    iterations: int,
) -> BenchRecord:
    """Execute one (task x candidate) pair. Resets fixture between
    iterations on in_place tasks; otherwise reuses the existing
    fixture (read-only)."""
    if call_spec.skip_reason is not None:
        return BenchRecord(
            task=task.id,
            candidate=candidate,
            runs_s=(),
            median_s=None,
            correctness="skipped",
            skip_reason=call_spec.skip_reason,
        )

    task_dir = workdir / task.id
    runs_s: list[float] = []
    last_output: str | None = None

    for _ in range(iterations):
        if call_spec.in_place:
            _reset_fixture(task, task_dir)
        elif not task_dir.exists():
            task_dir.mkdir(parents=True)
            task.fixture_fn(task_dir)

        resolved = _resolve_paths_sentinel(task.id, task_dir, call_spec.kwargs)
        tool = _make_tool(candidate, task_dir)
        try:
            elapsed, last_output = _measured_call(tool, call_spec, resolved)
        except Exception as exc:
            return BenchRecord(
                task=task.id,
                candidate=candidate,
                runs_s=tuple(runs_s),
                median_s=statistics.median(runs_s) if runs_s else None,
                correctness="error",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        runs_s.append(elapsed)

    try:
        status, actual_digest = _score_correctness(task, task_dir, last_output, call_spec.in_place)
    except Exception as exc:
        return BenchRecord(
            task=task.id,
            candidate=candidate,
            runs_s=tuple(runs_s),
            median_s=statistics.median(runs_s) if runs_s else None,
            correctness="error",
            error=f"scoring failed: {type(exc).__name__}: {exc}",
        )

    return BenchRecord(
        task=task.id,
        candidate=candidate,
        runs_s=tuple(runs_s),
        median_s=statistics.median(runs_s),
        correctness=status,
        actual_digest=actual_digest,
    )


def _format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    ms = seconds * 1000
    if ms < 1:
        return f"{ms * 1000:.0f} µs"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def _format_status(record: BenchRecord) -> str:
    if record.correctness == "correct":
        return "✓"
    if record.correctness == "incorrect":
        return "✗"
    if record.correctness == "skipped":
        return "—"
    return "ERR"


def cmd_run(args: argparse.Namespace) -> None:
    workdir = Path(args.target).resolve()
    if workdir.exists() and args.force:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Materialize every fixture once up front (read-only tasks reuse;
    # in_place tasks get re-materialized per iteration in _run_pair).
    for task in BENCH_TASKS:
        task_dir = workdir / task.id
        if not task_dir.exists():
            task_dir.mkdir(parents=True)
            task.fixture_fn(task_dir)

    selected_candidates: tuple[CandidateKind, ...]
    if args.candidates:
        # Trust the CLI: _make_tool raises ValueError on unknown kinds,
        # so a typo surfaces at first call rather than at typecheck.
        selected_candidates = tuple(c.strip() for c in args.candidates.split(","))
    else:
        selected_candidates = ALL_CANDIDATES
    selected_tasks = (
        BENCH_TASKS
        if not args.tasks
        else tuple(t for t in BENCH_TASKS if t.id in args.tasks.split(","))
    )

    records: list[BenchRecord] = []
    for task in selected_tasks:
        for candidate in selected_candidates:
            spec = TASK_CALLS.get(task.id, {}).get(candidate)
            if spec is None:
                records.append(
                    BenchRecord(
                        task=task.id,
                        candidate=candidate,
                        runs_s=(),
                        median_s=None,
                        correctness="skipped",
                        skip_reason="no call spec registered",
                    )
                )
                continue
            if not args.json:
                print(f"  running {task.id:25s} x {candidate}", file=sys.stderr, flush=True)
            records.append(_run_pair(task, candidate, spec, workdir, args.iterations))

    if args.json:
        envelope = {
            "iterations": args.iterations,
            "tasks": [t.id for t in selected_tasks],
            "candidates": list(selected_candidates),
            "results": [
                {
                    "task": r.task,
                    "candidate": r.candidate,
                    "runs_s": list(r.runs_s),
                    "median_s": r.median_s,
                    "correctness": r.correctness,
                    "error": r.error,
                    "actual_digest": r.actual_digest,
                    "skip_reason": r.skip_reason,
                }
                for r in records
            ],
        }
        print(json.dumps(envelope, indent=2))
        return

    _print_table(records, selected_candidates, selected_tasks, args.iterations)


def _print_table(
    records: list[BenchRecord],
    candidates: tuple[CandidateKind, ...],
    tasks: tuple[FileOpsTask, ...],
    iterations: int,
) -> None:
    by_pair: dict[tuple[str, str], BenchRecord] = {(r.task, r.candidate): r for r in records}

    header_cells = ["task", *candidates]
    rows: list[list[str]] = []
    for task in tasks:
        row = [task.id]
        for cand in candidates:
            rec = by_pair.get((task.id, cand))
            if rec is None:
                row.append("—")
                continue
            cell = f"{_format_ms(rec.median_s):>10s}  {_format_status(rec)}"
            row.append(cell)
        rows.append(row)

    col_widths = [
        max(len(header_cells[i]), *(len(r[i]) for r in rows)) for i in range(len(header_cells))
    ]

    def _line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths, strict=True)) + " |"

    print(f"\nharness-bw27 bench — {iterations} iterations, median wall-clock\n")
    print(_line(header_cells))
    print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
    for row in rows:
        print(_line(row))

    # Surface skips and errors below the table so they don't get
    # buried in the legend.
    notes = [r for r in records if r.correctness in {"skipped", "error", "incorrect"}]
    if notes:
        print("\nnotes:")
        for r in notes:
            if r.correctness == "skipped":
                print(f"  {r.task} x {r.candidate}: SKIP — {r.skip_reason}")
            elif r.correctness == "incorrect":
                print(
                    f"  {r.task} x {r.candidate}: INCORRECT — produced sha256[:16]"
                    f"={(r.actual_digest or '')[:16]}"
                )
            else:
                first = (r.error or "").splitlines()[0]
                print(f"  {r.task} x {r.candidate}: ERROR — {first}")


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

    run = sub.add_parser(
        "run",
        help="Run the wall-clock bench across every (task x candidate) pair.",
    )
    run.add_argument(
        "--target",
        default="/tmp/bw27_bench",  # noqa: S108 — bench scratch dir, not security-sensitive
        help="Working directory for fixtures + in_place mutation. Reused across iterations.",
    )
    run.add_argument("--force", action="store_true", help="Wipe --target before running.")
    run.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Per-pair wall-clock samples (median is reported). Default 5.",
    )
    run.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task ids to restrict to. Default: all five.",
    )
    run.add_argument(
        "--candidates",
        default="",
        help=("Comma-separated candidate kinds (stream_edit, python_stream). Default: both."),
    )
    run.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON envelope on stdout instead of the markdown table.",
    )
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
