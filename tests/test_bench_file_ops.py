"""Pin the Phase-1 file-ops bench fixtures + oracles against drift.

Each fixture generator must produce byte-identical output across runs and
across machines; each oracle must produce a digest matching the pin
below. If a fixture changes shape (different seed, different format) or
an oracle's semantics shift, the affected pin moves — but it must be a
deliberate edit, not silent drift, because the Phase-3 model-in-loop
eval scores candidate-tool outputs against these exact bytes.

Pins were captured on 2026-05-22 by running:

    uv run python scripts/bench_file_ops.py generate-fixtures /tmp/x --force
    uv run python scripts/bench_file_ops.py oracle /tmp/x
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_file_ops.py"
_SPEC = importlib.util.spec_from_file_location("bench_file_ops", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
bench_file_ops: Any = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_file_ops"] = bench_file_ops
_SPEC.loader.exec_module(bench_file_ops)

BENCH_TASKS: tuple[Any, ...] = bench_file_ops.BENCH_TASKS


# (task_id, oracle_sha256_full, oracle_size_bytes, oracle_line_count).
# Captured 2026-05-22 — drift is a deliberate edit, not silent.
EXPECTED_ORACLE: dict[str, tuple[str, int, int]] = {
    "extract-col3": (
        "413fe9fabbdb9700b05498a5ae5ef0de62dadba9cc0709905b060c54f193a6fd",
        1_671_046,
        100_000,
    ),
    "multi-file-replace": (
        "597ee6828a7c28de86f21cc2758265f6d83ef67bbe2a7ed2cb0dfa773e76531a",
        2_510,
        60,
    ),
    "filter-jsonl": (
        "92afea085e3660d462db2660571f395a90f1585e089a0629cb8e934e9f6b3b64",
        98_302,
        3_313,
    ),
    "function-body-rewrite": (
        "c40a8bd5d2859bc39c86aad86def0287ee3ad3cb894b9e3a03fc06e0f461e53d",
        115,
        9,
    ),
    "distinct-count": (
        "83c02ac2d48c863dab2ccf6870455aadfc2cec073b8db269b517c879d76aa6d9",
        5,
        1,
    ),
}


def _full_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(scope="module")
def fixtures_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize every task's fixture exactly once per test session."""
    root = tmp_path_factory.mktemp("bench_file_ops")
    for task in BENCH_TASKS:
        task_dir = root / task.id
        task_dir.mkdir()
        task.fixture_fn(task_dir)
    return root


def test_task_ids_unique() -> None:
    ids = [task.id for task in BENCH_TASKS]
    assert len(ids) == len(set(ids)), "task ids must be unique"


def test_each_task_has_three_prompts() -> None:
    for task in BENCH_TASKS:
        assert len(task.prompts) == 3, f"{task.id}: expected 3 prompts, got {len(task.prompts)}"
        assert all(p.strip() for p in task.prompts), f"{task.id}: empty prompt"


def test_fixture_is_deterministic(tmp_path: Path) -> None:
    """Run each fixture twice into separate dirs — every file must be
    byte-identical. Catches accidental nondeterminism (e.g. iterating a
    set, calling rng before seeding, picking up wall-clock time)."""
    for task in BENCH_TASKS:
        a = tmp_path / f"{task.id}_a"
        b = tmp_path / f"{task.id}_b"
        a.mkdir()
        b.mkdir()
        task.fixture_fn(a)
        task.fixture_fn(b)
        files_a = {p.relative_to(a): p.read_bytes() for p in sorted(a.rglob("*")) if p.is_file()}
        files_b = {p.relative_to(b): p.read_bytes() for p in sorted(b.rglob("*")) if p.is_file()}
        assert files_a.keys() == files_b.keys(), f"{task.id}: file set differs"
        for rel, content in files_a.items():
            assert files_b[rel] == content, f"{task.id}: {rel} content differs"


@pytest.mark.parametrize("task", BENCH_TASKS, ids=lambda t: str(t.id))
def test_oracle_digest_pinned(task: Any, fixtures_root: Path) -> None:
    """The oracle output for each task must match the recorded digest.

    Pin drift is allowed but must be deliberate — when this fails, the
    fix is to regenerate the pins from a clean fixture run and confirm
    the new bytes are the ones we want to score against."""
    task_dir = fixtures_root / task.id
    gold = task.oracle_fn(task_dir)
    expected_digest, expected_size, expected_lines = EXPECTED_ORACLE[task.id]
    assert len(gold) == expected_size, (
        f"{task.id}: oracle size drifted ({len(gold)} vs {expected_size})"
    )
    assert gold.count("\n") == expected_lines, (
        f"{task.id}: oracle line count drifted ({gold.count(chr(10))} vs {expected_lines})"
    )
    actual_digest = _full_digest(gold)
    assert actual_digest == expected_digest, (
        f"{task.id}: oracle digest drifted (got {actual_digest})"
    )


def test_extract_col3_oracle_shape(fixtures_root: Path) -> None:
    """Sanity: the first 5 oracle lines must look like host fields."""
    task = next(t for t in BENCH_TASKS if t.id == "extract-col3")
    gold = task.oracle_fn(fixtures_root / task.id)
    first = gold.splitlines()[:5]
    for host in first:
        assert host.startswith("ip-10-"), f"unexpected col-3 value: {host!r}"


def test_multi_file_replace_manifest_format(fixtures_root: Path) -> None:
    """Manifest format: ``<relpath> <16-hex>`` per line, 60 lines."""
    task = next(t for t in BENCH_TASKS if t.id == "multi-file-replace")
    gold = task.oracle_fn(fixtures_root / task.id)
    lines = gold.strip().split("\n")
    assert len(lines) == 60
    for line in lines:
        path, digest = line.split(" ")
        assert path.startswith("tests_ws/pkg_")
        assert path.endswith(".py")
        assert len(digest) == 16
        int(digest, 16)  # raises if not hex


def test_filter_jsonl_alice_fraction_in_range(fixtures_root: Path) -> None:
    """Three uniform users → ~33% alice. Tight bounds catch the case
    where the RNG drifts and the dataset stops looking like itself."""
    task = next(t for t in BENCH_TASKS if t.id == "filter-jsonl")
    gold = task.oracle_fn(fixtures_root / task.id)
    n_alice = len(gold.strip().split("\n"))
    assert 3_200 <= n_alice <= 3_500, f"alice count {n_alice} outside expected band"


def test_function_body_rewrite_oracle_is_exact_after_state() -> None:
    """The oracle ignores its root arg — confirm it returns the canonical
    post-edit string verbatim."""
    task = next(t for t in BENCH_TASKS if t.id == "function-body-rewrite")
    gold = task.oracle_fn(Path("/nonexistent"))  # arg is unused
    assert "def compute_total(items):\n    return sum(items)\n" in gold
    assert "def other():\n    return 42\n" in gold


def test_distinct_count_oracle_is_integer(fixtures_root: Path) -> None:
    task = next(t for t in BENCH_TASKS if t.id == "distinct-count")
    gold = task.oracle_fn(fixtures_root / task.id)
    n = int(gold.strip())
    # 100k lines * user-K with K in [0,999] -> close to 1000 distinct.
    assert 950 <= n <= 1_000


# --- Phase 3: timing-loop scaffolding --------------------------------------


TASK_CALLS = bench_file_ops.TASK_CALLS
ALL_CANDIDATES = bench_file_ops.ALL_CANDIDATES


def test_task_calls_has_entry_for_every_pair() -> None:
    """Every (task, candidate) pair must be either a real call spec or a
    documented skip. Missing entries default to a generic skip in the
    runner; tightening here so a new task added to BENCH_TASKS doesn't
    silently get scored as 'no call spec registered'."""
    for task in BENCH_TASKS:
        candidates = TASK_CALLS.get(task.id, {})
        for candidate in ALL_CANDIDATES:
            assert candidate in candidates, f"missing TASK_CALLS[{task.id}][{candidate}]"
            spec = candidates[candidate]
            if spec.skip_reason is not None:
                assert spec.skip_reason.strip(), f"{task.id} {candidate}: empty skip_reason"
            else:
                assert spec.kwargs, f"{task.id} {candidate}: empty kwargs but no skip_reason"


def test_make_tool_constructs_each_candidate(tmp_path: Path) -> None:
    for kind in ALL_CANDIDATES:
        tool = bench_file_ops._make_tool(kind, tmp_path)
        assert hasattr(tool, "spec"), f"{kind}: tool missing spec property"
        assert hasattr(tool, "call"), f"{kind}: tool missing call method"


def test_make_tool_uses_bumped_output_cap(tmp_path: Path) -> None:
    """The bench bumps each candidate's max_output_bytes well past the
    512 KB model-protection default so digest checks see the full
    transform."""
    for kind in ALL_CANDIDATES:
        tool = bench_file_ops._make_tool(kind, tmp_path)
        assert tool.max_output_bytes >= 8 * 1024 * 1024, f"{kind}: bench cap not bumped"


def test_all_py_paths_returns_sorted_relpaths(tmp_path: Path) -> None:
    """The <ALL_PY> sentinel expands to a deterministic sorted list of
    paths the multi-file-replace candidates can chew on."""
    task = next(t for t in BENCH_TASKS if t.id == "multi-file-replace")
    task.fixture_fn(tmp_path)
    paths = bench_file_ops._all_py_paths(tmp_path)
    assert len(paths) == 60
    assert paths == sorted(paths)
    assert all(p.startswith("tests_ws/pkg_") and p.endswith(".py") for p in paths)


def test_resolve_paths_sentinel_substitutes_all_py(tmp_path: Path) -> None:
    task = next(t for t in BENCH_TASKS if t.id == "multi-file-replace")
    task.fixture_fn(tmp_path)
    resolved = bench_file_ops._resolve_paths_sentinel(
        "multi-file-replace",
        tmp_path,
        {"tool": "sed", "args": ["s/foo/bar/g"], "paths": "<ALL_PY>", "in_place": True},
    )
    assert isinstance(resolved["paths"], list)
    assert len(resolved["paths"]) == 60


def test_resolve_paths_sentinel_passthrough_when_no_sentinel(tmp_path: Path) -> None:
    kwargs = {"tool": "awk", "args": ["{print $3}"], "paths": ["logs/sample.log"]}
    out = bench_file_ops._resolve_paths_sentinel("extract-col3", tmp_path, kwargs)
    assert out["paths"] == ["logs/sample.log"]
    # Other keys preserved as-is.
    assert out["tool"] == "awk"
    assert out["args"] == ["{print $3}"]


def test_run_pair_records_correct(tmp_path: Path) -> None:
    """python_stream + extract-col3 is the canonical 'correct' result —
    no destructive mutation, deterministic oracle. One iteration is
    enough to confirm the scoring path."""
    task = next(t for t in BENCH_TASKS if t.id == "extract-col3")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    task_dir = workdir / task.id
    task_dir.mkdir()
    task.fixture_fn(task_dir)
    spec = TASK_CALLS["extract-col3"]["python_stream"]
    rec = bench_file_ops._run_pair(task, "python_stream", spec, workdir, iterations=1)
    assert rec.correctness == "correct", f"expected correct, got {rec.correctness}: {rec.error}"
    assert rec.median_s is not None
    assert rec.median_s > 0
    assert len(rec.runs_s) == 1


def test_run_pair_records_skipped(tmp_path: Path) -> None:
    task = next(t for t in BENCH_TASKS if t.id == "filter-jsonl")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    spec = TASK_CALLS["filter-jsonl"]["stream_edit"]
    assert spec.skip_reason is not None
    rec = bench_file_ops._run_pair(task, "stream_edit", spec, workdir, iterations=3)
    assert rec.correctness == "skipped"
    assert rec.skip_reason == spec.skip_reason
    assert rec.runs_s == ()
    assert rec.median_s is None


def test_run_pair_records_error_when_call_raises(tmp_path: Path) -> None:
    task = next(t for t in BENCH_TASKS if t.id == "extract-col3")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    task_dir = workdir / task.id
    task_dir.mkdir()
    task.fixture_fn(task_dir)
    # Deliberately wrong shape: stream_edit needs a `tool` key.
    broken = bench_file_ops.CandidateCall(
        kwargs={"args": ["{print $3}"], "paths": ["logs/sample.log"]},
    )
    rec = bench_file_ops._run_pair(task, "stream_edit", broken, workdir, iterations=1)
    assert rec.correctness == "error"
    assert rec.error is not None
    assert "TypeError" in rec.error


def test_run_pair_in_place_resets_fixture_each_iteration(tmp_path: Path) -> None:
    """For multi-file-replace, iteration 2 must see the same starting
    state as iteration 1. Catches a regression where the runner reuses
    the post-mutation fixture and iteration 2 measures a no-op."""
    task = next(t for t in BENCH_TASKS if t.id == "multi-file-replace")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    spec = TASK_CALLS["multi-file-replace"]["stream_edit"]
    rec = bench_file_ops._run_pair(task, "stream_edit", spec, workdir, iterations=3)
    assert rec.correctness == "correct", rec.error
    # Three iterations: each elapsed time should be on the same order
    # of magnitude (no iteration drops to ~0 from a no-op).
    assert len(rec.runs_s) == 3
    min_s, max_s = min(rec.runs_s), max(rec.runs_s)
    assert max_s / max(min_s, 1e-6) < 5.0, (
        f"iteration times diverge: {rec.runs_s} — likely fixture not reset between runs"
    )


def test_format_ms_handles_none_and_ranges() -> None:
    assert bench_file_ops._format_ms(None) == "—"
    assert bench_file_ops._format_ms(0.0005).endswith("µs")
    assert bench_file_ops._format_ms(0.05).endswith("ms")
    assert bench_file_ops._format_ms(1.5).endswith("s")


def test_format_status_maps_correctness() -> None:
    rec_correct = bench_file_ops.BenchRecord(
        task="t", candidate="c", runs_s=(), median_s=None, correctness="correct"
    )
    rec_incorrect = bench_file_ops.BenchRecord(
        task="t", candidate="c", runs_s=(), median_s=None, correctness="incorrect"
    )
    rec_skipped = bench_file_ops.BenchRecord(
        task="t", candidate="c", runs_s=(), median_s=None, correctness="skipped"
    )
    rec_error = bench_file_ops.BenchRecord(
        task="t", candidate="c", runs_s=(), median_s=None, correctness="error"
    )
    assert bench_file_ops._format_status(rec_correct) == "✓"
    assert bench_file_ops._format_status(rec_incorrect) == "✗"
    assert bench_file_ops._format_status(rec_skipped) == "—"
    assert bench_file_ops._format_status(rec_error) == "ERR"
