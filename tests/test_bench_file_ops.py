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
