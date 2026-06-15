"""Tests for src/harness/driver/state.py — harness-ej42."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from harness.driver.state import LoopRunState


def _fresh(epic_id: str = "harness-e9oq", *, max_turns: int = 20) -> LoopRunState:
    return LoopRunState.fresh(epic_id=epic_id, max_turns=max_turns, started_at_sha="deadbeef")


def test_fresh_populates_required_fields() -> None:
    s = _fresh()
    assert len(s.loop_run_id) == 8
    assert s.started_at_sha == "deadbeef"
    assert s.epic_id == "harness-e9oq"
    assert s.max_turns == 20
    assert s.turns_used == 0
    assert s.closed_this_run == []
    assert s.attempt_counts == {}
    assert s.last_failure == {}
    assert s.started_at.tzinfo is UTC


def test_state_path_layout(tmp_path: Path) -> None:
    s = _fresh()
    expected = tmp_path / ".harness" / "loop_runs" / f"{s.loop_run_id}.json"
    assert LoopRunState.state_path(tmp_path, s.loop_run_id) == expected
    assert LoopRunState.state_dir(tmp_path) == expected.parent


def test_save_then_load_round_trips_every_field(tmp_path: Path) -> None:
    s = _fresh()
    s.turns_used = 7
    s.closed_this_run.extend(["harness-aaa", "harness-bbb"])
    s.attempt_counts["harness-ccc"] = 2
    s.attempt_counts["harness-ddd"] = 1
    s.last_failure["harness-ccc"] = "fabrication_fallback fired"
    # harness-smplj: cross-turn gate-suspect tail persists per issue.
    s.last_test_fail_tail["harness-ccc"] = "TypeError: Cannot set properties of undefined"

    path = LoopRunState.state_path(tmp_path, s.loop_run_id)
    s.save(path)

    loaded = LoopRunState.load(path)
    assert loaded.loop_run_id == s.loop_run_id
    assert loaded.started_at_sha == s.started_at_sha
    assert loaded.started_at == s.started_at
    assert loaded.epic_id == s.epic_id
    assert loaded.max_turns == s.max_turns
    assert loaded.turns_used == s.turns_used
    assert loaded.closed_this_run == s.closed_this_run
    assert loaded.attempt_counts == s.attempt_counts
    assert loaded.last_failure == s.last_failure
    assert loaded.last_test_fail_tail == s.last_test_fail_tail


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    # state_dir doesn't exist yet — save() must mkdir it.
    s = _fresh()
    path = LoopRunState.state_path(tmp_path, s.loop_run_id)
    assert not path.parent.exists()
    s.save(path)
    assert path.exists()


def test_load_raises_filenotfound_for_missing_run(tmp_path: Path) -> None:
    path = LoopRunState.state_path(tmp_path, "no-such-id")
    with pytest.raises(FileNotFoundError):
        LoopRunState.load(path)


def test_atomic_save_survives_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a crash after the tmp file is written but before
    # Path.replace() finalises the swap. The previously-persisted file
    # must remain intact (atomic swap contract). A stale `.tmp` sibling
    # may remain on disk — it gets overwritten on the next save.
    s = _fresh()
    path = LoopRunState.state_path(tmp_path, s.loop_run_id)
    s.save(path)
    baseline = path.read_text()

    s.turns_used = 999  # the mutation we're trying (and failing) to persist

    original_replace = Path.replace

    def boom(self: Path, target: str | Path) -> Path:
        raise OSError("simulated crash mid-replace")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError, match="simulated crash"):
        s.save(path)

    # Previous file untouched.
    assert path.read_text() == baseline
    assert LoopRunState.load(path).turns_used == 0
    # A stale tmp file may exist; that's fine — it'll be overwritten
    # by the next successful save.
    monkeypatch.setattr(Path, "replace", original_replace)


def test_load_tolerates_omitted_optional_fields(tmp_path: Path) -> None:
    # Older snapshots may predate fields with defaults. Synthesise a
    # minimal-shape JSON and confirm load() fills defaults rather than
    # raising KeyError.
    minimal: dict[str, Any] = {
        "loop_run_id": "abcd1234",
        "started_at_sha": "cafef00d",
        "started_at": datetime.now(UTC).isoformat(),
        "epic_id": "harness-e9oq",
        "max_turns": 20,
    }
    path = LoopRunState.state_path(tmp_path, "abcd1234")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(minimal))

    s = LoopRunState.load(path)
    assert s.turns_used == 0
    assert s.closed_this_run == []
    assert s.attempt_counts == {}
    assert s.last_failure == {}


def test_fresh_generates_distinct_ids() -> None:
    ids = {LoopRunState.fresh("e", 1, "sha")._to_dict()["loop_run_id"] for _ in range(50)}
    assert len(ids) == 50
