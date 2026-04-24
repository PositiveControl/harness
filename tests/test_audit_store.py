from __future__ import annotations

from pathlib import Path

import pytest

from harness.store.audit import AuditStore
from harness.tools.base import ToolHit


def _store(tmp_path: Path) -> AuditStore:
    return AuditStore(tmp_path / "harness.sqlite")


def test_record_minimal_turn(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rec = store.record(
        session="s1",
        character="airton_c1",
        user_id="mark",
        user_message="hi",
        model_reply="hello",
    )
    assert rec.id > 0
    assert rec.session == "s1"
    assert rec.character == "airton_c1"
    assert rec.user_message == "hi"
    assert rec.model_reply == "hello"
    assert rec.retrieval_top_score is None
    assert rec.retrieval_hits == ()
    assert rec.citations_grounded == frozenset()
    assert rec.citations_cited == frozenset()
    assert rec.tools_ran == frozenset()
    assert len(rec.turn_id) >= 8  # short UUID


def test_record_with_retrieval_hits_derives_top_score(tmp_path: Path) -> None:
    store = _store(tmp_path)
    hits = (
        ToolHit(source="episodic", external_id="a", title="A", score=0.42),
        ToolHit(source="episodic", external_id="b", title="B", score=0.81),
        ToolHit(source="episodic", external_id="c", title="C", score=0.55),
    )
    rec = store.record(
        session="s1",
        character="airton_c1",
        user_id=None,
        user_message="q",
        model_reply="r",
        retrieval_hits=hits,
    )
    assert rec.retrieval_top_score == 0.81
    assert rec.retrieval_hits == hits


def test_record_preserves_citations_and_tools(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rec = store.record(
        session="s1",
        character="airton_c1",
        user_id="mark",
        user_message="q",
        model_reply="r",
        citations_grounded=frozenset({"§4-1-1", "TBL 4-1-2"}),
        citations_cited=frozenset({"§4-1-1"}),
        tools_ran=frozenset({"search_memory"}),
    )
    # Round-trip via get() to ensure JSON serialisation is symmetric.
    fetched = store.get(rec.id)
    assert fetched.citations_grounded == frozenset({"§4-1-1", "TBL 4-1-2"})
    assert fetched.citations_cited == frozenset({"§4-1-1"})
    assert fetched.tools_ran == frozenset({"search_memory"})


def test_record_roundtrip_tool_hits(tmp_path: Path) -> None:
    store = _store(tmp_path)
    hits = (
        ToolHit(
            source="episodic",
            external_id="x",
            title="T",
            score=0.7,
            principle="P",
        ),
        ToolHit(source="web", external_id=None, title="U", score=0.3),
    )
    rec = store.record(
        session="s",
        character="c",
        user_id=None,
        user_message="q",
        model_reply="r",
        retrieval_hits=hits,
    )
    fetched = store.get(rec.id)
    assert fetched.retrieval_hits == hits


def test_list_filters_by_session_and_orders_newest_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for i in range(3):
        store.record(
            session="sA",
            character="c",
            user_id=None,
            user_message=f"mA{i}",
            model_reply="r",
        )
    for i in range(2):
        store.record(
            session="sB",
            character="c",
            user_id=None,
            user_message=f"mB{i}",
            model_reply="r",
        )
    rows_a = store.list(session="sA")
    rows_b = store.list(session="sB")
    assert [r.user_message for r in rows_a] == ["mA2", "mA1", "mA0"]
    assert [r.user_message for r in rows_b] == ["mB1", "mB0"]


def test_list_across_sessions_when_no_filter(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(
        session="sA", character="c", user_id=None, user_message="a", model_reply="r"
    )
    store.record(
        session="sB", character="c", user_id=None, user_message="b", model_reply="r"
    )
    rows = store.list()
    assert len(rows) == 2
    assert {r.user_message for r in rows} == {"a", "b"}


def test_list_respects_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for i in range(10):
        store.record(
            session="s", character="c", user_id=None, user_message=f"m{i}", model_reply="r"
        )
    assert len(store.list(limit=3)) == 3


def test_record_accepts_explicit_turn_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rec = store.record(
        session="s",
        character="c",
        user_id=None,
        user_message="q",
        model_reply="r",
        turn_id="custom-12345",
    )
    assert rec.turn_id == "custom-12345"


def test_get_raises_on_missing_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.get(9999)


def test_user_id_nullable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rec = store.record(
        session="s", character="c", user_id=None, user_message="q", model_reply="r"
    )
    fetched = store.get(rec.id)
    assert fetched.user_id is None


def test_persists_across_connections(tmp_path: Path) -> None:
    db = tmp_path / "harness.sqlite"
    store1 = AuditStore(db)
    store1.record(
        session="s", character="c", user_id=None, user_message="q", model_reply="r"
    )
    store1.close()
    store2 = AuditStore(db)
    rows = store2.list()
    assert len(rows) == 1
    assert rows[0].user_message == "q"
