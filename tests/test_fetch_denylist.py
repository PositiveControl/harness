"""Tests for FetchDenylistStore (harness-4dgm).

Real SQLite in tmp_path per harness convention — no mocks. The store
is small enough that round-tripping every method is cheap."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from harness.store.fetch_denylist import (
    DEFAULT_TTL_DAYS,
    DenylistEntry,
    FetchDenylistStore,
)


def _store(tmp_path: Path, **kwargs: object) -> FetchDenylistStore:
    db = tmp_path / "harness.sqlite"
    return FetchDenylistStore(db, **kwargs)  # type: ignore[arg-type]


def test_record_creates_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    entry = store.record(
        host="example.com", status=403, reason="Forbidden", url="https://example.com/a"
    )
    assert entry.host == "example.com"
    assert entry.last_status == 403
    assert entry.last_reason == "Forbidden"
    assert entry.last_url == "https://example.com/a"
    assert entry.count == 1
    assert entry.first_seen_at == entry.last_seen_at


def test_record_is_upsert_and_bumps_count(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.record(
        host="example.com", status=403, reason="Forbidden", url="https://example.com/a"
    )
    second = store.record(
        host="example.com", status=401, reason="Unauthorized", url="https://example.com/b"
    )
    assert second.count == 2
    assert second.last_status == 401
    assert second.last_reason == "Unauthorized"
    assert second.last_url == "https://example.com/b"
    # first_seen_at is sticky; last_seen_at advances.
    assert second.first_seen_at == first.first_seen_at
    assert second.last_seen_at >= first.last_seen_at


def test_host_key_is_lowercased(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(host="Example.COM", status=403, reason="Forbidden", url="https://Example.COM/")
    assert store.get("example.com") is not None
    assert store.get("EXAMPLE.com") is not None  # case-insensitive read


def test_is_blocked_returns_active_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(host="blocked.test", status=403, reason="Forbidden", url="https://blocked.test/")
    hit = store.is_blocked("blocked.test")
    assert hit is not None
    assert hit.host == "blocked.test"


def test_is_blocked_returns_none_for_unknown_host(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.is_blocked("never-seen.test") is None


def test_is_blocked_filters_expired_entries(tmp_path: Path) -> None:
    store = _store(tmp_path, ttl_days=30)
    # Backdate the last_seen_at to 31 days ago by passing now=...
    long_ago = datetime.now(UTC) - timedelta(days=31)
    store.record(
        host="stale.test", status=403, reason="Forbidden", url="https://stale.test/", now=long_ago
    )
    assert store.is_blocked("stale.test") is None
    # But the row is still on disk.
    raw = store.get("stale.test")
    assert raw is not None
    assert raw.host == "stale.test"


def test_list_all_hides_expired_by_default(tmp_path: Path) -> None:
    store = _store(tmp_path, ttl_days=30)
    now = datetime.now(UTC)
    store.record(
        host="fresh.test", status=403, reason="Forbidden", url="https://fresh.test/", now=now
    )
    store.record(
        host="stale.test",
        status=403,
        reason="Forbidden",
        url="https://stale.test/",
        now=now - timedelta(days=45),
    )
    active = store.list_all()
    assert [e.host for e in active] == ["fresh.test"]
    all_entries = store.list_all(include_expired=True)
    assert {e.host for e in all_entries} == {"fresh.test", "stale.test"}


def test_clear_one_host(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(host="a.test", status=403, reason="Forbidden", url="https://a.test/")
    store.record(host="b.test", status=403, reason="Forbidden", url="https://b.test/")
    removed = store.clear("a.test")
    assert removed == 1
    assert store.get("a.test") is None
    assert store.get("b.test") is not None


def test_clear_all(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record(host="a.test", status=403, reason="Forbidden", url="https://a.test/")
    store.record(host="b.test", status=401, reason="Unauthorized", url="https://b.test/")
    removed = store.clear()
    assert removed == 2
    assert store.list_all(include_expired=True) == []


def test_default_ttl_is_30_days() -> None:
    assert DEFAULT_TTL_DAYS == 30


def test_entry_is_active_window() -> None:
    now = datetime.now(UTC)
    entry = DenylistEntry(
        host="x.test",
        last_status=403,
        last_reason="Forbidden",
        last_url="https://x.test/",
        first_seen_at=now - timedelta(days=10),
        last_seen_at=now - timedelta(days=10),
        count=1,
    )
    assert entry.is_active(now=now, ttl_days=30) is True
    assert entry.is_active(now=now, ttl_days=5) is False
