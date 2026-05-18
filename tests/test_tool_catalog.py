"""Tests for the ToolCatalog data model + JSON persistence — harness-hfa7."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.tools.catalog import (
    BUILTIN_TOOL_METADATA,
    ToolCatalog,
    ToolCatalogEntry,
    load_catalog,
    save_catalog,
    seed_builtins_into,
)

# --- record + helpers ---------------------------------------------------


def test_entry_defaults() -> None:
    e = ToolCatalogEntry(name="now")
    assert e.family == ""
    assert e.tags == ()
    assert e.description == ""
    assert e.tier == "read"
    assert e.origin == "builtin"
    assert e.source_path is None
    assert e.registered_at == ""
    assert e.quarantined is False
    assert e.quarantine_reason is None


def test_register_inserts_and_replaces() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="now", family="reckon"))
    assert cat.get("now") is not None
    cat.register(ToolCatalogEntry(name="now", family="reckon-renamed"))
    assert cat.get("now").family == "reckon-renamed"  # type: ignore[union-attr]


def test_register_rejects_empty_name() -> None:
    cat = ToolCatalog()
    with pytest.raises(ValueError, match="non-empty"):
        cat.register(ToolCatalogEntry(name=""))


def test_drop_is_idempotent() -> None:
    cat = ToolCatalog()
    cat.drop("never-existed")  # no raise
    cat.register(ToolCatalogEntry(name="now"))
    cat.drop("now")
    cat.drop("now")  # second drop no raise


def test_get_returns_none_for_missing() -> None:
    assert ToolCatalog().get("missing") is None


def test_all_sorts_by_name() -> None:
    cat = ToolCatalog()
    for name in ("c", "a", "b"):
        cat.register(ToolCatalogEntry(name=name))
    assert [e.name for e in cat.all()] == ["a", "b", "c"]


# --- lookups -----------------------------------------------------------


def test_by_family_filters_by_family() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="now", family="reckon"))
    cat.register(ToolCatalogEntry(name="grep", family="filesystem"))
    cat.register(ToolCatalogEntry(name="calc", family="reckon"))
    names = [e.name for e in cat.by_family("reckon")]
    assert names == ["calc", "now"]


def test_by_tag_returns_matching_entries() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="now", tags=("time", "clock")))
    cat.register(ToolCatalogEntry(name="calc", tags=("math",)))
    cat.register(ToolCatalogEntry(name="tz_convert", tags=("time", "timezone")))
    time_tools = [e.name for e in cat.by_tag("time")]
    assert time_tools == ["now", "tz_convert"]


def test_by_origin_filters() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="builtin1", origin="builtin"))
    cat.register(ToolCatalogEntry(name="synth1", origin="synthesized"))
    cat.register(ToolCatalogEntry(name="ext1", origin="external"))
    assert [e.name for e in cat.by_origin("synthesized")] == ["synth1"]


def test_search_matches_name_substring() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="now"))
    cat.register(ToolCatalogEntry(name="snowman"))  # contains 'now'
    cat.register(ToolCatalogEntry(name="calc"))
    names = [e.name for e in cat.search("now")]
    assert set(names) == {"now", "snowman"}


def test_search_matches_description_substring() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="calc", description="Evaluate arithmetic expressions."))
    cat.register(ToolCatalogEntry(name="grep", description="Find text in files via regex."))
    names = [e.name for e in cat.search("regex")]
    assert names == ["grep"]


def test_search_matches_tag_substring() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="now", tags=("timezone",)))
    names = [e.name for e in cat.search("zone")]
    assert names == ["now"]


def test_search_empty_query_returns_empty() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="now"))
    assert cat.search("") == []
    assert cat.search("   ") == []


def test_search_tokenizes_multi_word_query() -> None:
    """harness-wwki: 'weather phoenix' must land on an entry whose
    tag includes 'weather', even though the literal phrase appears
    nowhere. Substring-match-the-whole-query semantics would miss this."""
    cat = ToolCatalog()
    cat.register(
        ToolCatalogEntry(
            name="search_web",
            description="Search the web for current info.",
            tags=("weather", "news", "web"),
        )
    )
    names = [e.name for e in cat.search("weather phoenix")]
    assert names == ["search_web"]


def test_search_drops_short_stopword_tokens() -> None:
    """'in', 'to', 'of' etc. produce noisy hits when matched
    individually — they substring-match unrelated tags ('in' hits
    'find', 'diff-summary'). Tokens shorter than 3 chars are dropped
    before the OR-match."""
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="finder", tags=("find", "files")))
    cat.register(ToolCatalogEntry(name="search_web", description="weather", tags=("weather",)))
    # 'in' alone would substring-match 'find' — but it's dropped.
    names = [e.name for e in cat.search("weather in phoenix")]
    assert names == ["search_web"]


def test_search_falls_back_to_phrase_when_all_tokens_short() -> None:
    """If every token is filtered out by the stopword length, fall
    back to a single whole-phrase substring match — at least let the
    literal query try."""
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="abc-tool", description="x"))
    cat.register(ToolCatalogEntry(name="other", description="y"))
    names = [e.name for e in cat.search("ab")]
    assert names == ["abc-tool"]


def test_search_is_case_insensitive() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="Now", description="Current TIME"))
    names = [e.name for e in cat.search("now")]
    assert names == ["Now"]
    names = [e.name for e in cat.search("TIME")]
    assert names == ["Now"]


# --- persistence -------------------------------------------------------


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    cat = load_catalog(tmp_path / "no-such-catalog.json")
    assert cat.entries == {}


def test_load_malformed_json_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    assert load_catalog(p).entries == {}


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    cat = ToolCatalog()
    cat.register(
        ToolCatalogEntry(
            name="now",
            family="reckon",
            tags=("time", "clock"),
            description="Current wall-clock.",
            tier="read",
            origin="builtin",
            registered_at="2026-05-18T20:00:00+00:00",
        )
    )
    cat.register(
        ToolCatalogEntry(
            name="custom",
            family="custom",
            tags=("user",),
            origin="synthesized",
            source_path=str(tmp_path / "custom.py"),
            registered_at="2026-05-18T20:01:00+00:00",
        )
    )
    p = tmp_path / "cat.json"
    save_catalog(cat, p)
    loaded = load_catalog(p)
    assert loaded.get("now") == cat.get("now")
    assert loaded.get("custom") == cat.get("custom")


def test_save_is_atomic(tmp_path: Path) -> None:
    """Pre-existing file with stale schema must be fully overwritten,
    not merged."""
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"entries": {"stale": {"old_field": True}}}))
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="new", family="reckon"))
    save_catalog(cat, p)
    on_disk = json.loads(p.read_text())
    assert "stale" not in on_disk["entries"]
    assert on_disk["entries"]["new"]["family"] == "reckon"


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "cat.json"
    save_catalog(ToolCatalog(), nested)
    assert nested.exists()


def test_load_drops_unknown_fields_for_forward_compat(tmp_path: Path) -> None:
    """A future schema-version field shouldn't crash older loaders."""
    p = tmp_path / "cat.json"
    p.write_text(
        json.dumps(
            {
                "entries": {
                    "now": {
                        "name": "now",
                        "family": "reckon",
                        "future_field": "ignored",
                    }
                }
            }
        )
    )
    cat = load_catalog(p)
    assert cat.get("now") is not None
    assert cat.get("now").family == "reckon"  # type: ignore[union-attr]


def test_load_skips_non_dict_entries(tmp_path: Path) -> None:
    """Defensive — a hand-edit that replaces a value with a string
    shouldn't crash the loader."""
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"entries": {"good": {"name": "good"}, "bad": "oops"}}))
    cat = load_catalog(p)
    assert set(cat.entries) == {"good"}


def test_load_unknown_origin_coerces_to_external(tmp_path: Path) -> None:
    """Forward-compat: an entry with origin='from-the-future' coerces
    to 'external' so the catalog stays consistent."""
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"entries": {"x": {"name": "x", "origin": "from-the-future"}}}))
    cat = load_catalog(p)
    entry = cat.get("x")
    assert entry is not None
    assert entry.origin == "external"


def test_save_serializes_tags_as_list(tmp_path: Path) -> None:
    """Tuple → list on disk so the JSON is natural; load restores
    the tuple."""
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="x", tags=("a", "b")))
    p = tmp_path / "cat.json"
    save_catalog(cat, p)
    on_disk = json.loads(p.read_text())
    assert on_disk["entries"]["x"]["tags"] == ["a", "b"]
    reloaded = load_catalog(p)
    assert reloaded.get("x").tags == ("a", "b")  # type: ignore[union-attr]


# --- seed_builtins_into ------------------------------------------------


def test_seed_builtins_inserts_every_known_tool() -> None:
    cat = ToolCatalog()
    inserted = seed_builtins_into(cat, now_iso="2026-05-18T20:00:00+00:00")
    assert inserted == len(BUILTIN_TOOL_METADATA)
    for name in BUILTIN_TOOL_METADATA:
        assert name in cat.entries
        entry = cat.get(name)
        assert entry is not None
        assert entry.origin == "builtin"


def test_seed_builtins_preserves_existing_entries() -> None:
    """A second seed-call after manual edits / synthesis must not
    overwrite custom entries."""
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="now", family="custom-reckon", tags=("user-edit",)))
    inserted = seed_builtins_into(cat, now_iso="2026-05-18T20:00:00+00:00")
    # `now` was already there → not re-inserted.
    assert inserted == len(BUILTIN_TOOL_METADATA) - 1
    entry = cat.get("now")
    assert entry is not None
    assert entry.family == "custom-reckon"
    assert entry.tags == ("user-edit",)


def test_seed_builtins_assigns_correct_family_and_tags() -> None:
    """Spot check a handful: the seeded metadata matches the table."""
    cat = ToolCatalog()
    seed_builtins_into(cat, now_iso="2026-05-18T20:00:00+00:00")
    now_entry = cat.get("now")
    assert now_entry is not None
    assert now_entry.family == "reckon"
    assert "time" in now_entry.tags
    grep_entry = cat.get("grep")
    assert grep_entry is not None
    assert grep_entry.family == "filesystem"
    assert "search" in grep_entry.tags
    plan_entry = cat.get("plan")
    assert plan_entry is not None
    assert plan_entry.family == "ops"


def test_builtin_metadata_includes_every_tool_family() -> None:
    """Sanity: the BUILTIN_TOOL_METADATA table covers every family
    the harness ships today. Catches the case where someone adds a
    new family to profiles.py but forgets the catalog entry."""
    families = {entry[0] for entry in BUILTIN_TOOL_METADATA.values()}
    # Pin the expected family set so a future addition is intentional.
    assert "reckon" in families
    assert "filesystem" in families
    assert "git" in families
    assert "memory" in families
    assert "research" in families
    assert "meta" in families
    assert "ops" in families
    assert "atc" in families


def test_builtin_metadata_descriptions_are_non_empty() -> None:
    """Regression for harness-huwj: when description fields were empty,
    tool_search returned matches like `search_web (research) —
    (no description)` and small models gave up rather than calling
    load_tool. Pin every builtin to have a non-empty description so
    the discovery flow works end-to-end."""
    missing = [name for name, entry in BUILTIN_TOOL_METADATA.items() if not entry[2].strip()]
    assert not missing, f"builtin tools without descriptions: {missing}"


def test_builtin_metadata_tiers_are_valid() -> None:
    """Every entry's tier must be 'read' or 'write' so the seeded
    catalog is consistent with the orchestrator's confirm gate."""
    bad = [
        name for name, entry in BUILTIN_TOOL_METADATA.items() if entry[3] not in ("read", "write")
    ]
    assert not bad, f"builtin tools with bad tier: {bad}"
