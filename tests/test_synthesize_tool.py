"""Tests for the synthesize_tool meta-tool (harness-yari).

Each happy-path test exercises the real sandbox validator + catalog
persistence chain — no mocks on the validator, no mocks on the
catalog. The candidate sources are tiny but real Tools (single class,
duck-typed Tool: spec + call).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from harness.tools.catalog import (
    ToolCatalog,
    ToolCatalogEntry,
    load_catalog,
    save_catalog,
)
from harness.tools.synth import SynthesizeToolTool

VALID_SOURCE_TEMPLATE = textwrap.dedent(
    """
    from harness.tools.base import ToolSpec


    class {cls}:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="{name}",
                description="{desc}",
                parameters={{
                    "type": "object",
                    "properties": {{"x": {{"type": "string"}}}},
                    "required": ["x"],
                }},
                tier="read",
            )

        def call(self, *, x: str) -> str:
            return f"echo: {{x}}"
    """
)


def _build_tool(tmp_path: Path) -> SynthesizeToolTool:
    catalog_path = tmp_path / "catalog.json"
    synth_dir = tmp_path / "synthesized"
    return SynthesizeToolTool(
        catalog=ToolCatalog(),
        catalog_path=catalog_path,
        synth_dir=synth_dir,
    )


# --- happy path -----------------------------------------------------------


def test_synthesize_tool_writes_source_and_registers_catalog_entry(
    tmp_path: Path,
) -> None:
    tool = _build_tool(tmp_path)
    source = VALID_SOURCE_TEMPLATE.format(cls="EchoTool", name="echo_me", desc="d")

    out = tool.call(
        name="echo_me",
        source=source,
        family="meta",
        tags=["echo", "demo"],
        smoke_args={"x": "hi"},
    )

    assert "registered 'echo_me'" in out
    assert "(none)" not in out  # tags shown
    assert "smoke_output" in out  # smoke captured

    # Source persisted under the synth dir.
    source_path = tmp_path / "synthesized" / "echo_me.py"
    assert source_path.exists()
    assert "EchoTool" in source_path.read_text()

    # Catalog persisted with a synthesized origin entry.
    persisted = load_catalog(tmp_path / "catalog.json")
    entry = persisted.get("echo_me")
    assert entry is not None
    assert entry.origin == "synthesized"
    assert entry.family == "meta"
    assert entry.tags == ("echo", "demo")
    assert entry.tier == "read"
    assert entry.source_path == str(source_path)
    assert entry.registered_at  # ISO timestamp populated


def test_synthesize_tool_with_no_tags_persists_empty_tuple(tmp_path: Path) -> None:
    tool = _build_tool(tmp_path)
    source = VALID_SOURCE_TEMPLATE.format(cls="EchoTool", name="echo_me", desc="d")
    tool.call(
        name="echo_me",
        source=source,
        family="meta",
        smoke_args={"x": "hi"},
    )
    persisted = load_catalog(tmp_path / "catalog.json")
    persisted_entry = persisted.get("echo_me")
    assert persisted_entry is not None
    assert persisted_entry.tags == ()


# --- failure: validator rejects → no mutation ----------------------------


def test_synthesize_tool_rejects_invalid_source_without_mutating_disk(
    tmp_path: Path,
) -> None:
    """A module that doesn't expose a Tool fails validation. Catalog
    must stay empty and no source file is written — the inverted order
    would corrupt the catalog with a dangling entry."""
    tool = _build_tool(tmp_path)
    bad_source = "x = 42\n"  # no Tool class

    out = tool.call(
        name="bad_tool",
        source=bad_source,
        family="meta",
        smoke_args={},
    )

    assert "refusing to register" in out
    assert "no Tool class" in out
    # Nothing under the synth dir.
    assert not (tmp_path / "synthesized" / "bad_tool.py").exists()
    # Catalog file still absent.
    assert not (tmp_path / "catalog.json").exists()


def test_synthesize_tool_rejects_spec_name_mismatch(tmp_path: Path) -> None:
    """If the candidate's ToolSpec.name disagrees with the catalog
    key, refuse — discoverability tools would lie about the tool's
    real identity otherwise."""
    tool = _build_tool(tmp_path)
    # Tool registers as `inner_name` but synthesize_tool gets called
    # with name="outer_name".
    source = VALID_SOURCE_TEMPLATE.format(cls="MismatchTool", name="inner_name", desc="d")
    out = tool.call(
        name="outer_name",
        source=source,
        family="meta",
        smoke_args={"x": "hi"},
    )
    assert "refusing to register" in out
    assert "inner_name" in out
    assert "outer_name" in out


# --- failure: name collision points at drop_tool -------------------------


def test_synthesize_tool_refuses_when_name_already_in_catalog(tmp_path: Path) -> None:
    tool = _build_tool(tmp_path)
    # Seed an existing entry so the collision check fires.
    tool.catalog.register(ToolCatalogEntry(name="echo_me", family="meta", origin="builtin"))
    source = VALID_SOURCE_TEMPLATE.format(cls="EchoTool", name="echo_me", desc="d")
    with pytest.raises(ValueError, match="already in catalog"):
        tool.call(
            name="echo_me",
            source=source,
            family="meta",
            smoke_args={"x": "hi"},
        )


# --- failure: shape guards before sandbox --------------------------------


def test_synthesize_tool_rejects_bad_name_shape(tmp_path: Path) -> None:
    tool = _build_tool(tmp_path)
    source = VALID_SOURCE_TEMPLATE.format(cls="Echo", name="echo_me", desc="d")
    # Uppercase, dashes, leading digit — all should be refused.
    for bad_name in ("EchoMe", "echo-me", "9_oops", ""):
        with pytest.raises(ValueError, match="snake_case"):
            tool.call(
                name=bad_name,
                source=source,
                family="meta",
                smoke_args={"x": "hi"},
            )


def test_synthesize_tool_requires_family_and_smoke_args_shape(
    tmp_path: Path,
) -> None:
    tool = _build_tool(tmp_path)
    source = VALID_SOURCE_TEMPLATE.format(cls="EchoTool", name="echo_me", desc="d")

    with pytest.raises(ValueError, match="family"):
        tool.call(
            name="echo_me",
            source=source,
            family="   ",
            smoke_args={"x": "hi"},
        )
    with pytest.raises(ValueError, match="smoke_args"):
        tool.call(
            name="echo_me",
            source=source,
            family="meta",
            smoke_args="not a dict",  # type: ignore[arg-type]
        )


# --- spec contract --------------------------------------------------------


def test_synthesize_tool_spec_is_write_tier(tmp_path: Path) -> None:
    """Write-tier flag drives the orchestrator's confirm gate — must
    not silently drift to read-tier under refactor."""
    tool = _build_tool(tmp_path)
    spec = tool.spec
    assert spec.name == "synthesize_tool"
    assert spec.tier == "write"
    required = set(spec.parameters["required"])
    assert required == {"name", "source", "family", "smoke_args"}


def test_synthesize_tool_appends_to_existing_catalog(tmp_path: Path) -> None:
    """Second synth call must coexist with prior entries — the
    catalog round-trips cleanly when save → load is exercised
    between two synth calls."""
    catalog_path = tmp_path / "catalog.json"
    synth_dir = tmp_path / "synthesized"

    catalog = ToolCatalog()
    catalog.register(ToolCatalogEntry(name="prior", family="meta", origin="builtin"))
    save_catalog(catalog, catalog_path)

    tool = SynthesizeToolTool(
        catalog=load_catalog(catalog_path),
        catalog_path=catalog_path,
        synth_dir=synth_dir,
    )
    source = VALID_SOURCE_TEMPLATE.format(cls="EchoTool", name="echo_me", desc="d")
    tool.call(
        name="echo_me",
        source=source,
        family="meta",
        smoke_args={"x": "hi"},
    )

    on_disk = json.loads(catalog_path.read_text())
    assert set(on_disk["entries"]) == {"prior", "echo_me"}
