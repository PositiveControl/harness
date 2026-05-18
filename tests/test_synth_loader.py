"""Tests for the synthesized-tool hot-reloader (harness-t5kx).

Real importlib path — no mocks. Each test writes a small candidate
source file to tmp_path, builds a matching catalog entry, and runs
`load_synthesized_tools`. The synth-tool path is symmetric with
synthesize_tool (yari), so several sources here mirror the templates
in `test_synthesize_tool.py`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from harness.tools.base import ToolRegistry
from harness.tools.catalog import (
    ToolCatalog,
    ToolCatalogEntry,
    load_catalog,
    save_catalog,
)
from harness.tools.loader import load_synthesized_tools

VALID_SOURCE_TEMPLATE = textwrap.dedent(
    """
    from harness.tools.base import ToolSpec


    class {cls}:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="{name}",
                description="loaded from disk",
                parameters={{"type": "object", "properties": {{}}}},
                tier="read",
            )

        def call(self) -> str:
            return "hello from {name}"
    """
)


def _seed_synth_entry(
    *, name: str, cls: str, tmp_path: Path, source_text: str | None = None
) -> tuple[ToolCatalog, Path, Path]:
    """Common scaffold: a catalog with one synthesized entry whose
    source file is freshly written to tmp_path."""
    synth_dir = tmp_path / "synthesized"
    synth_dir.mkdir(parents=True, exist_ok=True)
    source_path = synth_dir / f"{name}.py"
    source_path.write_text(
        source_text if source_text is not None else VALID_SOURCE_TEMPLATE.format(cls=cls, name=name)
    )
    catalog = ToolCatalog()
    catalog.register(
        ToolCatalogEntry(
            name=name,
            family="meta",
            origin="synthesized",
            source_path=str(source_path),
            registered_at="2026-05-18T00:00:00+00:00",
        )
    )
    catalog_path = tmp_path / "catalog.json"
    save_catalog(catalog, catalog_path)
    return catalog, catalog_path, source_path


# --- happy path -----------------------------------------------------------


def test_loader_registers_a_single_synthesized_tool(tmp_path: Path) -> None:
    catalog, catalog_path, _ = _seed_synth_entry(name="echo_a", cls="EchoA", tmp_path=tmp_path)
    registry = ToolRegistry()

    report = load_synthesized_tools(catalog=catalog, catalog_path=catalog_path, registry=registry)

    assert report.loaded == ("echo_a",)
    assert report.quarantined == ()
    assert "echo_a" in registry
    # Tool's call() runs through the live registry.
    result = registry.call("echo_a", {})
    assert result.success is True
    assert "echo_a" in result.output


def test_loader_handles_multiple_synthesized_tools(tmp_path: Path) -> None:
    """N synth entries → N registered tools, all callable."""
    synth_dir = tmp_path / "synthesized"
    synth_dir.mkdir()
    catalog = ToolCatalog()
    for n in ("alpha", "beta", "gamma"):
        path = synth_dir / f"{n}.py"
        path.write_text(VALID_SOURCE_TEMPLATE.format(cls=f"{n.capitalize()}Tool", name=n))
        catalog.register(
            ToolCatalogEntry(
                name=n,
                family="meta",
                origin="synthesized",
                source_path=str(path),
            )
        )
    catalog_path = tmp_path / "catalog.json"
    save_catalog(catalog, catalog_path)
    registry = ToolRegistry()

    report = load_synthesized_tools(catalog=catalog, catalog_path=catalog_path, registry=registry)

    assert set(report.loaded) == {"alpha", "beta", "gamma"}
    assert report.quarantined == ()
    assert {"alpha", "beta", "gamma"}.issubset(set(registry.names()))


# --- failure: import errors quarantine ----------------------------------


def test_loader_quarantines_module_with_syntax_error(tmp_path: Path) -> None:
    """A hand-edited synth file with a syntax error must not crash boot —
    quarantine + skip + persist."""
    catalog, catalog_path, _source_path = _seed_synth_entry(
        name="broken_syntax",
        cls="Doesnt",
        tmp_path=tmp_path,
        source_text="this is not python syntax !!!",
    )
    registry = ToolRegistry()

    report = load_synthesized_tools(catalog=catalog, catalog_path=catalog_path, registry=registry)

    assert report.loaded == ()
    assert len(report.quarantined) == 1
    name, reason = report.quarantined[0]
    assert name == "broken_syntax"
    assert "module import failed" in reason
    assert "broken_syntax" not in registry
    # Catalog mutated + persisted.
    persisted = load_catalog(catalog_path)
    entry = persisted.get("broken_syntax")
    assert entry is not None
    assert entry.quarantined is True
    assert entry.quarantine_reason is not None
    assert "module import failed" in entry.quarantine_reason


def test_loader_quarantines_module_with_no_tool_class(tmp_path: Path) -> None:
    """Same contract as the validator: a synth file that defines no
    Tool gets quarantined rather than skipped silently."""
    catalog, catalog_path, _ = _seed_synth_entry(
        name="not_a_tool",
        cls="X",
        tmp_path=tmp_path,
        source_text="THE_ANSWER = 42\n",
    )
    registry = ToolRegistry()

    report = load_synthesized_tools(catalog=catalog, catalog_path=catalog_path, registry=registry)

    assert report.loaded == ()
    assert report.quarantined
    assert report.quarantined[0][0] == "not_a_tool"
    assert "no Tool class" in report.quarantined[0][1]


def test_loader_quarantines_when_source_file_missing(tmp_path: Path) -> None:
    """Operator may have moved the synth dir; the catalog entry now
    points at a gone path. Don't crash; quarantine."""
    catalog = ToolCatalog()
    catalog.register(
        ToolCatalogEntry(
            name="ghost",
            family="meta",
            origin="synthesized",
            source_path=str(tmp_path / "nowhere.py"),
        )
    )
    catalog_path = tmp_path / "catalog.json"
    save_catalog(catalog, catalog_path)
    registry = ToolRegistry()

    report = load_synthesized_tools(catalog=catalog, catalog_path=catalog_path, registry=registry)

    assert report.loaded == ()
    assert report.quarantined
    assert report.quarantined[0][0] == "ghost"
    assert "source file missing" in report.quarantined[0][1]


# --- failure: name mismatch quarantines ----------------------------------


def test_loader_quarantines_when_spec_name_disagrees_with_catalog_key(
    tmp_path: Path,
) -> None:
    """The catalog key is `alpha` but the ToolSpec.name is `beta`. If
    we loaded it as-is, tool_search and operator CLI would lie. Refuse."""
    catalog, catalog_path, _ = _seed_synth_entry(
        name="alpha",
        cls="MismatchTool",
        tmp_path=tmp_path,
        source_text=VALID_SOURCE_TEMPLATE.format(cls="MismatchTool", name="beta"),
    )
    registry = ToolRegistry()

    report = load_synthesized_tools(catalog=catalog, catalog_path=catalog_path, registry=registry)

    assert report.loaded == ()
    assert report.quarantined
    assert report.quarantined[0][0] == "alpha"
    assert "disagrees with catalog key" in report.quarantined[0][1]


# --- already-quarantined entries skip without retry ---------------------


def test_loader_skips_already_quarantined_entries(tmp_path: Path) -> None:
    """An entry quarantined in a prior session must not be retried —
    that's the 'tried once, gave up' carryover."""
    catalog, catalog_path, _ = _seed_synth_entry(
        name="pre_broken",
        cls="EchoA",
        tmp_path=tmp_path,
        source_text="this is not python !!!",
    )
    # First pass quarantines.
    load_synthesized_tools(catalog=catalog, catalog_path=catalog_path, registry=ToolRegistry())
    # Second pass on a fresh registry must not even try.
    registry2 = ToolRegistry()
    report = load_synthesized_tools(catalog=catalog, catalog_path=catalog_path, registry=registry2)
    assert report.loaded == ()
    assert report.quarantined == ()  # no NEW quarantines this pass
    assert report.already_quarantined == ("pre_broken",)
    assert "pre_broken" not in registry2


# --- collision with builtin registry entry quarantines ------------------


def test_loader_quarantines_when_name_collides_with_already_registered_tool(
    tmp_path: Path,
) -> None:
    """Builtins are wired into the registry before hot-reload runs.
    A synth entry sharing a builtin's name must quarantine — the
    operator needs to rename or drop."""
    catalog, catalog_path, _ = _seed_synth_entry(
        name="echo_x",
        cls="EchoX",
        tmp_path=tmp_path,
    )
    # Seed the registry with a stub tool that uses the same name.
    registry = ToolRegistry()

    class _Stub:
        @property
        def spec(self) -> object:
            from harness.tools.base import ToolSpec

            return ToolSpec(
                name="echo_x",
                description="builtin stub",
                parameters={"type": "object", "properties": {}},
                tier="read",
            )

        def call(self) -> str:
            return "stub"

    registry.register(_Stub())  # type: ignore[arg-type]

    report = load_synthesized_tools(catalog=catalog, catalog_path=catalog_path, registry=registry)

    assert report.loaded == ()
    assert report.quarantined
    assert report.quarantined[0][0] == "echo_x"
    assert "already registered" in report.quarantined[0][1]
