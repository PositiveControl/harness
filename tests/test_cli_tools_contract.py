"""Behavioral contract for the CLI's tool-registry construction.

Step 4 of docs/cli-extraction-plan.md moves this cluster into
`cli_tools.py`. The coverage sweep for harness-z4k1.1 put the block at
27% — `_build_tool_registry_for_tui` alone is ~305 LOC at 2%, meaning
the function that decides WHICH TOOLS THE MODEL GETS was almost
entirely unexercised.

What's pinned here is the observable contract, not the wiring: which
names end up registered, which get dropped, and — the part that
actually bit a user (harness-akq) — whether a drop is silent or lands
in `warnings_out`.

Tests address the helpers through `harness.cli`, which keeps resolving
after the move via re-export, so they characterize the code unchanged
on both sides of the extraction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from harness.character import load_character
from harness.cli import (
    OPS_TOOL_NAMES,
    _ab_tool_builders,
    _build_tool_grounding_block,
    _build_tool_registry_for_tui,
    _missing_builder_reason,
    _resolve_router_tool_specs,
    _router_id_label,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AIRTON = _REPO_ROOT / "character" / "airton"


def _registry(**overrides: object) -> object:
    """Call the builder with the store-less defaults a bare TUI session
    would use. `character=None` keeps the builder from constructing a
    real bd adapter (that path is covered in test_airton_b_integration)."""
    kwargs: dict[str, object] = {
        "tools": True,
        "tool_set": "minimal",
        "tools_add": None,
        "tools_drop": None,
        "workspace_path": _REPO_ROOT,
        "memory_store": None,
        "semantic_store": None,
        "speaker": "mark",
        "session": "test",
        "character": None,
    }
    kwargs.update(overrides)
    return _build_tool_registry_for_tui(**kwargs)  # type: ignore[arg-type]


# ---------- _build_tool_registry_for_tui ----------


def test_registry_is_none_when_tools_are_off() -> None:
    """`--tools` off means no registry at all, not an empty one — the
    orchestrator branches on None to skip the whole tool loop."""
    assert _registry(tools=False) is None


def test_registry_is_none_when_the_profile_resolves_to_nothing() -> None:
    """`minimal` is deliberately empty. An empty registry is reported as
    None so callers don't hand the model a zero-tool schema."""
    assert _registry(tool_set="minimal") is None


def test_unknown_profile_becomes_a_typer_parameter_error() -> None:
    """A typo in --tool-set must surface as a CLI usage error, not a
    stack trace out of resolve_tool_names."""
    with pytest.raises(typer.BadParameter):
        _registry(tool_set="not-a-profile")


def test_registry_registers_the_requested_workspace_tools() -> None:
    registry = _registry(tools_add="read_file,grep,glob")

    assert registry is not None
    assert set(registry.names()) == {"read_file", "grep", "glob"}  # type: ignore[attr-defined]


def test_store_backed_tool_is_dropped_with_a_warning_not_silently() -> None:
    """harness-akq: a user passed names that couldn't be built and got
    an empty registry with no explanation. Every drop must leave a
    line in warnings_out."""
    warnings: list[str] = []

    registry = _registry(tools_add="read_file,search_memory", warnings_out=warnings)

    assert registry is not None
    assert "search_memory" not in registry.names()  # type: ignore[attr-defined]
    assert any("search_memory" in w and "store" in w for w in warnings)


def test_store_backed_tool_registers_once_its_store_exists(tmp_path: Path) -> None:
    """Same request, store present — proves the drop above is about the
    missing store and not about the name."""

    class _Embedder:
        id = "fake"
        dimension = 4

        def embed(self, texts: object) -> object:
            raise AssertionError("registry construction must not embed")

    from harness.store.episodic import EpisodicStore

    store = EpisodicStore(tmp_path / "e.db", embedder=_Embedder())  # type: ignore[arg-type]
    warnings: list[str] = []

    registry = _registry(tools_add="search_memory", memory_store=store, warnings_out=warnings)

    assert registry is not None
    assert "search_memory" in registry.names()  # type: ignore[attr-defined]
    assert warnings == []


def test_ops_tool_without_a_character_explains_the_bd_dependency() -> None:
    """Ops tools need a bd adapter, which needs a character. The warning
    has to name bd — 'not yet implemented' sent people looking in the
    wrong place."""
    warnings: list[str] = []

    _registry(tools_add="plan,close", warnings_out=warnings)

    assert len(warnings) == 2
    assert all("bd" in w for w in warnings)


def test_introspect_is_skipped_with_a_reason_when_the_adapter_is_missing() -> None:
    warnings: list[str] = []

    registry = _registry(tools_add="read_file,introspect", warnings_out=warnings)

    assert registry is not None
    assert "introspect" not in registry.names()  # type: ignore[attr-defined]
    assert any("introspect" in w and "adapter" in w for w in warnings)


def test_spawn_subagent_is_skipped_with_a_reason_when_the_adapter_is_missing() -> None:
    warnings: list[str] = []

    registry = _registry(tools_add="read_file,spawn_subagent", warnings_out=warnings)

    assert registry is not None
    assert "spawn_subagent" not in registry.names()  # type: ignore[attr-defined]
    assert any("spawn_subagent" in w and "adapter" in w for w in warnings)


def test_drop_list_removes_a_tool_the_profile_would_have_included() -> None:
    with_grep = _registry(tools_add="read_file,grep")
    without_grep = _registry(tools_add="read_file,grep", tools_drop="grep")

    assert with_grep is not None
    assert without_grep is not None
    assert "grep" in with_grep.names()  # type: ignore[attr-defined]
    assert "grep" not in without_grep.names()  # type: ignore[attr-defined]


def test_registry_construction_survives_a_missing_warnings_sink() -> None:
    """warnings_out is optional; the drop paths must not assume it."""
    registry = _registry(tools_add="read_file,search_memory", warnings_out=None)

    assert registry is not None
    assert set(registry.names()) == {"read_file"}  # type: ignore[attr-defined]


# ---------- _missing_builder_reason ----------


def test_missing_builder_reason_points_ops_tools_at_the_bd_dir() -> None:
    character = load_character(_AIRTON)

    reason = _missing_builder_reason("plan", character)

    assert "plan" in reason
    assert "bd init" in reason
    assert "bd dolt start" in reason


def test_missing_builder_reason_degrades_without_a_character() -> None:
    reason = _missing_builder_reason("plan", None)

    assert "<bd dir>" in reason


def test_missing_builder_reason_for_a_non_ops_tool_says_not_implemented() -> None:
    reason = _missing_builder_reason("some_future_tool", None)

    assert "not yet implemented" in reason
    assert "bd" not in reason


# ---------- _ab_tool_builders ----------


def test_ab_tool_builders_is_empty_without_an_adapter() -> None:
    """Empty dict, not None — the caller merges it unconditionally."""
    assert _ab_tool_builders(None) == {}


def test_ab_tool_builders_covers_the_ops_surface() -> None:
    """Every name the ops profile can ask for has a builder, so a bound
    bd adapter never produces a 'no builder' warning."""

    class _StubAdapter:
        pass

    builders = _ab_tool_builders(_StubAdapter())  # type: ignore[arg-type]

    assert set(builders) == set(OPS_TOOL_NAMES)


def test_ab_tool_builders_bind_the_adapter_they_were_given() -> None:
    class _StubAdapter:
        pass

    adapter = _StubAdapter()
    builders = _ab_tool_builders(adapter)  # type: ignore[arg-type]

    plan_tool = builders["plan"]()

    assert plan_tool is not None
    assert plan_tool.adapter is adapter  # type: ignore[attr-defined]


# ---------- _router_id_label ----------


def test_router_id_label_is_none_without_a_router() -> None:
    assert _router_id_label(None) is None


def test_router_id_label_names_the_mode_and_model() -> None:
    from harness.router import GrammarRouter, ModelRouter

    class _Adapter:
        id = "Hermes-3-3B-4bit"

        def complete(self, *args: object, **kwargs: object) -> str:
            return ""

        def complete_grammar(self, *args: object, **kwargs: object) -> str:
            return ""

    grammar = GrammarRouter(adapter=_Adapter())
    free = ModelRouter(adapter=_Adapter())  # type: ignore[arg-type]

    assert _router_id_label(grammar) == "grammar:Hermes-3-3B-4bit"
    assert _router_id_label(free) == "free:Hermes-3-3B-4bit"


def test_router_id_label_falls_back_when_the_adapter_has_no_id() -> None:
    from harness.router import ModelRouter

    class _Anonymous:
        def complete(self, *args: object, **kwargs: object) -> str:
            return ""

    label = _router_id_label(ModelRouter(adapter=_Anonymous()))  # type: ignore[arg-type]

    assert label == "free:router"


# ---------- _build_tool_grounding_block ----------


def test_grounding_block_names_the_sandbox_and_the_live_tool_set(tmp_path: Path) -> None:
    registry = _registry(tools_add="read_file,grep", workspace_path=tmp_path)
    assert registry is not None

    block = _build_tool_grounding_block(registry, tmp_path)  # type: ignore[arg-type]

    assert str(tmp_path) in block
    assert "read_file" in block
    assert "grep" in block
    # The forbidden-phrase list is the reason this block exists at all.
    assert "Would you like" in block


# ---------- _resolve_router_tool_specs ----------


def test_router_specs_come_from_real_tools_where_one_can_be_built() -> None:
    specs = _resolve_router_tool_specs(["read_file", "grep"], _REPO_ROOT)

    assert [s.name for s in specs] == ["read_file", "grep"]
    assert all(s.tier == "read" for s in specs)


def test_router_specs_substitute_schemas_for_store_backed_tools() -> None:
    """The router only needs the schema, so memory tools get a stub
    spec rather than a live store."""
    specs = _resolve_router_tool_specs(["search_memory", "search_facts"], _REPO_ROOT)

    assert [s.name for s in specs] == ["search_memory", "search_facts"]
    assert all(s.parameters["required"] == ["query"] for s in specs)


def test_router_specs_keep_the_write_tier_on_the_ingest_stub() -> None:
    """Tier is what the router uses to decide it may NOT auto-execute a
    call; a stub that lost it would make transcript_ingest auto-run."""
    (spec,) = _resolve_router_tool_specs(["transcript_ingest"], _REPO_ROOT)

    assert spec.tier == "write"


def test_router_specs_warn_and_skip_an_unknown_tool(
    capsys: pytest.CaptureFixture[str],
) -> None:
    specs = _resolve_router_tool_specs(["read_file", "no_such_tool"], _REPO_ROOT)

    assert [s.name for s in specs] == ["read_file"]
    assert "no_such_tool" in capsys.readouterr().out


# ---------- the character-present / adapter-present paths ----------
#
# Everything above runs with character=None and adapter=None, which is
# the store-less TUI boot. These three cover the other half: what gets
# registered once a session actually has a character, a model adapter
# and a bound bd adapter.


class _StubBeadsAdapter:
    """Enough of BeadsAdapter for the ops builders to construct. They
    only stash the reference; nothing here is called during build."""


def test_ops_tools_register_when_a_bd_adapter_is_supplied() -> None:
    character = load_character(_AIRTON)
    warnings: list[str] = []

    registry = _registry(
        tools_add="plan,close",
        character=character,
        ab_adapter=_StubBeadsAdapter(),
        warnings_out=warnings,
    )

    assert registry is not None
    assert {"plan", "close"} <= set(registry.names())  # type: ignore[attr-defined]
    assert warnings == []


def test_introspect_registers_once_adapter_and_character_are_present() -> None:
    from harness.model.echo import EchoAdapter

    character = load_character(_AIRTON)
    warnings: list[str] = []

    registry = _registry(
        tools_add="read_file,introspect",
        character=character,
        adapter=EchoAdapter(),
        ab_adapter=_StubBeadsAdapter(),
        warnings_out=warnings,
    )

    assert registry is not None
    assert "introspect" in registry.names()  # type: ignore[attr-defined]
    assert warnings == []


def test_spawn_subagent_registers_once_an_adapter_is_present() -> None:
    from harness.model.echo import EchoAdapter

    character = load_character(_AIRTON)

    registry = _registry(
        tools_add="read_file,spawn_subagent",
        character=character,
        adapter=EchoAdapter(),
        ab_adapter=_StubBeadsAdapter(),
    )

    assert registry is not None
    assert "spawn_subagent" in registry.names()  # type: ignore[attr-defined]
