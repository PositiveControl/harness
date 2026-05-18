"""End-to-end wiring tests for the bd-backed CLI path — harness-inj.6
extended by harness-55y to cover every character, not just airton_b.

Covers the CLI glue that turns a character's ops-tool request into a
bound BeadsAdapter:

- `_maybe_bd_adapter(character)` returns a BeadsAdapter for *any*
  character when that character's per-character bd dir is
  bootstrapped; returns None with a warning when the dir is missing.
  airton and airton_b resolve to distinct dirs so their thought
  graphs stay isolated.
- `_ab_tool_builders(adapter)` surfaces the 21 ops tool builders
  exactly when an adapter is present — independent of character.
- `_resolve_adapter` branches on character.name so airton_b gets the
  CavemanRewriter and every other character keeps PersonaAdapter.
- Full-surface smoke: loading either character + calling every ops
  tool builder produces the expected Tool instances with the right
  `ToolSpec.name` — no silent drops, no rename drift.

The bd subprocess is never spawned; tests either mock verify() or
supply a canned bd dir containing a `.beads/` subdirectory so verify
passes without hitting the network / subprocess path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness.character import load_character
from harness.cli import (
    _ab_tool_builders,
    _maybe_bd_adapter,
    _missing_builder_reason,
    _resolve_adapter,
)
from harness.persona import PersonaAdapter
from harness.persona.caveman_rewriter import CavemanRewriter
from harness.store.bd_adapter import BeadsAdapter, BeadsAdapterError

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRTON = REPO_ROOT / "character" / "airton"
AIRTON_B = REPO_ROOT / "character" / "airton_b"


def _stub_bd_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the bd-CLI surface so verify() clears without real bd.

    shutil.which returns a truthy path (bd-on-PATH gate) and
    subprocess.run returns exit 0 + a connection-successful stdout
    (Dolt-server-reachable gate)."""
    monkeypatch.setattr("harness.store.bd_adapter.shutil.which", lambda _: "/usr/local/bin/bd")

    def fake_run(*_a: object, **_k: object) -> MagicMock:
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "✓ Connection successful\n"
        proc.stderr = ""
        return proc

    monkeypatch.setattr("harness.store.bd_adapter.subprocess.run", fake_run)


def test_maybe_bd_adapter_returns_adapter_for_airton_b_when_dir_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".beads").mkdir()
    monkeypatch.setattr("harness.cli.settings.ab_bd_dir", tmp_path)
    _stub_bd_subprocess(monkeypatch)

    ab = load_character(AIRTON_B)
    adapter = _maybe_bd_adapter(ab)

    assert isinstance(adapter, BeadsAdapter)
    assert adapter.bd_dir == tmp_path


def test_maybe_bd_adapter_returns_adapter_for_airton_from_project_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Foundational shift (harness-55y): any character — not just
    airton_b — can bind an ops adapter. Post-Path-2, that adapter
    defaults to the project's bd dir (settings.root), which the
    project already maintains as a healthy Dolt instance. Here we
    point settings.root at a stand-in tmp_path with its own .beads/
    subtree to avoid mutating the real project store."""
    (tmp_path / ".beads").mkdir()
    monkeypatch.setattr("harness.cli.settings.root", tmp_path)
    # airton_b has no env override so it also resolves to the
    # project dir — the shared-dir invariant.
    monkeypatch.setattr("harness.cli.settings.ab_bd_dir", None)
    _stub_bd_subprocess(monkeypatch)

    airton = load_character(AIRTON)
    adapter = _maybe_bd_adapter(airton)

    assert isinstance(adapter, BeadsAdapter)
    assert adapter.bd_dir == tmp_path


def test_maybe_bd_adapter_returns_none_when_any_character_dir_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dir-existence is the only skip reason now; character name
    doesn't gate registration. Point settings.root at an empty
    tmp_path (no .beads/ subtree) and both characters come back
    empty-handed."""
    monkeypatch.setattr("harness.cli.settings.root", tmp_path)
    monkeypatch.setattr("harness.cli.settings.ab_bd_dir", None)
    monkeypatch.setattr("harness.store.bd_adapter.shutil.which", lambda _: "/usr/local/bin/bd")

    airton = load_character(AIRTON)
    ab = load_character(AIRTON_B)
    assert _maybe_bd_adapter(airton) is None
    assert _maybe_bd_adapter(ab) is None


def test_ab_tool_builders_empty_when_adapter_none() -> None:
    assert _ab_tool_builders(None) == {}


def test_ab_tool_builders_surface_matches_ops_profile() -> None:
    fake = MagicMock(spec=BeadsAdapter)
    builders = _ab_tool_builders(fake)
    assert set(builders) == {
        "plan",
        "capture",
        "status",
        "drift",
        "reprioritize",
        "close",
        "defer",
        "retro",
        "reopen",
        "delete",
        "update",
        "search",
        "list",
        "memories",
        "remember",
        "forget",
        "dep",
        "label",
        "comments",
        "find_duplicates",
        "persist_focus_note",
    }
    # Each builder produces a Tool whose spec.name matches the key —
    # no silent renames between builder map and tool class.
    for name, builder in builders.items():
        tool = builder()
        assert tool is not None
        assert tool.spec.name == name


def test_resolve_adapter_wraps_airton_b_with_caveman_rewriter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """airton_b must get the CavemanRewriter voice layer. Any other
    character keeps PersonaAdapter — branch verified explicitly so
    silent drift fails here."""
    base = MagicMock()
    base.id = "echo"
    base.context_window = 8192
    with patch("harness.cli.make_adapter", return_value=base):
        ab = load_character(AIRTON_B)
        adapter = _resolve_adapter("echo", persona=True, character=ab)

    assert isinstance(adapter, CavemanRewriter)
    # CavemanRewriter wraps the base; same id suffix discipline as
    # PersonaAdapter uses.
    assert adapter.id.startswith("caveman[")


def test_resolve_adapter_keeps_persona_adapter_for_airton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = MagicMock()
    base.id = "echo"
    base.context_window = 8192
    with patch("harness.cli.make_adapter", return_value=base):
        airton = load_character(AIRTON)
        adapter = _resolve_adapter("echo", persona=True, character=airton)

    assert isinstance(adapter, PersonaAdapter)
    assert not isinstance(adapter, CavemanRewriter)


def test_resolve_adapter_threads_chain_rewrites_into_persona_adapter() -> None:
    """Regression: `--chain-rewrites` on chat must reach `PersonaAdapter`
    so the second concrete-substitution pass actually runs. This used
    to silently no-op because the chat command never accepted the flag
    and _resolve_adapter never took the kwarg (harness-n9k). Verify
    both defaults-off and on-demand-on so a future refactor can't
    collapse the flag to a constant."""
    base = MagicMock()
    base.id = "echo"
    base.context_window = 8192
    with patch("harness.cli.make_adapter", return_value=base):
        airton = load_character(AIRTON)

        default_adapter = _resolve_adapter("echo", persona=True, character=airton)
        assert isinstance(default_adapter, PersonaAdapter)
        assert default_adapter.chain_rewrites is False

        chained_adapter = _resolve_adapter(
            "echo", persona=True, character=airton, chain_rewrites=True
        )
        assert isinstance(chained_adapter, PersonaAdapter)
        assert chained_adapter.chain_rewrites is True


def test_resolve_adapter_no_persona_bypasses_rewriter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = MagicMock()
    base.id = "echo"
    base.context_window = 8192
    # No eager .load method — getattr fallback path exercised.
    base.load = None
    with patch("harness.cli.make_adapter", return_value=base):
        ab = load_character(AIRTON_B)
        adapter = _resolve_adapter("echo", persona=False, character=ab)

    # persona=False keeps the raw base adapter regardless of character.
    assert adapter is base


def test_ab_path_end_to_end_shape() -> None:
    """Smoke check: ab's character loads, its ops tool names match the
    ops profile, and every ops tool builder produces a registerable
    Tool. Catches the 'profile says X but builder dict has no X' drift
    that tool-set changes are prone to."""
    from harness.tools.profiles import TOOL_PROFILES

    fake = MagicMock(spec=BeadsAdapter)
    builders = _ab_tool_builders(fake)
    # introspect + tool_search are shared meta-tools built in cli.py's
    # main builder dict, not the ab-specific builder.
    ops_profile = set(TOOL_PROFILES["ops"]) - {"introspect", "tool_search"}
    assert ops_profile.issubset(set(builders)), (
        f"ops profile has tools with no ab-builder: {ops_profile - set(builders)}"
    )


def test_bd_adapter_error_path_logs_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When verify() fails, the helper returns None rather than
    raising — a degraded ab session is still usable, just without ops
    tools. The warning path is exercised via BeadsAdapterError."""
    monkeypatch.setattr("harness.cli.settings.ab_bd_dir", tmp_path)

    def boom(self: object) -> None:
        raise BeadsAdapterError("bd not initialized")

    monkeypatch.setattr(BeadsAdapter, "verify", boom)
    ab = load_character(AIRTON_B)
    assert _maybe_bd_adapter(ab) is None


def test_missing_builder_reason_ops_tool_points_to_bd_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A requested ops tool that didn't bind must surface a bd-dir
    bootstrap hint, not the old 'not yet implemented' message. The
    hint names the actual resolved dir (= settings.root under
    Path 2) so the user knows exactly where bd should run."""
    monkeypatch.setattr("harness.cli.settings.root", tmp_path)
    monkeypatch.setattr("harness.cli.settings.ab_bd_dir", None)
    airton = load_character(AIRTON)
    reason = _missing_builder_reason("plan", airton)
    assert "bd init" in reason
    assert str(tmp_path) in reason
    assert "not yet implemented" not in reason


def test_maybe_bd_adapter_sets_scope_allowlist_for_airton_b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-j7y: airton_b's adapter gets the (professional, personal)
    scope allowlist wired by the factory. airton stays unconstrained
    so the same-dir setup doesn't accidentally narrow its view."""
    (tmp_path / ".beads").mkdir()
    monkeypatch.setattr("harness.cli.settings.root", tmp_path)
    monkeypatch.setattr("harness.cli.settings.ab_bd_dir", None)
    _stub_bd_subprocess(monkeypatch)

    ab_adapter = _maybe_bd_adapter(load_character(AIRTON_B))
    airton_adapter = _maybe_bd_adapter(load_character(AIRTON))

    assert ab_adapter is not None
    assert airton_adapter is not None
    # Private attr read here because it's the contract the _bd_crud
    # mixin consumes; exposing a property is more surface area than
    # this test rates.
    assert ab_adapter._default_scope_allowlist == ("professional", "personal")
    assert airton_adapter._default_scope_allowlist is None


def test_missing_builder_reason_unknown_tool_keeps_old_message() -> None:
    """Genuine typos / forward-compat placeholders still say 'not yet
    implemented' — conserves the message for the case where the user
    actually needs to check the spelling."""
    airton = load_character(AIRTON)
    reason = _missing_builder_reason("totally_made_up_tool", airton)
    assert "not yet implemented" in reason
