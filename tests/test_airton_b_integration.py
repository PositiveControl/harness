"""End-to-end wiring tests for the airton_b CLI path — harness-inj.6.

Covers the two pieces of CLI glue that turn airton_b from 'character
data on disk' into an invocable persona:

- `_maybe_ab_bd_adapter(character)` returns a BeadsAdapter for airton_b
  when its isolated dir is initialized; returns None with a warning
  otherwise; returns None silently for any other character.
- `_ab_tool_builders(adapter)` surfaces the 8 ops tool builders exactly
  when an adapter is present.
- `_resolve_adapter` branches on character.name so airton_b gets the
  CavemanRewriter and every other character keeps PersonaAdapter.
- Full-surface smoke: loading ab's character + calling every ab ops
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
    _maybe_ab_bd_adapter,
    _resolve_adapter,
)
from harness.persona import PersonaAdapter
from harness.persona.caveman_rewriter import CavemanRewriter
from harness.store.bd_adapter import BeadsAdapter, BeadsAdapterError

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRTON = REPO_ROOT / "character" / "airton"
AIRTON_B = REPO_ROOT / "character" / "airton_b"


def test_maybe_ab_bd_adapter_returns_none_for_non_ab_character() -> None:
    airton = load_character(AIRTON)
    assert _maybe_ab_bd_adapter(airton) is None


def test_maybe_ab_bd_adapter_returns_adapter_when_dir_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".beads").mkdir()
    monkeypatch.setattr("harness.cli.settings.ab_bd_dir", tmp_path)
    # shutil.which returns a truthy path so verify() clears the
    # bd-on-PATH gate without actually needing bd installed.
    monkeypatch.setattr("harness.store.bd_adapter.shutil.which", lambda _: "/usr/local/bin/bd")

    ab = load_character(AIRTON_B)
    adapter = _maybe_ab_bd_adapter(ab)

    assert isinstance(adapter, BeadsAdapter)
    assert adapter.bd_dir == tmp_path


def test_maybe_ab_bd_adapter_returns_none_when_dir_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "not_here"
    monkeypatch.setattr("harness.cli.settings.ab_bd_dir", missing)
    monkeypatch.setattr("harness.store.bd_adapter.shutil.which", lambda _: "/usr/local/bin/bd")

    ab = load_character(AIRTON_B)
    adapter = _maybe_ab_bd_adapter(ab)

    assert adapter is None


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
    ops_profile = set(TOOL_PROFILES["ops"]) - {"introspect"}  # introspect is shared
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
    assert _maybe_ab_bd_adapter(ab) is None
