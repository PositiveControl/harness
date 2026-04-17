"""Tests for the chat loading header renderer. Covers the on-state
and off-state paths so we can trust changes to the rendering without
spinning up the full chat loop."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from harness.cli import _render_chat_header


def _render(**overrides: object) -> str:
    """Call _render_chat_header with sensible defaults, allowing test
    overrides. Captures the Rich output to a buffer and strips ANSI so
    assertions can work with plain text."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    defaults: dict[str, object] = {
        "console": console,
        "character_name": "airton",
        "session": "local",
        "speaker": "mark",
        "adapter_id": "mlx:Qwen2.5-7B-Instruct-4bit",
        "lora_path": None,
        "persona": True,
        "top_k": 6,
        "retriever_active": True,
        "memories": 3,
        "memories_threshold": 0.5,
        "memories_active": True,
        "facts": 5,
        "facts_threshold": 0.45,
        "facts_active": True,
        "tools_enabled": True,
        "tool_set": "coding",
        "tool_names": [
            "read_file",
            "edit_file",
            "write_file",
            "shell",
            "list_dir",
            "grep",
            "glob",
            "git_status",
            "git_diff",
            "git_log",
            "search_memory",
            "search_facts",
        ],
        "workspace_path": Path("/Users/airton/workspace"),
        "rewrite_on_tools": False,
        "compact_at": 0.8,
        "compact_keep_recent": 10,
        "dev": False,
    }
    defaults.update(overrides)
    _render_chat_header(**defaults)  # type: ignore[arg-type]
    return buf.getvalue()


def test_header_includes_character_name() -> None:
    out = _render()
    assert "airton" in out
    assert "◈" in out  # pseudo-logo glyph


def test_header_shows_session_and_speaker() -> None:
    out = _render(session="nightly", speaker="alice")
    assert "nightly" in out
    assert "alice" in out


def test_header_shows_model_id() -> None:
    out = _render(adapter_id="ollama:qwen2.5-coder:32b-instruct")
    assert "ollama:qwen2.5-coder:32b-instruct" in out


def test_header_shows_lora_path_when_set() -> None:
    out = _render(lora_path="/Users/airton/adapters/lora-run-1")
    assert "+lora" in out
    assert "lora-run-1" in out


def test_header_persona_on() -> None:
    out = _render(persona=True)
    assert "persona" in out
    assert "on" in out


def test_header_persona_off() -> None:
    out = _render(persona=False)
    assert "persona" in out
    assert "off" in out


def test_header_retrieval_shows_counts_and_thresholds() -> None:
    out = _render(top_k=4, memories=2, memories_threshold=0.6, facts=3, facts_threshold=0.5)
    # \u00d7 = multiplication sign, matches the renderer's exact glyph.
    assert "voice\u00d74" in out
    assert "memories\u00d72" in out
    assert "0.60" in out
    assert "facts\u00d73" in out
    assert "0.50" in out


def test_header_retrieval_off_when_all_disabled() -> None:
    out = _render(
        retriever_active=False,
        memories_active=False,
        facts_active=False,
    )
    # 'off' appears somewhere in the retrieval row.
    assert "retrieval" in out
    assert "off" in out


def test_header_tools_on_shows_profile_and_count() -> None:
    out = _render()
    assert "coding" in out
    assert "12 tools" in out
    # At least the first few tool names visible
    assert "read_file" in out
    assert "edit_file" in out
    # Remainder count for the collapsed tail
    assert "…+6" in out


def test_header_tools_off() -> None:
    out = _render(tools_enabled=False, tool_names=[])
    assert "tools" in out
    assert "off" in out


def test_header_workspace_shown_only_when_tools_on() -> None:
    out_off = _render(tools_enabled=False, tool_names=[], workspace_path=None)
    assert "workspace" not in out_off
    out_on = _render(workspace_path=Path.home() / "dev" / "ideas" / "harness")
    assert "workspace" in out_on
    assert "~/dev/ideas/harness" in out_on


def test_header_rewrite_on_tools_visible_when_set() -> None:
    out = _render(rewrite_on_tools=True)
    assert "rewrite-on-tools" in out
    out_off = _render(rewrite_on_tools=False)
    # Off state doesn't clutter the header.
    assert "rewrite-on-tools" not in out_off


def test_header_compact_shows_percent_and_keep() -> None:
    out = _render(compact_at=0.75, compact_keep_recent=8)
    assert "75%" in out
    assert "keep 8" in out


def test_header_compact_off() -> None:
    out = _render(compact_at=0.0)
    assert "compact" in out
    assert "off" in out


def test_header_dev_mode_flag_shown() -> None:
    assert "dev" not in _render(dev=False).split("mode")[-1]
    out_on = _render(dev=True)
    assert "mode" in out_on
    assert "dev" in out_on


def test_header_handles_workspace_outside_home() -> None:
    out = _render(workspace_path=Path("/opt/harness"))
    assert "/opt/harness" in out
