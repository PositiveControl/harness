"""Regression: cli.py + cli_classic.py must register the same meta-tools.

Caught by Mark's transcript on --tool-set core_minimal: tool_search
was wired in both builder dicts, but load_tool only landed in the TUI
builder (then cli.py, now cli_tools.py). The classic-REPL path (cli_classic.py) silently
emitted 'tool load_tool not yet implemented — skipping' at session
start, breaking the discovery loop for the most-used chat path.

The right long-term fix is one shared builder-map source. Until that
refactor lands, this test pins the meta-tool surface so a future
addition doesn't drift between the two files again.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The TUI-side builder map moved out of cli.py in step 4 of
# docs/cli-extraction-plan.md (harness-z4k1.1). This pin broke loudly on
# the move, exactly as its docstring predicted; the target is the map's
# new home, not the old file.
CLI_TUI = REPO_ROOT / "src" / "harness" / "cli_tools.py"
CLI_CLASSIC = REPO_ROOT / "src" / "harness" / "cli_classic.py"

# Meta-tools the agent reaches for regardless of chat front-end:
# discovery (tool_search), activation (load_tool), self-inspection
# (introspect). introspect is wired separately as a deferred builder
# in both files, so this list intentionally excludes it.
_REQUIRED_META_TOOLS: tuple[str, ...] = ("tool_search", "load_tool")


def _builder_keys(source_path: Path) -> set[str]:
    """Crude but effective: grep for `"<name>": lambda` patterns in
    the source. Both builder dicts use that exact shape today, so the
    match set is stable. A future structural refactor will break this
    pin loudly — that's the intent."""
    text = source_path.read_text()
    keys: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith('"'):
            continue
        # Match `"name": lambda` (the only place builders go).
        if '": lambda' not in stripped:
            continue
        # Extract the quoted key.
        end = stripped.find('"', 1)
        if end <= 1:
            continue
        keys.add(stripped[1:end])
    return keys


def test_tui_builder_has_required_meta_tools() -> None:
    keys = _builder_keys(CLI_TUI)
    missing = [name for name in _REQUIRED_META_TOOLS if name not in keys]
    assert not missing, f"cli.py builders missing meta-tools: {missing}"


def test_classic_builder_has_required_meta_tools() -> None:
    keys = _builder_keys(CLI_CLASSIC)
    missing = [name for name in _REQUIRED_META_TOOLS if name not in keys]
    assert not missing, (
        f"cli_classic.py builders missing meta-tools: {missing}. "
        "These are the discovery primitives core_minimal depends on; "
        "a missing entry produces 'tool X not yet implemented — skipping' "
        "at session boot. Add it to the cli_classic.py builders dict."
    )


def test_builder_meta_tool_surface_matches_between_files() -> None:
    """The meta-tool set in both files must agree on what's reachable.
    Other tools may legitimately differ (the TUI carries some helpers
    the classic REPL doesn't), but discovery primitives must be
    symmetric — otherwise different chat frontends offer different
    capabilities to the same persona."""
    tui_keys = _builder_keys(CLI_TUI)
    classic_keys = _builder_keys(CLI_CLASSIC)
    for name in _REQUIRED_META_TOOLS:
        in_tui = name in tui_keys
        in_classic = name in classic_keys
        assert in_tui == in_classic, (
            f"meta-tool {name!r} surface diverged: "
            f"cli_tools.py={in_tui}, cli_classic.py={in_classic}"
        )
