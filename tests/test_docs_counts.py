"""Docs must agree with the registry on how many tools and profiles exist.

CLAUDE.md is agent-facing context: a stale count there misdirects every
session that reads it. The counts drifted 3x before anyone noticed
(README said 17 tools and ~470 tests while the catalog held 60 and the
suite 4,290 — harness-3if1). These tests fail the moment a tool or
profile lands without its doc line moving, which is cheaper than another
sweep.

Deliberately NOT asserted: the test count itself. It changes on every
commit that adds a test, and pinning it would make the suite fail on its
own growth — the docs say "~4,290" as an order-of-magnitude cue, and a
stale approximation there is not a correctness problem.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from harness.tools.catalog import BUILTIN_TOOL_METADATA
from harness.tools.profiles import TOOL_PROFILES

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_README = _REPO_ROOT / "README.md"
_ROADMAP = _REPO_ROOT / "docs" / "roadmap.md"


@pytest.mark.parametrize("doc", [_CLAUDE_MD, _README], ids=["CLAUDE.md", "README.md"])
def test_doc_states_current_tool_count(doc: Path) -> None:
    """Each doc names the catalog size verbatim, e.g. "60 built-in tools"."""
    expected = len(BUILTIN_TOOL_METADATA)
    text = doc.read_text()
    counts = {int(m) for m in re.findall(r"(\d+) built-in tools", text)}
    assert counts, f"{doc.name} no longer states a built-in tool count"
    assert counts == {expected}, (
        f"{doc.name} says {sorted(counts)} built-in tools; BUILTIN_TOOL_METADATA holds {expected}"
    )


@pytest.mark.parametrize(
    "doc", [_CLAUDE_MD, _README, _ROADMAP], ids=["CLAUDE.md", "README.md", "roadmap.md"]
)
def test_doc_states_current_profile_count(doc: Path) -> None:
    """Each doc names the profile count, e.g. "15 named tool-set profiles"."""
    expected = len(TOOL_PROFILES)
    text = doc.read_text()
    counts = {int(m) for m in re.findall(r"(\d+) (?:named )?(?:tool-set )?profiles", text)}
    assert counts, f"{doc.name} no longer states a tool-set profile count"
    assert counts == {expected}, (
        f"{doc.name} says {sorted(counts)} profiles; TOOL_PROFILES holds {expected}"
    )


@pytest.mark.parametrize(
    "doc", [_CLAUDE_MD, _README, _ROADMAP], ids=["CLAUDE.md", "README.md", "roadmap.md"]
)
def test_doc_names_every_profile(doc: Path) -> None:
    """Every profile name appears somewhere in the doc.

    Catches the drift shape that hurt most: a doc that lists profiles
    inline and quietly omits the character-scoped ones added later.
    """
    text = doc.read_text()
    missing = sorted(name for name in TOOL_PROFILES if f"`{name}`" not in text)
    assert not missing, f"{doc.name} never names these profiles: {missing}"


def test_claude_md_covers_every_source_subpackage() -> None:
    """CLAUDE.md's repo-layout list mentions every src/harness subpackage.

    driver/, plan/, notam/, runtime/, turn/, web/ and character_templates/
    all shipped without ever reaching the layout list.
    """
    text = _CLAUDE_MD.read_text()
    packages = sorted(
        p.name
        for p in (_REPO_ROOT / "src" / "harness").iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    )
    missing = [name for name in packages if f"`{name}/" not in text]
    assert not missing, f"CLAUDE.md repo layout omits: {missing}"


def test_readme_lists_every_character() -> None:
    """The README character table covers every character/ directory."""
    text = _README.read_text()
    characters = sorted(
        p.name
        for p in (_REPO_ROOT / "character").iterdir()
        if p.is_dir() and (p / "core.yaml").is_file()
    )
    missing = [name for name in characters if f"**{name}**" not in text]
    assert not missing, f"README character table omits: {missing}"
