"""Tests for `harness tool ...` — harness-fx24.

Subprocess-based integration coverage. The catalog commands wire
into the existing ToolCatalog (hfa7) via the same load/save +
seed-builtins path. Tests use --catalog-path on tmp_path so the
character-default catalog file (if any) isn't touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_BIN = Path(sys.executable).parent / "harness"
HARNESS_CMD = [str(HARNESS_BIN)]


def _env() -> dict[str, str]:
    base = os.environ.copy()
    for key in ("HARNESS_CHARACTER_NAME", "HARNESS_DATA_DIR"):
        base.pop(key, None)
    return base


def _norm(s: str) -> str:
    return " ".join(s.split())


# --- list -------------------------------------------------------------


def test_tool_list_shows_builtins_with_no_catalog_file(tmp_path: Path) -> None:
    """A never-saved catalog still surfaces the builtin metadata
    table — useful before any synthesize_tool call has written one."""
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "list",
            "--catalog-path",
            str(tmp_path / "no-such.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    out = _norm(proc.stdout)
    # At least the reckon builtins appear.
    assert "now" in out
    assert "reckon" in out


def test_tool_list_filters_by_family(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "list",
            "--family",
            "reckon",
            "--catalog-path",
            str(tmp_path / "cat.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    out = _norm(proc.stdout)
    # Every row is reckon family; pick a couple builtins to verify.
    assert "now" in out
    assert "calc" in out
    # A non-reckon builtin should NOT appear.
    assert "grep" not in out


def test_tool_list_filters_by_tag(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "list",
            "--tag",
            "arithmetic",
            "--catalog-path",
            str(tmp_path / "cat.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    out = _norm(proc.stdout)
    # 'arithmetic' tag is on calc + date_math.
    assert "calc" in out
    assert "date_math" in out
    assert "grep" not in out


def test_tool_list_filters_by_origin(tmp_path: Path) -> None:
    """No synthesized tools in a never-saved catalog → the filter
    matches nothing and we print the 'no entries match' line."""
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "list",
            "--origin",
            "synthesized",
            "--catalog-path",
            str(tmp_path / "cat.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    assert "no catalog entries match" in _norm(proc.stdout)


# --- show -------------------------------------------------------------


def test_tool_show_prints_metadata(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "show",
            "now",
            "--catalog-path",
            str(tmp_path / "cat.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    out = _norm(proc.stdout)
    assert "family : reckon" in out
    assert "time" in out  # tag
    assert "origin : builtin" in out


def test_tool_show_missing_name_exits_nonzero(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "show",
            "definitely-not-a-tool",
            "--catalog-path",
            str(tmp_path / "cat.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 1
    assert "no tool named" in _norm(proc.stdout)


# --- drop -------------------------------------------------------------


def test_tool_drop_refuses_builtin(tmp_path: Path) -> None:
    """Builtins are declared in code — dropping from the catalog file
    would be confusing because the next session would re-seed them.
    Refuse loudly."""
    # Persist a catalog with the seeded builtins so drop has something
    # to operate on (drop on a missing catalog is a no-op + warning,
    # which we test separately).
    cat_path = tmp_path / "cat.json"
    cat_path.write_text(
        json.dumps({"entries": {"now": {"name": "now", "family": "reckon", "origin": "builtin"}}})
    )
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "drop",
            "now",
            "--catalog-path",
            str(cat_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 1
    out = _norm(proc.stdout)
    assert "refusing to drop builtin" in out
    # Catalog file unchanged.
    on_disk = json.loads(cat_path.read_text())
    assert "now" in on_disk["entries"]


def test_tool_drop_removes_synthesized_entry(tmp_path: Path) -> None:
    cat_path = tmp_path / "cat.json"
    source = tmp_path / "fake_synth.py"
    source.write_text("# placeholder\n")
    cat_path.write_text(
        json.dumps(
            {
                "entries": {
                    "fake_synth": {
                        "name": "fake_synth",
                        "family": "meta",
                        "origin": "synthesized",
                        "source_path": str(source),
                    }
                }
            }
        )
    )
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "drop",
            "fake_synth",
            "--catalog-path",
            str(cat_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    on_disk = json.loads(cat_path.read_text())
    assert "fake_synth" not in on_disk["entries"]
    # Source file deleted by default.
    assert not source.exists()


def test_tool_drop_keeps_source_when_flag_set(tmp_path: Path) -> None:
    cat_path = tmp_path / "cat.json"
    source = tmp_path / "fake_synth.py"
    source.write_text("# placeholder\n")
    cat_path.write_text(
        json.dumps(
            {
                "entries": {
                    "fake_synth": {
                        "name": "fake_synth",
                        "origin": "synthesized",
                        "source_path": str(source),
                    }
                }
            }
        )
    )
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "drop",
            "fake_synth",
            "--catalog-path",
            str(cat_path),
            "--keep-source",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    assert source.exists()  # preserved


def test_tool_drop_missing_catalog_is_warning(tmp_path: Path) -> None:
    """Drop against a non-existent catalog file is a friendly no-op,
    not an error — operator may run this before any synthesize call
    has created the file."""
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "drop",
            "anything",
            "--catalog-path",
            str(tmp_path / "absent.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    assert "no catalog file at" in _norm(proc.stdout)


def test_tool_drop_missing_name_in_catalog_is_warning(tmp_path: Path) -> None:
    cat_path = tmp_path / "cat.json"
    cat_path.write_text(json.dumps({"entries": {}}))
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "drop",
            "missing",
            "--catalog-path",
            str(cat_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    assert "not in catalog" in _norm(proc.stdout)


# --- synth-rebuild ---------------------------------------------------


def test_tool_synth_rebuild_no_synthesized_entries(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "synth-rebuild",
            "--catalog-path",
            str(tmp_path / "cat.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    assert "no synthesized tools in catalog" in _norm(proc.stdout)


def test_tool_synth_rebuild_reports_synthesized_entries(tmp_path: Path) -> None:
    cat_path = tmp_path / "cat.json"
    cat_path.write_text(
        json.dumps(
            {
                "entries": {
                    "fake_synth": {
                        "name": "fake_synth",
                        "family": "meta",
                        "origin": "synthesized",
                    }
                }
            }
        )
    )
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "tool",
            "synth-rebuild",
            "--catalog-path",
            str(cat_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    out = _norm(proc.stdout)
    assert "1 synthesized tool(s):" in out
    assert "fake_synth (meta)" in out


# --- top-level + help ---------------------------------------------------


def test_tool_subcommand_listed_in_top_level_help() -> None:
    proc = subprocess.run(
        [*HARNESS_CMD, "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    assert "tool" in proc.stdout


def test_tool_help_lists_subcommands() -> None:
    proc = subprocess.run(
        [*HARNESS_CMD, "tool", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    for subcmd in ("list", "show", "drop", "synth-rebuild"):
        assert subcmd in proc.stdout, f"missing subcommand {subcmd!r} in tool --help"
