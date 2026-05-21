"""Registry tests for the fabrication-catcher hook surface (harness-tm8t).

Enforces three invariants:

1. Every hook in the full opt-in pipeline has a HOOK_SHAPES entry —
   adding a new hook without naming its shape fails fast here instead
   of silently emitting an unlabeled row in docs/hooks.md.

2. HOOK_SHAPES has no orphan entries — every documented name must
   actually appear in the pipeline (with all opt-ins enabled). Catches
   the reverse drift where a hook is removed but its shape lingers.

3. docs/hooks.md matches the current registry — the generator's
   `--check` mode is exercised here so a PR that adds a hook without
   regenerating docs fails in CI rather than shipping stale docs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from harness.orchestrator.hooks import HOOK_SHAPES, default_hook_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]

# Mirror scripts/gen_hook_docs.py:ALL_CATCHERS so the test and the
# generator share the same opt-in surface. Both lists must stay in
# sync; the no-orphans test catches drift between them.
ALL_CATCHERS: tuple[str, ...] = (
    "ab_fabrication",
    "ambiguous_context",
    "scope_redirect",
    "reserved_squawk_code",
    "fetch_url_guard",
    "assemble_context_once",
    "opinion_no_trigger",
    "post_search_grounding",
    "post_research_persist",
    "persist_body_citations",
    "source_count_inflation",
)


def test_every_pipeline_hook_has_a_shape_entry() -> None:
    """Every hook in the full opt-in pipeline must have a registered
    shape descriptor. A missing entry surfaces as '(no shape entry)'
    in describe() — the test pins on this sentinel so the failure
    points at the missing hook by name."""
    pipeline = default_hook_pipeline(catchers=ALL_CATCHERS)
    docs = pipeline.describe()
    missing = [doc.name for doc in docs if doc.shape == "(no shape entry)"]
    assert not missing, (
        f"Hooks in the pipeline lack a HOOK_SHAPES entry: {missing}. "
        "Add them to src/harness/orchestrator/hooks.py:HOOK_SHAPES."
    )


def test_hook_shapes_has_no_orphans() -> None:
    """Every HOOK_SHAPES entry must correspond to a hook that's actually
    registered in the full opt-in pipeline. Documented exceptions:
      - `tool_result_summarizer`: opt-in post_tool hook installed by
        the CLI at --summarize-tool-results, not by default_hook_pipeline.
      - `write_file_redirect` (harness-hnt7): opt-in pre_tool hook
        installed by the CLI when a session has a workspace + a
        registry containing write_file. Wiring lives in
        cli_classic._make_write_file_redirect_hook; the factory takes
        a parameter to install it.
      - `auto_load_on_unknown` (harness-2uso): opt-in pre_tool hook
        installed by the CLI when a session's registry contains
        load_tool. Wiring lives in
        cli_classic._make_auto_load_on_unknown_hook.
    """
    pipeline = default_hook_pipeline(catchers=ALL_CATCHERS)
    pipeline_names = {doc.name for doc in pipeline.describe()}
    documented = set(HOOK_SHAPES)
    cli_installed = {
        "tool_result_summarizer",
        "write_file_redirect",
        "auto_load_on_unknown",
    }
    orphans = documented - pipeline_names - cli_installed
    assert not orphans, (
        f"HOOK_SHAPES entries with no matching pipeline hook: "
        f"{sorted(orphans)}. Either register the hook in "
        "default_hook_pipeline or drop the entry from HOOK_SHAPES."
    )


def test_docs_hooks_md_matches_registry() -> None:
    """docs/hooks.md must match the current pipeline + shapes. The
    generator's --check mode does the comparison."""
    result = subprocess.run(  # noqa: S603  # fixed args, sys.executable + repo-local script
        [sys.executable, str(REPO_ROOT / "scripts" / "gen_hook_docs.py"), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"docs/hooks.md is out of date.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}\n"
        "Run: uv run python scripts/gen_hook_docs.py"
    )


def test_describe_groups_by_phase() -> None:
    """describe() must return entries in canonical phase order so the
    rendered docs and the attribution eval both see the same dispatch
    sequence."""
    pipeline = default_hook_pipeline(catchers=ALL_CATCHERS)
    docs = pipeline.describe()
    phase_order = [doc.phase for doc in docs]
    # post_model first, then bail, then pre_tool, then post_tool,
    # then finalize. Within each, hooks appear in pipeline order.
    seen_phases: list[str] = []
    for phase in phase_order:
        if not seen_phases or seen_phases[-1] != phase:
            seen_phases.append(phase)
    canonical = ["post_model", "bail", "pre_tool", "post_tool", "finalize"]
    # seen_phases is a subset of canonical preserving relative order.
    canonical_idx = [canonical.index(p) for p in seen_phases]
    assert canonical_idx == sorted(canonical_idx), (
        f"describe() emitted phases out of order: {seen_phases}"
    )
