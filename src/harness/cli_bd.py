"""bd adapter lifecycle for a chat session.

Step 3 of docs/cli-extraction-plan.md: constructing the per-character
bd adapter, the two session-start harvests that mirror bd state into
episodic memory, and the retro printed on the way out.

Shared failure discipline across the whole module: bd is a subprocess
and an optional one. Every helper here warns and returns rather than
raising, because a flaky `bd` must never stop the user opening — or
closing — a chat session.
"""

from __future__ import annotations

from rich.console import Console

from harness.character import Character
from harness.config import settings
from harness.store.bd_adapter import BeadsAdapter, BeadsAdapterError
from harness.store.episodic import EpisodicStore
from harness.tools.ab_ops import RetroTool

console = Console()


def _print_session_end_retro(ab_adapter: BeadsAdapter | None) -> None:
    """Render RetroTool's summary view at graceful session exit
    (/exit, :q, Ctrl-C). Skipped for non-ab sessions and on retro
    failure — the retro is a convenience, not a blocker on shutdown.
    Crash/kill paths intentionally don't hit this (no atexit); an
    unreliable retro is worse than none."""
    if ab_adapter is None:
        return
    try:
        summary = RetroTool(ab_adapter).call(mode="summary")
    except Exception:  # exit path; never raise on shutdown
        return
    console.print(f"[dim]{summary}[/dim]")


def _maybe_bd_adapter(
    character: Character,
    *,
    include_internal: bool = False,
) -> BeadsAdapter | None:
    """Construct a bd adapter for the active character's ops plane.
    Returns None (with a yellow warning) when bd isn't runnable or the
    character's bd dir hasn't been bootstrapped — ops tools then skip
    registration with a hint. Works for any character: per-character
    bd dirs (~/.harness/<name>/) mirror the memory-store isolation,
    so airton and airton_b never cross thought-graphs.

    `include_internal=False` (default) hides ab-owned thought-graph
    beads (assignee=airton_b) from read views. `--dev` or explicit
    `--include-internal` on the CLI flip this on. The filter is a
    no-op for characters with no airton_b-assigned beads; keeping it
    uniform avoids branching on character name here."""
    bd_dir = settings.bd_dir_for(character.name)
    # bd-graph identity comes from core.yaml (harness-a2sa). The
    # exclude assignee is suppressed when --include-internal is set;
    # the scope_allowlist + ab_assignee come straight from the loaded
    # character. Empty scope_allowlist becomes None so the adapter's
    # "no filter" path runs.
    exclude = None if include_internal else character.bd_exclude_assignee
    scope_allowlist = character.bd_scope_allowlist or None
    adapter = BeadsAdapter(
        bd_dir,
        default_exclude_assignee=exclude,
        ab_assignee=character.bd_assignee,
        default_scope_allowlist=scope_allowlist,
        turn_cap=settings.ab_turn_cap,
        inflight_cap=settings.ab_inflight_cap,
    )
    try:
        adapter.verify()
    except BeadsAdapterError as exc:
        console.print(f"[yellow]⚠ ops tools unavailable: {exc}[/yellow]")
        return None
    # Surface which path the adapter landed on so misconfigured
    # HARNESS_AB_BD_DIR (or missing bootstrap) is visible at session
    # start rather than silently writing to the wrong DB.
    console.print(f"[dim]{character.name} bd → {bd_dir}[/dim]")
    return adapter


# Back-compat alias — external callers (tests, scripts) still import
# the old name. Remove once all call sites are migrated.
_maybe_ab_bd_adapter = _maybe_bd_adapter


def _maybe_harvest_skills(
    ab_adapter: BeadsAdapter | None,
    episodic: EpisodicStore | None,
    *,
    enabled: bool = True,
) -> None:
    """Run the bd → episodic skill harvester at session start, when
    both halves of the substrate are available.

    Idempotent on external_id (bead id), so the steady-state cost is
    one `bd list` call + zero embeds. The first run after new
    decisions / observations close batches the new rows through the
    embedder — still fast (<1 s) for realistic working-set sizes.

    Failures log a yellow warning but never raise: self-improvement is
    a comfort, not a correctness requirement, and a flaky bd
    subprocess must not block the user from opening chat."""
    if not enabled or ab_adapter is None or episodic is None:
        return
    try:
        from harness.skills import harvest_bd_skills

        report = harvest_bd_skills(ab_adapter=ab_adapter, episodic=episodic)
    except Exception as exc:
        console.print(f"[yellow]⚠ skill harvest skipped: {exc}[/yellow]")
        return
    if report.newly_ingested > 0:
        console.print(
            f"[dim]harvested {report.newly_ingested} new skill(s) from bd "
            f"({report.already_present} already present)[/dim]"
        )


def _maybe_harvest_bd_memories(
    ab_adapter: BeadsAdapter | None,
    episodic: EpisodicStore | None,
    *,
    enabled: bool = True,
) -> None:
    """Mirror bd's persistent memories into the episodic store at
    session start so the main retrieval path can surface them on
    identity / biographical questions (harness-9yd).

    Same failure discipline as `_maybe_harvest_skills`: warn on
    failure, never raise — a flaky bd subprocess must not block chat
    startup. Idempotent on `external_id='bd-mem:<key>'`, so steady-
    state cost is one `bd memories --json` call plus zero embeds."""
    if not enabled or ab_adapter is None or episodic is None:
        return
    try:
        from harness.skills import harvest_bd_memories

        report = harvest_bd_memories(ab_adapter=ab_adapter, episodic=episodic)
    except Exception as exc:
        console.print(f"[yellow]⚠ bd memory harvest skipped: {exc}[/yellow]")
        return
    if report.newly_ingested > 0:
        console.print(
            f"[dim]harvested {report.newly_ingested} new bd memorie(s) "
            f"({report.already_present} already present)[/dim]"
        )
