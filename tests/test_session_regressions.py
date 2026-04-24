"""Regression tests for user-visible bugs caught during live chat.

Each test names the failing interaction, the root cause, and the
commit(s) that landed the fix — so a future breakage points straight
at the context instead of forcing the next owner to archaeology-dive
through git.

Two-layer strategy:

1. **Catcher / orchestrator regressions** (always run): scripted model
   adapter + mock registry + real hook pipeline. Fast, deterministic,
   no external state. Anchor the hook-pipeline behavior for the exact
   reply shapes users actually saw.

2. **Retrieval regressions** (auto-skip when the airton_c1 store isn't
   populated): use the real episodic store at
   `character/airton_c1/data/harness.sqlite`. Requires the corpus to
   have been ingested via `HARNESS_CHARACTER_NAME=airton_c1 uv run
   python scripts/atc_ingest.py` at least once. BM25 rankings depend
   on the full corpus vocabulary, so mocking a subset would give
   misleading ranks.

Growth pattern: each new user-visible miss caught in chat becomes a
new test function here before the fix ships. The test fails on the
pre-fix code and passes on the post-fix code — that's the permanent
regression anchor.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.evals.tool_loop import _run_scenario
from harness.orchestrator.hooks import TABLE_FABRICATION_FALLBACK

_ROOT = Path(__file__).resolve().parents[1]
_AIRTON_C1_DB = _ROOT / "character" / "airton_c1" / "data" / "harness.sqlite"


def _airton_c1_store_available() -> bool:
    """True when the airton_c1 episodic store exists and has rows.
    Tests that depend on real retrieval skip when False."""
    if not _AIRTON_C1_DB.exists():
        return False
    import sqlite3

    try:
        conn = sqlite3.connect(_AIRTON_C1_DB)
        cur = conn.execute("SELECT COUNT(*) FROM episodic")
        count = int(cur.fetchone()[0])
        conn.close()
    except sqlite3.Error:
        return False
    return count > 100


_REQUIRES_STORE = pytest.mark.skipif(
    not _airton_c1_store_available(),
    reason=(
        "airton_c1 episodic store empty or missing — run "
        "`HARNESS_CHARACTER_NAME=airton_c1 uv run python scripts/atc_ingest.py` "
        "first."
    ),
)


# ---------- orchestrator / catcher regressions (always run) ----------


# Tool output SearchMemoryTool would produce for the MH RBN repro with
# the harness-5uq post-fix cap (body[:1200]): both §4-1-1 chunks visible
# with the full TBL 4-1-2 rows intact. Reused by multiple scenarios
# below so a chunker change that truncates the MH row shows up here,
# not just in the live chat.
_MH_RBN_TOOL_OUTPUT = (
    "[0.033] ALTITUDE AND DISTANCE LIMITATIONS\n"
    "  lesson: JO_7110.65 §4-1-1\n"
    "  |Class|Power (watts)|Distance (miles)|\n"
    "|---|---|---|\n"
    "|CL|Under 25|15|\n"
    "|MH|Under 50|25|\n"
    "|H|50 - 1,999|50|\n"
    "|HH|2,000 or more|75|\n"
)


def test_mh_rbn_prose_fabrication_halts_via_numeric_fabrication() -> None:
    """Session 2026-04-24 repro: after the sanitizer cap bump, model
    still emits prose 'MH class RBN is 50 miles' when truth is 25.
    NumericFabricationHook must halt with the canned refusal.

    Fix landed in commit 3145126 (prose-form catcher + constitution
    quote-or-abstain rule).
    """
    result = _run_scenario(
        {
            "id": "_mh_rbn_prose_fabrication",
            "registry": [{"name": "search_memory", "output": _MH_RBN_TOOL_OUTPUT}],
            "messages": [
                {
                    "role": "user",
                    "content": "What is the usable distance for an MH class RBN?",
                }
            ],
            "scripted_replies": [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "search_memory",
                            "arguments": {"query": "MH class RBN usable distance"},
                        }
                    ],
                },
                {
                    "content": (
                        "The usable distance for an MH class RBN is 50 miles, "
                        "per JO 7110.65 §4-1-1."
                    ),
                },
            ],
        }
    )
    assert result.final_content == TABLE_FABRICATION_FALLBACK
    # The fabricated value must not leak through.
    assert "50 miles" not in result.final_content


def test_mh_rbn_table_fabrication_halts_via_table_fabrication() -> None:
    """Same root bug, table-shape reply: model echoes a TBL 4-1-2 with
    the MH row's distance wrong ('|MH|Under 50|50|'). TableFabrication
    halts on the verbatim-row compare.

    Fix landed in commit 79f355f (TableFabricationHook).
    """
    result = _run_scenario(
        {
            "id": "_mh_rbn_table_fabrication",
            "registry": [{"name": "search_memory", "output": _MH_RBN_TOOL_OUTPUT}],
            "messages": [{"role": "user", "content": "Show me TBL 4-1-2."}],
            "scripted_replies": [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "search_memory",
                            "arguments": {"query": "TBL 4-1-2"},
                        }
                    ],
                },
                {
                    "content": (
                        "Here is TBL 4-1-2:\n\n"
                        "| Class | Power (watts) | Distance (miles) |\n"
                        "|---|---|---|\n"
                        "| MH | Under 50 | 50 |\n"  # fabricated — truth is 25
                        "| H | 50 - 1,999 | 50 |\n"
                    ),
                },
            ],
        }
    )
    assert result.final_content == TABLE_FABRICATION_FALLBACK
    assert "| MH | Under 50 | 50 |" not in result.final_content


_PURPOSE_1_1_1_TOOL_OUTPUT = (
    "[0.033] PURPOSE OF THIS ORDER\n"
    "  lesson: JO_7110.65 §1-1-1 (Introduction — PURPOSE OF THIS ORDER)\n"
    "  This order prescribes air traffic control procedures and "
    "phraseology for use by persons providing air traffic control "
    "services. Controllers are required to be familiar with the "
    "provisions of this order that pertain to their operational "
    "responsibilities.\n"
)


_LOA_1_1_10_TOOL_OUTPUT = (
    "[0.033] PROCEDURAL LETTERS OF AGREEMENT (LOA)\n"
    "  lesson: JO_7110.65 §1-1-10 (Introduction — PROCEDURAL LETTERS OF AGREEMENT)\n"
    "  When procedures or minima are applied jointly or otherwise "
    "require the cooperation or concurrence of more than one facility "
    "or organization, they must be documented in an LOA. LOAs only "
    "supplement this order.\n"
)


def test_purpose_of_order_uncited_reply_gets_nudged() -> None:
    """Session 2026-04-24 repro: user asked 'What is the purpose of
    7110.65 as it pertains to ATC?', search_memory surfaced §1-1-1,
    model paraphrased correctly but omitted the §-anchor. The
    missing_citation bail hook must nudge; the retry round supplies
    the cited version.

    Fix landed in commit (TBD — this session): MissingCitationHook.
    """
    result = _run_scenario(
        {
            "id": "_purpose_of_order_uncited",
            "registry": [
                {"name": "search_memory", "output": _PURPOSE_1_1_1_TOOL_OUTPUT}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What is the purpose of 7110.65 as it pertains to "
                        "Air Traffic Control?"
                    ),
                }
            ],
            "scripted_replies": [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "search_memory",
                            "arguments": {"query": "purpose of 7110.65 ATC"},
                        }
                    ],
                },
                # Uncited reply — what the model actually produced.
                {
                    "content": (
                        "The purpose of FAA Order JO 7110.65 as it pertains "
                        "to Air Traffic Control is to prescribe air traffic "
                        "control procedures and phraseology for use by "
                        "persons providing air traffic control services. "
                        "The order is designed to ensure uniform ATC rules "
                        "across the National Airspace System."
                    ),
                },
                # Retry after the nudge — cited form.
                {
                    "content": (
                        "Per JO 7110.65 §1-1-1, the purpose of the order is "
                        "to prescribe air traffic control procedures and "
                        "phraseology for use by persons providing air "
                        "traffic control services."
                    ),
                },
            ],
        }
    )
    assert "§1-1-1" in result.final_content
    assert "JO 7110.65" in result.final_content


def test_loa_uncited_reply_gets_nudged() -> None:
    """Session 2026-04-24 repro 2: user asked 'What document is
    required concerning procedures that will be jointly applied
    between facilities?', search_memory surfaced §1-1-10 (Procedural
    Letters of Agreement), model answered 'According to JO 7110.65'
    with no section anchor. Missing citation nudged; retry produces
    the cited version."""
    result = _run_scenario(
        {
            "id": "_loa_uncited",
            "registry": [
                {"name": "search_memory", "output": _LOA_1_1_10_TOOL_OUTPUT}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What document is required concerning procedures "
                        "that will be jointly applied between facilities?"
                    ),
                }
            ],
            "scripted_replies": [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "search_memory",
                            "arguments": {
                                "query": "procedures jointly applied between facilities"
                            },
                        }
                    ],
                },
                # Uncited reply as observed.
                {
                    "content": (
                        "According to JO 7110.65, when procedures or minima "
                        "are applied jointly or otherwise require the "
                        "cooperation or concurrence of more than one "
                        "facility or organization, they must be documented "
                        "in a Procedural Letter of Agreement (LOA)."
                    ),
                },
                # Retry after the nudge — cited form.
                {
                    "content": (
                        "Per JO 7110.65 §1-1-10, procedures applied jointly "
                        "between facilities must be documented in a "
                        "Procedural Letter of Agreement (LOA)."
                    ),
                },
            ],
        }
    )
    assert "§1-1-10" in result.final_content
    assert "LOA" in result.final_content


_ATC_SERVICE_2_1_1_TOOL_OUTPUT = (
    "[0.031] ATC SERVICE\n"
    "  lesson: JO_7110.65 §2-1-1 (General — ATC SERVICE)\n"
    "  **a.** The primary purpose of the ATC system is to prevent a "
    "collision involving aircraft operating in the system.\n"
    "  **b.** In addition to its primary purpose, the ATC system also:\n"
    "    **1.** Provides a safe, orderly, and expeditious flow of air traffic.\n"
    "    **2.** Supports National Security and Homeland Defense missions.\n"
)


def test_four_primary_purposes_count_mismatch_gets_nudged() -> None:
    """Session 2026-04-24 repro: user asked for '4 specific Primary
    Purposes'. Source §2-1-1 has 3 items (1 primary + 2 additional
    roles). Model over-agreed with the wrong premise, wrote 'The four
    specific primary purposes ... are as follows:' then listed 3 —
    self-falsifying within one reply.

    Fix landed in this session via ListCountMismatchHook. Retry round
    must restate the count to match what the source actually lists,
    and should flag that the question's premise was off.
    """
    result = _run_scenario(
        {
            "id": "_four_purposes_count_mismatch",
            "registry": [
                {"name": "search_memory", "output": _ATC_SERVICE_2_1_1_TOOL_OUTPUT}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "List the 4 specific Primary Purposes of Air Traffic Control"
                    ),
                }
            ],
            "scripted_replies": [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "search_memory",
                            "arguments": {"query": "primary purposes of ATC"},
                        }
                    ],
                },
                # Over-agreeing reply (observed).
                {
                    "content": (
                        "The four specific primary purposes of Air Traffic "
                        "Control (ATC) are as follows:\n\n"
                        "1. Prevent a collision involving aircraft operating "
                        "in the system.\n"
                        "2. Provide a safe, orderly, and expeditious flow of "
                        "air traffic.\n"
                        "3. Support National Security and Homeland Defense "
                        "missions.\n\n"
                        "These are outlined in JO 7110.65 §2-1-1."
                    ),
                },
                # Retry after the count-mismatch nudge — restates count
                # to match the source and calls out the premise gap.
                {
                    "content": (
                        "Per JO 7110.65 §2-1-1, the source lists 3 items (1 "
                        "primary purpose plus 2 additional roles), not 4:\n\n"
                        "1. Prevent a collision involving aircraft (the primary "
                        "purpose).\n"
                        "2. Provide a safe, orderly, and expeditious flow of "
                        "air traffic.\n"
                        "3. Support National Security and Homeland Defense "
                        "missions."
                    ),
                },
            ],
        }
    )
    assert "3 items" in result.final_content
    assert "§2-1-1" in result.final_content


def test_mh_rbn_truthful_prose_passes_pipeline() -> None:
    """Counterfactual: truthful prose reply ('25 miles, §4-1-1') with
    the same tool output must NOT trip any catcher. Guards against
    NumericFabricationHook drifting into false-positive territory."""
    result = _run_scenario(
        {
            "id": "_mh_rbn_truthful_prose",
            "registry": [{"name": "search_memory", "output": _MH_RBN_TOOL_OUTPUT}],
            "messages": [
                {
                    "role": "user",
                    "content": "What is the usable distance for an MH class RBN?",
                }
            ],
            "scripted_replies": [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "search_memory",
                            "arguments": {"query": "MH class RBN usable distance"},
                        }
                    ],
                },
                {
                    "content": (
                        "The usable distance for an MH class RBN is 25 miles, "
                        "per JO 7110.65 §4-1-1."
                    ),
                },
            ],
        }
    )
    assert "25 miles" in result.final_content
    assert TABLE_FABRICATION_FALLBACK not in result.final_content


# ---------- retrieval regressions (skip when store empty) ----------


@_REQUIRES_STORE
def test_aircraft_to_aircraft_retrieval_lands_13_1_2() -> None:
    """Session 2026-04-24 repro: 'Where can a controller find
    information for aircraft-to-aircraft alerts?' must surface §13-1-2
    in the top-K default k the model sees via SearchMemoryTool.

    Fix landed in commits 20f2bce (sanitize_fts_query compound phrase
    pass) + 42a2f04 (SearchMemoryTool default k 5 -> 8). Guards both
    knobs from silent regression.
    """
    # Deferred imports so --collect-only doesn't pay the embedder load
    # cost when the skip marker fires.
    from harness.retrieval.st_embedder import SentenceTransformersEmbedder
    from harness.store.episodic import EpisodicStore
    from harness.tools.search_memory import SearchMemoryTool

    embedder = SentenceTransformersEmbedder()
    store = EpisodicStore(_AIRTON_C1_DB, embedder=embedder)
    tool = SearchMemoryTool(store=store)
    try:
        out = tool.call(
            query="Where can a controller find information for aircraft-to-aircraft alerts?"
        )
    finally:
        store.close()
    # §13-1-2 must appear in the default tool output so the model sees
    # it. Exact string match on the principle anchor.
    assert "§13-1-2" in out, (
        "§13-1-2 (CONFLICT DETECTION AND RESOLUTION) did not surface in "
        "the default SearchMemoryTool call — the sanitize_fts_query "
        "compound-phrase pass and/or the k=8 default are not landing."
    )


@_REQUIRES_STORE
def test_mh_rbn_tool_output_contains_full_tbl_412() -> None:
    """Session 2026-04-24 repro: tool output must include the full
    TBL 4-1-2 rows (CL/MH/H/HH with distance values). Guards the
    SearchMemoryTool body-cap bump (400 -> 1200) in commit 79f355f
    against a future regression that re-truncates the table before
    the MH row.
    """
    from harness.retrieval.st_embedder import SentenceTransformersEmbedder
    from harness.store.episodic import EpisodicStore
    from harness.tools.search_memory import SearchMemoryTool

    embedder = SentenceTransformersEmbedder()
    store = EpisodicStore(_AIRTON_C1_DB, embedder=embedder)
    tool = SearchMemoryTool(store=store)
    try:
        out = tool.call(query="What is the usable distance for an MH class RBN?")
    finally:
        store.close()
    # Normalize for the comparison the way NumericFabricationHook does
    # — whitespace + commas stripped — so trivial formatting variance
    # in the stored body doesn't false-positive this test.
    normalized = out.replace(" ", "").replace(",", "")
    for row_sig in ("|CL|Under25|15|", "|MH|Under50|25|", "|HH|2000ormore|75|"):
        assert row_sig in normalized, (
            f"Tool output missing TBL 4-1-2 row signature {row_sig!r} — "
            "SearchMemoryTool._BODY_CAP_CHARS may have regressed."
        )


# ---------- growth-guard meta-test ----------


def test_session_regressions_file_grows_with_bugs() -> None:
    """Lightweight sentinel: the docstring at the top of this module
    says bugs get added here. Assert the file contains at least N test
    functions so an accidental truncation is noisy."""
    source = Path(__file__).read_text()
    # Count test function definitions in THIS module. Plain substring
    # check is fine — no need for ast introspection.
    n = source.count("\ndef test_")
    assert n >= 4, (
        f"test_session_regressions.py has only {n} test functions — "
        "expected at least the 4 seeded session-2026-04-24 regressions. "
        "Did a truncation drop cases?"
    )


# ---------- self-skip hint ----------
# When the store is missing, the two gated tests skip cleanly. Print a
# hint once at import time so contributors know how to enable them.
if __name__ == "__main__":  # pragma: no cover — doc hint, not a runtime path
    avail = _airton_c1_store_available()
    print(f"airton_c1 store available: {avail} @ {_AIRTON_C1_DB}")
    if not avail:
        print(
            "Run `HARNESS_CHARACTER_NAME=airton_c1 uv run python "
            "scripts/atc_ingest.py` to enable retrieval regressions."
        )


# Used by the @pytest.mark.skipif decorator above — keep the name in
# the module so `ruff` doesn't flag the import as unused.
_ = os.environ  # no-op; decorator evaluates at collection time
