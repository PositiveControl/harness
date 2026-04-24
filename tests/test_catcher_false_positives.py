"""False-positive corpus for fabrication catchers.

The attribution eval (`tests/test_tool_loop_eval.py`) proves every
catcher saves at least one scenario from a known failure. This file
proves the complementary property: every catcher **passes** a corpus
of known-correct replies without Halting or Nudging.

Why the split matters: a catcher that never false-fires but also
never catches a real fabrication is useless (attribution covers that).
A catcher that catches real fabrications but ALSO halts truthful
replies is worse than useless — it hides legitimate answers behind a
canned refusal and the user can't tell the difference.

Corpus shape — each case is a (reply, tool_outputs, tools_ran,
memory_block_attached, description) tuple. Each case must produce
`Continue` from BOTH `run_bail` (with `tools_ran_this_turn=True` when
tools_ran is populated) and `run_finalize`. Grow by adding tuples —
do NOT parameterize by catcher (we want the full default pipeline's
verdict, not one-hook-at-a-time; false positives in combination are
the real risk).

Growth pattern: when tightening a catcher's regex in response to a
real fabrication, add the truthful counterfactual here before landing
the tighter regex. The test-first pressure forces the author to
consider the false-positive risk explicitly.
"""

from __future__ import annotations

import pytest

from harness.orchestrator.hooks import (
    BailContext,
    Continue,
    FinalizeContext,
    default_hook_pipeline,
)
from harness.tools.base import ModelReply


def _reply(content: str) -> ModelReply:
    return ModelReply(
        content=content,
        tool_calls=(),
        was_truncated=False,
        had_unparseable_call=False,
    )


# Tool output fixture: full TBL 4-1-2 visible (post harness-5uq cap
# bump). Used by the numeric-grounded cases. Kept identical to what
# the real SearchMemoryTool would emit for the MH RBN query so the
# assertions reflect production shape.
_TBL_412 = (
    "[0.033] ALTITUDE AND DISTANCE LIMITATIONS\n"
    "  lesson: JO_7110.65 §4-1-1\n"
    "  |Class|Power (watts)|Distance (miles)|\n"
    "|---|---|---|\n"
    "|CL|Under 25|15|\n"
    "|MH|Under 50|25|\n"
    "|H|50 - 1,999|50|\n"
    "|HH|2,000 or more|75|\n"
)


# EDST / §13-1-2 tool output — the aircraft-to-aircraft case's source.
# Used by citations that quote §13-1-2(a)'s 'scan EDST' guidance.
_EDST_13_1_2 = (
    "[0.045] CONFLICT DETECTION AND RESOLUTION\n"
    "  lesson: JO_7110.65 §13-1-2 (ERAM - En Route — CONFLICT DETECTION AND RESOLUTION)\n"
    "  **a.** Actively scan EDST information for predicted "
    "aircraft-to-aircraft and aircraft-to-airspace alerts.\n"
    "  **b.** When a conflict probe alert is displayed, evaluate the alert "
    "and take appropriate action as early as practical.\n"
)


# (id, reply, tool_outputs, tools_ran, memory_block_attached)
# Each row is a truthful reply shape the user might plausibly see.
# Catchers must NOT Halt or Nudge on any of them.
_CONTROL_CORPUS: tuple[
    tuple[str, str, tuple[str, ...], frozenset[str], bool], ...
] = (
    (
        "truthful_mh_rbn_prose",
        "The usable distance for an MH class RBN is 25 miles, per JO 7110.65 §4-1-1.",
        (_TBL_412,),
        frozenset({"search_memory"}),
        False,
    ),
    (
        "truthful_mh_rbn_table_echo",
        (
            "Here is TBL 4-1-2 from JO 7110.65 §4-1-1:\n\n"
            "| Class | Power (watts) | Distance (miles) |\n"
            "|---|---|---|\n"
            "| CL | Under 25 | 15 |\n"
            "| MH | Under 50 | 25 |\n"
            "| H | 50 - 1,999 | 50 |\n"
            "| HH | 2,000 or more | 75 |\n"
        ),
        (_TBL_412,),
        frozenset({"search_memory"}),
        False,
    ),
    (
        "truthful_h_class_range_cite_low_end",
        (
            "An H class RBN handles 50 to 1,999 watts and has a 50 mile "
            "usable distance, per JO 7110.65 §4-1-1."
        ),
        (_TBL_412,),
        frozenset({"search_memory"}),
        False,
    ),
    (
        "truthful_aircraft_to_aircraft_citation",
        (
            "Per JO 7110.65 §13-1-2(a), controllers must actively scan EDST "
            "information for predicted aircraft-to-aircraft and "
            "aircraft-to-airspace alerts."
        ),
        (_EDST_13_1_2,),
        frozenset({"search_memory"}),
        False,
    ),
    (
        "truthful_citation_with_memory_block",
        "Per §2-4-3, a controller must verify readback accuracy on every clearance.",
        (),
        frozenset(),
        True,  # retrieval memory block attached — disarms ungrounded_citation
    ),
    (
        "truthful_cite_then_prose_no_table",
        (
            "For aircraft-to-aircraft alerts, see JO 7110.65 §13-1-2. The "
            "controller's duty is to actively scan EDST information."
        ),
        (_EDST_13_1_2,),
        frozenset({"search_memory"}),
        False,
    ),
    (
        # Targets missing_citation's silent path — reply mentions the
        # order, includes a §-anchor, grounding tool ran. Must NOT
        # nudge for a second citation.
        "truthful_purpose_of_order_with_cite",
        (
            "Per JO 7110.65 §1-1-1, the purpose of the order is to prescribe "
            "air traffic control procedures and phraseology for use by "
            "persons providing air traffic control services. Controllers "
            "are required to be familiar with the provisions that pertain "
            "to their operational responsibilities."
        ),
        (
            "[0.033] PURPOSE OF THIS ORDER\n"
            "  lesson: JO_7110.65 §1-1-1\n"
            "  This order prescribes air traffic control procedures...\n",
        ),
        frozenset({"search_memory"}),
        False,
    ),
    (
        # TBL/FIG anchors substitute for §-anchors — this control
        # case proves missing_citation accepts them.
        "truthful_tbl_anchor_as_citation",
        (
            "Per JO 7110.65 TBL 4-1-2, MH class radio beacons have a "
            "usable distance of 25 miles. The table is given under §4-1-1."
        ),
        (_TBL_412,),
        frozenset({"search_memory"}),
        False,
    ),
    (
        # list_count_mismatch silent path: claim and list agree.
        "truthful_three_items_claim_matches_list",
        (
            "Per JO 7110.65 §2-1-1, the three items covered are:\n\n"
            "1. Prevent a collision involving aircraft.\n"
            "2. Provide a safe, orderly, and expeditious flow of traffic.\n"
            "3. Support National Security and Homeland Defense missions."
        ),
        (
            "[0.031] ATC SERVICE\n"
            "  lesson: JO_7110.65 §2-1-1\n",
        ),
        frozenset({"search_memory"}),
        False,
    ),
    (
        # reserved_squawk_code silent path: prose explanation of a
        # reserved code with no assignment-context wrapping.
        "truthful_7500_explanation_in_prose",
        (
            "Per JO 7110.65 §5-2-5, when you observe a Code 7500 display, "
            "apply the procedures in §10-2-6 (Hijacked Aircraft). "
            "7500 is a pilot-initiated code; controllers do not assign it."
        ),
        (
            "[0.031] HIJACK/UNLAWFUL INTERFERENCE\n"
            "  lesson: JO_7110.65 §5-2-5\n",
        ),
        frozenset({"search_memory"}),
        False,
    ),
    (
        # reserved_squawk_code silent path: correct VFR code assignment.
        "truthful_squawk_1200_vfr_termination",
        (
            'Per JO 7110.65 §7-6-11 and §5-2-7, the correct phraseology '
            'is: "Radar service terminated, squawk one two zero zero."'
        ),
        (
            "[0.029] TERMINATION OF SERVICE\n"
            "  lesson: JO_7110.65 §7-6-11\n",
        ),
        frozenset({"search_memory"}),
        False,
    ),
    (
        # list_count_mismatch silent path: section numbers in prose
        # must not trip the count-claim regex.
        "truthful_section_number_in_prose",
        (
            "Per JO 7110.65 §2-1-1, controllers must provide ATC service in "
            "accordance with the procedures and minima in the order. "
            "Additional services are required when the work situation permits."
        ),
        (
            "[0.031] ATC SERVICE\n  lesson: JO_7110.65 §2-1-1\n",
        ),
        frozenset({"search_memory"}),
        False,
    ),
    (
        "plain_non_citation_reply",
        "The answer is 42.",
        (),
        frozenset(),
        False,
    ),
    (
        "scope_redirect_no_numbers_no_citation",
        (
            "That's a pilot-side question — JO 7110.65 is controller-side. "
            "Ask airton_c for the pilot view."
        ),
        (),
        frozenset(),
        False,
    ),
    (
        "truthful_phraseology_quote",
        (
            'Phraseology per §7-9-2: "CLEARED TO ENTER (name) CLASS BRAVO '
            'AIRSPACE". The controller quotes this verbatim.'
        ),
        (),
        frozenset(),
        True,  # memory block attached
    ),
    (
        "wrap_up_after_tool_success",
        "I added the entry. It is now included in the list.",
        ('{"status": "ok"}',),
        frozenset({"remember_fact"}),
        False,
    ),
)


@pytest.mark.parametrize(
    ("case_id", "reply", "tool_outputs", "tools_ran", "memory_block"),
    _CONTROL_CORPUS,
    ids=[c[0] for c in _CONTROL_CORPUS],
)
def test_finalize_pipeline_passes_truthful_reply(
    case_id: str,
    reply: str,
    tool_outputs: tuple[str, ...],
    tools_ran: frozenset[str],
    memory_block: bool,
) -> None:
    """Every truthful reply shape must produce Continue from the full
    finalize pipeline (ungrounded_citation + table_fabrication +
    numeric_fabrication + fabrication_fallback). Any Halt here is a
    false positive we must understand before shipping."""
    pipe = default_hook_pipeline()
    ctx = FinalizeContext(
        reply=_reply(reply),
        last_outcome=Continue(),
        tools_ran=tools_ran,
        memory_block_attached=memory_block,
        tool_outputs=tool_outputs,
    )
    outcome = pipe.run_finalize(ctx, disabled=frozenset())
    assert isinstance(outcome, Continue), (
        f"finalize false positive on {case_id!r} — a catcher halted a "
        f"truthful reply. Replaced content: "
        f"{getattr(outcome, 'reply', None) and outcome.reply.content[:120]!r}"
    )


@pytest.mark.parametrize(
    ("case_id", "reply", "tool_outputs", "tools_ran", "memory_block"),
    _CONTROL_CORPUS,
    ids=[c[0] for c in _CONTROL_CORPUS],
)
def test_bail_pipeline_passes_truthful_reply(
    case_id: str,
    reply: str,
    tool_outputs: tuple[str, ...],
    tools_ran: frozenset[str],
    memory_block: bool,
) -> None:
    """Every truthful reply shape must produce Continue from the bail
    pipeline when `tools_ran_this_turn` is True (wrap-up after a tool
    ran). Catches shape-based false positives in fabricated_search /
    fabricated_itemization / meta_confirm / false_success etc."""
    pipe = default_hook_pipeline()
    ctx = BailContext(
        reply=_reply(reply),
        tools_ran_this_turn=bool(tools_ran),
        tools_ran=tools_ran,
    )
    outcome = pipe.run_bail(ctx, disabled=frozenset())
    assert isinstance(outcome, Continue), (
        f"bail false positive on {case_id!r} — a bail catcher "
        f"triggered on a truthful reply. Outcome: {outcome!r}"
    )


def test_control_corpus_covers_each_finalize_catcher() -> None:
    """Coverage guard: each finalize catcher must be structurally
    exercised by at least one control case — otherwise adding a new
    catcher could silently false-positive without this file noticing.

    Exercised = the case's context inputs are in the catcher's trigger
    surface (memory_block, tools_ran, tool_outputs, citation/table/
    numeric-claim shape in reply). We assert presence of each surface
    dimension rather than per-catcher name — the exact mapping drifts
    as catchers are added/refactored.
    """
    cases = _CONTROL_CORPUS
    # At least one case with a §-citation reply + grounding tool run
    # (exercises ungrounded_citation's grounded-disarm path).
    assert any(
        "§" in c[1] and "search_memory" in c[3]
        for c in cases
    ), "no control case exercises (citation + grounding-tool-ran)"
    # At least one case with a pipe table in the reply
    # (exercises table_fabrication's pass path).
    assert any(
        "\n|" in c[1] and "search_memory" in c[3]
        for c in cases
    ), "no control case exercises (pipe-table reply + grounding-tool-ran)"
    # At least one case with a (label, number, unit) prose claim
    # (exercises numeric_fabrication's pass path).
    assert any(
        "class" in c[1].lower()
        and ("miles" in c[1] or "feet" in c[1] or "watts" in c[1])
        and "search_memory" in c[3]
        for c in cases
    ), "no control case exercises (labeled-numeric prose + grounding-tool-ran)"
    # At least one case with memory_block_attached=True and §-citation
    # (exercises ungrounded_citation's memory-disarm path).
    assert any(
        c[4] is True and "§" in c[1]
        for c in cases
    ), "no control case exercises (citation + memory_block_attached)"
