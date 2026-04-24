"""Unit tests for LowConfidenceFallbackHook (harness-ywp.3).

The hook catches citation-bearing replies where a grounding tool ran
but scored below threshold AND the cited section wasn't in the
tool's declared grounded-citation set. Complementary to
UngroundedCitationHook, which covers the 'no tool ran' case.
"""

from __future__ import annotations

from harness.orchestrator.hooks import (
    LOW_CONFIDENCE_FALLBACK,
    Continue,
    FinalizeContext,
    Halt,
    LowConfidenceFallbackHook,
)
from harness.tools.base import ModelReply


def _ctx(
    *,
    reply_text: str,
    retrieval_top_score: float | None,
    citations_grounded: frozenset[str] = frozenset(),
    tools_ran: frozenset[str] = frozenset({"search_memory"}),
) -> FinalizeContext:
    from harness.orchestrator.hooks import Continue as _Continue

    return FinalizeContext(
        reply=ModelReply(content=reply_text),
        last_outcome=_Continue(),
        tools_ran=tools_ran,
        memory_block_attached=False,
        tool_outputs=(),
        retrieval_top_score=retrieval_top_score,
        citations_grounded=citations_grounded,
    )


def test_no_retrieval_metadata_continues() -> None:
    """Characters / turns with no retrieval telemetry (e.g. plain-str
    tools only, or no tool calls) slip past — the hook is a retrieval-
    signal gate, not a blanket citation check."""
    hook = LowConfidenceFallbackHook()
    out = hook.check(
        _ctx(reply_text="Per §4-1-1 the answer is X.", retrieval_top_score=None)
    )
    assert isinstance(out, Continue)


def test_high_confidence_continues_even_with_ungrounded_cite() -> None:
    """Top score at or above threshold disarms — UngroundedCitationHook
    and MissingCitationHook handle the other shapes."""
    hook = LowConfidenceFallbackHook(threshold=0.5)
    out = hook.check(
        _ctx(
            reply_text="Per §4-1-1 the answer is X.",
            retrieval_top_score=0.72,
            citations_grounded=frozenset(),
        )
    )
    assert isinstance(out, Continue)


def test_low_score_and_ungrounded_cite_halts() -> None:
    """Canonical failure mode: tool ran, scored 0.3, model still cited
    §4-1-1 which the tool never surfaced."""
    hook = LowConfidenceFallbackHook(threshold=0.5)
    out = hook.check(
        _ctx(
            reply_text="Per §4-1-1 the answer is X.",
            retrieval_top_score=0.3,
            citations_grounded=frozenset(),
        )
    )
    assert isinstance(out, Halt)
    assert out.reply.content == LOW_CONFIDENCE_FALLBACK


def test_low_score_but_cite_is_grounded_continues() -> None:
    """Tool scored weakly but DID ground the §4-1-1 citation the model
    used. The model is using what it was given; don't refuse."""
    hook = LowConfidenceFallbackHook(threshold=0.5)
    out = hook.check(
        _ctx(
            reply_text="Per §4-1-1 the answer is X.",
            retrieval_top_score=0.3,
            citations_grounded=frozenset({"§4-1-1"}),
        )
    )
    assert isinstance(out, Continue)


def test_low_score_but_no_citation_continues() -> None:
    """Clarifying questions, chatty non-citing replies, ack messages —
    there's nothing to be wrong about."""
    hook = LowConfidenceFallbackHook(threshold=0.5)
    out = hook.check(
        _ctx(
            reply_text="Could you clarify whether you mean manned or unmanned?",
            retrieval_top_score=0.2,
        )
    )
    assert isinstance(out, Continue)


def test_mixed_cites_one_grounded_one_not_halts() -> None:
    """Reply cites both §4-1-1 (grounded) and §5-5-4 (ungrounded) with
    weak retrieval — the ungrounded one alone is enough to halt. A
    single ungrounded citation is a fabrication risk; we don't require
    ALL to be ungrounded."""
    hook = LowConfidenceFallbackHook(threshold=0.5)
    out = hook.check(
        _ctx(
            reply_text="Per §4-1-1 and also §5-5-4 the answer is X.",
            retrieval_top_score=0.2,
            citations_grounded=frozenset({"§4-1-1"}),
        )
    )
    assert isinstance(out, Halt)


def test_threshold_param_respected() -> None:
    """Tighter threshold refuses replies the default would accept."""
    hook_tight = LowConfidenceFallbackHook(threshold=0.9)
    out = hook_tight.check(
        _ctx(
            reply_text="Per §4-1-1 the answer is X.",
            retrieval_top_score=0.7,
            citations_grounded=frozenset(),
        )
    )
    assert isinstance(out, Halt)

    hook_loose = LowConfidenceFallbackHook(threshold=0.3)
    out = hook_loose.check(
        _ctx(
            reply_text="Per §4-1-1 the answer is X.",
            retrieval_top_score=0.5,
            citations_grounded=frozenset(),
        )
    )
    assert isinstance(out, Continue)


def test_registered_in_default_pipeline_before_ungrounded_citation() -> None:
    """harness-ywp.3 ordering invariant: LowConfidenceFallbackHook must
    precede UngroundedCitationHook so the more-specific low-confidence
    refusal wins when both would fire (retrieval ran + low score + no
    grounded citation)."""
    from harness.orchestrator.hooks import (
        UngroundedCitationHook,
        default_hook_pipeline,
    )

    pipeline = default_hook_pipeline()
    names = [type(h).__name__ for h in pipeline.finalize]
    assert "LowConfidenceFallbackHook" in names
    assert "UngroundedCitationHook" in names
    assert names.index("LowConfidenceFallbackHook") < names.index(
        UngroundedCitationHook.__name__
    )
