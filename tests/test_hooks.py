"""Unit tests for the orchestrator hook pipeline.

These exercise each catcher in isolation — independently of the tool
loop — so a failure points at the hook, not the surrounding plumbing.
The end-to-end behavior is covered by `tests/test_tool_loop.py` and the
attribution eval (`tests/test_tool_loop_eval.py`).
"""

from __future__ import annotations

from harness.orchestrator.hooks import (
    AbFabricationHook,
    BailContext,
    Continue,
    DuplicateCallHook,
    FabricatedSearchHook,
    FabricationFallbackHook,
    FalseSuccessHook,
    FinalizeContext,
    Halt,
    HookPipeline,
    MetaConfirmHook,
    Nudge,
    PairedMetaConfirmStripHook,
    PostModelContext,
    PreToolContext,
    Replace,
    Skip,
    TeaserHook,
    ToolIntentHook,
    Truncated,
    TruncatedHook,
    UnparseableHook,
    default_hook_pipeline,
)
from harness.tools.base import ModelReply, ToolCall


def _reply(content: str = "", **kw: object) -> ModelReply:
    """Build a ModelReply with defaults for the fields we don't care about."""
    return ModelReply(
        content=content,
        tool_calls=kw.get("tool_calls", ()),  # type: ignore[arg-type]
        was_truncated=bool(kw.get("was_truncated", False)),
        had_unparseable_call=bool(kw.get("had_unparseable_call", False)),
    )


# ---------- individual bail hooks ----------


def test_truncated_hook_fires_on_truncated_reply() -> None:
    outcome = TruncatedHook().check(
        BailContext(reply=_reply(was_truncated=True), tools_ran_this_turn=False)
    )
    assert isinstance(outcome, Truncated)


def test_truncated_hook_passes_on_clean_reply() -> None:
    outcome = TruncatedHook().check(
        BailContext(reply=_reply("all good"), tools_ran_this_turn=False)
    )
    assert isinstance(outcome, Continue)


def test_unparseable_hook_emits_nudge() -> None:
    outcome = UnparseableHook().check(
        BailContext(reply=_reply(had_unparseable_call=True), tools_ran_this_turn=False)
    )
    assert isinstance(outcome, Nudge)
    assert "malformed" in outcome.text.lower()


def test_teaser_hook_fires_on_trailing_teaser() -> None:
    ctx = BailContext(reply=_reply("Now let me check the tests:"), tools_ran_this_turn=False)
    assert isinstance(TeaserHook().check(ctx), Nudge)


def test_teaser_hook_ignores_prose_without_trailing_marker() -> None:
    ctx = BailContext(
        reply=_reply("I'll explain in detail. The core idea is that..."),
        tools_ran_this_turn=False,
    )
    assert isinstance(TeaserHook().check(ctx), Continue)


def test_false_success_hook_only_fires_when_no_tool_ran() -> None:
    reply = _reply("The file has been updated with the new config.")
    assert isinstance(
        FalseSuccessHook().check(BailContext(reply=reply, tools_ran_this_turn=False)),
        Nudge,
    )
    # After a tool has run, completion claims are legitimate wrap-ups.
    assert isinstance(
        FalseSuccessHook().check(BailContext(reply=reply, tools_ran_this_turn=True)),
        Continue,
    )


def test_meta_confirm_hook_fires_on_would_you_like() -> None:
    ctx = BailContext(
        reply=_reply("Would you like me to proceed?"),
        tools_ran_this_turn=False,
    )
    assert isinstance(MetaConfirmHook().check(ctx), Nudge)


def test_fabricated_search_hook_fires_on_placeholder_domain() -> None:
    ctx = BailContext(
        reply=_reply("Top result: https://example.com/thing"),
        tools_ran_this_turn=False,
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Nudge)


def test_ab_fabrication_hook_fires_on_captured_colon() -> None:
    ctx = BailContext(reply=_reply("Captured: harness-abc"), tools_ran_this_turn=False)
    assert isinstance(AbFabricationHook().check(ctx), Nudge)


def test_tool_intent_hook_fires_on_stated_intent() -> None:
    ctx = BailContext(reply=_reply("I will search for that now."), tools_ran_this_turn=False)
    assert isinstance(ToolIntentHook().check(ctx), Nudge)


# ---------- pipeline semantics ----------


def test_pipeline_first_match_wins() -> None:
    """Truncation and unparseable can both be set on a single reply;
    truncated is checked first so it should win."""
    pipe = default_hook_pipeline()
    reply = _reply("partial", was_truncated=True, had_unparseable_call=True)
    outcome = pipe.run_bail(
        BailContext(reply=reply, tools_ran_this_turn=False),
        disabled=frozenset(),
    )
    assert isinstance(outcome, Truncated)


def test_pipeline_skips_disabled_hooks() -> None:
    """Disabling truncated lets the next hook (unparseable) fire."""
    pipe = default_hook_pipeline()
    reply = _reply("partial", was_truncated=True, had_unparseable_call=True)
    outcome = pipe.run_bail(
        BailContext(reply=reply, tools_ran_this_turn=False),
        disabled=frozenset({"truncated"}),
    )
    assert isinstance(outcome, Nudge)
    assert "malformed" in outcome.text.lower()


def test_pipeline_returns_continue_when_nothing_matches() -> None:
    pipe = default_hook_pipeline()
    outcome = pipe.run_bail(
        BailContext(reply=_reply("here is the final answer"), tools_ran_this_turn=False),
        disabled=frozenset(),
    )
    assert isinstance(outcome, Continue)


def test_pipeline_names_match_expected_surface() -> None:
    """Canonical catcher names, ordered as bail → post_model → pre_tool → finalize."""
    pipe = default_hook_pipeline()
    assert pipe.names() == (
        "truncated",
        "unparseable",
        "teaser",
        "false_success",
        "meta_confirm",
        "fabricated_search",
        "ab_fabrication",
        "tool_intent",
        "paired_meta_confirm_strip",
        "duplicate_call",
        "fabrication_fallback",
    )


# ---------- post-model hook ----------


def test_paired_meta_confirm_strip_clears_narrative_on_tool_call_reply() -> None:
    reply = _reply(
        "Would you like me to proceed? I'll run the tool.",
        tool_calls=(ToolCall(name="list_dir", arguments={"path": "/workdir"}),),
    )
    outcome = PairedMetaConfirmStripHook().check(PostModelContext(reply=reply))
    assert isinstance(outcome, Replace)
    assert outcome.reply.content == ""
    assert outcome.reply.tool_calls == reply.tool_calls


def test_paired_meta_confirm_strip_leaves_reply_without_tool_calls_alone() -> None:
    """When there's no tool call to preserve, the bail pipeline is the
    right place to fire — this hook is a no-op."""
    reply = _reply("Would you like me to proceed?")  # no tool_calls
    outcome = PairedMetaConfirmStripHook().check(PostModelContext(reply=reply))
    assert isinstance(outcome, Continue)


# ---------- pre-tool hook ----------


def test_duplicate_call_hook_skips_seen_call() -> None:
    call = ToolCall(name="list_dir", arguments={"path": "/workdir"})
    # The canonical key is (name, json-sorted arguments).
    seen = frozenset({("list_dir", '{"path": "/workdir"}')})
    outcome = DuplicateCallHook().check(PreToolContext(call=call, seen_calls=seen))
    assert isinstance(outcome, Skip)
    assert "duplicate call" in outcome.result.output


def test_duplicate_call_hook_passes_first_time() -> None:
    call = ToolCall(name="list_dir", arguments={"path": "/workdir"})
    outcome = DuplicateCallHook().check(PreToolContext(call=call, seen_calls=frozenset()))
    assert isinstance(outcome, Continue)


# ---------- finalize hook ----------


def test_fabrication_fallback_fires_on_exhausted_nudge() -> None:
    """When bail retries are exhausted and the last outcome was a
    fabrication-shaped Nudge, substitute the canned refusal."""
    reply = _reply("I'll search for that now.")
    outcome = FabricationFallbackHook().check(
        FinalizeContext(reply=reply, last_outcome=Nudge("...nudge..."))
    )
    assert isinstance(outcome, Halt)
    assert "I couldn't answer that without calling a tool" in outcome.reply.content


def test_fabrication_fallback_passes_on_truncated_outcome() -> None:
    """A truncated reply is a partial answer, not a fabrication —
    keep the partial instead of swapping in the canned refusal."""
    reply = _reply("partial", was_truncated=True)
    outcome = FabricationFallbackHook().check(
        FinalizeContext(reply=reply, last_outcome=Truncated())
    )
    assert isinstance(outcome, Continue)


def test_fabrication_fallback_passes_on_clean_reply() -> None:
    """No bail outcome fired → nothing to rescue; stay out of the way."""
    reply = _reply("the final answer")
    outcome = FabricationFallbackHook().check(FinalizeContext(reply=reply, last_outcome=Continue()))
    assert isinstance(outcome, Continue)


# ---------- custom pipeline construction ----------


def test_custom_pipeline_can_omit_phases() -> None:
    """A subagent-style caller can construct a pipeline with only a
    subset of phases — run_bail returns Continue in that case."""
    pipe = HookPipeline(bail=[], post_model=[], pre_tool=[], finalize=[])
    outcome = pipe.run_bail(
        BailContext(reply=_reply("anything"), tools_ran_this_turn=False),
        disabled=frozenset(),
    )
    assert isinstance(outcome, Continue)
