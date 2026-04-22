"""Unit tests for the orchestrator hook pipeline.

These exercise each catcher in isolation — independently of the tool
loop — so a failure points at the hook, not the surrounding plumbing.
The end-to-end behavior is covered by `tests/test_tool_loop.py` and the
attribution eval (`tests/test_tool_loop_eval.py`).
"""

from __future__ import annotations

from harness.orchestrator.hooks import (
    AbFabricationHook,
    ArgumentGroundingHook,
    BailContext,
    Continue,
    DuplicateCallHook,
    FabricatedItemizationHook,
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


def test_fabricated_search_hook_fires_on_past_tense_web_claim_no_tool() -> None:
    """harness-78z: model narrates 'I've searched the web for X and found…'
    without any tool actually running. Must trip the catcher."""
    ctx = BailContext(
        reply=_reply(
            'I\'ve searched the web for "Brad Hintze Phoenix" and found no specific biography.'
        ),
        tools_ran_this_turn=False,
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Nudge)


def test_fabricated_search_hook_fires_on_web_claim_after_search_memory() -> None:
    """harness-78z: search_memory ran and returned empty; model still
    narrates 'I've searched the web'. Narrow gate: only search_web /
    fetch_url disarms the web-claim branch, so a search_memory call
    does NOT legitimize the web-claim."""
    ctx = BailContext(
        reply=_reply("I've searched the web for brad hintze and found no biography."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_memory"}),
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Nudge)


def test_fabricated_search_hook_skipped_on_web_claim_after_real_search_web() -> None:
    """The web-claim pattern is legit wrap-up narration after search_web
    actually ran — the catcher must NOT fire."""
    ctx = BailContext(
        reply=_reply("I've searched the web for that and found two relevant pages."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Continue)


def test_fabricated_search_hook_skipped_on_web_claim_after_real_fetch_url() -> None:
    """fetch_url also counts as a web-fetch tool — disarms the
    web-claim branch."""
    ctx = BailContext(
        reply=_reply("After searching the web I've gathered the following."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"fetch_url"}),
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Continue)


def test_fabricated_search_hook_skipped_on_result_list_after_any_tool() -> None:
    """Legacy gate preserved: the result-list patterns ('here are the
    results', placeholder URLs) are disarmed by ANY tool running, not
    just web tools. A non-web tool followed by a wrap-up list is still
    legitimate."""
    ctx = BailContext(
        reply=_reply("Here are the results: the data shows X."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"noop"}),
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Continue)


def test_ab_fabrication_hook_fires_on_captured_colon() -> None:
    ctx = BailContext(reply=_reply("Captured: harness-abc"), tools_ran_this_turn=False)
    assert isinstance(AbFabricationHook().check(ctx), Nudge)


def test_fabricated_itemization_hook_fires_on_here_are_articles() -> None:
    """harness-f5x: user asks for more detail on item N from a prior
    fetch; model regenerates a fake summary list without calling
    fetch_url. Gated on no-tool-ran."""
    ctx = BailContext(
        reply=_reply(
            "Sure — here are the first 5 articles from the BBC News website:\n"
            "1. Iran blockade...\n2. US-Iran standoff..."
        ),
        tools_ran_this_turn=False,
    )
    outcome = FabricatedItemizationHook().check(ctx)
    assert isinstance(outcome, Nudge)
    assert "regenerated summary list" in outcome.text


def test_fabricated_itemization_hook_fires_on_lets_focus_on() -> None:
    ctx = BailContext(
        reply=_reply("Let's focus on one of the top stories. 1. Story... 2. ..."),
        tools_ran_this_turn=False,
    )
    assert isinstance(FabricatedItemizationHook().check(ctx), Nudge)


def test_fabricated_itemization_hook_fires_on_rundown_of_headlines() -> None:
    ctx = BailContext(
        reply=_reply("Here's a rundown of the headlines: ..."),
        tools_ran_this_turn=False,
    )
    assert isinstance(FabricatedItemizationHook().check(ctx), Nudge)


def test_fabricated_itemization_hook_skipped_when_tool_ran() -> None:
    """After a real fetch_url / search_web, a list of articles is a
    legitimate wrap-up — the catcher must not fire."""
    ctx = BailContext(
        reply=_reply("Here are the first 5 articles from the BBC News website: 1. ..."),
        tools_ran_this_turn=True,
    )
    assert isinstance(FabricatedItemizationHook().check(ctx), Continue)


def test_fabricated_itemization_hook_passes_on_clean_reply() -> None:
    ctx = BailContext(
        reply=_reply("Your next meeting is at 3pm."),
        tools_ran_this_turn=False,
    )
    assert isinstance(FabricatedItemizationHook().check(ctx), Continue)


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
        "fabricated_itemization",
        "ab_fabrication",
        "tool_intent",
        "paired_meta_confirm_strip",
        "duplicate_call",
        "argument_grounding",
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


# ---------- argument grounding hook ----------


def test_argument_grounding_hook_flags_unrelated_domain_in_args() -> None:
    """The observed failure from harness-7od: user asks for stackoverflow,
    router picks search_web with a query naming a totally unrelated
    domain ('dailydrop.fm'). The grounding hook must Skip with a nudge."""
    call = ToolCall(name="search_web", arguments={"query": "dailydrop.fm", "max_results": 1})
    ctx = PreToolContext(
        call=call,
        seen_calls=frozenset(),
        user_message="go to stackoverflow, fetch the first question, summarize it",
    )
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Skip)
    assert outcome.result.success is False
    assert outcome.result.error == "argument_grounding"
    assert "dailydrop" in outcome.result.output
    assert "stackoverflow" in outcome.result.output  # echoes the user msg


def test_argument_grounding_hook_passes_when_domain_root_matches() -> None:
    """User names the domain, args use it — legitimate fetch_url call."""
    call = ToolCall(name="fetch_url", arguments={"url": "https://stackoverflow.com/questions/1"})
    ctx = PreToolContext(
        call=call,
        seen_calls=frozenset(),
        user_message="go to stackoverflow and fetch the first question",
    )
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_argument_grounding_hook_passes_when_args_have_no_domain() -> None:
    """Plain prose queries ('python best practices') should pass — the
    hook only flags args with an explicit domain/URL token that
    conflicts with the user message. Broader overlap checks are
    out of scope for this narrow hook."""
    call = ToolCall(name="search_web", arguments={"query": "python best practices 2026"})
    ctx = PreToolContext(
        call=call,
        seen_calls=frozenset(),
        user_message="search for python tips",
    )
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_argument_grounding_hook_passes_when_user_message_is_none() -> None:
    """No user message threaded through → no ground-truth to check
    against. The hook stays out of the way (bootstrap / subagent cases)."""
    call = ToolCall(name="search_web", arguments={"query": "example.com"})
    ctx = PreToolContext(call=call, seen_calls=frozenset(), user_message=None)
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_argument_grounding_hook_matches_case_insensitively() -> None:
    """User message can spell the domain any case; args can too. Matching
    must be case-insensitive on both sides."""
    call = ToolCall(name="fetch_url", arguments={"url": "https://GitHub.com/foo/bar"})
    ctx = PreToolContext(
        call=call,
        seen_calls=frozenset(),
        user_message="Check github for the latest release",
    )
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_argument_grounding_hook_walks_nested_args() -> None:
    """Nested argument structures (lists, dicts) get their leaf strings
    walked. A rogue domain buried in a filters list is still a leak."""
    call = ToolCall(
        name="some_tool",
        arguments={"filters": ["keyword", "dailydrop.fm"], "limit": 5},
    )
    ctx = PreToolContext(
        call=call,
        seen_calls=frozenset(),
        user_message="look for articles about gardening",
    )
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Skip)
    assert "dailydrop" in outcome.result.output


def test_argument_grounding_hook_flags_all_ungrounded_roots() -> None:
    """When multiple domains in args are all ungrounded, the nudge lists
    them all sorted (deterministic) so the model sees the full scope of
    the drift."""
    call = ToolCall(
        name="search_web",
        arguments={"query": "results from zebra.io and alpha.co"},
    )
    ctx = PreToolContext(
        call=call,
        seen_calls=frozenset(),
        user_message="find news about the election",
    )
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Skip)
    # Sorted alphabetically → alpha before zebra
    assert outcome.result.output.index("alpha") < outcome.result.output.index("zebra")


def test_argument_grounding_hook_ignores_file_extensions() -> None:
    """File paths with common extensions (`.txt`, `.yaml`, `.json`, `.md`,
    `.py`) are NOT domains and must not trip the hook — write_file /
    edit_file / read_file calls would otherwise get blocked constantly.
    Fixed by keeping file extensions out of the TLD whitelist."""
    for path in ("new.txt", "config.yaml", "data.json", "README.md", "main.py"):
        call = ToolCall(name="write_file", arguments={"path": path, "content": "x"})
        ctx = PreToolContext(
            call=call,
            seen_calls=frozenset(),
            user_message="write something",
        )
        outcome = ArgumentGroundingHook().check(ctx)
        assert isinstance(outcome, Continue), f"{path} was incorrectly flagged"


def test_argument_grounding_hook_passes_mixed_grounded_and_prose_args() -> None:
    """If ALL domain-like args are grounded in the user message, the
    hook stays out of the way even when other (non-domain) arg values
    look arbitrary."""
    call = ToolCall(
        name="fetch_url",
        arguments={
            "url": "https://stackoverflow.com/questions/tagged/python",
            "header": "Accept: text/html",
        },
    )
    ctx = PreToolContext(
        call=call,
        seen_calls=frozenset(),
        user_message="summarize the first python question on stackoverflow",
    )
    outcome = ArgumentGroundingHook().check(ctx)
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
