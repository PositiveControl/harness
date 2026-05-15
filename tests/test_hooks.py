"""Unit tests for the orchestrator hook pipeline.

These exercise each catcher in isolation — independently of the tool
loop — so a failure points at the hook, not the surrounding plumbing.
The end-to-end behavior is covered by `tests/test_tool_loop.py` and the
attribution eval (`tests/test_tool_loop_eval.py`).
"""

from __future__ import annotations

from pathlib import Path

from harness.character import load_character
from harness.orchestrator.hooks import (
    TABLE_FABRICATION_FALLBACK,
    UNGROUNDED_CITATION_FALLBACK,
    AbFabricationHook,
    AmbiguousContextHook,
    ArgumentGroundingHook,
    AssembleContextOnceHook,
    BailContext,
    Continue,
    DuplicateCallHook,
    FabricatedItemizationHook,
    FabricatedSearchHook,
    FabricatedSectionHook,
    FabricationFallbackHook,
    FalseSuccessHook,
    FetchUrlGuardHook,
    FinalizeContext,
    Halt,
    HookPipeline,
    ListCountMismatchHook,
    MetaConfirmHook,
    MissingCitationHook,
    Nudge,
    NumericFabricationHook,
    PairedMetaConfirmStripHook,
    PostModelContext,
    PreToolContext,
    Replace,
    ReservedSquawkCodeHook,
    ScopeRedirectHook,
    Skip,
    TableFabricationHook,
    TeaserHook,
    ToolIntentHook,
    Truncated,
    TruncatedHook,
    UncitedSubstantiveReplyHook,
    UngroundedCitationHook,
    UnparseableHook,
    default_hook_pipeline,
)
from harness.tools.base import ModelReply, ToolCall

# FAA citation grammar for MissingCitationHook tests — non-citation
# characters skip the hook entirely (grammar=None).
_REPO = Path(__file__).resolve().parents[1]
_GRAMMAR = load_character(_REPO / "character" / "airton_c1").citation_grammar
assert _GRAMMAR is not None


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
    """Canonical catcher names, ordered as bail → post_model → pre_tool → finalize.

    Constructed with the full atc + ab opt-in roster (harness-qvwq) so
    the catcher-name surface is exhaustive — runtime characters install
    only the subset they declare in core.yaml."""
    pipe = default_hook_pipeline(
        catchers=(
            "ab_fabrication",
            "ambiguous_context",
            "scope_redirect",
            "reserved_squawk_code",
        ),
    )
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
        "missing_citation",
        "fabricated_section",
        "list_count_mismatch",
        "reserved_squawk_code",
        "scope_redirect",
        "ambiguous_context",
        "paired_meta_confirm_strip",
        "duplicate_call",
        "argument_grounding",
        "low_confidence_fallback",
        "ungrounded_citation",
        "uncited_substantive_reply",
        "table_fabrication",
        "numeric_fabrication",
        "fabrication_fallback",
    )


def test_pipeline_default_excludes_opt_in_catchers() -> None:
    """No-arg pipeline omits the four opt-in catchers (harness-qvwq)
    — non-corpus characters get a clean default with no FAA / ab
    domain checks running."""
    pipe = default_hook_pipeline()
    names = pipe.names()
    for opt_in in (
        "ab_fabrication",
        "ambiguous_context",
        "scope_redirect",
        "reserved_squawk_code",
    ):
        assert opt_in not in names


def test_pipeline_unknown_catcher_raises() -> None:
    """Typos in core.yaml `catchers:` surface immediately rather than
    silently dropping a catcher."""
    import pytest

    with pytest.raises(ValueError, match=r"Unknown opt-in catchers"):
        default_hook_pipeline(catchers=("nonsense",))


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


# ---------- fetch_url_guard (harness-ygvg follow-up) ----------


def test_fetch_url_guard_skips_speculative_call_without_user_url() -> None:
    """harness-ygvg follow-up: airton_c_tfr observed 2026-05-15
    speculatively calling fetch_url with a guessed FAA listing URL
    after the user pasted only NOTAM text. The constitution says
    fetch_url is paste-a-URL-only. The guard must Skip with a
    re-plan tool result so the model pivots to decoding the pasted
    text instead of treating the 403 as authoritative."""
    call = ToolCall(
        name="fetch_url",
        arguments={"url": "https://www.faa.gov/air_traffic/air_facts/notams/"},
    )
    ctx = PreToolContext(
        call=call,
        seen_calls=frozenset(),
        user_message="FDC 5/2809 ZMA PALM BEACH FL TFR 14 CFR 99.7 ...",
    )
    outcome = FetchUrlGuardHook().check(ctx)
    assert isinstance(outcome, Skip)
    assert outcome.result.success is False
    assert outcome.result.error == "fetch_url_guard"
    assert "did not contain a URL" in outcome.result.output


def test_fetch_url_guard_allows_call_when_user_pastes_url() -> None:
    """Truthful counterfactual: when the user DOES paste an http(s)
    URL the design intent (paste a tfr.faa.gov link) is exercised
    and the guard stays out of the way. Downstream argument_grounding
    + the tool's host allowlist remain in charge of validating which
    URL is acceptable."""
    call = ToolCall(
        name="fetch_url",
        arguments={"url": "https://tfr.faa.gov/save_pages/detail_5_2809.html"},
    )
    ctx = PreToolContext(
        call=call,
        seen_calls=frozenset(),
        user_message=("Decode this TFR for me: https://tfr.faa.gov/save_pages/detail_5_2809.html"),
    )
    outcome = FetchUrlGuardHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_fetch_url_guard_ignores_other_tool_names() -> None:
    """The guard only checks fetch_url-family calls. read_file, grep,
    assemble_context, etc. pass through untouched even when the user
    message lacks a URL."""
    for tool_name in ("read_file", "grep", "assemble_context", "search_memory"):
        call = ToolCall(name=tool_name, arguments={"x": "y"})
        ctx = PreToolContext(
            call=call,
            seen_calls=frozenset(),
            user_message="no urls in this message",
        )
        outcome = FetchUrlGuardHook().check(ctx)
        assert isinstance(outcome, Continue), f"guard fired on unrelated tool {tool_name!r}"


def test_fetch_url_guard_skips_when_user_message_missing() -> None:
    """Bootstrap / subagent contexts may not thread a user_message
    through. Default to skipping the call rather than allowing it —
    paste-only characters never have a legitimate fetch_url without a
    user-pasted URL."""
    call = ToolCall(name="fetch_url", arguments={"url": "https://example.com"})
    ctx = PreToolContext(call=call, seen_calls=frozenset(), user_message=None)
    outcome = FetchUrlGuardHook().check(ctx)
    assert isinstance(outcome, Skip)


def test_fetch_url_guard_wires_into_pipeline_when_opted_in() -> None:
    """Pipeline composition pin: passing 'fetch_url_guard' in the
    catchers roster installs the hook in the pre_tool list. Absence
    keeps the pre_tool surface to (duplicate_call, argument_grounding)
    so non-paste-only characters aren't affected."""
    on = default_hook_pipeline(catchers=("fetch_url_guard",))
    assert "fetch_url_guard" in on.names()
    off = default_hook_pipeline(catchers=())
    assert "fetch_url_guard" not in off.names()


# ---------- assemble_context_once (harness-ygvg follow-up) ----------


def test_assemble_context_once_skips_when_forced_call_already_ran() -> None:
    """harness-ygvg follow-up: airton_c_tfr observed 2026-05-15 issuing
    its own assemble_context call with role='TFR_interpreter' after
    the orchestrator's forced-grounding prelude had already run with
    role='airton_c_tfr'. The forced call adds (name, args) to
    seen_calls before round 0; the hook must see that and Skip the
    redundant call regardless of the new call's role argument."""
    forced_key = ("assemble_context", '{"role":"airton_c_tfr","variables":{"...":""}}')
    bad_call = ToolCall(
        name="assemble_context",
        arguments={"role": "TFR_interpreter", "variables": {"notam_text": "..."}},
    )
    ctx = PreToolContext(
        call=bad_call,
        seen_calls=frozenset({forced_key}),
        user_message="FDC 5/2809 ZMA PALM BEACH FL TFR ...",
    )
    outcome = AssembleContextOnceHook().check(ctx)
    assert isinstance(outcome, Skip)
    assert outcome.result.success is False
    assert outcome.result.error == "assemble_context_once"
    assert "already ran" in outcome.result.output


def test_assemble_context_once_passes_on_first_call() -> None:
    """Truthful counterfactual: if assemble_context has NOT run yet
    this turn (seen_calls empty or has only other tool names), the
    hook stays out of the way. Otherwise the forced-grounding prelude
    itself would be blocked."""
    call = ToolCall(name="assemble_context", arguments={"role": "airton_c_tfr"})
    ctx = PreToolContext(
        call=call,
        seen_calls=frozenset({("search_memory", "{}")}),  # different tool ran
        user_message="some notam text",
    )
    outcome = AssembleContextOnceHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_assemble_context_once_ignores_other_tool_names() -> None:
    """The hook only checks assemble_context calls. Other tools (even
    if they're being re-issued) pass through — duplicate_call handles
    that broader case."""
    for tool_name in ("search_memory", "fetch_url", "read_file"):
        call = ToolCall(name=tool_name, arguments={"x": "y"})
        ctx = PreToolContext(
            call=call,
            seen_calls=frozenset({("assemble_context", "{}")}),
            user_message="hi",
        )
        outcome = AssembleContextOnceHook().check(ctx)
        assert isinstance(outcome, Continue), (
            f"hook fired on non-assemble_context tool {tool_name!r}"
        )


def test_assemble_context_once_wires_into_pipeline_when_opted_in() -> None:
    """Pipeline composition pin: opt-in via the catchers roster."""
    on = default_hook_pipeline(catchers=("assemble_context_once",))
    assert "assemble_context_once" in on.names()
    off = default_hook_pipeline(catchers=())
    assert "assemble_context_once" not in off.names()


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


# ---------- ungrounded citation hook ----------


_FAKE_REPLY = (
    "Standard IFR minimums in Class B airspace are 1,200 ft AGL during "
    "the day and 1,500 ft AGL at night. See JO 7110.65 §12-1-2 for the "
    "full airspace classification."
)


def test_ungrounded_citation_fires_when_no_grounding() -> None:
    """Section citation + no memory block + no grounding tool =>
    replace the reply with the character-agnostic refusal."""
    reply = _reply(_FAKE_REPLY)
    outcome = UngroundedCitationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset(),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Halt)
    assert outcome.reply.content == UNGROUNDED_CITATION_FALLBACK
    assert outcome.reply.tool_calls == ()


def test_ungrounded_citation_passes_when_memory_block_attached() -> None:
    """If retrieval surfaced a memory block, the model had grounding
    material to work from — don't second-guess the citation."""
    reply = _reply(_FAKE_REPLY)
    outcome = UngroundedCitationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset(),
            memory_block_attached=True,
        )
    )
    assert isinstance(outcome, Continue)


def test_ungrounded_citation_passes_when_search_memory_ran() -> None:
    """A successful search_memory call counts as grounding."""
    reply = _reply(_FAKE_REPLY)
    outcome = UngroundedCitationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Continue)


def test_ungrounded_citation_passes_when_fact_search_ran() -> None:
    """fact_search is also in the grounding set."""
    reply = _reply(_FAKE_REPLY)
    outcome = UngroundedCitationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"fact_search"}),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Continue)


def test_ungrounded_citation_ignores_non_grounding_tools() -> None:
    """fetch_url / search_web / read_file are NOT grounding for
    section-citation purposes."""
    reply = _reply(_FAKE_REPLY)
    outcome = UngroundedCitationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"fetch_url", "search_web", "read_file"}),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Halt)


def test_ungrounded_citation_passes_on_reply_without_section_pattern() -> None:
    """No `§N-N-N` => no trigger; let the reply through untouched."""
    reply = _reply("Class B airspace surrounds major airports. No citations here.")
    outcome = UngroundedCitationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset(),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Continue)


def test_ungrounded_citation_regex_tolerates_en_dash_and_minus() -> None:
    """Real corpus uses hyphen, en-dash, and unicode-minus
    interchangeably — all three must match."""
    for sep in ("-", "–", "−"):  # noqa: RUF001 — intentional unicode-dash variants
        reply = _reply(f"See JO 7110.65 §12{sep}1{sep}2 for details.")
        outcome = UngroundedCitationHook().check(
            FinalizeContext(
                reply=reply,
                last_outcome=Continue(),
                tools_ran=frozenset(),
                memory_block_attached=False,
            )
        )
        assert isinstance(outcome, Halt), f"separator {sep!r} did not match"


def test_ungrounded_citation_ignores_currency_and_short_anchors() -> None:
    """Narrow-regex contract: prices ($19.99), plain section marks
    (§2), and two-segment refs (§3-1) must NOT trip the hook."""
    for content in ("The fee is $19.99 per month.", "See §2 for preamble.", "Ref §3-1 only."):
        reply = _reply(content)
        outcome = UngroundedCitationHook().check(
            FinalizeContext(
                reply=reply,
                last_outcome=Continue(),
                tools_ran=frozenset(),
                memory_block_attached=False,
            )
        )
        assert isinstance(outcome, Continue), f"false positive on: {content!r}"


def test_ungrounded_citation_respects_disabled_toggle() -> None:
    """The attribution eval disables catchers by name — pipeline must
    honor 'ungrounded_citation' in the `disabled` frozenset."""
    pipe = default_hook_pipeline()
    reply = _reply(_FAKE_REPLY)
    ctx = FinalizeContext(
        reply=reply,
        last_outcome=Continue(),
        tools_ran=frozenset(),
        memory_block_attached=False,
    )
    # Enabled: the hook halts and replaces the reply.
    enabled = pipe.run_finalize(ctx, disabled=frozenset())
    assert isinstance(enabled, Halt)
    # Disabled: the hook is skipped; downstream fabrication_fallback
    # doesn't fire either because last_outcome is Continue.
    disabled = pipe.run_finalize(ctx, disabled=frozenset({"ungrounded_citation"}))
    assert isinstance(disabled, Continue)


def test_ungrounded_citation_runs_before_fabrication_fallback() -> None:
    """Pipeline ordering: when BOTH hooks would fire (citation + a
    surviving Nudge bail outcome), ungrounded_citation wins and its
    scope-aware refusal is what the user sees."""
    pipe = default_hook_pipeline()
    reply = _reply(_FAKE_REPLY)
    outcome = pipe.run_finalize(
        FinalizeContext(
            reply=reply,
            last_outcome=Nudge("...surviving nudge..."),
            tools_ran=frozenset(),
            memory_block_attached=False,
        ),
        disabled=frozenset(),
    )
    assert isinstance(outcome, Halt)
    assert outcome.reply.content == UNGROUNDED_CITATION_FALLBACK


# ---------- uncited_substantive_reply (harness-zsbz) ----------

# Live repro from the 2026-05-14 wedding-ring session: retry response
# prefixes "JO 7110.65 doesn't cover it" (matching _SCOPE_REDIRECT_MARKER_RE)
# but then continues with three paragraphs of general-knowledge content.
# The finalize hook is the safety net for what survives the bail-phase
# retry budget.
_REDIRECT_THEN_ANSWER_REPLY = (
    "That's a pilot-facing question — airton_c is the generalist. "
    "JO 7110.65 doesn't cover it. In real-life practice, it's a "
    "personal decision based on cultural norms and local customs. "
    "Typically, the second ring or any unused ring on the left "
    "hand is used for subsequent marriages. However, there's no "
    "universal rule and many people choose differently based on "
    "their cultural background or family tradition."
)

_SCOPE_REDIRECT_TEMPLATE = "Outside JO 7110.65 — ask airton_c."


def test_uncited_substantive_reply_halts_on_redirect_then_answer() -> None:
    """The motivating failure mode: a long reply that opens with a
    scope-redirect marker then continues with general-knowledge content.
    The bail-phase bypass only covers short clean redirects; the finalize
    hook catches what slips through."""
    outcome = UncitedSubstantiveReplyHook(
        grammar=_GRAMMAR,
        scope_redirect_template=_SCOPE_REDIRECT_TEMPLATE,
    ).check(
        FinalizeContext(
            reply=_reply(_REDIRECT_THEN_ANSWER_REPLY),
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Halt)
    assert outcome.reply.content == _SCOPE_REDIRECT_TEMPLATE


def test_uncited_substantive_reply_silent_without_template() -> None:
    """Non-corpus characters (no scope_redirect_template) get no
    enforcement — the hook is opt-in via character data."""
    outcome = UncitedSubstantiveReplyHook(
        grammar=_GRAMMAR,
        scope_redirect_template=None,
    ).check(
        FinalizeContext(
            reply=_reply(_REDIRECT_THEN_ANSWER_REPLY),
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Continue)


def test_uncited_substantive_reply_silent_without_grammar() -> None:
    """Same default: characters without citation_grammar (Airton, ab,
    echo) skip the hook entirely."""
    outcome = UncitedSubstantiveReplyHook(
        grammar=None,
        scope_redirect_template=_SCOPE_REDIRECT_TEMPLATE,
    ).check(
        FinalizeContext(
            reply=_reply(_REDIRECT_THEN_ANSWER_REPLY),
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Continue)


def test_uncited_substantive_reply_silent_when_no_grounding_tool_ran() -> None:
    """No grounding tool ran → that's UngroundedCitationHook's domain,
    not this one. The two hooks partition the failure space cleanly:
    no-tool-ran → ungrounded_citation; tool-ran-no-cite → this hook."""
    outcome = UncitedSubstantiveReplyHook(
        grammar=_GRAMMAR,
        scope_redirect_template=_SCOPE_REDIRECT_TEMPLATE,
    ).check(
        FinalizeContext(
            reply=_reply(_REDIRECT_THEN_ANSWER_REPLY),
            last_outcome=Continue(),
            tools_ran=frozenset(),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Continue)


def test_uncited_substantive_reply_silent_on_short_redirect() -> None:
    """A short clean scope-redirect is the right answer — don't replace
    it with the template (would be a no-op at best, and could downgrade
    a more specific refusal). The hook fires only when the redirect
    marker is followed by substantial body content."""
    short_redirect = "That's outside JO 7110.65 — ask airton_c for the pilot view."
    outcome = UncitedSubstantiveReplyHook(
        grammar=_GRAMMAR,
        scope_redirect_template=_SCOPE_REDIRECT_TEMPLATE,
    ).check(
        FinalizeContext(
            reply=_reply(short_redirect),
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Continue)


def test_uncited_substantive_reply_silent_when_citation_present() -> None:
    """Truthful counterfactual: the same long body but with a §-anchor
    must NOT fire — the cite-discipline gate already passed."""
    grounded_reply = (
        "Per JO 7110.65 §1-1-1, the purpose of the order is to prescribe "
        "air traffic control procedures and phraseology for use by persons "
        "providing air traffic control services. It serves to prevent "
        "collisions, provide a safe, orderly, and expeditious flow of air "
        "traffic, and support national security and homeland defense."
    )
    outcome = UncitedSubstantiveReplyHook(
        grammar=_GRAMMAR,
        scope_redirect_template=_SCOPE_REDIRECT_TEMPLATE,
    ).check(
        FinalizeContext(
            reply=_reply(grounded_reply),
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Continue)


def test_uncited_substantive_reply_silent_on_clarifying_question() -> None:
    """A reply that asks the user to clarify ('manned or unmanned?')
    isn't asserting anything — same exemption MissingCitationHook
    gives, mirrored here so a long valid clarifying reply isn't
    Halt-replaced at finalize."""
    clarifier = (
        "Could you specify whether you mean a manned or unmanned free "
        "balloon? JO 7110.65 handles each differently — manned balloons "
        "are treated as general aircraft, while unmanned free balloons "
        "fall under §9-6 with distinct traffic-advisory and altitude-"
        "verification rules. Before I answer, which one are you asking "
        "about?"
    )
    outcome = UncitedSubstantiveReplyHook(
        grammar=_GRAMMAR,
        scope_redirect_template=_SCOPE_REDIRECT_TEMPLATE,
    ).check(
        FinalizeContext(
            reply=_reply(clarifier),
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
        )
    )
    assert isinstance(outcome, Continue)


def test_uncited_substantive_reply_respects_disabled_toggle() -> None:
    """Attribution eval disables catchers by name. Pipeline must honor
    'uncited_substantive_reply' in the disabled frozenset."""
    pipe = default_hook_pipeline(
        citation_grammar=_GRAMMAR,
        scope_redirect_template=_SCOPE_REDIRECT_TEMPLATE,
    )
    ctx = FinalizeContext(
        reply=_reply(_REDIRECT_THEN_ANSWER_REPLY),
        last_outcome=Continue(),
        tools_ran=frozenset({"search_memory"}),
        memory_block_attached=False,
    )
    enabled = pipe.run_finalize(ctx, disabled=frozenset())
    assert isinstance(enabled, Halt)
    assert enabled.reply.content == _SCOPE_REDIRECT_TEMPLATE
    disabled = pipe.run_finalize(ctx, disabled=frozenset({"uncited_substantive_reply"}))
    assert isinstance(disabled, Continue)


# ---------- table_fabrication (harness-5uq) ----------


# The exact table fabrication from the session 2026-04-24 MH RBN repro:
# cited §4-1-1 correctly, reconstructed the MH row from priors with
# `Under 50` → `50` (collapsed MH's label with the H row's distance).
# Keep this fixture identical to the live repro so any regression here
# would show up the same way.
_FABRICATED_MH_TABLE_REPLY = (
    "The usable distance for an MH class RBN is 50 miles, per "
    "JO 7110.65 §4-1-1 TBL 4-1-2:\n\n"
    "| Class | Power (watts) | Distance (miles) |\n"
    "|---|---|---|\n"
    "| MH | Under 50 | 50 |\n"
    "| H | 50 − 1,999 | 50 |\n"  # noqa: RUF001 — unicode-minus from the corpus
)


# Truthful table from the source — same shape, with the correct MH
# distance (25 miles). The catcher must NOT fire on this.
_TRUTHFUL_MH_TABLE_REPLY = (
    "The usable distance for an MH class RBN is 25 miles, per "
    "JO 7110.65 §4-1-1 TBL 4-1-2:\n\n"
    "| Class | Power (watts) | Distance (miles) |\n"
    "|---|---|---|\n"
    "| MH | Under 50 | 25 |\n"
    "| H | 50 − 1,999 | 50 |\n"  # noqa: RUF001 — unicode-minus from the corpus
)


# Tool output stand-in representing what SearchMemoryTool would return
# post-cap-bump: both tables chunks' body with the full TBL 4-1-2.
# Whitespace between the pipes varies from the model's emitted reply
# (corpus has no interior spaces; replies typically add them), so the
# catcher's normalized-compare is what makes the check pass.
_TOOL_OUTPUT_WITH_FULL_TABLE = (
    "[0.033] ALTITUDE AND DISTANCE LIMITATIONS\n"
    "  lesson: JO_7110.65 §4-1-1 (NAVAID Use Limitations — ALTITUDE AND DISTANCE LIMITATIONS)\n"
    "  |**Class**|**Power (watts)**|**Distance**<br>**(miles)**|\n"
    "|---|---|---|\n"
    "|CL|Under 25|15|\n"
    "|MH|Under 50|25|\n"
    "|H|50 − 1,999|50|\n"  # noqa: RUF001 — unicode-minus from the corpus
    "|HH|2,000 or more|75|\n"
)


def test_table_fabrication_fires_on_repro_row() -> None:
    """Session 2026-04-24 repro: `|MH|Under 50|50|` does not appear in
    tool output (truth is `|MH|Under 50|25|`) → Halt with refusal."""
    reply = _reply(_FABRICATED_MH_TABLE_REPLY)
    outcome = TableFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            tool_outputs=(_TOOL_OUTPUT_WITH_FULL_TABLE,),
        )
    )
    assert isinstance(outcome, Halt)
    assert outcome.reply.content == TABLE_FABRICATION_FALLBACK
    assert outcome.reply.tool_calls == ()


def test_table_fabrication_passes_on_truthful_table() -> None:
    """Every data row in the reply appears verbatim (whitespace-
    normalized) in a tool output → Continue."""
    reply = _reply(_TRUTHFUL_MH_TABLE_REPLY)
    outcome = TableFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            tool_outputs=(_TOOL_OUTPUT_WITH_FULL_TABLE,),
        )
    )
    assert isinstance(outcome, Continue)


def test_table_fabrication_silent_without_grounding_tool() -> None:
    """No grounding tool ran → defer to ungrounded_citation, don't
    fire. A reply with a fabricated table AND no grounding is the
    'ungrounded citation' shape; this catcher only covers the narrower
    grounded-but-fabricated-row case."""
    reply = _reply(_FABRICATED_MH_TABLE_REPLY)
    outcome = TableFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset(),  # no grounding tool ran
            memory_block_attached=False,
            tool_outputs=(),
        )
    )
    assert isinstance(outcome, Continue)


def test_table_fabrication_skips_non_table_replies() -> None:
    """Prose reply with numbers but no pipe table — no pipe rows to
    check, Continue."""
    reply = _reply("The MH class RBN has a usable distance of 25 miles, per JO 7110.65 §4-1-1.")
    outcome = TableFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            tool_outputs=(_TOOL_OUTPUT_WITH_FULL_TABLE,),
        )
    )
    assert isinstance(outcome, Continue)


def test_table_fabrication_tolerates_whitespace_variation() -> None:
    """Reply uses `| MH | Under 50 | 25 |` (spaces); tool output uses
    `|MH|Under 50|25|` (no spaces). Normalization makes them match."""
    reply = _reply("| Class | Power | Distance |\n|---|---|---|\n| MH | Under 50 | 25 |\n")
    outcome = TableFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            tool_outputs=("|MH|Under 50|25|",),
        )
    )
    assert isinstance(outcome, Continue)


def test_table_fabrication_tolerates_comma_grouping() -> None:
    """`1,999` in the reply vs `1999` in a hypothetical tool output —
    commas are stripped at normalize time so the row still matches."""
    reply = _reply(
        "| Class | Power | Distance |\n|---|---|---|\n| H | 50 − 1,999 | 50 |\n"  # noqa: RUF001 — unicode-minus
    )
    outcome = TableFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            tool_outputs=("|H|50 − 1999|50|",),  # noqa: RUF001 — unicode-minus
        )
    )
    assert isinstance(outcome, Continue)


def test_table_fabrication_respects_disabled_toggle() -> None:
    """Attribution eval disables catchers by name — pipeline must honor
    'table_fabrication' in the disabled frozenset."""
    pipe = default_hook_pipeline()
    reply = _reply(_FABRICATED_MH_TABLE_REPLY)
    ctx = FinalizeContext(
        reply=reply,
        last_outcome=Continue(),
        # A grounding tool ran, so ungrounded_citation won't fire and
        # we can isolate table_fabrication's contribution.
        tools_ran=frozenset({"search_memory"}),
        memory_block_attached=False,
        tool_outputs=(_TOOL_OUTPUT_WITH_FULL_TABLE,),
    )
    # Enabled: table_fabrication halts.
    enabled = pipe.run_finalize(ctx, disabled=frozenset())
    assert isinstance(enabled, Halt)
    assert enabled.reply.content == TABLE_FABRICATION_FALLBACK
    # Disabled: table_fabrication's contribution is isolated by also
    # disabling numeric_fabrication (which independently catches the
    # prose "MH class ... 50 miles" lead-in of the same reply). With
    # both disabled, last_outcome=Continue leaves no hook firing.
    disabled = pipe.run_finalize(
        ctx,
        disabled=frozenset({"table_fabrication", "numeric_fabrication"}),
    )
    assert isinstance(disabled, Continue)


def test_table_fabrication_ignores_label_only_rows() -> None:
    """A pipe row with no digits can't be a numeric-fabrication target.
    The header row and any all-label row must not trip the catcher."""
    reply = _reply(
        "| Col A | Col B | Col C |\n"
        "|---|---|---|\n"
        "| alpha | beta | gamma |\n"
        "| delta | epsilon | zeta |\n"
    )
    outcome = TableFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            # Empty tool output — but none of the rows have digits, so
            # none are data rows by our definition, so Continue.
            tool_outputs=(),
        )
    )
    assert isinstance(outcome, Continue)


# ---------- numeric_fabrication (harness-5uq prose-shape) ----------


# Tool output containing the full TBL 4-1-2 pipe table. Header tags
# the third column as "Distance (miles)" so the hook can associate
# the numeric values with the 'mile' unit. Corpus uses **bold**
# markdown in headers; the cell stripper handles it.
_TOOL_OUTPUT_RBN_TABLE = (
    "[0.033] ALTITUDE AND DISTANCE LIMITATIONS\n"
    "  lesson: JO_7110.65 §4-1-1\n"
    "  |**Class**|**Power (watts)**|**Distance (miles)**|\n"
    "|---|---|---|\n"
    "|CL|Under 25|15|\n"
    "|MH|Under 50|25|\n"
    "|H|50 - 1,999|50|\n"
    "|HH|2,000 or more|75|\n"
)


def test_numeric_fabrication_fires_on_cross_row_prose() -> None:
    """Reply says 'MH class ... 50 miles' but tool output has
    |MH|Under 50|25|. Distance-column unit matches ('miles'), label
    MH has a row, but 50 isn't the MH row's distance value — it's
    the H row's. Halt with canned refusal."""
    reply = _reply(
        "The usable distance for an MH class RBN is 50 miles for all "
        "altitudes, per JO 7110.65 §4-1-1."
    )
    outcome = NumericFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            tool_outputs=(_TOOL_OUTPUT_RBN_TABLE,),
        )
    )
    assert isinstance(outcome, Halt)
    assert outcome.reply.content == TABLE_FABRICATION_FALLBACK


def test_numeric_fabrication_passes_on_truthful_prose() -> None:
    """'MH class ... 25 miles' agrees with |MH|Under 50|25|. The value
    25 is present in the MH row's distance column → Continue."""
    reply = _reply("The usable distance for an MH class RBN is 25 miles, per JO 7110.65 §4-1-1.")
    outcome = NumericFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            tool_outputs=(_TOOL_OUTPUT_RBN_TABLE,),
        )
    )
    assert isinstance(outcome, Continue)


def test_numeric_fabrication_silent_without_grounding_tool() -> None:
    """No grounding tool ran → defer to ungrounded_citation, don't
    fire."""
    reply = _reply("The usable distance for an MH class RBN is 50 miles.")
    outcome = NumericFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset(),
            memory_block_attached=False,
            tool_outputs=(),
        )
    )
    assert isinstance(outcome, Continue)


def test_numeric_fabrication_silent_when_label_not_in_tool_table() -> None:
    """Claim references a label the tool-output tables don't have —
    the catcher can't structurally verify, so it lets the reply
    through. Out-of-scope, not a fabrication the catcher can claim."""
    reply = _reply("The XX class device is 99 miles per JO 7110.65 §4-1-1.")
    outcome = NumericFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            tool_outputs=(_TOOL_OUTPUT_RBN_TABLE,),
        )
    )
    assert isinstance(outcome, Continue)


def test_numeric_fabrication_silent_when_unit_column_absent() -> None:
    """Tool output's table has no distance column but reply claims a
    'miles' value — without a unit-tagged column, the catcher can't
    structurally verify. Continue."""
    reply = _reply("The MH class needs 50 miles per JO 7110.65 §4-1-1.")
    # Header lacks a 'miles' / 'ft' / 'watts' column, so row-map is empty.
    tool_out = "  |**Class**|**Description**|\n|---|---|\n|MH|Medium power|\n"
    outcome = NumericFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            tool_outputs=(tool_out,),
        )
    )
    assert isinstance(outcome, Continue)


def test_numeric_fabrication_handles_comma_grouping() -> None:
    """'1,999' in reply should canonicalize to '1999' for lookup
    against the tool row's '1,999'. The catcher should accept a
    legitimate claim about the H row (`50 - 1,999 watts`, 50 miles)."""
    reply = _reply(
        "An H class RBN operates between 50 and 1,999 watts and has a "
        "usable distance of 50 miles, per JO 7110.65 §4-1-1."
    )
    outcome = NumericFabricationHook().check(
        FinalizeContext(
            reply=reply,
            last_outcome=Continue(),
            tools_ran=frozenset({"search_memory"}),
            memory_block_attached=False,
            tool_outputs=(_TOOL_OUTPUT_RBN_TABLE,),
        )
    )
    assert isinstance(outcome, Continue)


def test_numeric_fabrication_respects_disabled_toggle() -> None:
    """Attribution eval disables catchers by name — pipeline must honor
    'numeric_fabrication' in the disabled frozenset."""
    pipe = default_hook_pipeline()
    reply = _reply("The usable distance for an MH class RBN is 50 miles, per JO 7110.65 §4-1-1.")
    ctx = FinalizeContext(
        reply=reply,
        last_outcome=Continue(),
        tools_ran=frozenset({"search_memory"}),
        memory_block_attached=False,
        tool_outputs=(_TOOL_OUTPUT_RBN_TABLE,),
    )
    # Enabled: numeric_fabrication halts.
    enabled = pipe.run_finalize(ctx, disabled=frozenset())
    assert isinstance(enabled, Halt)
    assert enabled.reply.content == TABLE_FABRICATION_FALLBACK
    # Disabled: no other finalize hook fires (ungrounded_citation is
    # disarmed by tools_ran; table_fabrication doesn't see a pipe
    # table in the reply; last_outcome is Continue).
    disabled = pipe.run_finalize(ctx, disabled=frozenset({"numeric_fabrication"}))
    assert isinstance(disabled, Continue)


# ---------- missing_citation (harness-5uq follow-up) ----------


def test_missing_citation_fires_on_jo_reference_without_section() -> None:
    """Session 2026-04-24 repro 1: 'The purpose of FAA Order JO 7110.65
    as it pertains to Air Traffic Control is to prescribe...' —
    substantive, references the order, no §-anchor. Grounding tool
    ran. Must Nudge."""
    reply_text = (
        "The purpose of FAA Order JO 7110.65 as it pertains to Air "
        "Traffic Control is to prescribe air traffic control procedures "
        "and phraseology for use by persons providing air traffic "
        "control services. Specifically, it serves to prevent collisions, "
        "provide a safe, orderly, and expeditious flow of air traffic, "
        "and support national security and homeland defense missions."
    )
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Nudge)
    assert "section citation" in outcome.text.lower()


def test_missing_citation_fires_on_according_to_jo_without_section() -> None:
    """Session 2026-04-24 repro 2: 'According to JO 7110.65, when
    procedures or minima are applied jointly...' — substantive,
    references the order, no §-anchor. Must Nudge."""
    reply_text = (
        "According to JO 7110.65, when procedures or minima are applied "
        "jointly or otherwise require the cooperation or concurrence of "
        "more than one facility or organization, they must be documented "
        "in a Procedural Letter of Agreement (LOA). LOAs only supplement "
        "this order and any minima they specify must not be less than "
        "that specified in the order."
    )
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Nudge)


def test_missing_citation_passes_when_section_anchor_present() -> None:
    """Truthful counterfactual: same reply with an inline §-anchor must
    NOT fire. Guards against the catcher drifting into demanding a
    specific citation FORM."""
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(
                "Per JO 7110.65 §1-1-1, the purpose of the order is to "
                "prescribe air traffic control procedures and phraseology."
            ),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_missing_citation_passes_when_table_anchor_present() -> None:
    """TBL / Table / FIG anchors count as citations — they're always
    section-scoped in the JO, so 'TBL 4-1-2' unambiguously points at
    §4-1-1's second table."""
    for anchor in ("TBL 4-1-2", "Table 4-1-2", "FIG 3-9-1", "Figure 3-9-1"):
        outcome = MissingCitationHook(grammar=_GRAMMAR).check(
            BailContext(
                reply=_reply(
                    f"Per JO 7110.65 {anchor}, usable radius distances for "
                    "L/MF Radio Beacons are given by class and power."
                ),
                tools_ran_this_turn=True,
                tools_ran=frozenset({"search_memory"}),
            )
        )
        assert isinstance(outcome, Continue), (
            f"missing_citation false-positived on anchor form {anchor!r}"
        )


def test_missing_citation_silent_without_grounding_tool() -> None:
    """No grounding tool ran → this isn't the hook's job. A reply
    referencing JO 7110.65 with no §-anchor AND no grounding is
    ungrounded-citation territory (if it also has a §) or just
    a parametric answer (if it doesn't)."""
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(
                "FAA Order JO 7110.65 governs air traffic control operations "
                "and covers every controller-facing procedure."
            ),
            tools_ran_this_turn=False,
            tools_ran=frozenset(),
        )
    )
    assert isinstance(outcome, Continue)


def test_missing_citation_fires_on_jo_phraseology_without_citation() -> None:
    """Session 2026-04-24 repro (retry round): reply contained the
    JO-normative all-caps phraseology 'RADAR SERVICE TERMINATED,
    SQUAWK VFR' without naming the order OR citing a section. The
    phraseology-marker fallback in the in-scope check must catch this.

    Fix landed this session: _JO_PHRASEOLOGY_MARKERS_RE as a secondary
    signal alongside _ORDER_REFERENCE_RE.
    """
    reply_text = (
        "The correct phraseology for terminating radar service to a VFR "
        "aircraft is:\n\n"
        "RADAR SERVICE TERMINATED, SQUAWK VFR,\n\nor\n\n"
        "RADAR SERVICE TERMINATED, SQUAWK ONE TWO ZERO ZERO.\n\n"
        "Do not assign the code as a routine phraseology."
    )
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Nudge)


def test_missing_citation_phraseology_signal_passes_on_prose_without_markers() -> None:
    """Counter-check: a reply that uses no phraseology markers and no
    order reference stays silent. Guards against the phraseology regex
    being too loose (e.g. matching generic all-caps acronyms)."""
    reply_text = (
        "The ATC IFR VFR and FAA are common acronyms in aviation. "
        "Their meanings cover a wide range of procedures and services."
    )
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_missing_citation_silent_on_clarifying_question_reply() -> None:
    """Session 2026-04-24 repro (retry after ambiguous_context): the
    clarifying reply mentioned 'JO 7110.65' but had no §-anchor,
    tripping missing_citation despite being a valid clarifying
    question. Exempt via _CLARIFYING_QUESTION_RE."""
    reply_text = (
        'The term "balloon" can refer to both unmanned free balloons and '
        "manned balloons, which are handled differently according to JO "
        "7110.65. Could you please clarify whether you are referring to "
        "an unmanned free balloon or a manned balloon? This will help me "
        "provide the correct answer."
    )
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_missing_citation_silent_on_do_you_mean_clarifier() -> None:
    """Variant clarifying shape — 'Do you mean X or Y?'"""
    reply_text = (
        "Your question about JO 7110.65 and balloons is ambiguous. Do you "
        "mean an unmanned free balloon or a manned balloon? The rules "
        "differ substantially, so the answer depends on which you intend."
    )
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_missing_citation_silent_on_short_reply() -> None:
    """Sub-80-char replies are usually refusals or scope-redirects
    that don't need a citation — Continue."""
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply("I don't have that indexed from JO 7110.65."),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_missing_citation_silent_when_order_not_referenced() -> None:
    """Reply doesn't mention JO 7110.65 — hook must not demand a
    citation. Non-airton_c1 characters naturally fall into this path."""
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(
                "The user asked about Python dict merging. The answer: "
                "in modern Python, `{**a, **b}` merges two dicts with "
                "right-hand keys winning."
            ),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_missing_citation_silent_on_short_scope_redirect() -> None:
    """A short, clean scope-redirect references the order to say what's
    NOT covered ('That question is outside JO 7110.65') and shouldn't
    be re-nudged into a pointless retry. Bypass holds for short bodies."""
    reply_text = (
        "That's a pilot-side question — outside JO 7110.65, which is the "
        "controller's handbook. Ask airton_c for the pilot view."
    )
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_missing_citation_fires_on_redirect_then_answer_shape() -> None:
    """harness-zsbz repro (2026-05-14): retry response prefixes 'JO 7110.65
    doesn't cover it' then writes three paragraphs of general-knowledge
    content. The redirect marker bypass only holds when the body is
    short. A long marker-bearing reply gets nudged so the retry can
    produce a clean refusal (and the finalize-phase
    UncitedSubstantiveReplyHook catches what survives)."""
    reply_text = (
        "That's a pilot-facing question — airton_c is the generalist. "
        "JO 7110.65 doesn't cover it. In real-life practice, it's a "
        "personal decision based on cultural norms and local customs. "
        "Typically, the second ring or any unused ring on the left "
        "hand is used for subsequent marriages. However, there's no "
        "universal rule and many people choose differently based on "
        "their cultural background, family traditions, or personal "
        "preference about visibility of marital status."
    )
    outcome = MissingCitationHook(grammar=_GRAMMAR).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Nudge)


def test_missing_citation_accepts_cfr_dot_form_via_character_grammar() -> None:
    """harness-ygvg regression: airton_c_tfr cites 14 CFR sections in
    dot-form (§91.141, §91.137 etc.). The hardcoded
    `_CITATION_PRESENT_RE` only knows the JO/AIM hyphen-form, so a
    correctly-cited TFR decode was falsely tripping missing_citation
    and getting discarded. The hook must consult the character's own
    `citation_grammar.surface_patterns` so each persona's native shape
    counts as 'cited'."""
    tfr_grammar = load_character(_REPO / "character" / "airton_c_tfr").citation_grammar
    assert tfr_grammar is not None
    reply_text = (
        "Per 14 CFR §91.141, this NOTAM is a VIP movement TFR. The "
        "restriction defines a 5 NM radius cylinder centered at "
        "390333N0771929W from 1230Z to 1800Z on 2026-05-16, surface "
        "to 9999 ft MSL. Part 91 operations inside the inner core "
        "are prohibited except for the approved categories the NOTAM "
        "enumerates."
    )
    outcome = MissingCitationHook(grammar=tfr_grammar).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"assemble_context"}),
        )
    )
    assert isinstance(outcome, Continue), (
        "14 CFR §91.141 dot-form must count as a citation via airton_c_tfr's grammar"
    )


def test_missing_citation_respects_disabled_toggle() -> None:
    """Attribution eval disables catchers by name. Pipeline must honor
    'missing_citation' in the disabled frozenset."""
    pipe = default_hook_pipeline(citation_grammar=_GRAMMAR)
    reply_text = (
        "According to JO 7110.65, when procedures are applied jointly "
        "they must be documented in a Procedural Letter of Agreement. "
        "LOAs supplement the order and specify minima no less than "
        "what the order itself requires."
    )
    ctx = BailContext(
        reply=_reply(reply_text),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_memory"}),
    )
    enabled = pipe.run_bail(ctx, disabled=frozenset())
    assert isinstance(enabled, Nudge)
    disabled = pipe.run_bail(ctx, disabled=frozenset({"missing_citation"}))
    assert isinstance(disabled, Continue)


# ---------- list_count_mismatch (harness-5uq follow-up #2) ----------


def test_list_count_mismatch_fires_on_4_purposes_repro() -> None:
    """Session 2026-04-24 repro: 'The four specific primary purposes
    of ATC are as follows:' followed by a 3-item list. Must Nudge."""
    reply_text = (
        "The four specific primary purposes of Air Traffic Control (ATC) "
        "are as follows:\n\n"
        "1. Prevent a collision involving aircraft operating in the system.\n"
        "2. Provide a safe, orderly, and expeditious flow of air traffic.\n"
        "3. Support National Security and Homeland Defense missions.\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Nudge)
    assert "count mismatch" in outcome.text.lower()


def test_list_count_mismatch_passes_when_claim_matches_list() -> None:
    """Truthful counterfactual: claim=3, list=3 -> Continue."""
    reply_text = (
        "Per JO 7110.65 §2-1-1, the three items covered are:\n\n"
        "1. Prevent a collision involving aircraft.\n"
        "2. Provide a safe, orderly, and expeditious flow.\n"
        "3. Support National Security and Homeland Defense missions.\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_silent_when_no_enumerated_list() -> None:
    """Narrative prose with count but no list-shaped enumeration
    should NOT fire — the reply may be correctly counting an inline
    enumeration that's prose-joined ('X, Y, and Z')."""
    reply_text = (
        "Per §2-1-1, the four main goals of ATC are collision prevention, "
        "safe and orderly traffic flow, national security support, and "
        "additional controller services."
    )
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_silent_when_no_count_claim() -> None:
    """Reply with a list but no count claim — Continue. The model is
    enumerating without pre-committing to a specific count."""
    reply_text = (
        "Per §2-1-1, the ATC system's roles are:\n\n"
        "1. Collision prevention.\n"
        "2. Safe and orderly flow.\n"
        "3. National security.\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_ignores_section_number_digits() -> None:
    """Digits embedded in section numbers ('JO 7110.65 §2-1-1') must
    not trip the count claim — they're identifiers, not counts.
    Guard on the `\\s+[A-Za-z]` suffix in the regex."""
    reply_text = "Per JO 7110.65 §2-1-1, the requirements are strict and comprehensive."
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_ignores_multiple_count_claims() -> None:
    """When a reply has >1 distinct count claim, the catcher can't
    know which one is 'the' claim for the list — conservative Continue
    rather than guess."""
    reply_text = (
        "The four purposes of ATC are broad; there are five reasons to study "
        "them, and seven sections cover them. Here are the items:\n\n"
        "1. one\n2. two\n3. three\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_matches_bullet_lists() -> None:
    """Bulleted items count too (`- foo`, `* foo`, `• foo`). Nudge on
    claim=4 with a 2-bullet list."""
    reply_text = "The four main reasons are:\n\n- alpha\n- beta\n"
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Nudge)


def test_list_count_mismatch_ignores_cfr_citation_digits() -> None:
    """harness-ygvg follow-up #2: airton_c_tfr observed 2026-05-15
    discarding a clean §99.7 UAS TFR decode with two restricted areas.
    The model's paraphrase line was 'According to 14 CFR §99.7, all
    UAS flight operations are prohibited' — the count-claim regex
    matched '14 CFR §99.7, all UAS flight operations are' with '14'
    treated as a count, then compared to the 12-bullet two-area list
    and fired the mismatch. Citation patterns (§ / CFR / USC / AIM /
    JO / AC) inside a count-claim span are now filtered."""
    reply_text = (
        "Lead with the verdict: All UAS flight operations are prohibited.\n\n"
        "Geometry:\n\n"
        "- First Restricted Area:\n"
        "  - Center: 403946N0735805W\n"
        "  - Radius: 1 NM\n"
        "  - Floor: SFC\n"
        "  - Ceiling: 400FT AGL\n"
        "  - Active Window: 2605161000 UTC to 2605161500 UTC\n"
        "- Second Restricted Area:\n"
        "  - Center: 403502N0735801W\n"
        "  - Radius: 1 NM\n"
        "  - Floor: SFC\n"
        "  - Ceiling: 400FT AGL\n"
        "  - Active Window: 2605161130 UTC to 2605161800 UTC\n\n"
        "According to 14 CFR §99.7, all UAS flight operations are prohibited "
        "within the defined areas during the specified times."
    )
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"assemble_context"}),
        )
    )
    assert isinstance(outcome, Continue), (
        "'14 CFR §99.7' must not be matched as a count claim of 14 items"
    )


def test_list_count_mismatch_ignores_usc_citation_digits() -> None:
    """'Per 49 USC 40103(B)(3), the relevant restrictions are listed
    below' fits the count-claim shape if you read '49' as a count.
    USC inside the matched span filters it out, matching the CFR
    behaviour. The list has 5 bullets; without the filter the hook
    would fire on 49 ≠ 5."""
    reply_text = (
        "Some prose.\n\n"
        "Per 49 USC 40103(B)(3), the relevant restrictions are listed below.\n\n"
        "- alpha\n- beta\n- gamma\n- delta\n- epsilon\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"assemble_context"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_ignores_phone_and_frequency_digits() -> None:
    """harness-ygvg follow-up: airton_c_tfr observed 2026-05-15
    discarding a clean TFR decode because the model echoed the NOTAM's
    'TEL 406-444-4242 OR FREQ 123.725 JERICHO CREEK IS IN CHARGE'
    contact line. The old regex matched '4242 or … is' as a count
    claim of 4242 items against the reply's 5-bullet geometry list.
    Capping the digit group to 1-2 chars rules out phone numbers,
    frequencies, timestamps, and other multi-digit runs without
    losing legitimate count claims like 'the 12 sections are'."""
    reply_text = (
        "Lead with the verdict: TFR is in effect per 14 CFR §91.137(a)(2).\n\n"
        "Geometry:\n\n"
        "- Center: 462830N1122030W\n"
        "- Radius: 3NM\n"
        "- Floor: SFC\n"
        "- Ceiling: 8000FT MSL\n"
        "- Active window: 2605151548-2605180400\n\n"
        "The Helena Interagency Dispatch Center at 406-444-4242 or on the "
        "frequency 123.725 is in charge. Salt Lake City ARTCC is 801-320-2560."
    )
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"assemble_context"}),
        )
    )
    assert isinstance(outcome, Continue), (
        "phone/frequency digits must not be matched as a count claim"
    )


def test_list_count_mismatch_respects_disabled_toggle() -> None:
    """Attribution eval disables catchers by name."""
    pipe = default_hook_pipeline()
    reply_text = (
        "The four specific primary purposes of ATC are as follows:\n\n"
        "1. Prevent a collision.\n"
        "2. Provide safe flow.\n"
        "3. Support national security.\n\n"
        "Per JO 7110.65 §2-1-1."
    )
    ctx = BailContext(
        reply=_reply(reply_text),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_memory"}),
    )
    enabled = pipe.run_bail(ctx, disabled=frozenset())
    assert isinstance(enabled, Nudge)
    disabled = pipe.run_bail(
        ctx,
        disabled=frozenset({"list_count_mismatch"}),
    )
    assert isinstance(disabled, Continue)


# ---------- reserved_squawk_code (harness-5uq follow-up #3) ----------


def _bail_ctx(reply_text: str) -> BailContext:
    return BailContext(
        reply=_reply(reply_text),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_memory"}),
    )


def test_reserved_squawk_fires_on_quoted_phonetic_repro() -> None:
    """Session 2026-04-24 repro: model proposed
    '"Radar service terminated, squawk seven five hundred."' —
    quoted phraseology assigning the 7500 hijack code. Must Nudge."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx(
            'The correct phraseology is:\n\n"Radar service terminated, '
            'squawk seven five hundred."\n\nPer JO 7110.65 §7-6-11.'
        )
    )
    assert isinstance(outcome, Nudge)
    assert "reserved-squawk-code" in outcome.text.lower()


def test_reserved_squawk_fires_on_digit_form_in_quotes() -> None:
    """Digit-form code inside a quoted phraseology line."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx('The corrected phraseology: "Radar service terminated, squawk 7500."')
    )
    assert isinstance(outcome, Nudge)


def test_reserved_squawk_fires_on_all_caps_phraseology() -> None:
    """All-caps SQUAWK is the JO phraseology convention — fires even
    without quotes around it."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx("PHRASEOLOGY:\n\nSQUAWK 7600\n\nPer §7-6-11.")
    )
    assert isinstance(outcome, Nudge)


def test_reserved_squawk_fires_on_italic_phraseology() -> None:
    """Markdown italic `_squawk 7700_` — matches even though `_` is a
    word char (handled by the custom left/right lookarounds)."""
    outcome = ReservedSquawkCodeHook().check(_bail_ctx("Per §7-6-11 the line is _squawk 7700_."))
    assert isinstance(outcome, Nudge)


def test_reserved_squawk_fires_on_bold_phraseology() -> None:
    """Markdown bold **squawk 7500**."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx("The reply should quote **squawk 7500** verbatim (as an error).")
    )
    assert isinstance(outcome, Nudge)


def test_reserved_squawk_fires_on_lay_seventy_five_hundred_form() -> None:
    """Lay paraphrase 'seventy five hundred' (student phrasing)."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx('"Radar service terminated, squawk seventy five hundred."')
    )
    assert isinstance(outcome, Nudge)


def test_reserved_squawk_fires_on_digit_by_digit_readback() -> None:
    """JO phraseology convention: digit-by-digit ('seven five zero zero')."""
    outcome = ReservedSquawkCodeHook().check(_bail_ctx('Quoted: "squawk seven five zero zero"'))
    assert isinstance(outcome, Nudge)


def test_reserved_squawk_passes_on_prose_description() -> None:
    """Narrative 'when you observe Code 7500' — not an assignment.
    Must pass; this is §5-2-5 describing controller procedure."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx("When you observe a Code 7500 display, apply the procedures in §10-2-6.")
    )
    assert isinstance(outcome, Continue)


def test_reserved_squawk_passes_on_warning_phrasing() -> None:
    """'Remember that squawk 7500 is the hijack code' — warning, not
    assignment. No quote/italic/bold/caps wrapping. Pass."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx("Remember that squawk 7500 is the hijack code — never assign it.")
    )
    assert isinstance(outcome, Continue)


def test_reserved_squawk_passes_on_correct_1200_assignment() -> None:
    """Correct VFR code in assignment context must pass."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx(
            'The corrected phraseology: "Radar service terminated, squawk one two zero zero."'
        )
    )
    assert isinstance(outcome, Continue)


def test_reserved_squawk_passes_on_squawk_vfr() -> None:
    """'SQUAWK VFR' — correct per §5-2-7."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx('Per §5-2-7: "SQUAWK VFR" is the correct form.')
    )
    assert isinstance(outcome, Continue)


def test_reserved_squawk_does_not_match_resquawk_compound() -> None:
    """'resquawk' is one word — left-anchor must reject the prefix."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx(
            "Please resquawk your code once identified. For hijack, the "
            "aircraft will set 7500; controllers do not assign it."
        )
    )
    assert isinstance(outcome, Continue)


def test_reserved_squawk_silent_when_match_echoes_user_message() -> None:
    """Session 2026-04-24 repro (second pass): user asked to correct a
    phraseology containing a reserved code; model quoted the user's
    exact bad phrase inline to flag it as wrong. The quoted echo is
    NOT a new assignment — ReservedSquawk must pass."""
    user_msg = (
        'Correct the following radar phraseology: "Services stopped, squawk seventy five hundred"'
    )
    reply_text = (
        'The phrase "Services stopped, squawk seventy five hundred" is '
        "incorrect. The correct phraseology per §7-6-11 is: "
        '"RADAR SERVICE TERMINATED, SQUAWK ONE TWO ZERO ZERO."'
    )
    outcome = ReservedSquawkCodeHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
            user_message=user_msg,
        )
    )
    assert isinstance(outcome, Continue)


def test_reserved_squawk_still_fires_when_model_proposes_different_reserved_form() -> None:
    """Regression guard on the Fix-A echo check: if the model reformats
    the user's reserved-code phrase into a different form (e.g.
    'seventy five hundred' -> 'seven five hundred') and places it in
    an assignment context, that's still a new unsafe assignment, NOT
    an echo. Must still Nudge."""
    user_msg = (
        'Correct the following radar phraseology: "Services stopped, squawk seventy five hundred"'
    )
    # Model modified the code form from the user's input — propagation,
    # not echo.
    reply_text = 'The corrected phraseology: "Radar service terminated, squawk seven five hundred."'
    outcome = ReservedSquawkCodeHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
            user_message=user_msg,
        )
    )
    assert isinstance(outcome, Nudge)


def test_reserved_squawk_respects_disabled_toggle() -> None:
    """Attribution eval disables catchers by name."""
    pipe = default_hook_pipeline(catchers=("reserved_squawk_code",))
    ctx = _bail_ctx(
        'The correct phraseology is: "Radar service terminated, '
        'squawk seven five hundred." Per JO 7110.65 §7-6-11.'
    )
    enabled = pipe.run_bail(ctx, disabled=frozenset())
    assert isinstance(enabled, Nudge)
    disabled = pipe.run_bail(
        ctx,
        disabled=frozenset({"reserved_squawk_code"}),
    )
    assert isinstance(disabled, Continue)


# ---------- scope_redirect (harness-5uq follow-up #4) ----------


def _scope_ctx(user: str, reply: str) -> BailContext:
    return BailContext(
        reply=_reply(reply),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_memory"}),
        user_message=user,
    )


def test_scope_redirect_fires_on_roosters_with_atc_reply() -> None:
    """Session 2026-04-24 repro: user asked 'do roosters lay eggs'
    (zero aviation vocab), model replied with phraseology-correction
    content from a prior turn (context bleed)."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "do roosters lay eggs",
            'The phraseology "Services stopped, squawk seventy five hundred" '
            "is incorrect. Per JO 7110.65 §7-6-11, the correct phrase is "
            '"Radar service terminated."',
        )
    )
    assert isinstance(outcome, Nudge)
    assert "scope mismatch" in outcome.text.lower()


def test_scope_redirect_passes_in_scope_question() -> None:
    """User message HAS aviation vocab ('MH class RBN') → silent."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "What is the usable distance for an MH class RBN?",
            "Per §4-1-1 TBL 4-1-2, MH class has a 25 mile usable distance.",
        )
    )
    assert isinstance(outcome, Continue)


def test_scope_redirect_passes_short_followup() -> None:
    """Short follow-ups ('tell me more') legitimately lack aviation
    vocab but continue an in-scope thread. Length guard disarms."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "tell me more",
            "Per §4-1-1, the H class RBN has a 50 mile usable distance.",
        )
    )
    assert isinstance(outcome, Continue)


def test_scope_redirect_passes_correct_refusal_reply() -> None:
    """A reply that IS already scope-redirecting (even if it names JO
    7110.65 to decline) must pass — exempt via scope-redirect marker."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "can you help me write a Python function to sort a list",
            "That question is outside JO 7110.65. I am a specialist for "
            "FAA Air Traffic Control procedures only.",
        )
    )
    assert isinstance(outcome, Continue)


def test_scope_redirect_fires_when_python_answered_then_atc_bleed() -> None:
    """User asks non-ATC (Python), model answers Python but ALSO adds
    an unrelated ATC claim. The ATC claim is the bleed."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "can you write a Python function to sort a list",
            "Here is the answer: sorted(my_list) returns a new list. "
            "Per JO 7110.65, controllers must verify readback accuracy.",
        )
    )
    assert isinstance(outcome, Nudge)


def test_scope_redirect_fires_on_non_aviation_reply_to_non_aviation() -> None:
    """User asks something non-ATC, model answers in kind without ATC
    vocab. airton_c1 is scoped to JO 7110.65 only — answering an
    off-topic prompt with an off-topic answer is still a scope
    failure, even absent context-bleed. The catcher must nudge so
    the retry produces a proper scope-redirect."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "do roosters lay eggs",
            "No, roosters are male chickens and do not lay eggs. Only hens lay eggs.",
        )
    )
    assert isinstance(outcome, Nudge)
    assert "scope mismatch" in outcome.text.lower()


def test_scope_redirect_fires_on_walks_into_bar_joke() -> None:
    """Session 2026-04-24 repro: 'three men walk into a bar. One says
    hi to the bartender. What does the bartender say in relpy?' — a
    classic bar-joke opener with no aviation vocab and a non-ATC
    reply. `_JOKE_FRAME_RE` must catch the 'walks into a bar' shape
    so scope_redirect fires even when the reply stays off-topic."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "three men walk into a bar. One says hi to the bartender. "
            "What does the bartender say in reply?",
            'The bartender would likely respond with a greeting, such as "Hi, how can I help you?"',
        )
    )
    assert isinstance(outcome, Nudge)
    assert "scope mismatch" in outcome.text.lower()


def test_scope_redirect_fires_on_computer_race_riddle() -> None:
    """Session 2026-04-24 repro: 'Five Macs and PC join a computer
    race. Who won?' — a consumer-tech riddle. 'Macs' / 'PC' / 'race'
    aren't in the vocab list (bare 'Mac'/'PC' collide with aviation
    acronyms like MAC=Military Airlift Command), but the 'Who won?'
    tag is characteristic of race / contest jokes. Model produced
    a context-bleed reply about VFR balloon right-of-way.
    `_JOKE_FRAME_RE` must catch the 'who won' riddle ending so
    scope_redirect fires before missing_citation has to clean up."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "Five Macs and PC join a computer race. Who won?",
            "The right of way between a VFR single prop aircraft and a "
            "manned balloon can be determined by the specific "
            "circumstances and the principles of visual flight rules "
            "(VFR) operations. Per JO 7110.65 the aircraft on the "
            "right has the right of way.",
        )
    )
    assert isinstance(outcome, Nudge)
    assert "scope mismatch" in outcome.text.lower()


def test_scope_redirect_fires_on_consumer_tech_vocab() -> None:
    """Consumer-tech brand terms ('iphone', 'ipad', 'laptop',
    'smartphone') are unambiguously non-aviation and do not collide
    with any JO 7110.65 acronym — they trip Signal A even when the
    prompt lacks any joke-frame structure."""
    for user, term in (
        ("does my iphone interfere with cockpit radios", "iphone"),
        ("can I charge my laptop on the runway", "laptop"),
        ("why is my ipad battery draining so fast", "ipad"),
        ("smartphone apps for tracking flights", "smartphone"),
    ):
        outcome = ScopeRedirectHook().check(
            _scope_ctx(
                user,
                "Per JO 7110.65 §2-1-1, controllers must provide ATC service.",
            )
        )
        assert isinstance(outcome, Nudge), f"scope_redirect did not fire on {term!r} user message"


def test_scope_redirect_silent_without_user_message() -> None:
    """No user_message threaded through (bootstrap path) → Continue."""
    outcome = ScopeRedirectHook().check(
        BailContext(
            reply=_reply("Per JO 7110.65 §2-1-1, the ATC system prevents collisions."),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
            user_message=None,
        )
    )
    assert isinstance(outcome, Continue)


def test_scope_redirect_fires_on_joke_frame_structure() -> None:
    """Session 2026-04-24 repro: 'If a ghost, a turtle, and a
    refrigerator buy plane tickets, where do they go?' is a joke
    frame. `_CLEARLY_NON_AVIATION_RE` covers ghost/turtle/refrigerator
    individually, but the structural `_JOKE_FRAME_RE` is the durable
    signal — fires even when specific nouns aren't in the vocab list."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "If a pilot, a controller, and a mechanic walk into a bar, what do they order?",
            "Per JO 7110.65 §4-1-1, the usable distance for an MH class RBN is 25 miles.",
        )
    )
    assert isinstance(outcome, Nudge)


def test_scope_redirect_fires_on_reply_content_bleed() -> None:
    """Defense in depth: if the reply mixes aviation vocab AND clearly
    non-aviation vocab (e.g. 'Per §4-1-1 ... Also roosters don't lay
    eggs'), that's confusion/context-bleed regardless of the user's
    message shape. Signal C catches this even when A (user vocab)
    and B (joke frame) both miss."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "please summarize what that section says about beacons",
            "Per JO 7110.65 §4-1-1, the usable distance for an MH class "
            "RBN is 25 miles. Also, roosters do not lay eggs.",
        )
    )
    assert isinstance(outcome, Nudge)


def test_scope_redirect_expanded_vocab_catches_household_and_supernatural() -> None:
    """Expanded `_CLEARLY_NON_AVIATION_RE` covers household and
    supernatural terms that show up in comedy / riddle prompts."""
    for user, nonaviation_term in (
        ("what does a refrigerator need to fly an IFR approach", "refrigerator"),
        ("can a ghost squawk IDENT on a radar display", "ghost"),
        ("tell me what a turtle sees when approaching the runway", "turtle"),
    ):
        outcome = ScopeRedirectHook().check(
            _scope_ctx(
                user,
                "Per JO 7110.65 §5-2-7, VFR radar service termination uses 'squawk 1200'.",
            )
        )
        assert isinstance(outcome, Nudge), (
            f"scope_redirect did not fire on {nonaviation_term!r} user message"
        )


def test_scope_redirect_passes_bird_strike_question() -> None:
    """Session 2026-04-28 false positive: 'In what publication and
    section can I find information on bird strikes?' tripped the
    `bird` token in `_CLEARLY_NON_AVIATION_RE` even though bird
    strikes are documented in AIM §7-5-1..4, 14 CFR §25.631 / §29.631
    / §33.76 / §35.36, and JO 7110.65 §2-1-23.'bird' was dropped
    from the non-aviation vocab so this in-scope hazard question
    lands silently."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "In what publication and section can I find information on bird strikes?",
            "Bird-strike risks are covered in AIM §7-5-2 (Reducing Bird "
            "Strike Risks) and JO 7110.65 §2-1-23 (bird-activity "
            "advisories).",
        )
    )
    assert isinstance(outcome, Continue)


def test_scope_redirect_passes_full_bird_strike_section_dump() -> None:
    """Session 2026-04-28 follow-up: a thorough answer recites every
    relevant AIM / 14 CFR / JO 7110.65 section, naming 'bird' in many
    shapes — 'Bird Strike Risks', 'Bird and Other Wildlife
    Activities', '8-pound bird', 'bird species', 'bird migration',
    'bird type', 'bird or other wildlife', 'bird concentration'. No
    collocation list catches them all. With 'bird' out of the
    non-aviation vocab, every shape lands silently."""
    outcome = ScopeRedirectHook().check(
        _scope_ctx(
            "What publications, chapters and sections relate to bird strikes?",
            "The publications, chapters, and sections related to bird "
            "strikes include:\n"
            "1. AIM §7-5-2: Reducing Bird Strike Risks\n"
            "  - The most serious strikes are those involving ingestion "
            "into an engine or windshield strikes.\n"
            "  - Avoid overflight of known areas of bird concentration "
            "and flying at low altitudes during bird migration.\n"
            "2. AIM §7-5-3: Reporting Bird Strikes\n"
            "  - Pilots are urged to report any bird or other wildlife "
            "strike using FAA Form 5200-7.\n"
            "3. AIM §7-5-1: Migratory Bird Activity\n"
            "  - Bird strike risk increases during migration months.\n"
            "  - Altitudes of migrating birds vary with environmental "
            "variables.\n"
            "4. AIM §7-5-4: Bird Hazards and Flight Over National "
            "Refuges\n"
            "  - Report geographic location, bird type, numbers, and "
            "altitude to airport management.\n"
            "5. 14 CFR §25.631: Bird Strike Damage\n"
            "  - Empennage structure must be designed for continued "
            "safe flight after impact with an 8-pound bird.\n"
            "6. 14 CFR §29.631: Bird Strike for Rotorcraft\n"
            "  - Rotorcraft must withstand impact with a 2.2-lb bird.\n"
            "7. 14 CFR §35.36: Bird Impact\n"
            "  - Propeller must withstand impact of a 4-pound bird.\n"
            "8. 14 CFR §33.76: Bird Ingestion\n"
            "  - Compliance with large bird ingestion tests for "
            "engines.\n"
            "9. JO 7110.65 §2-1-23: Issue Advisory Information on "
            "Pilot-Reported Bird Activity\n"
            "  - Issue advisory information on pilot-reported bird "
            "activity.\n"
            "10. AIM §11-8-6: Some bird species may attack UAS.\n",
        )
    )
    assert isinstance(outcome, Continue)


def test_scope_redirect_passes_bird_hazard_and_activity_collocations() -> None:
    """Variant phrasings for bird hazards / activity / ingestion all
    pass — 'bird' is no longer a non-aviation vocab token, so any
    aviation-context bird question lands silently regardless of
    head-noun shape."""
    for prompt in (
        "What does the AIM say about bird hazards on final approach?",
        "How should controllers issue bird activity advisories?",
        "Where is bird ingestion certification documented for engines?",
        "What's the procedure for reporting a bird strike to ATC?",
    ):
        outcome = ScopeRedirectHook().check(
            _scope_ctx(
                prompt,
                "Per AIM §7-5-2, bird hazards near airports are mitigated "
                "by pilot reports and tower-issued advisories.",
            )
        )
        assert isinstance(outcome, Continue), f"false positive on {prompt!r}"


def test_scope_redirect_fires_on_bird_joke_via_joke_frame() -> None:
    """Dropping 'bird' from the non-aviation vocab does not lose
    joke detection — `_JOKE_FRAME_RE` catches structural shapes
    ('why did the bird cross the road', 'if a bird, a fish, and a
    horse...') regardless of which animal noun appears."""
    for prompt in (
        "Why did the bird cross the road in front of the runway?",
        "If a bird, a fish, and a horse all walked into a control tower",
    ):
        outcome = ScopeRedirectHook().check(
            _scope_ctx(
                prompt,
                "The bird crossed because it heard the chicken did it first.",
            )
        )
        assert isinstance(outcome, Nudge), f"joke-frame missed on {prompt!r}"


def test_scope_redirect_respects_disabled_toggle() -> None:
    """Attribution eval disables catchers by name."""
    pipe = default_hook_pipeline(catchers=("scope_redirect",))
    ctx = _scope_ctx(
        "do roosters lay eggs",
        "Per JO 7110.65 §7-6-11, radar service is terminated when...",
    )
    enabled = pipe.run_bail(ctx, disabled=frozenset())
    assert isinstance(enabled, Nudge)
    disabled = pipe.run_bail(ctx, disabled=frozenset({"scope_redirect"}))
    assert isinstance(disabled, Continue)


def test_scope_redirect_passes_geometry_in_tfr_reply() -> None:
    """harness-ygvg regression: airton_c_tfr's core.yaml directive
    explicitly tells the model to 'Surface the geometry (center,
    radius, floor, ceiling)' when decoding a NOTAM. The reply mixes
    that word with aviation vocab. Before the fix, 'geometry' was
    in `_CLEARLY_NON_AVIATION_RE` (intended to flag math-class) and
    Signal C tripped on every TFR decode."""
    user = (
        "FDC 6/0550 ZDC VA..AIRSPACE STERLING, VIRGINIA..TEMPORARY "
        "FLIGHT RESTRICTIONS. MAY 16, 2026 LOCAL."
    )
    reply = (
        "Per 14 CFR §91.141, this is a VIP movement TFR. Surface the "
        "geometry: center at 390333N0771929W, radius 5 NM, surface to "
        "9999 ft MSL. All Part 91 aircraft flight operations inside the "
        "inner core are prohibited from 1230Z to 1800Z on 2026-05-16."
    )
    outcome = ScopeRedirectHook().check(_scope_ctx(user, reply))
    assert isinstance(outcome, Continue), (
        "scope_redirect must not fire on a TFR decode that uses the in-domain word 'geometry'"
    )


def test_scope_redirect_nudge_uses_character_template() -> None:
    """harness-ygvg regression: when scope_redirect fires for a non-
    airton_c1 character, the nudge text must NOT contain airton_c1's
    'JO 7110.65 specialist' identity. Instead it must quote the
    character's own `scope_redirect_template` so the model has a
    correct example refusal to emit."""
    tfr_template = (
        "I only decode published TFR/NOTAM text. That question is "
        "outside my scope. For operational decisions, check with your "
        "CFI or call 1-800-WX-BRIEF."
    )
    hook = ScopeRedirectHook(
        character_name="airton_c_tfr",
        scope_redirect_template=tfr_template,
    )
    outcome = hook.check(
        _scope_ctx(
            "tell me a joke about a rooster and an egg",
            "Per JO 7110.65 §7-6-11, radar service is terminated...",
        )
    )
    assert isinstance(outcome, Nudge)
    assert "airton_c_tfr" in outcome.text
    assert "1-800-WX-BRIEF" in outcome.text, (
        "nudge must echo the character's scope_redirect_template so the "
        "model has the correct refusal sentence to emit"
    )
    assert "JO 7110.65 specialist" not in outcome.text, (
        "nudge must not leak airton_c1's identity into other characters"
    )


def test_scope_redirect_nudge_falls_back_to_generic_without_template() -> None:
    """When no character_name / scope_redirect_template is threaded
    through (e.g. legacy callers, the bare `_DEFAULT_PIPELINE`), the
    nudge is generic — no airton_c1-specific phrasing leaks through."""
    hook = ScopeRedirectHook()
    outcome = hook.check(
        _scope_ctx(
            "tell me a joke about a rooster and an egg",
            "Per JO 7110.65 §7-6-11, radar service is terminated...",
        )
    )
    assert isinstance(outcome, Nudge)
    assert "JO 7110.65 specialist" not in outcome.text
    assert "scope mismatch" in outcome.text


# ---------- ambiguous_context (harness-5uq follow-up #5) ----------


def _ambig_ctx(user: str, reply: str) -> BailContext:
    return BailContext(
        reply=_reply(reply),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_memory"}),
        user_message=user,
    )


def test_ambiguous_balloon_user_bare_reply_assumes_unmanned_free() -> None:
    """Session 2026-04-24 repro: user asked 'a balloon are intersecting
    ... right of way?'. Model silently assumed 'unmanned free balloon'
    and answered for §9-6 procedures. Hook must nudge."""
    outcome = AmbiguousContextHook().check(
        _ambig_ctx(
            "a single prop squawking 1200 and a balloon are intersecting, "
            "who has the right of way?",
            "When a single-engine propeller aircraft and an unmanned free "
            "balloon are intersecting, per JO 7110.65 §9-6-1 the aircraft "
            "typically has the right of way.",
        )
    )
    assert isinstance(outcome, Nudge)
    assert "ambiguous context" in outcome.text.lower()


def test_ambiguous_balloon_user_specifies_passes() -> None:
    """If the user already said 'unmanned', there's no ambiguity —
    the reply's 'unmanned free balloon' is legitimate."""
    outcome = AmbiguousContextHook().check(
        _ambig_ctx(
            "an unmanned balloon and an aircraft are intersecting, who has right of way?",
            "Per JO 7110.65 §9-6-1, unmanned free balloons are handled by "
            "traffic advisory; the aircraft has right of way.",
        )
    )
    assert isinstance(outcome, Continue)


def test_ambiguous_balloon_reply_asks_for_clarification_passes() -> None:
    """Reply that asks 'manned or unmanned?' — both alternatives in
    axis 1 — is the correct shape. Don't nudge; don't loop."""
    outcome = AmbiguousContextHook().check(
        _ambig_ctx(
            "a balloon and an aircraft are intersecting — right of way?",
            "Before I answer: are you asking about a manned balloon or an "
            "unmanned balloon? JO 7110.65 handles them differently.",
        )
    )
    assert isinstance(outcome, Continue)


def test_ambiguous_balloon_reply_free_vs_tethered_ask_passes() -> None:
    """Second axis: free vs. tethered — mentioning both offers
    choice, not commitment."""
    outcome = AmbiguousContextHook().check(
        _ambig_ctx(
            "an unmanned balloon near my flight path — how do I handle it?",
            "Do you mean a free balloon or a tethered balloon? The JO treats them differently.",
        )
    )
    # User already said 'unmanned' so the hook short-circuits before
    # even checking the reply — still Continue via the user-qualifier gate.
    assert isinstance(outcome, Continue)


def test_ambiguous_balloon_reply_no_qualifier_passes() -> None:
    """Bare user term + bare reply — no qualifier picked, nothing to
    challenge. Don't nudge (the model might be answering generally
    or asking later in the reply)."""
    outcome = AmbiguousContextHook().check(
        _ambig_ctx(
            "what rules apply when a balloon and an aircraft meet?",
            "Per JO 7110.65 §9-6, balloons near aircraft require traffic "
            "advisory procedures. The controller must coordinate separation.",
        )
    )
    assert isinstance(outcome, Continue)


def test_ambiguous_balloon_single_axis_commit_fires() -> None:
    """Reply that commits to only one qualifier in a single axis
    (e.g., 'unmanned' without 'free/tethered') still fires — the
    commitment to 'unmanned' alone is already an assumption the
    user didn't provide."""
    outcome = AmbiguousContextHook().check(
        _ambig_ctx(
            "a balloon crosses the traffic pattern — what does ATC do?",
            "Per §9-6-1, an unmanned balloon crossing the traffic pattern "
            "triggers the controller's traffic advisory procedures.",
        )
    )
    assert isinstance(outcome, Nudge)


def test_ambiguous_no_match_on_non_balloon_term() -> None:
    """User doesn't mention balloon — hook stays silent regardless of
    what the reply says. Prevents cross-term false positives."""
    outcome = AmbiguousContextHook().check(
        _ambig_ctx(
            "how do I terminate radar service for a VFR aircraft?",
            "Per §7-6-11: 'Radar service terminated, squawk VFR.'",
        )
    )
    assert isinstance(outcome, Continue)


def test_ambiguous_context_silent_without_user_message() -> None:
    outcome = AmbiguousContextHook().check(
        BailContext(
            reply=_reply("Per §9-6-1, unmanned free balloons are handled by traffic advisory."),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
            user_message=None,
        )
    )
    assert isinstance(outcome, Continue)


def test_ambiguous_context_respects_disabled_toggle() -> None:
    """Attribution eval disables catchers by name."""
    pipe = default_hook_pipeline(catchers=("ambiguous_context",))
    ctx = _ambig_ctx(
        "a balloon intersects a VFR aircraft — who has right of way?",
        "Per §9-6-1, unmanned free balloons are handled by traffic advisory.",
    )
    enabled = pipe.run_bail(ctx, disabled=frozenset())
    assert isinstance(enabled, Nudge)
    disabled = pipe.run_bail(ctx, disabled=frozenset({"ambiguous_context"}))
    assert isinstance(disabled, Continue)


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


# ---------- FabricatedSectionHook (harness-aise) ----------


def _ctx_with_reply(content: str) -> BailContext:
    return BailContext(
        reply=_reply(content),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_memory"}),
    )


def test_fabricated_section_silent_with_empty_anchor_set() -> None:
    """No corpus → hook silent regardless of reply content. Default
    construction yields an empty frozenset; airton / airton_b / airton_c
    rely on this."""
    outcome = FabricatedSectionHook().check(
        _ctx_with_reply("Per JO 7110.65 §99-99-99, hijack codes are blue.")
    )
    assert isinstance(outcome, Continue)


def test_fabricated_section_silent_when_no_citations() -> None:
    outcome = FabricatedSectionHook(
        valid_anchors=frozenset({"§3-10-3"}),
    ).check(_ctx_with_reply("That's outside JO 7110.65 — talk to airton_c."))
    assert isinstance(outcome, Continue)


def test_fabricated_section_passes_when_all_cited_anchors_valid() -> None:
    outcome = FabricatedSectionHook(
        valid_anchors=frozenset({"§3-10-3", "§4-1-1", "§3-10"}),
    ).check(_ctx_with_reply("Per JO 7110.65 §3-10-3, same-runway separation applies to arrivals."))
    assert isinstance(outcome, Continue)


def test_fabricated_section_fires_on_invented_anchor() -> None:
    """The §3-99-3 case — model invented a section that doesn't exist."""
    outcome = FabricatedSectionHook(
        valid_anchors=frozenset({"§3-10-3", "§4-1-1"}),
    ).check(
        _ctx_with_reply("Per JO 7110.65 §3-99-3, hijack squawk handling requires immediate vector.")
    )
    assert isinstance(outcome, Nudge)
    assert "§3-99-3" in outcome.text
    assert "search_memory" in outcome.text


def test_fabricated_section_fires_on_mixed_valid_and_invalid() -> None:
    """One real cite + one invented cite — must Nudge with the invented
    one named, not silenced by the presence of the real one."""
    outcome = FabricatedSectionHook(
        valid_anchors=frozenset({"§3-10-3"}),
    ).check(
        _ctx_with_reply(
            "Per JO 7110.65 §3-10-3 (same runway) and §3-99-3 (made up), see the table."
        )
    )
    assert isinstance(outcome, Nudge)
    assert "§3-99-3" in outcome.text
    # The valid anchor should not appear in the nudge — only the offender.
    assert "§3-10-3" not in outcome.text


def test_fabricated_section_fires_on_multiple_invented_anchors() -> None:
    """Multi-offender shape uses the plural nudge variant."""
    outcome = FabricatedSectionHook(
        valid_anchors=frozenset({"§3-10-3"}),
    ).check(_ctx_with_reply("Per JO 7110.65 §3-99-3 and §99-1-1, both procedures apply."))
    assert isinstance(outcome, Nudge)
    assert "§3-99-3" in outcome.text
    assert "§99-1-1" in outcome.text
    # Plural form names "none of these".
    assert "none of these" in outcome.text


def test_fabricated_section_accepts_parent_section_when_only_paragraph_in_set() -> None:
    """Valid-anchor set is built with §N-N parents when §N-N-N is
    present (see section_index.collect_valid_anchors). A reply citing
    the parent without paragraph passes."""
    outcome = FabricatedSectionHook(
        valid_anchors=frozenset({"§3-10-3", "§3-10"}),
    ).check(_ctx_with_reply("Per JO 7110.65 §3-10, same-runway separation applies."))
    assert isinstance(outcome, Continue)


def test_fabricated_section_skips_tbl_and_fig_citations() -> None:
    """TBL/FIG citations need a different index (table-level enumeration)
    so we deliberately let them pass even if they look fabricated to a
    section-only set."""
    outcome = FabricatedSectionHook(
        valid_anchors=frozenset({"§4-1-1"}),
    ).check(
        _ctx_with_reply("Per JO 7110.65 TBL 4-1-2, an MH class RBN has 25 mile usable distance.")
    )
    assert isinstance(outcome, Continue)


def test_fabricated_section_normalises_dash_variants() -> None:
    """Corpus uses unicode-minus and en-dash variants. The model may
    emit either. Canonical form (used in valid_anchors) has ASCII
    hyphens, so a cite using a non-ASCII dash should still match."""
    minus = "−"  # noqa: RUF001 — unicode-minus dash variant is the test premise
    cited = f"§3{minus}10{minus}3"
    outcome = FabricatedSectionHook(
        valid_anchors=frozenset({"§3-10-3"}),
    ).check(_ctx_with_reply(f"Per JO 7110.65 {cited}, same-runway separation."))
    assert isinstance(outcome, Continue)


def test_default_pipeline_threads_anchors_into_fabricated_section_hook() -> None:
    """End-to-end: build the shipping pipeline with a non-empty anchor
    set, run it against a reply that cites a fabricated section, and
    confirm the bail outcome is a Nudge attributed to fabricated_section."""
    pipeline = default_hook_pipeline(
        valid_section_anchors=frozenset({"§3-10-3"}),
    )
    outcome = pipeline.run_bail(
        BailContext(
            reply=_reply(
                "Per JO 7110.65 §99-99-99, the procedure applies to all controlled airports."
            ),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        ),
        disabled=frozenset(),
    )
    assert isinstance(outcome, Nudge)
    assert outcome.catcher == "fabricated_section"


def test_default_pipeline_silent_on_empty_anchors_for_invented_cite() -> None:
    """Default construction (no anchors) → FabricatedSectionHook stays
    quiet. A reply with an invented §-anchor and a real cite passes
    bail; the invented-cite case is then covered by finalize hooks
    (UngroundedCitation / LowConfidenceFallback) on a per-tool basis."""
    pipeline = default_hook_pipeline()
    outcome = pipeline.run_bail(
        BailContext(
            reply=_reply(
                "Per JO 7110.65 §3-10-3, same-runway separation applies "
                "to arrivals — see also §99-99-99 for unrelated cases."
            ),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        ),
        disabled=frozenset(),
    )
    # No fabricated_section nudge; ListCountMismatch / others won't
    # match either. Continue.
    assert isinstance(outcome, Continue)
