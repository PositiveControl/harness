"""Unit tests for the orchestrator hook pipeline.

These exercise each catcher in isolation — independently of the tool
loop — so a failure points at the hook, not the surrounding plumbing.
The end-to-end behavior is covered by `tests/test_tool_loop.py` and the
attribution eval (`tests/test_tool_loop_eval.py`).
"""

from __future__ import annotations

from harness.orchestrator.hooks import (
    TABLE_FABRICATION_FALLBACK,
    UNGROUNDED_CITATION_FALLBACK,
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
    Skip,
    TableFabricationHook,
    TeaserHook,
    ToolIntentHook,
    Truncated,
    TruncatedHook,
    UngroundedCitationHook,
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
        "missing_citation",
        "list_count_mismatch",
        "reserved_squawk_code",
        "paired_meta_confirm_strip",
        "duplicate_call",
        "argument_grounding",
        "ungrounded_citation",
        "table_fabrication",
        "numeric_fabrication",
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
    reply = _reply(
        "The MH class RBN has a usable distance of 25 miles, "
        "per JO 7110.65 §4-1-1."
    )
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
    reply = _reply(
        "| Class | Power | Distance |\n"
        "|---|---|---|\n"
        "| MH | Under 50 | 25 |\n"
    )
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
        "| Class | Power | Distance |\n"
        "|---|---|---|\n"
        "| H | 50 − 1,999 | 50 |\n"  # noqa: RUF001 — unicode-minus
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
    reply = _reply(
        "The usable distance for an MH class RBN is 25 miles, per "
        "JO 7110.65 §4-1-1."
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


def test_numeric_fabrication_silent_without_grounding_tool() -> None:
    """No grounding tool ran → defer to ungrounded_citation, don't
    fire."""
    reply = _reply(
        "The usable distance for an MH class RBN is 50 miles."
    )
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
    tool_out = (
        "  |**Class**|**Description**|\n|---|---|\n|MH|Medium power|\n"
    )
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
    reply = _reply(
        "The usable distance for an MH class RBN is 50 miles, per "
        "JO 7110.65 §4-1-1."
    )
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
    disabled = pipe.run_finalize(
        ctx, disabled=frozenset({"numeric_fabrication"})
    )
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
    outcome = MissingCitationHook().check(
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
    outcome = MissingCitationHook().check(
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
    outcome = MissingCitationHook().check(
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
        outcome = MissingCitationHook().check(
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
    outcome = MissingCitationHook().check(
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
    outcome = MissingCitationHook().check(
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
    outcome = MissingCitationHook().check(
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
    outcome = MissingCitationHook().check(
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
    outcome = MissingCitationHook().check(
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


def test_missing_citation_respects_disabled_toggle() -> None:
    """Attribution eval disables catchers by name. Pipeline must honor
    'missing_citation' in the disabled frozenset."""
    pipe = default_hook_pipeline()
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
    reply_text = (
        "The four main reasons are:\n\n"
        "- alpha\n"
        "- beta\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_memory"}),
        )
    )
    assert isinstance(outcome, Nudge)


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
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx("Per §7-6-11 the line is _squawk 7700_.")
    )
    assert isinstance(outcome, Nudge)


def test_reserved_squawk_fires_on_bold_phraseology() -> None:
    """Markdown bold **squawk 7500**."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx(
            "The reply should quote **squawk 7500** verbatim (as an error)."
        )
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
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx('Quoted: "squawk seven five zero zero"')
    )
    assert isinstance(outcome, Nudge)


def test_reserved_squawk_passes_on_prose_description() -> None:
    """Narrative 'when you observe Code 7500' — not an assignment.
    Must pass; this is §5-2-5 describing controller procedure."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx(
            "When you observe a Code 7500 display, apply the procedures "
            "in §10-2-6."
        )
    )
    assert isinstance(outcome, Continue)


def test_reserved_squawk_passes_on_warning_phrasing() -> None:
    """'Remember that squawk 7500 is the hijack code' — warning, not
    assignment. No quote/italic/bold/caps wrapping. Pass."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx(
            "Remember that squawk 7500 is the hijack code — never assign it."
        )
    )
    assert isinstance(outcome, Continue)


def test_reserved_squawk_passes_on_correct_1200_assignment() -> None:
    """Correct VFR code in assignment context must pass."""
    outcome = ReservedSquawkCodeHook().check(
        _bail_ctx(
            'The corrected phraseology: "Radar service terminated, '
            'squawk one two zero zero."'
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
            'Please resquawk your code once identified. For hijack, the '
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
        'Correct the following radar phraseology: '
        '"Services stopped, squawk seventy five hundred"'
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
        'Correct the following radar phraseology: '
        '"Services stopped, squawk seventy five hundred"'
    )
    # Model modified the code form from the user's input — propagation,
    # not echo.
    reply_text = (
        'The corrected phraseology: "Radar service terminated, '
        'squawk seven five hundred."'
    )
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
    pipe = default_hook_pipeline()
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
