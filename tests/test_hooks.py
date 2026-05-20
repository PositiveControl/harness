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
    ConfidentFactualClaimHook,
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
    IncompleteMultipartHook,
    ListCountMismatchHook,
    MetaConfirmHook,
    MissingCitationHook,
    Nudge,
    NumericFabricationHook,
    OpinionWithoutTriggerHook,
    PairedMetaConfirmStripHook,
    PersistBodyCitationsHook,
    PostModelContext,
    PostResearchPersistHook,
    PostSearchGroundingHook,
    PreToolContext,
    RawResultsDumpHook,
    Replace,
    ReservedSquawkCodeHook,
    ScopeRedirectHook,
    ScopeViolationHook,
    SelfContradictingRankHook,
    Skip,
    SourceCountInflationHook,
    TableFabricationHook,
    TeaserHook,
    ThinSourceFabricationHook,
    ToolIntentHook,
    ToolSearchLoopHook,
    Truncated,
    TruncatedHook,
    UncitedSubstantiveReplyHook,
    UngroundedCitationHook,
    UnparseableHook,
    WriteFileRedirectHook,
    default_hook_pipeline,
)
from harness.tools.base import ModelReply, ToolCall, ToolResult

_STUB_RESULT = ToolResult(tool_name="stub", output="", success=True)

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
    # After a CONTENT tool has run, completion claims are legitimate
    # wrap-ups (harness-q7kn: meta-tools alone don't legitimize).
    assert isinstance(
        FalseSuccessHook().check(
            BailContext(
                reply=reply,
                tools_ran_this_turn=True,
                tools_ran=frozenset({"write_file"}),
            )
        ),
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


# ---------- passive-voice web-claim coverage (harness-bylm) ----------


def test_fabricated_search_hook_fires_on_passive_attribution_no_tool() -> None:
    """Session repro 2026-05-19 (avocado query, 2nd session): model
    emitted 'This information is from a web search' on round 2 — before
    any web tool had run (only tool_search, a meta-tool, executed).
    The original active-voice-only regex missed it. Passive-voice /
    attribution forms must trip the catcher just like active-voice
    'I searched the web' does."""
    ctx = BailContext(
        reply=_reply(
            "I found that the top South American avocado exporter is Chile. "
            "This information is from a web search."
        ),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"tool_search"}),  # meta-only — _content_tools_ran=False
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Nudge)


def test_fabricated_search_hook_fires_on_based_on_web_search() -> None:
    """'Based on a web search ...' — preposition-led attribution, same
    fabrication shape. Must fire when no web tool ran."""
    ctx = BailContext(
        reply=_reply("Based on a recent web search, the answer is Chile."),
        tools_ran_this_turn=False,
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Nudge)


def test_fabricated_search_hook_fires_on_according_to_search_results() -> None:
    """'According to search results' — another attribution form claiming
    a search happened. Must fire when no web tool ran."""
    ctx = BailContext(
        reply=_reply("According to search results, Chile is the top exporter."),
        tools_ran_this_turn=False,
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Nudge)


def test_fabricated_search_hook_fires_on_the_web_search_indicates() -> None:
    """Reporting-verb attribution: 'the web search indicates/showed/
    returned/tells/reveals/says'. Same fabrication shape — model
    treating an imagined search as the source."""
    for verb in ("indicates", "showed", "returned", "reveals", "says"):
        ctx = BailContext(
            reply=_reply(f"The web search {verb} that Chile is the top exporter."),
            tools_ran_this_turn=False,
        )
        assert isinstance(FabricatedSearchHook().check(ctx), Nudge), (
            f"web-search {verb!r} attribution should fire when no web tool ran"
        )


def test_fabricated_search_hook_silent_on_passive_attribution_after_web_tool() -> None:
    """The same passive attribution AFTER a real web tool ran is
    legitimate wrap-up narration — the catcher must NOT fire."""
    for phrase in (
        "This information is from a web search.",
        "Based on a web search, the answer is Chile.",
        "According to search results, Chile leads.",
        "The web search indicates Chile is top.",
    ):
        ctx = BailContext(
            reply=_reply(phrase),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_web"}),
        )
        assert isinstance(FabricatedSearchHook().check(ctx), Continue), (
            f"phrase {phrase!r} should be silent after real search_web"
        )


def test_fabricated_search_hook_silent_on_unrelated_uses_of_from() -> None:
    """The 'from' alternative must not over-fire on benign phrases that
    happen to start with 'from' — only the specific 'from <a/the/my>
    web-search-flavored noun phrase' shape qualifies."""
    benign_phrases = [
        "From the data you provided, the answer is X.",
        "From my reading of the article, Chile is third.",
        "I learned this from the article you shared.",
    ]
    for phrase in benign_phrases:
        ctx = BailContext(reply=_reply(phrase), tools_ran_this_turn=False)
        assert isinstance(FabricatedSearchHook().check(ctx), Continue), (
            f"benign phrase {phrase!r} must not trip the web-claim branch"
        )


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


def test_meta_confirm_catches_post_search_read_question() -> None:
    """Mark's repro: agent finished search_web, listed 5 URLs, asked
    'Would you like to read the full details from one of these
    sources?' instead of calling fetch_url. The original META_CONFIRM_RE
    had a fixed verb list (add/proceed/edit/...) that excluded
    read/see/fetch — adding those caught this case."""
    ctx = BailContext(
        reply=_reply("Would you like to read the full details from one of these sources?"),
        tools_ran_this_turn=False,
    )
    assert isinstance(MetaConfirmHook().check(ctx), Nudge)


def test_meta_confirm_catches_noun_phrase_follow_up() -> None:
    """Same repro shape, different post-action question:
    'Would you like more precision or a different format?'. No verb
    in the original pattern; the new 'would you like + noun phrase'
    alternation handles it."""
    ctx = BailContext(
        reply=_reply("Would you like more precision or a different format?"),
        tools_ran_this_turn=False,
    )
    assert isinstance(MetaConfirmHook().check(ctx), Nudge)


def test_meta_confirm_catches_i_recommend_punt() -> None:
    """'I recommend checking one of these sources' is the polite
    version of meta-confirm: agent has data but kicks the work back
    to the user."""
    ctx = BailContext(
        reply=_reply("I recommend checking the most recent forecast from one of these sources."),
        tools_ran_this_turn=False,
    )
    assert isinstance(MetaConfirmHook().check(ctx), Nudge)


def test_meta_confirm_does_not_fire_on_substantive_read_mention() -> None:
    """Negative: 'I read the file and found...' is a legitimate report,
    not a meta-confirm. The 'read' verb only matters inside the
    'would you like (me )? to read' shape."""
    ctx = BailContext(
        reply=_reply("I read your config and found three issues to address."),
        tools_ran_this_turn=False,
    )
    assert isinstance(MetaConfirmHook().check(ctx), Continue)


# ---------- raw_results_dump (harness-s451) ----------

# Shared payload — a 5-item news list. Used as both the tool output
# and (verbatim) as the model's reply on the failure-path tests. Keeps
# the overlap clearly above the Jaccard threshold.
_RAW_DUMP_NEWS = (
    "1. Wildfire spreads in Northern California, mass evacuations ordered\n"
    "2. Earthquake magnitude 6.4 hits Tokyo region\n"
    "3. Brazil election results announced\n"
    "4. Stock market closes at new high\n"
    "5. Hurricane forming in Atlantic, watch issued\n"
)


def test_raw_results_dump_hook_fires_on_synthesis_prompt_with_verbatim_dump() -> None:
    """Positive case: search_web ran, the user asked to 'prioritize',
    and the reply is the tool output near-verbatim with no ordering
    markers imposed. The catcher fires and nudges."""
    ctx = BailContext(
        reply=_reply(_RAW_DUMP_NEWS),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="search for breaking news and prioritize by location and severity",
        prior_tool_outputs=(_RAW_DUMP_NEWS,),
    )
    outcome = RawResultsDumpHook().check(ctx)
    assert isinstance(outcome, Nudge)
    assert "synthesis" in outcome.text.lower()


def test_raw_results_dump_hook_silent_without_synthesis_verb() -> None:
    """User asked for a plain search. Even though the reply repeats the
    tool output verbatim, no synthesis verb means stopping is the
    correct exit — catcher stays silent."""
    ctx = BailContext(
        reply=_reply(_RAW_DUMP_NEWS),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="search the web for the latest news headlines",
        prior_tool_outputs=(_RAW_DUMP_NEWS,),
    )
    assert isinstance(RawResultsDumpHook().check(ctx), Continue)


def test_raw_results_dump_hook_silent_when_no_content_tool_ran() -> None:
    """No content tool succeeded — the no-tool case is fabricated_search
    / false_success / etc.'s job. RawResultsDump only targets
    real-but-unsynthesized output, so it disarms when no tool ran."""
    ctx = BailContext(
        reply=_reply(_RAW_DUMP_NEWS),
        tools_ran_this_turn=False,
        tools_ran=frozenset(),
        user_message="rank these news items by severity",
        prior_tool_outputs=(),
    )
    assert isinstance(RawResultsDumpHook().check(ctx), Continue)


def test_raw_results_dump_hook_silent_when_meta_tools_only() -> None:
    """Meta-tools (tool_search, load_tool) are plumbing, not content
    producers. Their presence in tools_ran must NOT disarm this catcher
    — _content_tools_ran is False, so the no-tool gate keeps it
    silent and the fabrication catchers handle the case."""
    ctx = BailContext(
        reply=_reply(_RAW_DUMP_NEWS),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"tool_search", "load_tool"}),
        user_message="rank news items by severity",
        prior_tool_outputs=(_RAW_DUMP_NEWS,),
    )
    assert isinstance(RawResultsDumpHook().check(ctx), Continue)


def test_raw_results_dump_hook_silent_with_ranked_by_marker() -> None:
    """Reply imposes structure via 'Ranked by severity:' — that's the
    synthesis we wanted, so the catcher must stay silent even though
    overlap with the tool data is still high (item titles reused)."""
    reply = (
        "Ranked by severity:\n\n"
        "1. High — Earthquake in Tokyo region: immediate safety risk.\n"
        "2. High — Wildfire in Northern California: active evacuations.\n"
        "3. Low — Stock market new high: economic, no urgency.\n"
    )
    ctx = BailContext(
        reply=_reply(reply),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="rank the latest news items by severity",
        prior_tool_outputs=(_RAW_DUMP_NEWS,),
    )
    assert isinstance(RawResultsDumpHook().check(ctx), Continue)


def test_raw_results_dump_hook_silent_with_severity_heading_marker() -> None:
    """'High severity:' / 'Medium severity:' grouping headings disarm
    the catcher — explicit grouping IS the synthesis."""
    reply = (
        "High severity:\n"
        "- Tokyo earthquake — immediate safety risk\n"
        "- California wildfire — evacuations underway\n\n"
        "Medium severity:\n"
        "- Brazil election results\n\n"
        "Low severity:\n"
        "- Stock market new high\n"
    )
    ctx = BailContext(
        reply=_reply(reply),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="prioritize by severity",
        prior_tool_outputs=(_RAW_DUMP_NEWS,),
    )
    assert isinstance(RawResultsDumpHook().check(ctx), Continue)


def test_raw_results_dump_hook_silent_on_low_overlap_reply() -> None:
    """Reply doesn't repeat the tool output — the model wrote
    something substantively different. Overlap below threshold means
    no dump, no catcher fire (even with synthesis verb in prompt)."""
    ctx = BailContext(
        reply=_reply(
            "Based on what came back: I can't tell which item is most "
            "critical from the headlines alone. Want me to fetch the "
            "Tokyo earthquake article for more detail?"
        ),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="rank these news items by severity",
        prior_tool_outputs=(_RAW_DUMP_NEWS,),
    )
    assert isinstance(RawResultsDumpHook().check(ctx), Continue)


def test_raw_results_dump_hook_silent_without_prior_tool_outputs() -> None:
    """tools_ran reports a content tool succeeded, but prior_tool_outputs
    is empty — degenerate state we don't fire on (the comparison anchor
    is missing)."""
    ctx = BailContext(
        reply=_reply(_RAW_DUMP_NEWS),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="rank these items",
        prior_tool_outputs=(),
    )
    assert isinstance(RawResultsDumpHook().check(ctx), Continue)


def test_raw_results_dump_hook_silent_without_user_message() -> None:
    """No user message captured — we can't check for synthesis verbs,
    so we don't fire. (System-only bootstrap doesn't trip this catcher.)"""
    ctx = BailContext(
        reply=_reply(_RAW_DUMP_NEWS),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message=None,
        prior_tool_outputs=(_RAW_DUMP_NEWS,),
    )
    assert isinstance(RawResultsDumpHook().check(ctx), Continue)


def test_raw_results_dump_hook_fires_on_compare_verb_variant() -> None:
    """The verb regex covers inflections — 'compare' / 'comparing' /
    'compared' all arm the catcher equivalently."""
    ctx = BailContext(
        reply=_reply(_RAW_DUMP_NEWS),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="comparing the latest news headlines side by side",
        prior_tool_outputs=(_RAW_DUMP_NEWS,),
    )
    assert isinstance(RawResultsDumpHook().check(ctx), Nudge)


# ---------- thin_source_fabrication (harness-xszc) ----------

# Approximation of what fetch_url returns when pointed at a JS-rendered
# SPA (weather.com Nairobi repro 2026-05-19). Extract-text strips most
# scripts, leaving navigation + footer + a notice that JS is required.
# Zero numeric-with-unit tokens — the body can't support specific
# numeric claims.
_THIN_WEATHER_BODY = (
    "10-Day Weather Forecast for Nairobi, Kenya - The Weather Channel\n\n"
    "Sign In | Skip to navigation. JavaScript required to view this page. "
    "Privacy | Terms | About | Contact. Today's forecast | Hourly | "
    "Tomorrow | Weekend. Allergy | Pollen | Air quality. Maps | Radar | "
    "Satellite. © 2026 Weather Group, LLC."
)

# Repro shape: model emits a 4-day forecast off the thin body.
_THIN_WEATHER_REPLY = (
    "The 4-day forecast for Nairobi, Kenya:\n"
    "Today: high 85°F, low 65°F, humidity 65%\n"
    "Tomorrow: high 87°F, low 66°F, humidity 68%\n"
    "Day 3: high 88°F, low 67°F, humidity 70%\n"
    "Day 4: high 89°F, low 68°F, humidity 72%\n"
)


def test_thin_source_fabrication_hook_fires_on_weather_repro() -> None:
    """The motivating repro: fetch_url returned a JS-rendered SPA body
    with no forecast data, model fabricated four days of temps +
    humidity. Thin body (0 numeric tokens) + numeric reply (12 tokens)
    must Nudge."""
    ctx = BailContext(
        reply=_reply(_THIN_WEATHER_REPLY),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web", "fetch_url"}),
        user_message="4-day weather forecast for Nairobi, Kenya",
        prior_tool_outputs=(_THIN_WEATHER_BODY,),
    )
    outcome = ThinSourceFabricationHook().check(ctx)
    assert isinstance(outcome, Nudge)
    assert "thin" in outcome.text.lower() or "fabrication" in outcome.text.lower()


def test_thin_source_fabrication_hook_silent_when_body_has_numbers() -> None:
    """Body carries real data — defer to numeric_fabrication /
    table_fabrication for the cross-row drift case. This catcher
    targets the no-anchor fabrication path, not the wrong-anchor
    path."""
    rich_body = (
        "Nairobi, Kenya — 4-day forecast:\n"
        "Day 1: high 75°F low 55°F humidity 60%\n"
        "Day 2: high 76°F low 56°F humidity 62%\n"
        "Day 3: high 77°F low 57°F humidity 63%\n"
        "Day 4: high 78°F low 58°F humidity 64%\n"
    )
    ctx = BailContext(
        reply=_reply(_THIN_WEATHER_REPLY),  # reply doesn't match, but
        # this catcher's job isn't to check that. It only filters out
        # the no-anchor case so other catchers can do their work.
        tools_ran_this_turn=True,
        tools_ran=frozenset({"fetch_url"}),
        user_message="4-day forecast for Nairobi",
        prior_tool_outputs=(rich_body,),
    )
    assert isinstance(ThinSourceFabricationHook().check(ctx), Continue)


def test_thin_source_fabrication_hook_silent_when_reply_has_no_numeric_claims() -> None:
    """Reply doesn't make specific numeric claims — fabrication isn't
    the failure shape. A polite 'I couldn't extract the forecast'
    reply over a thin body must NOT trigger."""
    ctx = BailContext(
        reply=_reply(
            "I couldn't extract a forecast from that page. The Weather "
            "Channel site needs JavaScript. Try weather.gov for US "
            "locations or wttr.in for global coverage."
        ),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"fetch_url"}),
        user_message="4-day forecast for Nairobi",
        prior_tool_outputs=(_THIN_WEATHER_BODY,),
    )
    assert isinstance(ThinSourceFabricationHook().check(ctx), Continue)


def test_thin_source_fabrication_hook_silent_without_prior_tool_outputs() -> None:
    """No tool output to compare against — the no-tool fabrication
    case is fabricated_search / false_success / teaser territory."""
    ctx = BailContext(
        reply=_reply(_THIN_WEATHER_REPLY),
        tools_ran_this_turn=False,
        tools_ran=frozenset(),
        user_message="4-day forecast",
        prior_tool_outputs=(),
    )
    assert isinstance(ThinSourceFabricationHook().check(ctx), Continue)


def test_thin_source_fabrication_hook_silent_on_paraphrase_reply() -> None:
    """Reply paraphrases the body without inventing specific numbers —
    legitimate summarization of a thin or prose-only source."""
    prose_body = (
        "Nairobi has a subtropical highland climate due to its elevation. "
        "The city experiences mild temperatures year-round with cool nights. "
        "Rainfall peaks during the long rains in March-May and short rains "
        "in October-November."
    )
    paraphrase_reply = (
        "Nairobi has a mild climate year-round thanks to its elevation. "
        "There are two rainy seasons — long rains in spring and short rains "
        "in autumn. Nights tend to be cool."
    )
    ctx = BailContext(
        reply=_reply(paraphrase_reply),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"fetch_url"}),
        user_message="what's the climate in Nairobi like",
        prior_tool_outputs=(prose_body,),
    )
    assert isinstance(ThinSourceFabricationHook().check(ctx), Continue)


def test_thin_source_fabrication_hook_silent_on_low_claim_count_reply() -> None:
    """Reply mentions only one or two numbers — below the
    reply-floor. A single incidental claim isn't the fabrication
    shape this catcher targets."""
    ctx = BailContext(
        reply=_reply(
            "I see roughly 70°F mentioned on the page. The rest of the "
            "forecast didn't render — try a different source."
        ),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"fetch_url"}),
        user_message="4-day forecast for Nairobi",
        prior_tool_outputs=(_THIN_WEATHER_BODY,),
    )
    assert isinstance(ThinSourceFabricationHook().check(ctx), Continue)


def test_thin_source_fabrication_hook_numeric_regex_coverage() -> None:
    """The numeric-claim regex must cover the common units that get
    fabricated: temperatures (°F/°C), percentages, speeds (mph/knots),
    distances (mi/km/ft), pressure (hPa/inHg), currency. Check via
    the helper used by the catcher."""
    from harness.orchestrator.hooks import _thin_source_numeric_count

    samples = [
        "high 85°F low 65°F humidity 65%",  # temps + percent
        "wind 12 mph gusting 25 mph",  # mph
        "visibility 5 miles ceiling 2000 ft",  # miles + ft
        "pressure 1013 hPa or 29.91 inHg",  # pressure
        "price $19.99 or €17.50",  # currency
    ]
    for sample in samples:
        assert _thin_source_numeric_count(sample) >= 2, (
            f"numeric regex missed tokens in: {sample!r}"
        )


def test_thin_source_fabrication_nudge_names_fallback_tools() -> None:
    """harness-qzxq upgrade: the nudge must teach what-to-do-instead,
    not just what-not-to-do. Concretely it must name search_web AND
    fetch_url AND a non-JS-rendered alternative so the model has a
    clear recovery path. Without this, the bail converts fabrication
    into refusal — the user wanted the data, not an apology."""
    ctx = BailContext(
        reply=_reply(_THIN_WEATHER_REPLY),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"fetch_url"}),
        user_message="4-day weather forecast for Nairobi, Kenya",
        prior_tool_outputs=(_THIN_WEATHER_BODY,),
    )
    outcome = ThinSourceFabricationHook().check(ctx)
    assert isinstance(outcome, Nudge)
    nudge = outcome.text.lower()
    # Names BOTH data-side tools.
    assert "search_web" in nudge, "nudge must name search_web as a recovery option"
    assert "fetch_url" in nudge, "nudge must name fetch_url as a recovery option"
    # Calls out the non-JS-rendered alternative pattern.
    assert "non-js" in nudge or "api endpoint" in nudge or "wttr.in" in nudge, (
        "nudge must point at non-JS / API alternatives so the agent "
        "knows the SPA-page failure has a known workaround"
    )


# ---------- incomplete_multipart (harness-111v) ----------

# Repro 2026-05-19: user asked TWO things in one prompt. Agent
# searched, found the population, then gave up on the gender ratio
# instead of issuing another search_web. The reply explicitly
# acknowledged it couldn't fulfill the second sub-ask.
_NAIROBI_MULTIPART_USER = (
    "Find the population count of Nairobi, Kenya. What percent are female vs male?"
)
_NAIROBI_GIVEUP_REPLY = (
    "The population of Nairobi, Kenya, is approximately 5,545,000 as of "
    "2026. According to the source, the gender breakdown is not "
    "explicitly provided. However, the source does not mention any "
    "specific data on the percentage of males and females. For a precise "
    "breakdown, we would need to look at a more detailed demographic "
    "report or a specific source that provides gender statistics."
)


def test_incomplete_multipart_hook_fires_on_nairobi_repro() -> None:
    """The motivating repro. Multi-part user prompt + content tool ran +
    reply ends with explicit give-up phrasing must Nudge."""
    ctx = BailContext(
        reply=_reply(_NAIROBI_GIVEUP_REPLY),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web", "fetch_url"}),
        user_message=_NAIROBI_MULTIPART_USER,
        prior_tool_outputs=("Nairobi Population 2026 — 5,545,000 People...",),
    )
    outcome = IncompleteMultipartHook().check(ctx)
    assert isinstance(outcome, Nudge)
    assert "multi" in outcome.text.lower() or "another tool call" in outcome.text.lower()


def test_incomplete_multipart_hook_silent_on_single_ask_refusal() -> None:
    """Single-part prompt that the model legitimately can't answer:
    one ask, give-up phrasing — the hook must stay silent so the
    refusal passes through. Not the failure mode this catcher targets."""
    ctx = BailContext(
        reply=_reply(
            "The source does not mention any gender breakdown for Nairobi. "
            "We would need a more detailed demographic report."
        ),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="What percent of Nairobi residents are female?",
        prior_tool_outputs=("Nairobi general info — no demographics tables.",),
    )
    assert isinstance(IncompleteMultipartHook().check(ctx), Continue)


def test_incomplete_multipart_hook_silent_when_reply_answers_both() -> None:
    """Multi-part prompt + content tool ran + reply answers everything
    without give-up phrasing. Nothing to nudge about."""
    ctx = BailContext(
        reply=_reply(
            "Population: 5,545,000. Gender split per the 2019 KNBS census: "
            "approximately 50.5% female and 49.5% male."
        ),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web", "fetch_url"}),
        user_message=_NAIROBI_MULTIPART_USER,
        prior_tool_outputs=("KNBS Nairobi census: pop 5,545,000; F 50.5% M 49.5%",),
    )
    assert isinstance(IncompleteMultipartHook().check(ctx), Continue)


def test_incomplete_multipart_hook_silent_on_meta_tool_only() -> None:
    """Meta-tools (tool_search / load_tool / introspect / spawn_subagent)
    don't count as 'a content tool ran' — defer to the fabricated_*
    family which targets the no-real-tool case."""
    ctx = BailContext(
        reply=_reply(_NAIROBI_GIVEUP_REPLY),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"tool_search", "load_tool"}),
        user_message=_NAIROBI_MULTIPART_USER,
        prior_tool_outputs=("tool_search results...",),
    )
    assert isinstance(IncompleteMultipartHook().check(ctx), Continue)


def test_incomplete_multipart_hook_silent_without_tools() -> None:
    """No tools ran at all → fabricated_search / teaser / false_success
    territory. This catcher only nudges when a search was attempted but
    the model bailed early on a sub-ask."""
    ctx = BailContext(
        reply=_reply(_NAIROBI_GIVEUP_REPLY),
        tools_ran_this_turn=False,
        tools_ran=frozenset(),
        user_message=_NAIROBI_MULTIPART_USER,
        prior_tool_outputs=(),
    )
    assert isinstance(IncompleteMultipartHook().check(ctx), Continue)


def test_incomplete_multipart_hook_silent_on_empty_user_message() -> None:
    """Guard: empty user_message can't be multi-part. Belt-and-braces
    in case the context arrives with no prompt (continuation turns,
    forced-grounding preludes)."""
    ctx = BailContext(
        reply=_reply(_NAIROBI_GIVEUP_REPLY),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="",
        prior_tool_outputs=("anything",),
    )
    assert isinstance(IncompleteMultipartHook().check(ctx), Continue)


def test_incomplete_multipart_hook_multi_ask_regex_coverage() -> None:
    """The ask-counter must recognize the common multi-ask shapes
    the user sends. Verified via the helper used by the catcher."""
    from harness.orchestrator.hooks import _multipart_ask_count

    samples = [
        ("Find X. What is Y?", 2),
        ("What's X? What's Y?", 2),
        ("Find X and what is Y?", 2),
        ("List X, then show Y.", 2),
        ("Search X. Also tell me Y.", 2),
        ("Calculate X and compare Y.", 2),
    ]
    for prompt, minimum in samples:
        actual = _multipart_ask_count(prompt)
        assert actual >= minimum, (
            f"ask-counter missed in {prompt!r}: got {actual}, expected >= {minimum}"
        )


def test_incomplete_multipart_hook_single_ask_under_threshold() -> None:
    """Single-ask prompts must count as 1 (or less) so the catcher
    stays silent on legitimate one-part questions even with give-up
    phrasing. Targets the false-positive guard."""
    from harness.orchestrator.hooks import _multipart_ask_count

    for prompt in (
        "Find the population of Nairobi.",
        "What is the gender ratio in Nairobi?",
        "Tell me about Nairobi's climate.",
        "Compare the climate in Nairobi to that in Cairo.",  # single 'compare' verb
    ):
        assert _multipart_ask_count(prompt) < 2, f"ask-counter overfired on single-ask {prompt!r}"


# ---------- confident_factual_claim (harness-wpo0) ----------

# Session repro 2026-05-19: user asked 'what is the national bird of
# kenya?'. Model called tool_search seven times (all meta), zero
# content tools, then emitted the bare 'X is Y' claim. Factually wrong.


def test_confident_factual_claim_fires_on_kenya_ostrich_repro() -> None:
    """The motivating repro shape, capitalized. User asks a question
    about a named entity; only meta-tools ran; reply makes a bare
    confident claim with no hedge. The catcher must Nudge."""
    ctx = BailContext(
        reply=_reply("The national bird of Kenya is the ostrich."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"tool_search"}),
        user_message="What is the national bird of Kenya?",
    )
    outcome = ConfidentFactualClaimHook().check(ctx)
    assert isinstance(outcome, Nudge)
    assert "confident factual claim" in outcome.text.lower()
    # Recovery path must name BOTH content-tool options + the
    # hedge alternative.
    assert "search_web" in outcome.text
    assert "fetch_url" in outcome.text
    assert "hedge" in outcome.text.lower()


def test_confident_factual_claim_fires_on_nairobi_population_no_tool() -> None:
    """Different named entity, different copula shape, still bare
    confident claim with no content tool. Catcher must Nudge."""
    ctx = BailContext(
        reply=_reply("The population of Nairobi is approximately 5,000,000."),
        tools_ran_this_turn=False,  # no tool at all this turn
        tools_ran=frozenset(),
        user_message="What is the population of Nairobi?",
    )
    assert isinstance(ConfidentFactualClaimHook().check(ctx), Nudge)


def test_confident_factual_claim_silent_when_content_tool_ran() -> None:
    """A content tool (search_web / fetch_url / search_memory / …)
    legitimizes the claim — defer to thin_source_fabrication /
    raw_results_dump / numeric_fabrication / missing_citation
    instead."""
    ctx = BailContext(
        reply=_reply("The national bird of Kenya is the ostrich."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="What is the national bird of Kenya?",
    )
    assert isinstance(ConfidentFactualClaimHook().check(ctx), Continue)


def test_confident_factual_claim_silent_on_hedged_reply() -> None:
    """A reply that already hedges its claim ('I believe', 'I'm not
    sure', 'unofficially', 'according to my training') is signalling
    uncertainty correctly — nothing to nudge."""
    base_user = "What is the national bird of Kenya?"
    hedged_replies = [
        "I believe the national bird of Kenya is the ostrich.",
        "I think the national bird of Kenya is the ostrich.",
        "Unofficially, the national bird of Kenya is the lilac-breasted roller.",
        (
            "I'm not certain, but based on my training data, the national "
            "bird of Kenya is the lilac-breasted roller."
        ),
        "The national bird of Kenya might be the lilac-breasted roller.",
        (
            "I cannot verify this without a tool call, but the national "
            "bird of Kenya is the lilac-breasted roller."
        ),
        "The national bird of Kenya is allegedly the lilac-breasted roller.",
        "To my knowledge, the national bird of Kenya is the lilac-breasted roller.",
    ]
    for reply in hedged_replies:
        ctx = BailContext(
            reply=_reply(reply),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"tool_search"}),
            user_message=base_user,
        )
        assert isinstance(ConfidentFactualClaimHook().check(ctx), Continue), (
            f"hedged reply tripped the catcher: {reply!r}"
        )


def test_confident_factual_claim_silent_on_generic_question() -> None:
    """No proper noun in the user message ('what is the capital of a
    country?') — the question is generic enough that a parametric
    answer is often fine. Catcher stays silent."""
    ctx = BailContext(
        reply=_reply("The capital of a country is its primary administrative city."),
        tools_ran_this_turn=False,
        tools_ran=frozenset(),
        user_message="What is the capital of a country?",
    )
    assert isinstance(ConfidentFactualClaimHook().check(ctx), Continue)


def test_confident_factual_claim_silent_on_non_question() -> None:
    """User message isn't a factual question ('please summarize what
    we did today', 'thanks for the help'). The catcher's question-
    shape gate must filter these out so chit-chat and meta turns
    don't trip the named-entity check."""
    ctx = BailContext(
        reply=_reply("The plan is the next step."),
        tools_ran_this_turn=False,
        tools_ran=frozenset(),
        user_message="Please summarize what we did with the Nairobi plan.",
    )
    assert isinstance(ConfidentFactualClaimHook().check(ctx), Continue)


def test_confident_factual_claim_silent_on_empty_user_message() -> None:
    """No user message — the catcher can't tell whether the reply is
    a factual claim about something the user asked about. Silent."""
    ctx = BailContext(
        reply=_reply("Kenya is in East Africa."),
        tools_ran_this_turn=False,
        tools_ran=frozenset(),
        user_message=None,
    )
    assert isinstance(ConfidentFactualClaimHook().check(ctx), Continue)


def test_confident_factual_claim_silent_on_reply_without_copula() -> None:
    """Reply has no copula assertion — it's a question, a refusal, or
    a deferral, not a claim. Catcher must stay silent."""
    refusals = [
        "I can't answer that without checking a source. Want me to search?",
        "What time frame did you have in mind for Kenya?",
        "Let me look that up for Kenya.",
    ]
    for reply in refusals:
        ctx = BailContext(
            reply=_reply(reply),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"tool_search"}),
            user_message="What is the national bird of Kenya?",
        )
        assert isinstance(ConfidentFactualClaimHook().check(ctx), Continue), (
            f"non-claim reply tripped the catcher: {reply!r}"
        )


def test_confident_factual_claim_has_named_entity_helper() -> None:
    """Direct exercise of `_has_named_entity` to pin the proper-noun
    heuristic. Capitalized 3+ letter tokens NOT at sentence start AND
    NOT in the stop-set must register; sentence-initial interrogatives
    and articles must not."""
    from harness.orchestrator.hooks import _has_named_entity

    # Positive: mid-sentence capitalized entity
    assert _has_named_entity("What is the national bird of Kenya?")
    assert _has_named_entity("What is the population of Nairobi?")
    assert _has_named_entity("Tell me about Mount Kilimanjaro.")
    assert _has_named_entity("Who is the president of France?")
    # Positive: entity appears both sentence-initially AND mid-sentence
    assert _has_named_entity("Kenya is a country. What is its capital, Kenya?")

    # Negative: lowercase entity tokens
    assert not _has_named_entity("what is the national bird of kenya?")
    # Negative: generic question with no proper noun
    assert not _has_named_entity("What is the capital of a country?")
    # Negative: only sentence-initial capitalized words (interrogatives,
    # articles, and other stop-set tokens)
    assert not _has_named_entity("What is the answer?")
    assert not _has_named_entity("Who is the leader?")
    # Negative: empty / whitespace
    assert not _has_named_entity("")
    assert not _has_named_entity("   ")


def test_confident_factual_claim_silent_on_meta_only_with_lowercase_entity() -> None:
    """Deliberate conservative gap: lowercase entity tokens in
    user_message don't trip the named-entity check. Documented in
    the hook docstring as a false-negative tradeoff to keep
    false-positives bounded. Pinned here so the behavior change is
    intentional, not accidental."""
    ctx = BailContext(
        reply=_reply("The national bird of Kenya is the ostrich."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"tool_search"}),
        user_message="what is the national bird of kenya?",
    )
    assert isinstance(ConfidentFactualClaimHook().check(ctx), Continue)


def test_fabricated_search_hook_fires_after_meta_tool_only() -> None:
    """harness-q7kn: load_tool / tool_search / introspect are meta-tools
    (plumbing for the discovery loop). They must NOT disarm the
    fabrication catcher — the model still hasn't produced any real
    content. Mark's repro: load_tool succeeded, the model emitted a
    fabricated numbered list of weather sites BEFORE search_web
    actually ran. The catcher has to fire on the fab list."""
    fab_reply = (
        "1. **Weather.com — <https://weather.com/weather/today/LNKN9999> — Nairobi**\n"
        "2. **AccuWeather — <https://www.accuweather.com/en/ke/nairobi/3489/cw> — Nairobi**\n"
        "3. **BBC Weather — <https://www.bbc.co.uk/weather/world/lnkn9999> — Nairobi**\n"
    )
    ctx = BailContext(
        reply=_reply(fab_reply),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"load_tool"}),
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Nudge)


def test_fabricated_search_hook_skipped_after_real_search_web() -> None:
    """The same numbered-URL-list pattern is legitimate wrap-up after
    a real search_web run — must NOT fire."""
    real_reply = (
        "1. AccuWeather — https://www.accuweather.com/...\n"
        "   Current: 65 F\n"
        "2. Weather.com — https://weather.com/...\n"
        "   Forecast: light rain\n"
    )
    ctx = BailContext(
        reply=_reply(real_reply),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Continue)


def test_fabricated_search_hook_skipped_after_meta_plus_real_tool() -> None:
    """Mixed turn — load_tool ran THEN search_web ran. The real
    content tool legitimizes wrap-up; meta-tool presence doesn't
    invalidate it."""
    real_reply = (
        "1. AccuWeather — https://accuweather.com/...\n2. Weather.com — https://weather.com/...\n"
    )
    ctx = BailContext(
        reply=_reply(real_reply),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"load_tool", "search_web"}),
    )
    assert isinstance(FabricatedSearchHook().check(ctx), Continue)


def test_fabricated_itemization_hook_fires_after_meta_tool_only() -> None:
    """Same q7kn principle for itemization-shaped fabrications:
    meta-tool execution is transparent to the gate."""
    ctx = BailContext(
        reply=_reply("Here are some of the top stories on weather:\n1. Storm in...\n2. Heat..."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"tool_search"}),
    )
    assert isinstance(FabricatedItemizationHook().check(ctx), Nudge)


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
        tools_ran=frozenset({"fetch_url"}),
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


# ---------- tool_intent indirect-phrasing coverage (harness-27zr) ----------


def test_tool_intent_hook_fires_on_will_need_to_search() -> None:
    """Session 2026-05-19 (gum-export query) trace: model emitted 'I
    will need to search the web' as a final text-only reply. The
    interposing 'need to' between 'will' and 'search' slipped past
    the original direct-adjacency regex; the new interposing-cluster
    branch must catch it."""
    ctx = BailContext(
        reply=_reply(
            "I found that no specific tool matched. To find the data, I will "
            "need to search the web. Let's proceed with that."
        ),
        tools_ran_this_turn=False,
    )
    assert isinstance(ToolIntentHook().check(ctx), Nudge)


def test_tool_intent_hook_fires_on_will_use_tool_phrasing() -> None:
    """Same session: 'I will use the `search_web` tool to gather this
    information.' Three-word window between 'use' and 'tool' so the
    new 'use ... tool' branch must land."""
    ctx = BailContext(
        reply=_reply("I will use the `search_web` tool to gather this information."),
        tools_ran_this_turn=False,
    )
    assert isinstance(ToolIntentHook().check(ctx), Nudge)


def test_tool_intent_hook_fires_on_lets_proceed_with_search() -> None:
    """Same session: 'Let's proceed with the search.' 'proceed' as a
    top-level alternative, with the required (with|to|by) preposition
    that distinguishes 'proceed' as a tool-intent idiom from generic
    use."""
    ctx = BailContext(
        reply=_reply("Let's proceed with the search."),
        tools_ran_this_turn=False,
    )
    assert isinstance(ToolIntentHook().check(ctx), Nudge)


def test_tool_intent_hook_silent_on_use_data_benign() -> None:
    """'I will use the data' is benign — no tool call intended. The
    'use ... tool' branch requires a tool-naming object ('tool',
    'function', 'command', 'utility', 'primitive') after 'use', so
    'use the data' / 'use my training' must NOT fire."""
    for phrase in (
        "I will use the data you provided to compute the answer.",
        "I'll use my training to estimate that.",
        "Let me use the table from the source.",
    ):
        ctx = BailContext(reply=_reply(phrase), tools_ran_this_turn=False)
        assert isinstance(ToolIntentHook().check(ctx), Continue), (
            f"benign 'use' phrase {phrase!r} must not trip tool_intent"
        )


def test_tool_intent_hook_silent_on_proceed_without_preposition() -> None:
    """'I will proceed' without the with/to/by preposition is too
    generic to be a tool-intent signal. The new branch requires the
    preposition to qualify."""
    ctx = BailContext(
        reply=_reply("I will proceed and tell you the answer directly."),
        tools_ran_this_turn=False,
    )
    assert isinstance(ToolIntentHook().check(ctx), Continue)


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
        "raw_results_dump",
        "fabricated_search",
        "fabricated_itemization",
        "thin_source_fabrication",
        "incomplete_multipart",
        "confident_factual_claim",
        "ab_fabrication",
        "tool_intent",
        "missing_citation",
        "fabricated_section",
        "list_count_mismatch",
        "self_contradicting_rank",
        "scope_violation",
        "reserved_squawk_code",
        "scope_redirect",
        "ambiguous_context",
        "paired_meta_confirm_strip",
        "duplicate_call",
        "tool_search_loop",
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


def test_duplicate_call_hook_reissues_prior_success() -> None:
    """A cross-round duplicate of a successful call re-issues the prior
    result with a duplicate prefix; success/error preserved."""
    call = ToolCall(name="list_dir", arguments={"path": "/workdir"})
    prior = ToolResult(
        tool_name="list_dir",
        output="entries:\n - README.md\n - src/",
        success=True,
    )
    seen = {("list_dir", '{"path": "/workdir"}'): prior}
    outcome = DuplicateCallHook().check(PreToolContext(call=call, seen_calls=seen))
    assert isinstance(outcome, Skip)
    assert outcome.result.success is True
    assert outcome.result.error is None
    assert "duplicate of an earlier call" in outcome.result.output
    # Prior output preserved verbatim (just prefixed).
    assert "entries:" in outcome.result.output
    assert "README.md" in outcome.result.output


def test_duplicate_call_hook_preserves_prior_failure() -> None:
    """Duplicate of a failed call: re-issues the failure (NOT success=True).
    See harness-v5w — feeding success=True for a duplicate of a failed
    call made the model paraphrase 'duplicate' as 'captured/done'."""
    call = ToolCall(name="capture", arguments={"title": "x"})
    prior_failure = ToolResult(
        tool_name="capture",
        output="bd command failed: --add-label vs --labels",
        success=False,
        error="bd_command_failed",
    )
    seen = {("capture", '{"title": "x"}'): prior_failure}
    outcome = DuplicateCallHook().check(PreToolContext(call=call, seen_calls=seen))
    assert isinstance(outcome, Skip)
    # Failure is preserved — the model now sees a tool message that
    # cannot be paraphrased as success.
    assert outcome.result.success is False
    assert outcome.result.error == "bd_command_failed"
    # The failure body is intact so the model can read what went wrong.
    assert "bd command failed" in outcome.result.output
    assert "duplicate of an earlier call" in outcome.result.output


def test_duplicate_call_hook_passes_first_time() -> None:
    call = ToolCall(name="list_dir", arguments={"path": "/workdir"})
    outcome = DuplicateCallHook().check(PreToolContext(call=call, seen_calls={}))
    assert isinstance(outcome, Continue)


def test_duplicate_call_hook_falls_through_for_unknown_tool_error() -> None:
    """harness-cck4 carve-out: the registry's 'unknown_tool' error is
    the one error code load_tool/synthesize_tool can fix mid-turn. A
    duplicate after load_tool activated the missing tool MUST execute,
    not replay the stale absent-from-registry result."""
    call = ToolCall(name="read_file", arguments={"path": "docs/spec.md"})
    prior_unknown = ToolResult(
        tool_name="read_file",
        output=(
            "unknown tool: 'read_file'. The tool exists in the catalog but "
            "is not active in this session. Call load_tool(name='read_file') "
            "first..."
        ),
        success=False,
        error="unknown_tool",
    )
    seen = {("read_file", '{"path": "docs/spec.md"}'): prior_unknown}
    outcome = DuplicateCallHook().check(PreToolContext(call=call, seen_calls=seen))
    # Falls through — the call goes to the registry for a real execution.
    assert isinstance(outcome, Continue)


def test_duplicate_call_hook_still_dedups_other_error_codes() -> None:
    """The unknown_tool carve-out is narrow. TypeError / FileNotFoundError /
    HTTP errors / bd_command_failed / unknown_kwarg etc. remain
    persistent for the same args — re-running them produces the same
    failure, so dedup still applies."""
    call = ToolCall(name="read_file", arguments={"path": "/nope.md"})
    for err in (
        "FileNotFoundError: /nope.md",
        "TypeError: bad arg",
        "unknown_kwarg:foo",
        "bd_command_failed",
    ):
        prior = ToolResult(
            tool_name="read_file",
            output=f"error calling read_file: {err}",
            success=False,
            error=err,
        )
        seen = {("read_file", '{"path": "/nope.md"}'): prior}
        outcome = DuplicateCallHook().check(PreToolContext(call=call, seen_calls=seen))
        assert isinstance(outcome, Skip), f"non-unknown_tool error {err!r} should still dedup"
        assert outcome.result.error == err


# ---------- tool_search loop (harness-lmwm) ----------


def _seen_with(names: list[str]) -> dict[tuple[str, str], ToolResult]:
    """Build a seen_calls map with one entry per requested name. Args
    are differentiated so each entry has a unique key — mirrors the
    repro's pattern of refining the query across attempts."""
    out: dict[tuple[str, str], ToolResult] = {}
    for i, name in enumerate(names):
        out[(name, f'{{"query":"variant_{i}"}}')] = ToolResult(
            tool_name=name, output=f"prior result {i}", success=True
        )
    return out


def _attempted_with(names: list[str]) -> dict[tuple[str, str], int]:
    """Build an attempted_calls map with one entry per requested name.
    Mirrors _seen_with's differentiated args. Each entry has count=1
    (one attempt per unique args)."""
    out: dict[tuple[str, str], int] = {}
    for i, name in enumerate(names):
        out[(name, f'{{"query":"variant_{i}"}}')] = 1
    return out


def test_tool_search_loop_hook_skips_third_call_without_load_tool() -> None:
    """Session 2026-05-19 repro: model called tool_search 7 times in
    one turn with refined / search-engine-syntax queries (site:...) and
    never called load_tool, eventually fabricating an answer. The 3rd
    tool_search call must Skip with a directive nudge naming load_tool
    as the required next step."""
    call = ToolCall(name="tool_search", arguments={"query": "kenya national bird wiki"})
    ctx = PreToolContext(
        call=call,
        seen_calls=_seen_with(["tool_search", "tool_search"]),
        attempted_calls=_attempted_with(["tool_search", "tool_search"]),
    )
    outcome = ToolSearchLoopHook().check(ctx)
    assert isinstance(outcome, Skip)
    assert "load_tool" in outcome.result.output
    assert outcome.result.success is False


def test_tool_search_loop_hook_passes_first_call() -> None:
    """First tool_search call this turn — legitimate discovery."""
    call = ToolCall(name="tool_search", arguments={"query": "x"})
    ctx = PreToolContext(call=call, seen_calls={})
    assert isinstance(ToolSearchLoopHook().check(ctx), Continue)


def test_tool_search_loop_hook_passes_second_call() -> None:
    """Second tool_search call — one refinement is fine. Threshold
    is 3rd call. Conservative on the early-call side to avoid blocking
    legitimate iterate-then-load patterns."""
    call = ToolCall(name="tool_search", arguments={"query": "y"})
    ctx = PreToolContext(
        call=call,
        seen_calls=_seen_with(["tool_search"]),
        attempted_calls=_attempted_with(["tool_search"]),
    )
    assert isinstance(ToolSearchLoopHook().check(ctx), Continue)


def test_tool_search_loop_hook_resets_after_load_tool() -> None:
    """Once load_tool has run, the discovery flow is engaging
    correctly. A subsequent tool_search (e.g. the loaded tool didn't
    fit, model is looking for an alternative) is legitimate. Catcher
    must NOT skip even with 3+ prior tool_search calls."""
    call = ToolCall(name="tool_search", arguments={"query": "different topic"})
    ctx = PreToolContext(
        call=call,
        seen_calls=_seen_with(["tool_search", "tool_search", "load_tool", "tool_search"]),
        attempted_calls=_attempted_with(["tool_search", "tool_search", "load_tool", "tool_search"]),
    )
    assert isinstance(ToolSearchLoopHook().check(ctx), Continue)


def test_tool_search_loop_hook_ignores_non_tool_search_calls() -> None:
    """The hook only gates tool_search re-invocation. Other tool calls
    pass through unaffected, even if many tool_search calls preceded
    them — those would already have hit the threshold but this call
    isn't tool_search."""
    call = ToolCall(name="search_web", arguments={"query": "kenya national bird"})
    ctx = PreToolContext(
        call=call,
        seen_calls=_seen_with(["tool_search", "tool_search", "tool_search"]),
        attempted_calls=_attempted_with(["tool_search", "tool_search", "tool_search"]),
    )
    assert isinstance(ToolSearchLoopHook().check(ctx), Continue)


def test_tool_search_loop_hook_nudge_carries_call_count() -> None:
    """The nudge text includes the actual repeat count so the model
    can see the magnitude of its loop ('called 7 times' is more
    forceful than 'called several times')."""
    call = ToolCall(name="tool_search", arguments={"query": "site:example.com"})
    ctx = PreToolContext(
        call=call,
        seen_calls=_seen_with(["tool_search"] * 6),  # this would be the 7th
        attempted_calls=_attempted_with(["tool_search"] * 6),
    )
    outcome = ToolSearchLoopHook().check(ctx)
    assert isinstance(outcome, Skip)
    assert "7" in outcome.result.output


def test_tool_search_loop_hook_fires_on_dedup_masked_attempts() -> None:
    """harness-delk repro: model emitted the SAME tool_search args 6
    times. duplicate_call (which runs before this hook) Skipped 5 of
    them — seen_calls has only 1 entry. The OLD hook missed this loop
    because it counted executions. The NEW hook counts attempts:
    attempted_calls[(tool_search, args)] = 6, so the threshold of 2
    prior attempts is crossed on the 3rd emission."""
    # Same args every time → seen_calls would have just 1 entry under
    # the executed-only count. attempted_calls tracks all 6 attempts.
    args_key = ("tool_search", '{"query":"ev production 2023","tag":"search"}')
    call = ToolCall(
        name="tool_search",
        arguments={"query": "ev production 2023", "tag": "search"},
    )
    ctx = PreToolContext(
        call=call,
        seen_calls={
            args_key: ToolResult(tool_name="tool_search", output="prior", success=True)
        },  # only 1 execution from N attempts
        attempted_calls={args_key: 5},  # 5 prior emissions, this would be the 6th
    )
    outcome = ToolSearchLoopHook().check(ctx)
    assert isinstance(outcome, Skip)
    # Nudge reports 6 — total attempts including this call.
    assert "6" in outcome.result.output


def test_tool_search_loop_hook_recognizes_load_tool_from_attempted_calls() -> None:
    """harness-delk: load_tool's RESET-the-counter behavior must work
    even when the load_tool attempt itself was deduped — the model's
    intent (tried to load) is what matters, not whether the call
    executed."""
    args_key = ("load_tool", '{"name":"search_web"}')
    ts_key = ("tool_search", '{"query":"x"}')
    call = ToolCall(name="tool_search", arguments={"query": "still looking"})
    ctx = PreToolContext(
        call=call,
        seen_calls={},  # nothing executed
        attempted_calls={ts_key: 3, args_key: 1},  # load_tool was attempted (deduped/skip)
    )
    # Even with 3 prior tool_search attempts, the deduped load_tool
    # attempt counts as 'engaging the discovery flow' → Continue.
    assert isinstance(ToolSearchLoopHook().check(ctx), Continue)


def test_tool_search_spec_description_calls_out_load_tool_step() -> None:
    """harness-lmwm preventive layer: the tool_search spec
    description must explicitly tell the model that after a non-empty
    result the next call is load_tool, AND that search-engine
    operators like 'site:' don't apply to the catalog. Without these
    cues the model loops on refined queries as if tool_search were
    the search engine."""
    from harness.tools import ToolCatalog, ToolSearchTool

    spec = ToolSearchTool(catalog=ToolCatalog()).spec
    desc = spec.description
    assert "load_tool" in desc, (
        "tool_search description must name load_tool as the required next step"
    )
    assert "TOOL NAMES" in desc or "tool names" in desc, (
        "tool_search description must clarify it returns tool names, not answers"
    )
    assert "site:" in desc, (
        "tool_search description must call out that search-engine "
        "operators ('site:') don't apply to the catalog"
    )


# ---------- argument grounding hook ----------


def test_argument_grounding_hook_flags_unrelated_domain_in_args() -> None:
    """The observed failure from harness-7od: user asks for stackoverflow,
    router picks search_web with a query naming a totally unrelated
    domain ('dailydrop.fm'). The grounding hook must Skip with a nudge."""
    call = ToolCall(name="search_web", arguments={"query": "dailydrop.fm", "max_results": 1})
    ctx = PreToolContext(
        call=call,
        seen_calls={},
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
        seen_calls={},
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
        seen_calls={},
        user_message="search for python tips",
    )
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_argument_grounding_hook_passes_when_user_message_is_none() -> None:
    """No user message threaded through → no ground-truth to check
    against. The hook stays out of the way (bootstrap / subagent cases)."""
    call = ToolCall(name="search_web", arguments={"query": "example.com"})
    ctx = PreToolContext(call=call, seen_calls={}, user_message=None)
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_argument_grounding_hook_matches_case_insensitively() -> None:
    """User message can spell the domain any case; args can too. Matching
    must be case-insensitive on both sides."""
    call = ToolCall(name="fetch_url", arguments={"url": "https://GitHub.com/foo/bar"})
    ctx = PreToolContext(
        call=call,
        seen_calls={},
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
        seen_calls={},
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
        seen_calls={},
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
            seen_calls={},
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
        seen_calls={},
        user_message="summarize the first python question on stackoverflow",
    )
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_argument_grounding_hook_passes_url_from_prior_tool_result() -> None:
    """harness-jm9p: a fetch_url targeting a domain that appeared in a
    prior search_web tool result is legitimate even if the user never
    named it. Without this, the discovery loop (search_web → fetch_url
    one of the results) is broken."""
    call = ToolCall(
        name="fetch_url",
        arguments={"url": "https://metar-taf.com/metar/RJTA"},
    )
    prior_search_result = (
        "search_web(query='weather RJTA') returned 1 result:\n"
        "  1. RJTA METAR — https://metar-taf.com/metar/RJTA"
    )
    ctx = PreToolContext(
        call=call,
        seen_calls={},
        user_message="weather at RJTA",
        prior_tool_outputs=(prior_search_result,),
    )
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Continue), (
        "URL grounded in a prior tool result must pass — the discovery "
        "loop (search_web → fetch_url) depends on this."
    )


def test_argument_grounding_hook_still_flags_url_not_in_any_source() -> None:
    """harness-jm9p inverse: a fetch_url targeting a domain that
    appears in NEITHER the user message NOR any prior tool result
    is still flagged. This is the original fabrication shape — the
    model invented metar-taf.com from training data when the search
    only returned Chandler-AZ weather URLs."""
    call = ToolCall(
        name="fetch_url",
        arguments={"url": "https://metar-taf.com/metar/RJTA"},
    )
    # Prior search returned a different domain entirely (the bug's
    # actual repro: geolocation-polluted results).
    prior_search_result = (
        "search_web(query='weather RJTA') returned 1 result:\n"
        "  1. Chandler AZ weather — https://example-weather.com/chandler"
    )
    ctx = PreToolContext(
        call=call,
        seen_calls={},
        user_message="weather at RJTA",
        prior_tool_outputs=(prior_search_result,),
    )
    outcome = ArgumentGroundingHook().check(ctx)
    assert isinstance(outcome, Skip)
    assert "'metar-taf'" in outcome.result.output


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
        seen_calls={},
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
        seen_calls={},
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
            seen_calls={},
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
    ctx = PreToolContext(call=call, seen_calls={}, user_message=None)
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
        seen_calls={forced_key: _STUB_RESULT},
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
        seen_calls={("search_memory", "{}"): _STUB_RESULT},  # different tool ran
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
            seen_calls={("assemble_context", "{}"): _STUB_RESULT},
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


def test_missing_citation_silent_when_reply_has_url_citations() -> None:
    """JEPA smoke repro 2026-05-15 (search_scholar variant): the model
    summarised search_scholar results with `[arxiv:..]` / `[doi:..]`
    URL citations, with `§Abstract:` paper-internal section markers
    inside each cited paragraph. document_reference (`\\bsection\\b`)
    matched the `§Abstract` token, surface_patterns required §-form
    paren-doc and didn't match — false-positive Nudge. URL citations
    are valid grounding for scholar-style characters; the hook now
    accepts them as evidence of citation."""
    af = load_character(_REPO / "character" / "airton_f").citation_grammar
    assert af is not None
    reply_text = (
        "The corpus doesn't cover JEPA in the context of predictive models — "
        "it has a greeting protocol unrelated to ML.\n\n"
        "[arxiv:2305.17493] §Abstract: Dynamic / ME-JEPA v2.0.0-rc1 is an "
        "audited single-binary JEPA-style world-model runtime that runs three "
        "domains by TOML manifest alone. Every persisted operation flows "
        "through 25 typed RocksDB families.\n\n"
        "[arxiv:2403.00504] §Abstract: Joint-Embedding Predictive Architecture "
        "(JEPA) has emerged as a promising self-supervised approach that learns "
        "by leveraging a world model. We explore how to generalize the JEPA "
        "prediction task to a broader set of corruptions."
    )
    outcome = MissingCitationHook(grammar=af).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"assemble_context", "search_scholar"}),
        )
    )
    assert isinstance(outcome, Continue), (
        "URL citations should satisfy the citation-required check — the "
        "reply is grounded in [arxiv:..] / [doi:..] form even though it "
        "doesn't use §-form paren-doc syntax"
    )


def test_missing_citation_fires_on_bare_anchor_for_non_faa_character() -> None:
    """Smoke 2026-05-15 repro (airton_f): the model wrote `§1-5
    explicitly disclaims security` — bare FAA-style hyphen anchor, no
    paren-doc suffix. The character's surface_patterns require
    `§<anchor> (<doc>)` and don't match this shape. Previously, the
    global `_CITATION_PRESENT_RE` accepted bare `§N-N` regardless of
    grammar and let the reply through uncited. Root fix: the global
    fallback is now gated on `citation_grammar.accept_faa_bare_anchor`
    (default False). airton_f leaves it default → bare §1-5 nudged."""
    af = load_character(_REPO / "character" / "airton_f").citation_grammar
    assert af is not None
    assert af.accept_faa_bare_anchor is False
    reply_text = (
        "The corpus doesn't cover modern cryptographic key exchange "
        "mechanisms in detail — it has a greeting protocol with no "
        "key-agreement step (§1-5 explicitly disclaims security). "
        "Want me to search the web instead?"
    )
    outcome = MissingCitationHook(grammar=af).check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"assemble_context"}),
        )
    )
    assert isinstance(outcome, Nudge)


def test_missing_citation_silent_on_bare_anchor_for_faa_character() -> None:
    """Counterfactual to the previous test: when the character's
    grammar opts INTO the FAA bare-anchor fallback (airton_c1 / c /
    c_tfr), a bare `§N-N-N` continues to count as a citation even
    without the JO 7110.65 prefix. Preserves the pre-root-fix
    behavior for FAA-flavored personas."""
    c1 = load_character(_REPO / "character" / "airton_c1").citation_grammar
    assert c1 is not None
    assert c1.accept_faa_bare_anchor is True
    reply_text = (
        "Per JO 7110.65 §1-1-1, the order prescribes ATC procedures. "
        "Same-runway separation is covered under §3-10-3 of the order, "
        "with separation minima depending on weight class and runway "
        "configuration. The relevant table is TBL 3-9-1 for category "
        "I, II, and III combinations."
    )
    outcome = MissingCitationHook(grammar=c1).check(
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


# ---------- list_count_mismatch subset carve-out (harness-9gzk) ----------


def test_list_count_mismatch_silent_on_orchid_subset_repro() -> None:
    """Session 2026-05-19: model summarized an article saying 'Kenya is
    home to 283 species of orchids belonging to 50 genera. Among these,
    some species are unique and potentially valuable for ornamental
    purposes' then listed 4 named species (potential cut-flower
    cultivars). Without the subset carve-out the catcher fired
    50-vs-4 on 'genera ... are' — wrong reading: 50 was the population
    total, 4 was a labelled subset. Repro must stay silent so the
    legitimate summary passes through."""
    reply_text = (
        "Based on the information provided in the article, Kenya is home to "
        "283 species of orchids belonging to 50 genera. Among these, some "
        "species are unique and potentially valuable for ornamental purposes. "
        "The article mentions a few species that have potential for "
        "cut-flower production:\n\n"
        "1. **Ansellia africana**\n"
        "2. **Angraecum eburneum**\n"
        "3. **Calanthea sylvatica**\n"
        "4. **Eulophia horsfalii**\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"fetch_url"}),
        )
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_silent_with_among_these_marker() -> None:
    """Minimal 'among these' subset pattern — 50 things in the
    population, then 4 specific examples called out. Must stay silent."""
    reply_text = (
        "The state has 50 counties. Among these, four are coastal:\n\n"
        "- Alpha\n- Beta\n- Gamma\n- Delta\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=False)
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_silent_with_some_of_them_marker() -> None:
    """'Some of them are' is canonical subset language. 50 vs 3
    enumeration with this marker should NOT fire."""
    reply_text = "There are 50 species. Some of them are notable:\n\n- A\n- B\n- C\n"
    outcome = ListCountMismatchHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=False)
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_silent_with_for_purpose_marker() -> None:
    """'for cut-flower production' / 'for ornamental use' etc. label
    a list as a purpose-filtered subset of the count claim. Stay
    silent — the list isn't promising to enumerate the count."""
    reply_text = (
        "The country has 12 native species. For ornamental cultivation, "
        "the following are notable:\n\n"
        "- A\n- B\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=False)
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_silent_with_including_marker() -> None:
    """'including' opens an open-ended subset. '10 X, including A, B, C'
    is not promising an exhaustive 10-item list. Stay silent."""
    reply_text = (
        "The class has 10 students, including these top performers:\n\n- Alice\n- Bob\n- Charlie\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=False)
    )
    assert isinstance(outcome, Continue)


def test_list_count_mismatch_still_fires_on_real_mismatch_in_repro_shape() -> None:
    """Critical: the second draft of the orchid repro said 'lists five
    orchids that are unique to or particularly notable in Kenya. Here
    are the orchids mentioned:' followed by 4 items. That's a real 5
    vs 4 mismatch (no subset language between 'five' and the list).
    Must still fire — the subset carve-out shouldn't disarm genuine
    count fabrications."""
    reply_text = (
        "The source lists five orchids that are unique to or particularly "
        "notable in Kenya. Here are the orchids mentioned:\n\n"
        "1. Ansellia africana\n"
        "2. Angraecum eburneum\n"
        "3. Calanthea sylvatica\n"
        "4. Eulophia horsfalii\n"
    )
    outcome = ListCountMismatchHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=True)
    )
    assert isinstance(outcome, Nudge)


def test_list_count_mismatch_still_fires_when_subset_marker_far_from_claim() -> None:
    """Subset language must be inside the count-claim's match window
    (120 chars). A 'including' that appears late in the reply, after
    the count + list, can't disarm the catcher — the claim already
    committed to an exhaustive enumeration shape."""
    reply_text = (
        "The four primary purposes of ATC are as follows:\n\n"
        "1. Prevent collisions.\n"
        "2. Safe and orderly flow.\n"
        "3. National security.\n\n"
        "These are non-exhaustive — additional services may apply, "
        "including secondary missions."
    )
    outcome = ListCountMismatchHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=True)
    )
    assert isinstance(outcome, Nudge)


# ---------- self_contradicting_rank (harness-1jca) ----------


def test_self_contradicting_rank_fires_on_chile_repro() -> None:
    """Session 2026-05-19 (avocado query, 2nd session): final reply
    said Chile is BOTH 'the top South American avocado exporter' AND
    'the third highest exporter in South America'. Internally self-
    falsifying. The 'making it the third' idiom uses a back-reference
    pronoun ('it') to the previously-named subject (Chile)."""
    reply_text = (
        "The top South American avocado exporter is Chile, as reported by "
        "the article 'Avocados Exports by Country 2024' from WorldStopExports. "
        "The data shows that Chile exported $291.6 million worth of avocados "
        "in 2024, making it the third highest exporter in South America."
    )
    outcome = SelfContradictingRankHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=True)
    )
    assert isinstance(outcome, Nudge)
    # Nudge text must name the failure shape.
    assert "contradicts" in outcome.text.lower()


def test_self_contradicting_rank_fires_on_named_subject_in_both() -> None:
    """Shared proper-noun across both windows (no pronoun needed).
    'Brazil is the top exporter' + 'Brazil is the second largest
    producer' — same proper noun in both windows, conflicting rank."""
    reply_text = (
        "Per the source, Brazil is the top exporter of coffee. Brazil is "
        "the second largest producer overall."
    )
    outcome = SelfContradictingRankHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=True)
    )
    assert isinstance(outcome, Nudge)


def test_self_contradicting_rank_silent_when_subjects_differ() -> None:
    """'Brazil is the top exporter, Colombia is the third.' — both
    rank markers fire but different subjects. Catcher must stay
    silent — no contradiction; each statement is consistent."""
    reply_text = (
        "Per the source, Brazil is the top exporter of coffee. Colombia "
        "is the third largest producer in the region."
    )
    outcome = SelfContradictingRankHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=True)
    )
    assert isinstance(outcome, Continue)


def test_self_contradicting_rank_silent_with_only_top_claim() -> None:
    """One TOP claim, no NTH claim — nothing to contradict."""
    reply_text = "Per the source, Brazil is the top exporter of coffee in the world."
    outcome = SelfContradictingRankHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=True)
    )
    assert isinstance(outcome, Continue)


def test_self_contradicting_rank_silent_with_only_nth_claim() -> None:
    """One NTH claim, no TOP claim — also no contradiction."""
    reply_text = "Per the source, Chile is the third largest exporter of avocados in South America."
    outcome = SelfContradictingRankHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=True)
    )
    assert isinstance(outcome, Continue)


def test_self_contradicting_rank_silent_across_far_paragraphs() -> None:
    """TOP and NTH claims must be near each other (within the
    proximity window) to count as a single-paragraph contradiction.
    Claims in widely-separated paragraphs may legitimately be about
    different scopes ('top in the world' vs 'third in 2010')."""
    reply_text = (
        "Brazil is the top exporter of coffee globally as of 2024.\n\n"
        + ("---\n" * 30)  # ~120 chars of separator
        + (
            "Background: in 2010, before the recent boom, Brazil was the third "
            "largest exporter. Output has grown significantly since then."
        )
    )
    outcome = SelfContradictingRankHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=True)
    )
    assert isinstance(outcome, Continue)


def test_self_contradicting_rank_silent_with_no_shared_subject_and_no_pronoun() -> None:
    """TOP and NTH markers within window but no shared proper noun AND
    no back-reference pronoun. Catcher can't establish that the two
    claims refer to the same entity → conservative Continue."""
    reply_text = "the top exporter is reported. The third is also reported."
    outcome = SelfContradictingRankHook().check(
        BailContext(reply=_reply(reply_text), tools_ran_this_turn=True)
    )
    assert isinstance(outcome, Continue)


# ---------- scope_violation (harness-lyyr) ----------


def test_scope_violation_fires_on_mexico_for_south_america() -> None:
    """Session 2026-05-19 (avocado query, 1st session): user asked for
    'south american' top exporter; reply named Mexico. Mexico is in
    North America, not South America. Catcher must Nudge."""
    ctx = BailContext(
        reply=_reply(
            "The top avocado exporter in South America is Mexico. Mexico "
            "exported $4 billion worth of avocados, accounting for 41.9% "
            "of global exports."
        ),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web", "fetch_url"}),
        user_message="which south american country exports the most avocados?",
    )
    outcome = ScopeViolationHook().check(ctx)
    assert isinstance(outcome, Nudge)
    assert "Mexico" in outcome.text
    assert "south america" in outcome.text.lower()


def test_scope_violation_silent_when_country_is_in_region() -> None:
    """User asked for top SA country; reply names Peru. Peru IS in
    South America. Catcher must stay silent — correct answer."""
    ctx = BailContext(
        reply=_reply("The top South American avocado exporter is Peru."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="which south american country exports the most avocados?",
    )
    assert isinstance(ScopeViolationHook().check(ctx), Continue)


def test_scope_violation_silent_when_user_named_no_region() -> None:
    """Question without a named region — catcher has no scope to
    verify against. Stay silent."""
    ctx = BailContext(
        reply=_reply("The top avocado exporter is Mexico."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="which country exports the most avocados?",
    )
    assert isinstance(ScopeViolationHook().check(ctx), Continue)


def test_scope_violation_silent_without_top_rank_trigger() -> None:
    """User asked a region-scoped question but not a top/most/largest
    one — the catcher only arms for top-pick questions. 'which south
    american countries grow avocados?' has no top trigger."""
    ctx = BailContext(
        reply=_reply("Avocados are grown in South American countries including Mexico."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="which south american countries grow avocados?",
    )
    assert isinstance(ScopeViolationHook().check(ctx), Continue)


def test_scope_violation_silent_when_no_content_tool_ran() -> None:
    """No content tool ran — fabricated_search territory, not scope
    territory. Conservative skip."""
    ctx = BailContext(
        reply=_reply("The top South American avocado exporter is Mexico."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"tool_search"}),  # meta-only
        user_message="which south american country exports the most avocados?",
    )
    assert isinstance(ScopeViolationHook().check(ctx), Continue)


def test_scope_violation_silent_when_reply_names_no_country() -> None:
    """Reply doesn't name a country — nothing to check. Stay silent."""
    ctx = BailContext(
        reply=_reply("The data didn't surface a clear top exporter for that region."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="which south american country exports the most avocados?",
    )
    assert isinstance(ScopeViolationHook().check(ctx), Continue)


def test_scope_violation_silent_when_user_names_multiple_regions() -> None:
    """Ambiguity guard: user message names MULTIPLE regions and the
    named country is in one of them. Skip — the question's scope is
    unclear and the country isn't unambiguously out-of-scope."""
    ctx = BailContext(
        reply=_reply("The top avocado exporter in the region is Mexico."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message=("compare top avocado exporters from north america and south america."),
    )
    assert isinstance(ScopeViolationHook().check(ctx), Continue)


def test_scope_violation_fires_on_asian_country_for_african_question() -> None:
    """Generalized shape: user asks for top African, reply names an
    Asian country. The gazetteer covers more regions than just SA."""
    ctx = BailContext(
        reply=_reply("The top African coffee producer is Vietnam."),
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_web"}),
        user_message="which african country produces the most coffee?",
    )
    outcome = ScopeViolationHook().check(ctx)
    assert isinstance(outcome, Nudge)
    assert "Vietnam" in outcome.text
    assert "africa" in outcome.text.lower()


def test_geography_module_has_expected_membership() -> None:
    """Smoke test the gazetteer itself: known-good memberships for
    the failure modes the catcher targets."""
    from harness.geography import (
        REGION_COUNTRIES,
        countries_in_region,
        in_region,
    )

    # South America membership: Mexico must NOT be in; Peru must be in.
    assert in_region("Peru", "south america")
    assert in_region("Peru", "South America")  # case-insensitive
    assert not in_region("Mexico", "south america")

    # Latin America DOES include Mexico (broader scope).
    assert in_region("Mexico", "latin america")

    # Lookup with unknown region: empty.
    assert countries_in_region("xenadu") == frozenset()
    # Every populated region has entries.
    for region in REGION_COUNTRIES:
        assert countries_in_region(region), f"region {region!r} unexpectedly empty"


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


# ---------- source_count_inflation (harness-lynw) ----------


def test_source_count_inflation_fires_on_1_to_5_pad() -> None:
    """harness-lynw repro: search_web returned 1 result; reply
    enumerates 5 sources. Padding 4 fabricated entries is the
    failure shape."""
    reply_text = (
        "Here are the weather sources for Mombasa:\n\n"
        "1. **Time and Date**: Provides a 14-day forecast.\n"
        "2. **AccuWeather**: Offers a 3-day forecast.\n"
        "3. **Weather.com**: Shows today's conditions.\n"
        "4. **Weather.co.ke**: Provides updates on temperature.\n"
        "5. **BBC Weather**: Offers a 14-day forecast.\n"
    )
    tool_output = (
        "search_web(query='weather in Mombasa, Kenya') returned 1 result:\n"
        "  1. Mombasa 14 day forecast - https://timeanddate.com/...\n"
    )
    outcome = SourceCountInflationHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_web"}),
            prior_tool_outputs=(tool_output,),
        )
    )
    assert isinstance(outcome, Nudge)
    assert "5" in outcome.text
    assert "1" in outcome.text


def test_source_count_inflation_silent_within_tolerance() -> None:
    """Reply enumerates exactly tool_count + 1 — within the tolerance
    (a model may add a single 'also worth mentioning' line on top of
    the tool's items). Don't fire."""
    reply_text = "From the search:\n\n1. **Source A**\n2. **Source B**\n3. **Source C**\n"
    tool_output = (
        "search_web returned 2 results:\n  1. A — https://a.example\n  2. B — https://b.example\n"
    )
    outcome = SourceCountInflationHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_web"}),
            prior_tool_outputs=(tool_output,),
        )
    )
    assert isinstance(outcome, Continue)


def test_source_count_inflation_silent_when_reply_below_floor() -> None:
    """A 2-item reply when the tool returned 0 items is a 'no
    matches' wrap-up shape, not fabrication. Don't fire."""
    reply_text = "Found nothing.\n\n1. A\n2. B\n"
    tool_output = "search_web returned 0 results."
    outcome = SourceCountInflationHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_web"}),
            prior_tool_outputs=(tool_output,),
        )
    )
    assert isinstance(outcome, Continue)


def test_source_count_inflation_silent_when_no_prior_tool_output() -> None:
    """No prior tool output → no comparison anchor. Don't fire."""
    reply_text = "Here are the steps:\n\n1. First\n2. Second\n3. Third\n4. Fourth\n"
    outcome = SourceCountInflationHook().check(
        BailContext(
            reply=_reply(reply_text),
            tools_ran_this_turn=False,
            tools_ran=frozenset(),
            prior_tool_outputs=(),
        )
    )
    assert isinstance(outcome, Continue)


def test_source_count_inflation_opt_in_via_catchers_roster() -> None:
    """Hook installs only when 'source_count_inflation' is in the
    character's catchers roster — characters that don't do web
    research provably skip it."""
    from harness.orchestrator.hooks import default_hook_pipeline

    # Default (empty catchers) — hook NOT installed.
    pipe_default = default_hook_pipeline()
    hook_names = {h.name for h in pipe_default.bail}
    assert "source_count_inflation" not in hook_names

    # Opt-in — hook installed.
    pipe_optin = default_hook_pipeline(catchers=("source_count_inflation",))
    hook_names_optin = {h.name for h in pipe_optin.bail}
    assert "source_count_inflation" in hook_names_optin


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


# ---------- OpinionWithoutTriggerHook ----------


def test_opinion_no_trigger_fires_on_unprompted_opinion() -> None:
    """Reply contains `Opinion:` paragraph but user's message has none
    of the trigger words. Nudge.

    Reproduces the 2026-05-15 airton_f smoke: user said "Search the web
    to find modern key exchange mechanisms and contrast them with
    diffie-hellman" — no `opinion` / `thoughts` / `view`. Model produced
    an Opinion paragraph anyway."""
    reply = _reply(
        "§5 (01-example-rfc-style): the document disclaims security.\n\n"
        "Opinion: the disclaimer is doing too much work for two sentences."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_web"}),
        user_message=(
            "Search the web to find modern cryptographic key exchange "
            "mechanisms and contrast them with diffie-hellman"
        ),
    )
    outcome = OpinionWithoutTriggerHook().check(ctx)
    assert isinstance(outcome, Nudge)
    assert "Opinion" in outcome.text


def test_opinion_no_trigger_allows_when_trigger_word_present() -> None:
    """User asks for an opinion explicitly → Opinion paragraph is
    legitimate. Hook stays out of the way regardless of substance."""
    reply = _reply(
        "§5 (01-example-rfc-style): the document disclaims security.\n\n"
        "Opinion: the disclaimer is doing too much work for two sentences."
    )
    for trigger in (
        "what's your opinion on this disclaimer?",
        "thoughts?",
        "what do you think of section 5?",
        "your view on the security model?",
        "your take?",
    ):
        ctx = BailContext(
            reply=reply,
            tools_ran_this_turn=True,
            tools_ran=frozenset({"assemble_context"}),
            user_message=trigger,
        )
        outcome = OpinionWithoutTriggerHook().check(ctx)
        assert isinstance(outcome, Continue), f"hook fired on legitimate trigger {trigger!r}"


def test_opinion_no_trigger_silent_when_no_opinion_paragraph() -> None:
    """Reply has no `Opinion:` token → hook stays quiet. The token is
    the canonical paragraph header; lowercase "opinion" inside prose
    is fine and shouldn't trip the check."""
    reply = _reply(
        "§5 (01-example-rfc-style): the document explicitly disclaims "
        "any security model. It names unauthenticated names, replay "
        "susceptibility, and unencrypted transport as known gaps. "
        "There is a public opinion on this kind of disclaimer pattern."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context"}),
        user_message="what does section 5 say?",
    )
    outcome = OpinionWithoutTriggerHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_opinion_no_trigger_continues_when_user_message_missing() -> None:
    """Bootstrap / subagent contexts may not thread user_message
    through. Conservative default: don't nudge — better to let a
    legitimate opinion through than to false-positive on a context the
    hook can't reason about."""
    reply = _reply("§5 (01-example-rfc-style): summary.\n\nOpinion: thoughts.")
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=False,
        tools_ran=frozenset(),
        user_message=None,
    )
    outcome = OpinionWithoutTriggerHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_opinion_no_trigger_wires_into_pipeline_when_opted_in() -> None:
    """Composition pin: 'opinion_no_trigger' in catchers installs the
    hook; absence keeps it out. Non-scholar characters aren't affected."""
    on = default_hook_pipeline(catchers=("opinion_no_trigger",))
    assert "opinion_no_trigger" in on.names()
    off = default_hook_pipeline(catchers=())
    assert "opinion_no_trigger" not in off.names()


# ---------- PostSearchGroundingHook ----------


def test_post_search_grounding_fires_when_search_ran_but_no_followup() -> None:
    """search_web ran, no fetch_url, substantive reply, no URL citation,
    no raw URL → Nudge.

    Reproduces the 2026-05-15 airton_f smoke: search_web returned a
    Wikipedia hit; model didn't fetch, didn't cite, just produced
    training-data prose."""
    reply = _reply(
        "Diffie-Hellman is a foundational protocol for establishing a "
        "shared secret over an insecure channel. Modern variants build "
        "on it with forward secrecy and MITM-resistance via additional "
        "authentication binding."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_web"}),
        user_message="find papers on modern key exchange",
    )
    outcome = PostSearchGroundingHook().check(ctx)
    assert isinstance(outcome, Nudge)
    assert "search_web" in outcome.text


def test_post_search_grounding_allows_when_fetch_url_followed() -> None:
    """Model did the right thing: search_web → fetch_url → answer.
    fetch_url in tools_ran means the model grounded the search.
    Hook stays out."""
    reply = _reply(
        "[arxiv:2401.12345] The authors report a 12% improvement on the "
        "standard benchmark, replacing the affine layer with…"
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_web", "fetch_url"}),
        user_message="find papers on the topic",
    )
    outcome = PostSearchGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_post_search_grounding_allows_url_citation() -> None:
    """Reply with a `[arxiv:…]` / `[scholar:…]` / `[doi:…]` / `[wiki:…]`
    citation counts as grounding even when fetch_url didn't run this
    turn (the model may be citing from search_web snippet directly)."""
    for cite in (
        "[arxiv:2401.12345] the paper proves…",
        "[scholar:Diffie-Hellman 1976] the original…",
        "[doi:10.1145/12345.67890] the standard binds…",
        "[wiki:Diffie-Hellman_key_exchange] the overview names…",
    ):
        reply = _reply(
            cite + " " + ("filler " * 20)  # > 80 chars
        )
        ctx = BailContext(
            reply=reply,
            tools_ran_this_turn=True,
            tools_ran=frozenset({"assemble_context", "search_web"}),
            user_message="find papers on key exchange",
        )
        outcome = PostSearchGroundingHook().check(ctx)
        assert isinstance(outcome, Continue), f"hook fired on legit citation {cite!r}"


def test_post_search_grounding_allows_raw_url() -> None:
    """Raw http(s) URL in the reply counts as grounding-after-search.
    The model may surface a search-result URL inline rather than in
    bracket-citation form."""
    reply = _reply(
        "The top search result was https://arxiv.org/abs/2401.12345, "
        "which covers exactly this question. Let me know if you want "
        "me to fetch it."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_web"}),
        user_message="find papers on key exchange",
    )
    outcome = PostSearchGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_post_search_grounding_silent_when_search_did_not_run() -> None:
    """No search_web in tools_ran → hook out of scope. Other tools
    running (assemble_context, search_memory) are irrelevant; this
    catcher pairs specifically with the search-then-stay-ungrounded
    failure mode."""
    reply = _reply(
        "I have nothing from the corpus to cite for this question. "
        "Want me to narrow the question or search the web?"
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context"}),
        user_message="what's the modern key exchange?",
    )
    outcome = PostSearchGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_post_search_grounding_silent_on_short_reply() -> None:
    """Short refusals like 'no allowlisted results, want to broaden?'
    don't need a URL — the user is being asked to take the next move."""
    reply = _reply("No allowlisted source covered the query. Broaden?")
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_web"}),
        user_message="find papers on key exchange",
    )
    outcome = PostSearchGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_post_search_grounding_fires_when_search_scholar_ran_but_no_followup() -> None:
    """JEPA smoke repro 2026-05-15 (search_scholar variant): the model
    called search_scholar, got 5 real papers back, then produced
    "you might want to search the web" with no citations and no
    substance. Hook must fire on search_scholar the same way it
    fires on search_web — both are "search-and-ground" tools."""
    reply = _reply(
        "The corpus doesn't cover JEPA in the context of predictive "
        "models. To proceed, you might want to search the web or "
        "academic papers for more information on JEPA."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_scholar"}),
        user_message="Search and review what JEPA is in the context of predictive models.",
    )
    outcome = PostSearchGroundingHook().check(ctx)
    assert isinstance(outcome, Nudge)


def test_post_search_grounding_allows_search_scholar_with_inline_citation() -> None:
    """When search_scholar ran and the reply summarizes with a
    `[scholar:...]` / `[arxiv:...]` / `[doi:...]` citation, the hook
    stays out of the way — that's the legitimate grounding move."""
    reply = _reply(
        "JEPA (Joint-Embedding Predictive Architecture) is a self-supervised "
        "learning framework. The original I-JEPA paper "
        "[arxiv:2301.08243] introduces it for image representation; "
        "V-JEPA [doi:10.x/vjepa] extends the framework to video."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_scholar"}),
        user_message="Search and review what JEPA is.",
    )
    outcome = PostSearchGroundingHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_post_search_grounding_wires_into_pipeline_when_opted_in() -> None:
    """Composition pin."""
    on = default_hook_pipeline(catchers=("post_search_grounding",))
    assert "post_search_grounding" in on.names()
    off = default_hook_pipeline(catchers=())
    assert "post_search_grounding" not in off.names()


# ---------- PostResearchPersistHook ----------


def _persist_hook() -> PostResearchPersistHook:
    """Fixed-date hook for deterministic nudge-text assertions."""
    return PostResearchPersistHook(today_provider=lambda: "2026-05-15")


def test_post_research_persist_fires_on_multi_source_summary_without_remember() -> None:
    """search_scholar ran, the reply has 3 URL citations (multi-source
    synthesis), but the model never called remember_event. Hook nudges
    with the persist instructions — today's date stamped in."""
    reply = _reply(
        "JEPA spans several domains. [arxiv:2403.00504] frames it for "
        "world-model SSL; [doi:10.1016/j.isprsjprs.2024.09.013] applies "
        "it to SAR ATR; [arxiv:2502.03933] extends to HEP collider data."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_scholar"}),
        user_message="Search and review what JEPA is in the context of predictive models.",
    )
    outcome = _persist_hook().check(ctx)
    assert isinstance(outcome, Nudge)
    assert "remember_event" in outcome.text
    assert "Captured: 2026-05-15" in outcome.text


def test_post_research_persist_silent_when_remember_event_ran() -> None:
    """Happy path: model called remember_event after search_scholar.
    Hook stays out — the work was persisted."""
    reply = _reply(
        "JEPA spans several domains. [arxiv:2403.00504] (world models); "
        "[arxiv:2502.03933] (HEP-JEPA)."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_scholar", "remember_event"}),
        user_message="Search and review JEPA",
    )
    outcome = _persist_hook().check(ctx)
    assert isinstance(outcome, Continue)


def test_post_research_persist_silent_on_single_citation_drill_down() -> None:
    """One URL citation = single-paper recap, not multi-source synthesis.
    Single-source replies stay in transcript; the hook only catches the
    multi-source omissions worth durable storage."""
    reply = _reply(
        "[arxiv:2301.08243] introduces I-JEPA. The paper proposes joint "
        "embedding for self-supervised image representation, training "
        "the predictor and target encoders in a single pass."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_scholar"}),
        user_message="What is I-JEPA?",
    )
    outcome = _persist_hook().check(ctx)
    assert isinstance(outcome, Continue)


def test_post_research_persist_silent_when_search_scholar_did_not_run() -> None:
    """search_web alone is not enough — its hits are too transient to
    earn a durable row. Only search_scholar (academic-paper search)
    triggers the persist requirement."""
    reply = _reply(
        "Modern key exchange uses ECDH variants. [wiki:Diffie-Hellman_key_exchange] "
        "and [wiki:Elliptic-curve_Diffie-Hellman] both cover the topic."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_web"}),
        user_message="What's the modern key exchange?",
    )
    outcome = _persist_hook().check(ctx)
    assert isinstance(outcome, Continue)


def test_post_research_persist_silent_when_no_tools_ran() -> None:
    """Bootstrap / no-tool turns are out of scope."""
    reply = _reply(
        "Multi-paper claim with [arxiv:1] and [arxiv:2] citations would "
        "trigger persist — but no search ran this turn so nothing to save."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=False,
        tools_ran=frozenset(),
        user_message="anything",
    )
    outcome = _persist_hook().check(ctx)
    assert isinstance(outcome, Continue)


def test_post_research_persist_today_provider_injects_date_into_nudge() -> None:
    """The nudge text MUST quote today's date verbatim so the model has
    no excuse to guess. today_provider is the injection seam."""
    hook = PostResearchPersistHook(today_provider=lambda: "2099-12-31")
    reply = _reply("Two-source summary: [arxiv:1234.5678] one paper; [doi:10/abc] another.")
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_scholar"}),
        user_message="Search and find papers on topic X",
    )
    outcome = hook.check(ctx)
    assert isinstance(outcome, Nudge)
    assert "Captured: 2099-12-31" in outcome.text


def test_post_research_persist_default_today_provider_returns_iso_date() -> None:
    """The default provider reads `datetime.now(UTC).date().isoformat()`
    at check-time (not import-time), so long-running sessions still get
    today's date. Format pin: YYYY-MM-DD only — no time component, no
    timezone suffix in the body stamp."""
    import re as _re

    hook = PostResearchPersistHook()
    today = hook.today_provider()
    assert _re.fullmatch(r"\d{4}-\d{2}-\d{2}", today)


def test_post_research_persist_wires_into_pipeline_when_opted_in() -> None:
    """Composition pin."""
    on = default_hook_pipeline(catchers=("post_research_persist",))
    assert "post_research_persist" in on.names()
    off = default_hook_pipeline(catchers=())
    assert "post_research_persist" not in off.names()


def test_post_research_persist_silent_on_recall_lead() -> None:
    """harness-i5pk smoke: model led with 'From prior discussion
    (captured 2026-05-15):' — a recall, not new research. Hook must
    skip even though search_scholar ran (router-fronted) and the
    reply quoted memory URL tokens, because writing remember_event
    again would create a duplicate row."""
    reply = _reply(
        "From prior discussion (captured 2026-05-15): JEPA spans "
        "diverse SSL tasks. [doi:10.48550/arxiv.2403.06432] (Choi et al, "
        "brain networks), [doi:10.1145/3678717.3691271] (Li et al, "
        "trajectory similarity), [arxiv:2309.16014] (Skenderi et al, "
        "graph-level), [arxiv:2409.15803] (Hu et al, 3D), "
        "[doi:10.1109/waspaa66052.2025.11230951] (Pilataki et al, music)."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"assemble_context", "search_scholar"}),
        user_message="what do you know about JEPA?",
    )
    outcome = _persist_hook().check(ctx)
    assert isinstance(outcome, Continue)


def test_post_research_persist_silent_on_recall_lead_variants() -> None:
    """The recall-lead regex accepts 'From prior session', 'From a
    prior discussion', 'Found in prior discussion' — same intent,
    minor phrasing variation. Case-insensitive."""
    for lead in (
        "From prior session (captured 2026-05-15): JEPA spans diverse tasks. ",
        "From a prior discussion (captured 2026-05-15): JEPA spans diverse tasks. ",
        "from PRIOR Discussion (captured 2026-05-15): JEPA spans diverse tasks. ",
        "Found in prior discussion (captured 2026-05-15): JEPA spans diverse tasks. ",
    ):
        reply = _reply(lead + "[arxiv:1] (A), [doi:2] (B), [arxiv:3] (C).")
        ctx = BailContext(
            reply=reply,
            tools_ran_this_turn=True,
            tools_ran=frozenset({"search_scholar"}),
            user_message="what do you know about X?",
        )
        outcome = _persist_hook().check(ctx)
        assert isinstance(outcome, Continue), f"hook fired on recall lead {lead!r}"


def test_post_research_persist_fires_when_no_recall_lead() -> None:
    """Negative-of-negative pin: a reply that does NOT lead with a
    recall marker still trips persist when the other gates apply.
    This is the genuine-new-research path — the hook should keep
    nudging, the recall gate must not break the happy path."""
    reply = _reply(
        "JEPA spans several domains. [arxiv:2403.00504] frames it for "
        "world-model SSL; [doi:10.1016/j.isprsjprs.2024.09.013] applies "
        "it to SAR ATR; [arxiv:2502.03933] extends to HEP collider data."
    )
    ctx = BailContext(
        reply=reply,
        tools_ran_this_turn=True,
        tools_ran=frozenset({"search_scholar"}),
        user_message="research JEPA",
    )
    outcome = _persist_hook().check(ctx)
    assert isinstance(outcome, Nudge)


# ---------- PersistBodyCitationsHook (harness-gumn) ----------


def _persist_body_call(body: str) -> ToolCall:
    return ToolCall(
        name="remember_event",
        arguments={"title": "JEPA in predictive models", "body": body},
    )


def test_persist_body_citations_skips_when_body_strips_url_tokens() -> None:
    """harness-gumn smoke: model wrote a remember_event body using
    parenthetical author descriptions only, dropping every [doi:..]
    token from the reply. Hook must Skip the call (pre-write) so the
    bad row never reaches the user's approval dialog."""
    body = (
        "Captured: 2026-05-15 — JEPA spans diverse self-supervised "
        "learning tasks. Cross-validated papers:  (Choi et al, GNNs "
        "for brain networks),  (Li et al, trajectory similarity), "
        " (Skenderi et al, graph-level representation),  (Hu et al, "
        "3D representation),  (Pilataki et al, music transcription)."
    )
    call = _persist_body_call(body)
    ctx = PreToolContext(
        call=call,
        seen_calls={("search_scholar", '{"query":"jepa"}'): _STUB_RESULT},
        user_message="research JEPA",
    )
    outcome = PersistBodyCitationsHook().check(ctx)
    assert isinstance(outcome, Skip)
    assert outcome.result.success is False
    assert outcome.result.error == "persist_body_citations"
    assert "URL citation tokens" in outcome.result.output


def test_persist_body_citations_allows_when_body_has_url_tokens() -> None:
    """Happy path: body carries every [doi:..]/[arxiv:..] token from
    the reply verbatim. Hook stays out — the row is fully sourceable
    from its body alone, which is the contract the rule enforces."""
    body = (
        "Captured: 2026-05-15 — JEPA spans diverse self-supervised "
        "learning tasks. [doi:10.48550/arxiv.2403.06432] (Choi et al, "
        "brain networks), [doi:10.1145/3678717.3691271] (Li et al, "
        "trajectory similarity), [arxiv:2309.16014] (Skenderi et al, "
        "graph-level)."
    )
    call = _persist_body_call(body)
    ctx = PreToolContext(
        call=call,
        seen_calls={("search_scholar", '{"query":"jepa"}'): _STUB_RESULT},
        user_message="research JEPA",
    )
    outcome = PersistBodyCitationsHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_persist_body_citations_allows_single_token_path_to_post_research() -> None:
    """Threshold pins at _PERSIST_BODY_MIN_URL_TOKENS=2 (matches
    post_research_persist). A body with exactly 1 URL token is below
    threshold — but a body with exactly 2 passes. Boundary check."""
    body_one_url = "Captured: 2026-05-15 — Single paper recap [arxiv:1]."
    body_two_urls = "Captured: 2026-05-15 — Two papers: [arxiv:1] (A) and [doi:2] (B)."
    ctx_one = PreToolContext(
        call=_persist_body_call(body_one_url),
        seen_calls={("search_scholar", "{}"): _STUB_RESULT},
        user_message="x",
    )
    ctx_two = PreToolContext(
        call=_persist_body_call(body_two_urls),
        seen_calls={("search_scholar", "{}"): _STUB_RESULT},
        user_message="x",
    )
    assert isinstance(PersistBodyCitationsHook().check(ctx_one), Skip)
    assert isinstance(PersistBodyCitationsHook().check(ctx_two), Continue)


def test_persist_body_citations_silent_when_no_search_scholar_in_turn() -> None:
    """The hook only enforces the citation contract on the research
    persist path (search_scholar ran). User-asked 'remember that ...'
    chit-chat persists without URL citations are fine."""
    body = "Captured: 2026-05-15 — Mark prefers MLX over Ollama for daily chat."
    call = _persist_body_call(body)
    ctx = PreToolContext(
        call=call,
        seen_calls={("assemble_context", '{"role":"airton_f"}'): _STUB_RESULT},
        user_message="remember that I prefer MLX",
    )
    outcome = PersistBodyCitationsHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_persist_body_citations_ignores_other_tool_names() -> None:
    """Hook only checks remember_event. remember_fact, scribe, etc.
    pass through untouched."""
    for tool_name in ("remember_fact", "scribe_session", "search_scholar", "fetch_url"):
        call = ToolCall(name=tool_name, arguments={"x": "y"})
        ctx = PreToolContext(
            call=call,
            seen_calls={("search_scholar", "{}"): _STUB_RESULT},
            user_message="x",
        )
        outcome = PersistBodyCitationsHook().check(ctx)
        assert isinstance(outcome, Continue), f"hook fired on {tool_name!r}"


def test_persist_body_citations_handles_non_string_body() -> None:
    """Defensive: a malformed call where body isn't a string (model
    drift, tokenizer mishap) — don't crash, just pass through. The
    tool's own schema validation will reject it."""
    call = ToolCall(name="remember_event", arguments={"body": 12345})
    ctx = PreToolContext(
        call=call,
        seen_calls={("search_scholar", "{}"): _STUB_RESULT},
        user_message="x",
    )
    outcome = PersistBodyCitationsHook().check(ctx)
    assert isinstance(outcome, Continue)


def test_persist_body_citations_wires_into_pipeline_when_opted_in() -> None:
    """Composition pin."""
    on = default_hook_pipeline(catchers=("persist_body_citations",))
    assert "persist_body_citations" in on.names()
    off = default_hook_pipeline(catchers=())
    assert "persist_body_citations" not in off.names()


# ---------- WriteFileRedirectHook (harness-hnt7) ----------


def _make_redirect_hook(
    *,
    existing: dict[str, str],
    edit_active: bool = True,
    edit_log: list[tuple[str, str, str]] | None = None,
) -> WriteFileRedirectHook:
    """Build a WriteFileRedirectHook wired against an in-memory file
    map. `existing` maps path → current content; absent paths return
    None from read_existing. `edit_active=True` means
    ensure_edit_file_active reports success. Captures edit_file
    invocations into `edit_log` if provided."""

    def read_existing(path: str) -> str | None:
        return existing.get(path)

    def ensure_edit_file_active() -> bool:
        return edit_active

    def invoke_edit_file(path: str, old: str, new: str) -> ToolResult:
        if edit_log is not None:
            edit_log.append((path, old, new))
        return ToolResult(
            tool_name="edit_file",
            output=f"edited {path}: 1 replacement(s), {len(new) - len(old):+d} bytes",
            success=True,
        )

    return WriteFileRedirectHook(
        read_existing=read_existing,
        ensure_edit_file_active=ensure_edit_file_active,
        invoke_edit_file=invoke_edit_file,
    )


def test_write_file_redirect_passes_through_non_write_file() -> None:
    """Hook MUST NOT fire on calls other than write_file — Continue is
    the only safe outcome for unrelated tools."""
    hook = _make_redirect_hook(existing={})
    call = ToolCall(name="read_file", arguments={"path": "x"})
    outcome = hook.check(PreToolContext(call=call, seen_calls={}))
    assert isinstance(outcome, Continue)


def test_write_file_redirect_passes_through_when_path_absent() -> None:
    """write_file on a NEW path is the legitimate create case;
    Continue so write_file's normal create-and-write path runs."""
    hook = _make_redirect_hook(existing={})
    call = ToolCall(
        name="write_file",
        arguments={"path": "new.txt", "content": "hello world"},
    )
    outcome = hook.check(PreToolContext(call=call, seen_calls={}))
    assert isinstance(outcome, Continue)


def test_write_file_redirect_passes_through_on_explicit_overwrite() -> None:
    """`overwrite=True` is explicit user intent to replace; the hook
    must NOT intercept — write_file's own overwrite path (with the
    harness-2tq shrink guard) runs."""
    hook = _make_redirect_hook(existing={"game.js": "old content"})
    call = ToolCall(
        name="write_file",
        arguments={
            "path": "game.js",
            "content": "new content of about the same length",
            "overwrite": True,
        },
    )
    outcome = hook.check(PreToolContext(call=call, seen_calls={}))
    assert isinstance(outcome, Continue)


def test_write_file_redirect_dispatches_edit_file_on_existing_path() -> None:
    """The core fix: write_file(path, content) where path exists and
    overwrite is unset → Skip with the edit_file result, prefixed so
    the model knows the redirect happened. edit_file was invoked with
    old_string=<current>, new_string=<content>."""
    existing = {"game.js": "// existing 50+ bytes of placeholder content here"}
    log: list[tuple[str, str, str]] = []
    hook = _make_redirect_hook(existing=existing, edit_log=log)
    new_content = "// new content also at least 50+ bytes of placeholder code here"
    call = ToolCall(
        name="write_file",
        arguments={"path": "game.js", "content": new_content},
    )
    outcome = hook.check(PreToolContext(call=call, seen_calls={}))
    assert isinstance(outcome, Skip)
    assert outcome.result.success is True
    assert outcome.result.tool_name == "write_file"
    assert "write_file → edit_file" in outcome.result.output
    assert "game.js existed" in outcome.result.output
    # edit_file invoked with old=existing, new=content
    assert len(log) == 1
    path, old, new = log[0]
    assert path == "game.js"
    assert old == existing["game.js"]
    assert new == new_content


def test_write_file_redirect_preserves_safety_shrink_guard() -> None:
    """harness-2tq: refuse a write that shrinks the file by > half AND
    is under 1KB — the model almost certainly meant to append, not
    replace. Hook must Skip with the same error write_file would have
    raised; no redirect happens (edit_log stays empty)."""
    existing = {"notes.md": "x" * 4000}
    log: list[tuple[str, str, str]] = []
    hook = _make_redirect_hook(existing=existing, edit_log=log)
    call = ToolCall(
        name="write_file",
        arguments={"path": "notes.md", "content": "one more line"},
    )
    outcome = hook.check(PreToolContext(call=call, seen_calls={}))
    assert isinstance(outcome, Skip)
    assert outcome.result.success is False
    assert outcome.result.error == "suspicious_shrink"
    assert "looks like you meant to append" in outcome.result.output
    # No edit_file dispatch — the shrink guard fired first.
    assert log == []


def test_write_file_redirect_idempotent_on_identical_content() -> None:
    """Content already in the file — edit_file would raise no-op;
    return a clean success Skip instead so the model doesn't see a
    confusing error and the round still 'counts' as a successful write."""
    existing = {"a.txt": "same content here"}
    log: list[tuple[str, str, str]] = []
    hook = _make_redirect_hook(existing=existing, edit_log=log)
    call = ToolCall(
        name="write_file",
        arguments={"path": "a.txt", "content": "same content here"},
    )
    outcome = hook.check(PreToolContext(call=call, seen_calls={}))
    assert isinstance(outcome, Skip)
    assert outcome.result.success is True
    assert "no-op" in outcome.result.output
    # No edit_file dispatch — we short-circuited before invoking it.
    assert log == []


def test_write_file_redirect_falls_through_when_edit_file_inactive() -> None:
    """If edit_file is neither active nor activatable, the hook must
    fall through so the existing multi-round recovery path runs. The
    fix is opt-in on edit_file availability — no worse than today
    when the dependency is missing."""
    existing = {"f.py": "def foo(): pass\n# more lines here for length floor"}
    log: list[tuple[str, str, str]] = []
    hook = _make_redirect_hook(existing=existing, edit_active=False, edit_log=log)
    call = ToolCall(
        name="write_file",
        arguments={
            "path": "f.py",
            "content": "def foo(): pass\n# slightly different content here too",
        },
    )
    outcome = hook.check(PreToolContext(call=call, seen_calls={}))
    assert isinstance(outcome, Continue)
    assert log == []


def test_write_file_redirect_is_registered_by_default_pipeline_when_passed() -> None:
    """Composition pin: the pipeline registers the hook ONLY when a
    wired instance is passed; absent it, pre_tool stays at its prior
    shape so existing characters don't get surprise behavior."""
    on = default_hook_pipeline(write_file_redirect_hook=_make_redirect_hook(existing={}))
    assert "write_file_redirect" in on.names()
    off = default_hook_pipeline()
    assert "write_file_redirect" not in off.names()
