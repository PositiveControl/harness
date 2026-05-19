"""Hook pipeline for orchestrator catchers.

The tool loop has ~11 named catchers — regex guards + flag checks that
intercept fabrication-shaped model replies. They used to live as inline
`if` branches in `tool_loop.py`; this module hosts them as typed hooks
organized into four phases:

- `post_model` — runs on every model reply, before tool dispatch.
  Today: `paired_meta_confirm_strip` (strip narrative from a reply that
  also carries a valid tool call).
- `bail` — runs on 0-tool-calls replies. First matching hook wins.
  Today: `truncated` · `unparseable` · `teaser` · `false_success` ·
  `meta_confirm` · `fabricated_search` · `ab_fabrication` · `tool_intent`.
- `pre_tool` — runs before each individual tool call.
  Today: `duplicate_call` (short-circuit identical-args re-invocation).
- `finalize` — runs after bail retries are exhausted.
  Today: `fabrication_fallback` (replace lingering fabricated reply with
  canned refusal).

Behavior preservation is the non-negotiable success criterion: every
catcher below is a direct port of its inline predecessor, and the
attribution eval (`harness.evals.tool_loop`) keeps running via the
`disabled: frozenset[str]` argument threaded through each pipeline
method — toggle a name, watch which scenarios it uniquely saves.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from harness.citation import CitationGrammar
from harness.model.adapter import ChatMessage
from harness.tools.base import ModelReply, ToolCall, ToolResult, ToolSpec

# ---------- canned strings ----------


DUPLICATE_CALL_NUDGE = (
    "[REJECTED — duplicate call. You already called this tool with identical "
    "arguments earlier this turn; the previous result is in your message "
    "history (re-read it). DO NOT emit another tool call. Either: (a) answer "
    "the user using the data you already have, or (b) tell the user plainly "
    "what you could not find. Emitting another tool call here will be rejected "
    "again and waste the round.]"
)


# Prefix the prior tool result with this annotation when re-issuing on a
# duplicate. Preserves the original success/error/output so the model
# can't paraphrase a prior failure as success (harness-v5w).
_DUPLICATE_CALL_PREFIX = (
    "[duplicate of an earlier call this turn — re-issuing the prior "
    "result; do NOT emit this call again]\n\n"
)


# Final fallback when bail-retries are exhausted but the reply still trips
# a fabrication catcher. Before this, we surfaced the last fabricated reply
# verbatim (harness-24xj: "what is the date?" produced three stacked
# fabricated plans because the final retry still fabricated and we returned
# it). Swap to a canned refusal so the user never sees hallucinated tool
# output as an answer.
EXHAUSTED_FABRICATION_FALLBACK = (
    "I couldn't answer that without calling a tool, and my attempts to "
    "call one didn't land cleanly. Try rephrasing, or ask me something I "
    "can answer without live data."
)


# ---------- regex library (shared with cli stream filter) ----------


# Matches "Let me check…", "I'll now read…", "Next, I'll…" etc. — the
# model announcing more work without actually emitting tool calls. Anchored
# to end of content so a teaser mid-paragraph (followed by real prose) doesn't
# trigger.
TEASER_RE = re.compile(
    r"\b(let me|i'?ll|now i'?ll|now let me|next,?\s+i'?ll?)\b[^.\n]*[:.]\s*$",
    re.IGNORECASE,
)

# Matches past-tense / present-perfect claims that an action was completed —
# "has been added", "is now included", "I've created", "successfully updated",
# etc. When the reply contains one of these AND no tool was executed in the
# turn, the model is hallucinating success (harness-3fn).
_ACTION_VERBS = (
    r"(?:added|included|updated|created|written|modified|"
    r"replaced|removed|set|appended|saved|deleted)"
)
FALSE_SUCCESS_RE = re.compile(
    r"\b(?:"
    rf"has been\s+{_ACTION_VERBS}"
    r"|"
    r"(?:is|are)\s+now\s+(?:in|included|added|excluded|"
    r"set|present|updated|available|saved)"
    r"|"
    rf"(?:i(?:'ve|\shave))(?:\s+(?:just|now|successfully))?\s+{_ACTION_VERBS}"
    r"|"
    rf"successfully\s+{_ACTION_VERBS}"
    r"|"
    r"the\s+\S+\s+(?:has\s+been|is\s+now|will\s+be)\s+"
    r"(?:added|included|updated|created|excluded|modified|replaced)"
    r")\b",
    re.IGNORECASE,
)

# Matches chat-level meta-confirm prompts — "Would you like me to…?",
# "Should I…?", "Please confirm…", "Shall I…?" etc. Small models default
# to this pattern when they misunderstand that the user's request IS
# the instruction and the tool layer handles confirmation. Broad on
# purpose: 7B Qwen hit several variants in a single reply
# ("let's confirm", "would you like to", "please confirm your approval")
# so we catch all of them.
META_CONFIRM_RE = re.compile(
    r"(?:"
    r"would you like (?:(?:me|us|you)\s+)?to\s+"
    r"(?:add|proceed|continue|update|create|edit|write|change|append|"
    r"remove|modify|delete|run|install|make|do|confirm|go\s+ahead|"
    # Read-ish action verbs added after Mark's Mombasa transcript:
    # the agent finished search_web, listed 5 URLs, and asked
    # 'Would you like to read the full details?' instead of fetching.
    r"read|see|check|fetch|find|search|review|look|visit)"
    r"|"
    # 'Would you like + noun phrase' shape — caught the time-format
    # follow-up: 'Would you like more precision or a different format?'.
    r"would you like (?:more|a\s+different|some|the|another|further|"
    r"additional|specific|detailed|fewer)"
    r"|"
    r"shall i\b"
    r"|"
    r"should i (?:proceed|go ahead|continue|update|add|edit|change|"
    r"write|do|run|read|fetch|check)"
    r"|"
    r"do you want me to"
    r"|"
    r"please confirm"
    r"|"
    r"confirm (?:your |the |my )?(?:approval|request|intent|instruction)"
    r"|"
    r"let(?:'s|\s+us)\s+confirm"
    r"|"
    r"we\s+need\s+to\s+make\s+sure\s+(?:the\s+user|you)\s+confirms?"
    r"|"
    r"(?:please\s+)?approve\s+(?:the\s+|this\s+)?action"
    r"|"
    # 'I recommend (checking|reading|reviewing|consulting|visiting)'
    # the post-search punt: agent has results but kicks the work back
    # to the user instead of fetching the most authoritative URL.
    r"i\s+recommend\s+(?:checking|reading|reviewing|consulting|visiting|"
    r"looking)"
    r")",
    re.IGNORECASE,
)

# Matches replies that look like fabricated search-tool output — a
# numbered results intro or URLs hosted on classic placeholder domains
# (example.com/org/net, your-site, localhost, etc.). Small models
# sometimes respond to 'search the web for X' by inventing a result
# list with a made-up URL and snippet rather than calling search_web.
# Paired with 'no tool has run this turn' this is a strong fabrication
# tell (see harness-q27, harness-j1d).
FABRICATED_SEARCH_RE = re.compile(
    r"(?:"
    r"here\s+are\s+the\s+results"
    r"|"
    r"here\s+(?:is|are)\s+(?:what\s+)?i\s+found"
    r"|"
    r"(?:top|first|search)\s+results?\s*:"
    r"|"
    r"https?://(?:www\.)?(?:example|your-?site|your-?domain|"
    r"placeholder|localhost|test|dummy|fake)\.(?:com|org|net|io)\b"
    r"|"
    # Numbered-list entry whose content ends with terminal punctuation
    # plus a closing double-quote (`."`, `!"`, `?"`) and contains no
    # URL. Classic fabricated-snippet shape: 7B-class models imitate
    # search-tool output in an HTML-excerpt style instead of emitting
    # a <tool_call>. Tempered match rules out real `1. TITLE —
    # https://...` search_web entries. See harness-j1d.
    r"(?:\A|\n)\s*\d+\.\s+(?:(?!https?://).)*?[.!?]\"(?=\s|$)"
    r")",
    re.IGNORECASE,
)

# Past-tense claim of having searched the web / online / the internet
# when no web-fetch tool actually ran. Harness-78z shape: search_memory
# returned empty, model narrates "I've searched the web for X and
# found..." without ever calling search_web. Split from
# FABRICATED_SEARCH_RE because its gate is narrower: any tool running
# disarms the result-list patterns (legit wrap-up), but NO amount of
# non-web tool activity legitimizes a "I've searched the web" claim —
# only search_web / fetch_url actually running does.
_APOS_CLASS = "['’]"  # noqa: RUF001 — straight + curly apostrophe in a char class
FABRICATED_WEB_CLAIM_RE = re.compile(
    rf"(?:"
    rf"\bi(?:{_APOS_CLASS}ve|\s+have|\s+just)?\s+searched\s+"
    rf"(?:the\s+(?:web|internet)|online)"
    rf"|"
    rf"\b(?:after|upon)\s+searching\s+(?:the\s+(?:web|internet)|online)"
    rf")",
    re.IGNORECASE,
)

# Matches replies that regenerate a "here are the articles/stories/
# headlines…" summary list without calling any tool this turn — the
# classic follow-up fabrication shape (harness-f5x). User asks "more
# details on 8" from an earlier fetch result; 3B model skips the
# index→URL lookup and fakes a new summary. Distinct from
# FABRICATED_SEARCH_RE because the intro phrasing names articles /
# stories / headlines rather than "results" / "what I found".
# Alternatives listed first for regex efficiency (Python re backs off).
# Numbered list of URL-bearing entries — 2+ entries where each entry
# starts with `N.` and contains a URL. The classic shape the model
# fabricates when it KNOWS it should search but reaches for memory
# instead. Mark's q7kn repro:
#   1. **Weather.com — <https://weather.com/...> — Nairobi Current...**
#   2. **AccuWeather — <https://www.accuweather.com/...> — Nairobi Current...**
#   3. **BBC Weather — <https://www.bbc.co.uk/...> — Nairobi Weather**
# Pattern is conservative: must see at least the "1." and "2." entries
# (with URLs) in the same reply. False-positive risk on a real
# search_web wrap-up is mitigated by the meta-tool-only gate on
# FabricatedSearchHook — when search_web actually ran, this whole
# branch is skipped.
FABRICATED_URL_LIST_RE = re.compile(
    r"(?:\A|\n)\s*1\.\s+[^\n]*?https?://"
    r"(?:[^\n]|\n(?!\s*\d+\.\s))*?"
    r"\n\s*2\.\s+[^\n]*?https?://",
    re.IGNORECASE,
)


FABRICATED_ITEMIZATION_RE = re.compile(
    r"(?:"
    r"here\s+are\s+(?:some|the|a\s+few)\s+"
    r"(?:of\s+the\s+)?(?:first|top|latest|popular|recent|main)?\s*"
    r"(?:\d+\s+)?"
    r"(?:articles|stories|headlines|items|posts|entries|news|"
    r"top\s+stories)"
    r"|"
    r"let'?s\s+focus\s+on\s+(?:one|a\s+few)\s+of\s+the"
    r"|"
    r"here'?s\s+(?:a\s+)?(?:summary|rundown|overview|recap)\s+of\s+"
    r"(?:the\s+)?(?:articles|stories|headlines|items|posts)"
    r")",
    re.IGNORECASE,
)


# Matches ab_ops-specific fabrication shapes — ab imitates its own
# tool receipts (capture / plan) without emitting a <tool_call>.
# Distinct tells:
#   - `^Captured[.:]` — real CaptureTool output is `Captured <id> — [scope]…`
#     (id, not punctuation, follows `Captured`). A colon OR period after
#     `Captured` at line start is fabrication (harness-lbh colon form;
#     harness-ce2x period form, "Captured. Scope: personal. Outcome: …").
#   - `[prof/…]` / `[pers/…]` — real scopes are `professional` / `personal`,
#     never abbreviated. The abbreviation is a telltale of imitation.
#   - Two+ tier-label line headers (Shall/Should/Shmaybe/Watching) in one
#     reply — the `_render_plan` shape. A single mention is prose; two
#     together is plan imitation.
FABRICATED_AB_CAPTURE_RE = re.compile(r"(?:\A|\n)\s*Captured\s*[.:]", re.IGNORECASE)
# `Remembered:` receipt from ab_ops RememberTool. Real shape is
# `Remembered: <insight>` (colon form), which is exactly what the model
# imitates (harness-z734: user asked ab to remember a birthday; round 0
# bail-retry fired on 'Updated.' bare-claim, round 1 fabricated
# `Remembered: dad Steve's birthday is Oct 8th` and — because no
# 'Remembered'-shaped catcher existed — slipped through as the final
# answer). Safe because bail runs gated on `tools_ran_this_turn=False`;
# real RememberTool output disarms this.
FABRICATED_REMEMBER_RE = re.compile(r"(?:\A|\n)\s*Remembered\s*[.:]", re.IGNORECASE)
FABRICATED_AB_SCOPE_RE = re.compile(r"\[(?:prof|pers)/[^\]]+\]", re.IGNORECASE)
# `Bead id: harness-xxx` in free text. Real CaptureTool output never
# uses this phrasing — the id appears bare as the second token after
# `Captured`. Any "Bead id:" / "Bead:" label with an id is the model
# imitating an imagined schema and is always a fabrication tell
# (harness-ce2x: user saw `Bead id: harness-abc123..`).
FABRICATED_BEAD_ID_RE = re.compile(
    r"\bbead\s*(?:id)?\s*[:=]\s*(?:harness|bd|ab)-[a-z0-9]+",
    re.IGNORECASE,
)
# Domain-like token inside a tool argument's string value: matches
# `example.com`, `dailydrop.fm`, `sub.example.co.uk`, etc. Captures the
# leftmost label ("example", "dailydrop") — that's the "root" the
# grounding hook matches against the user message. TLD whitelist is
# intentionally narrow: catches the public-web TLDs that show up in
# real fetch_url / search_web args without false-positiving on
# filenames like `new.txt`, `config.yaml`, `data.json` (file extensions
# are NOT in the list). A missing TLD is cheap — legitimate calls pass;
# only "leaked fabricated domain" cases get caught. Extend as needed
# when a legitimate domain appears that isn't covered.
_KNOWN_TLDS = (
    "com|org|net|io|ai|dev|me|co|edu|gov|fm|tv|app|info|biz|xyz|"
    "uk|us|de|fr|jp|ca|au|nz|nl|es|it|ru|in|cn|br|mx|ly|to|cc|so|"
    "gg|pro|tech|site|online|store|cloud|news|blog|wiki"
)
ARG_DOMAIN_RE = re.compile(
    rf"\b([a-z0-9][a-z0-9-]*)\.(?:[a-z]{{2,}}\.)*(?:{_KNOWN_TLDS})\b",
    re.IGNORECASE,
)


def _arg_string_values(arguments: dict[str, object]) -> list[str]:
    """Flatten a tool-call's argument dict into its string values (one
    level deep). Non-string values are stringified via `str()` so a
    `max_results=1` doesn't count as a URL but also doesn't silently
    drop any string-like content. Nested dicts/lists get their leaf
    strings pulled too — covers `arguments={"filters": ["python"]}`
    shapes without recursion blow-ups on cycles (tool args are
    JSON-serializable, so no cycles by construction)."""

    out: list[str] = []

    def _walk(value: object) -> None:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, list | tuple):
            for v in value:
                _walk(v)

    _walk(arguments)
    return out


def _arg_domain_roots(arg_values: list[str]) -> set[str]:
    """Collect the lowercased leftmost labels of every domain-like
    token embedded in `arg_values`. Duplicates are collapsed; order
    doesn't matter."""
    roots: set[str] = set()
    for value in arg_values:
        for match in ARG_DOMAIN_RE.finditer(value):
            roots.add(match.group(1).lower())
    return roots


# Bare past-tense success claim as a standalone sentence — "Updated.",
# "Captured.", "Created." — word flanked by whitespace/quote and
# followed by end-of-sentence. Gated on `tools_ran_this_turn=False`
# by the caller, so legitimate wrap-ups after a real tool ran never
# trip. Observed fabrication (harness-ce2x):
#   `Missed "…". Updated. Rerun /plan.` — no tool call, pure claim.
BARE_CLAIM_RE = re.compile(
    r"(?:\A|[\s\"'])"
    r"(?:Updated|Captured|Created|Deleted|Removed|Added|Saved|Noted|Done|Remembered)"
    r"\.(?:\s|$)",
    re.IGNORECASE,
)
# 'Today — YYYY-MM-DD' is the exact _render_plan header shape; natural
# prose basically never emits this prefix. Accept em-dash or hyphen
# (harness-jj9: the observed fabrication reproduces the em-dash verbatim).
AB_DATE_HEADER_RE = re.compile(
    r"\bToday\s*[—\-]\s*\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
# Tier labels anywhere in the reply (not just line-start): the user's
# observed fabrication emitted them inline after the date header, so the
# earlier line-start anchor missed the case entirely. Two+ labels with a
# colon/dash suffix is a plan-imitation signal regardless of layout
# (harness-jj9).
AB_TIER_HEADER_RE = re.compile(
    r"\b(?:Shall|Should|Shmaybe|Watching)\b[-:]",
    re.IGNORECASE,
)


def looks_like_ab_fabrication(content: str) -> bool:
    if FABRICATED_AB_CAPTURE_RE.search(content):
        return True
    if FABRICATED_REMEMBER_RE.search(content):
        return True
    if FABRICATED_BEAD_ID_RE.search(content):
        return True
    if FABRICATED_AB_SCOPE_RE.search(content):
        return True
    if AB_DATE_HEADER_RE.search(content):
        return True
    if len(AB_TIER_HEADER_RE.findall(content)) >= 2:
        return True
    return bool(BARE_CLAIM_RE.search(content))


# Matches tool-intent statements that should be accompanied by an
# actual <tool_call>. Broader than the trailing-teaser regex: doesn't
# require end-of-content anchoring, and includes 'I will <verb>' not
# just "I'll <verb>". When a reply contains one of these AND no tool
# was called this turn, the model has announced intent without acting
# on it — classic 7B failure mode (harness-q27 follow-up).
TOOL_INTENT_RE = re.compile(
    r"\b(?:"
    r"i['\u2019]?ll|i\s+will|i\s+need\s+to|i\s+should|i'?m\s+going\s+to|"
    r"let\s+me|let['\u2019]?s|"
    r"now\s+i['\u2019]?ll|now\s+let\s+me|next,?\s+i['\u2019]?ll"
    r")\s+"
    r"(?:search|look\s+(?:up|for|at)|find|check|read|run|fetch|call|"
    r"invoke|execute|list|grep|edit|write|open|browse|query|"
    r"retrieve|download|inspect|examine)\b",
    re.IGNORECASE,
)


# ---------- outcomes ----------


@dataclass(frozen=True)
class Continue:
    """Default outcome — this hook had nothing to say."""


@dataclass(frozen=True)
class Nudge:
    """0-tool-call reply tripped a catcher. Caller appends `text` as a
    user-role message and re-runs the round.

    `catcher` names the hook that produced this Nudge. Populated by
    `HookPipeline.run_bail` at return time (individual hooks construct
    Nudge with text only — the pipeline attaches the name). Surfaced
    in `ToolLoopEvent(kind="bail_retry")` so the CLI / TUI can show
    which rule triggered the retry ('⋯ discarding draft, retrying
    (ambiguous_context)…')."""

    text: str
    catcher: str = ""


@dataclass(frozen=True)
class Truncated:
    """Reply stopped mid-stream at the token cap. Caller widens caps
    and re-runs without appending anything."""


@dataclass(frozen=True)
class Replace:
    """Mutate the round's reply in place (used by
    `paired_meta_confirm_strip` to drop narrative when a tool call is
    already present)."""

    reply: ModelReply


@dataclass(frozen=True)
class Halt:
    """Return this reply as the final turn output; no more retries."""

    reply: ModelReply


@dataclass(frozen=True)
class Skip:
    """Skip executing this tool call; feed `result` back as the tool-
    role message instead."""

    result: ToolResult


@dataclass(frozen=True)
class ReplaceResult:
    """Replace a just-executed tool's result before it's appended to
    the model-visible message thread. Used by the tool-result
    summarizer (sota punch #3, harness-zoz) to compress high-noise
    outputs before they eat through context. The tool DID run; only
    the string the model sees is rewritten."""

    result: ToolResult


BailOutcome = Continue | Nudge | Truncated
PostModelOutcome = Continue | Replace
PreToolOutcome = Continue | Skip
PostToolOutcome = Continue | ReplaceResult
FinalizeOutcome = Continue | Halt


# ---------- hook contexts + protocols ----------


@dataclass(frozen=True)
class BailContext:
    """Inputs to a bail hook. `tools_ran_this_turn` distinguishes
    first-round bails (no tools executed yet → completion claims are
    hallucinations) from wrap-up rounds (tools ran earlier → claims
    are legitimate). `tools_ran` carries the finer signal — the set
    of tool names that succeeded this turn — so hooks can gate on
    specific tools (e.g. fabricated_search only cares whether a
    web-fetch tool ran; search_memory running doesn't legitimize
    a fabricated web-search narration).

    `user_message` is the verbatim content of the most recent
    user-role turn (same value that feeds `PreToolContext.user_message`).
    Hooks use it to tell 'model echoed the user's input' apart from
    'model is making a new claim' — e.g. reserved_squawk_code
    disarms when the reserved-code match in the reply is a verbatim
    echo of the user's quoted question. Defaults to None so callers
    that don't plumb it through fall through untouched.

    `prior_tool_outputs` is the tuple of tool-role message contents
    executed earlier this turn (in order). Catchers compare the reply
    against actual tool output — e.g. SourceCountInflationHook flags
    a 5-item reply when the search returned 1 (harness-lynw)."""

    reply: ModelReply
    tools_ran_this_turn: bool
    tools_ran: frozenset[str] = frozenset()
    user_message: str | None = None
    prior_tool_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PostModelContext:
    reply: ModelReply


@dataclass(frozen=True)
class PreToolContext:
    """Inputs to a pre-tool hook. `user_message` is the verbatim content
    of the most recent user-role turn in the loop's working thread — the
    grounding hook uses it to verify that entity-specific arguments
    (URLs, domains) trace back to something the user actually named. Nil
    when no user turn exists yet (system-only bootstrap).

    `seen_calls` maps each (name, args-json) key the loop has executed
    this turn to the ToolResult it produced. DuplicateCallHook uses the
    stored result to re-issue prior outcomes preserving success/error
    (harness-v5w) — feeding back a generic success=True nudge made the
    model hallucinate success after a failed retry. Other hooks that
    only care about which tools ran iterate the keys.

    `prior_tool_outputs` is the tuple of tool-role message contents
    already in the working thread this turn. ArgumentGroundingHook
    widens its grounding corpus to include them (harness-jm9p), so a
    URL the model picked up from a prior search result isn't flagged
    as ungrounded just because the user never spelled the domain out."""

    call: ToolCall
    seen_calls: Mapping[tuple[str, str], ToolResult]
    user_message: str | None = None
    prior_tool_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PostToolContext:
    """Inputs to a post-tool hook. `spec` is the ToolSpec of the
    tool that just executed — the summarizer hook uses `spec.high_noise`
    to decide whether to compress, so we pass it in rather than have
    the hook hold a registry reference."""

    call: ToolCall
    result: ToolResult
    spec: ToolSpec


@dataclass(frozen=True)
class FinalizeContext:
    """Inputs to a finalize hook. `last_outcome` is the last non-Continue
    bail outcome the pipeline produced this turn, so `fabrication_fallback`
    can fire only when a fabrication-shaped bail outcome survived
    retries (truncated outcomes are exempt).

    `tools_ran` carries the set of tool names that succeeded this turn —
    used by `ungrounded_citation` to check whether any grounding tool
    (search_memory / fact_search) ran before a section-citation reply.
    `memory_block_attached` reports whether the CLI (or equivalent
    call-site) attached a retrieval memory block to the system prompt
    for this turn — the other grounding signal the citation catcher
    needs.

    `tool_outputs` carries the concatenated text of every successful
    tool-role message this turn, in execution order — the grounding
    corpus against which `table_fabrication` verifies that every
    pipe-table data row in the reply appears verbatim (harness-5uq).
    Defaults to the empty tuple so callers that don't plumb tool
    outputs through fall through untouched."""

    reply: ModelReply
    last_outcome: BailOutcome
    tools_ran: frozenset[str] = frozenset()
    memory_block_attached: bool = False
    tool_outputs: tuple[str, ...] = ()
    # Max retrieval score across every tool call this turn, or None
    # when no tool surfaced retrieval metadata. Populated by the
    # tool loop from ToolResult.hits (harness-ywp.4). Consumed by
    # LowConfidenceFallbackHook (harness-ywp.3) to refuse citations
    # that ride on weak retrieval.
    retrieval_top_score: float | None = None
    # Union of citations_grounded across every tool call this turn —
    # canonicalised (harness-ywp.5). Consumed by the low-confidence
    # fallback to distinguish citations the tool actually grounded
    # from ones the model made up.
    citations_grounded: frozenset[str] = frozenset()


class BailHook(Protocol):
    # Declared as a read-only property so frozen-dataclass implementations
    # (every concrete hook below) structurally match. A bare `name: str`
    # in the Protocol would require a writable attribute.
    @property
    def name(self) -> str: ...

    def check(self, ctx: BailContext) -> BailOutcome: ...


class PostModelHook(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, ctx: PostModelContext) -> PostModelOutcome: ...


class PreToolHook(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, ctx: PreToolContext) -> PreToolOutcome: ...


class PostToolHook(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, ctx: PostToolContext) -> PostToolOutcome: ...


class FinalizeHook(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome: ...


# ---------- bail hooks ----------


@dataclass(frozen=True)
class TruncatedHook:
    name: str = "truncated"

    def check(self, ctx: BailContext) -> BailOutcome:
        return Truncated() if ctx.reply.was_truncated else Continue()


_UNPARSEABLE_NUDGE = (
    "Your last <tool_call> block was malformed and could not be parsed. "
    "Re-emit it as a single line of valid JSON inside <tool_call>…</tool_call>: "
    '<tool_call>{"name": "...", "arguments": {...}}</tool_call>'
)


@dataclass(frozen=True)
class UnparseableHook:
    name: str = "unparseable"

    def check(self, ctx: BailContext) -> BailOutcome:
        return Nudge(_UNPARSEABLE_NUDGE) if ctx.reply.had_unparseable_call else Continue()


_TEASER_NUDGE = (
    "Your reply announced more work but didn't include any tool calls. "
    "Either call the tool now, or give the user your final answer."
)


@dataclass(frozen=True)
class TeaserHook:
    name: str = "teaser"

    def check(self, ctx: BailContext) -> BailOutcome:
        if TEASER_RE.search(ctx.reply.content.strip()):
            return Nudge(_TEASER_NUDGE)
        return Continue()


_FALSE_SUCCESS_NUDGE = (
    "Your reply claims that a file was changed / created / updated, "
    "but you did not call any tool this turn. You CANNOT modify the "
    "workspace without calling a write-tier tool (edit_file, "
    "write_file, shell). Either call the appropriate tool now, or "
    "tell the user you cannot make that change."
)


@dataclass(frozen=True)
class FalseSuccessHook:
    name: str = "false_success"

    def check(self, ctx: BailContext) -> BailOutcome:
        if _content_tools_ran(ctx):
            return Continue()
        if FALSE_SUCCESS_RE.search(ctx.reply.content):
            return Nudge(_FALSE_SUCCESS_NUDGE)
        return Continue()


_META_CONFIRM_NUDGE = (
    "Do NOT ask the user to confirm in chat. The user's previous "
    "message IS the instruction — call the tool right now. Write-tier "
    "tools have their own approve/decline UX at the tool layer; "
    "re-asking in chat just wastes a round."
)


@dataclass(frozen=True)
class MetaConfirmHook:
    name: str = "meta_confirm"

    def check(self, ctx: BailContext) -> BailOutcome:
        if _content_tools_ran(ctx):
            return Continue()
        if META_CONFIRM_RE.search(ctx.reply.content):
            return Nudge(_META_CONFIRM_NUDGE)
        return Continue()


_RAW_RESULTS_DUMP_NUDGE = (
    "Your reply repeats the tool result without producing the synthesis "
    "the user asked for (rank / prioritize / compare / summarize / group / "
    "score / categorize / order / sort / filter). The tool output is "
    "input, not the answer — continue the turn and produce the requested "
    "output (grouped, ranked, or scored as the prompt specified) using "
    "the data you already gathered. Do NOT call the tool again."
)


# User-prompt synthesis verbs. A match flips the catcher armed — without
# one, raw-dumping the tool output is the user's actual request and the
# catcher must stay silent (see synthesis_completion.yaml negative case).
SYNTHESIS_VERB_RE = re.compile(
    r"\b(?:"
    r"rank(?:ing|ed|s)?"
    r"|prioritiz(?:e|ing|ed|es)"
    r"|compar(?:e|ing|ed|es)"
    r"|summariz(?:e|ing|ed|es)"
    r"|group(?:ing|ed|s)?"
    r"|scor(?:e|ing|ed|es)"
    r"|categoriz(?:e|ing|ed|es)"
    r"|order(?:ing|ed|s)?"
    r"|sort(?:ing|ed|s)?"
    r"|filter(?:ing|ed|s)?"
    r")\b",
    re.IGNORECASE,
)


# Ordering markers that signal the model imposed synthesis structure
# on top of the data. Numbered lists alone don't count — raw tool
# outputs are often numbered lists already. We look for:
#   - pipe-table rows (two+ pipes per line)
#   - severity / priority grouping headings ("High severity:", "Top
#     priority:")
#   - "by X:" headings ("by region:", "by date:")
#   - explicit "ranked by" / "prioritized by" / "grouped by" phrases
RAW_RESULTS_DUMP_ORDERING_RE = re.compile(
    r"(?:"
    r"\|[^|\n]+\|[^|\n]+\|"
    r"|"
    r"^\s*(?:high|medium|low|critical|severe|major|minor|"
    r"top|bottom|most|least)\s+"
    r"(?:severity|priority|importance|critical(?:ity)?|urgency)\s*[:.]"
    r"|"
    r"^\s*by\s+\w+\s*[:.]"
    r"|"
    r"\b(?:ranked|prioritized|grouped|sorted|categorized|scored|ordered)\s+by\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


# Jaccard threshold above which the reply counts as "mostly the tool
# output." Tuned biased toward false-negatives (harness-s451 design):
# only fire when overlap is unambiguous; missed catches just land the
# same failure we already have, false positives force extra rounds on
# legitimate single-step completions.
_RAW_RESULTS_DUMP_OVERLAP_THRESHOLD = 0.5


def _content_token_set(text: str) -> frozenset[str]:
    """Lowercased alphanumeric tokens of length >= 3 from `text`.
    Short tokens (a, is, to, the) dominate Jaccard otherwise and
    inflate overlap on near-disjoint replies."""
    return frozenset(t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3)


@dataclass(frozen=True)
class RawResultsDumpHook:
    """Catch the synthesis-completion failure mode (harness-akpx /
    s451): a content-producing tool ran, the user asked for synthesis
    (rank / prioritize / compare / summarize / etc.), and the reply
    is mostly a verbatim regurgitation of the tool output with no
    ordering / grouping / scoring structure imposed on top.

    All four signals must hold to fire:
      1. A content tool succeeded this turn (`_content_tools_ran`).
         Opposite gate from the fabrication catchers (which fire on
         no-tool turns) — RawResultsDump targets real-but-unsynthesized
         output, not invented content.
      2. The user message carries a synthesis verb (rank / prioritize /
         compare / summarize / group / score / categorize / order /
         sort / filter).
      3. Reply tokens overlap with the most recent prior tool output
         above `_RAW_RESULTS_DUMP_OVERLAP_THRESHOLD` (Jaccard).
      4. The reply lacks ordering markers — pipe tables, severity /
         priority headings, "by X:" groupings, or explicit "ranked
         by" phrases. The presence of any marker disarms the catcher;
         the model imposed structure, that's the synthesis we wanted.

    Defensive layer behind the synthesis-continue preamble nudge
    (harness-b7yd). The nudge is preventive (steer the model away
    from raw-dump in the first place); this hook is the recovery
    path when the model emits the dump anyway."""

    name: str = "raw_results_dump"

    def check(self, ctx: BailContext) -> BailOutcome:
        if not _content_tools_ran(ctx):
            return Continue()
        if not ctx.user_message:
            return Continue()
        if not SYNTHESIS_VERB_RE.search(ctx.user_message):
            return Continue()
        if not ctx.prior_tool_outputs:
            return Continue()
        if RAW_RESULTS_DUMP_ORDERING_RE.search(ctx.reply.content):
            return Continue()
        reply_tokens = _content_token_set(ctx.reply.content)
        if not reply_tokens:
            return Continue()
        # Compare against the most recent tool output — the one the
        # model is most likely echoing. Older outputs may share
        # incidental tokens but aren't the dump source.
        tool_tokens = _content_token_set(ctx.prior_tool_outputs[-1])
        if not tool_tokens:
            return Continue()
        union = len(reply_tokens | tool_tokens)
        if union == 0:
            return Continue()
        overlap = len(reply_tokens & tool_tokens) / union
        if overlap < _RAW_RESULTS_DUMP_OVERLAP_THRESHOLD:
            return Continue()
        return Nudge(_RAW_RESULTS_DUMP_NUDGE)


_FABRICATED_SEARCH_NUDGE = (
    "Your reply looks like fabricated tool output (search results / "
    "placeholder URLs / 'here are the results'). You did NOT call any "
    "tool this turn — you cannot know results without actually calling "
    "search_web / fetch_url / read_file. Call the appropriate tool now, "
    "or tell the user you cannot answer without live data."
)


_WEB_FETCH_TOOLS: frozenset[str] = frozenset({"search_web", "fetch_url"})

# Meta-tools — plumbing rather than content-producing. The discovery
# loop (tool_search → load_tool → real tool call) means a meta-tool
# can succeed BEFORE the content tool runs; treating that as 'a tool
# ran this turn' disarms fabrication catchers prematurely. Mark's
# core_minimal repro (harness-q7kn): load_tool's success let the model
# emit a fabricated numbered list of weather sites before search_web
# ever ran. Hooks that check `tools_ran_this_turn` to disarm should
# use `_content_tools_ran(ctx)` instead so meta-tool execution stays
# transparent to the fabrication gate.
_META_TOOLS: frozenset[str] = frozenset(
    {"tool_search", "load_tool", "introspect", "spawn_subagent"}
)


def _content_tools_ran(ctx: BailContext) -> bool:
    """True iff a content-producing (non-meta) tool succeeded this
    turn. Meta-tools are excluded — they're plumbing for the discovery
    loop, not output the model can summarize."""
    return bool(ctx.tools_ran - _META_TOOLS)


@dataclass(frozen=True)
class FabricatedSearchHook:
    """Catches two shapes of web-search fabrication.

    1. Result-list patterns (FABRICATED_SEARCH_RE): "here are the
       results", placeholder-domain URLs, quoted-snippet numbered
       lists. Gated on `tools_ran_this_turn` — ANY tool running
       disarms the catcher because the wrap-up may legitimately
       summarize that tool's output.
    2. Past-tense web-claim (FABRICATED_WEB_CLAIM_RE): "I've
       searched the web", "after searching online". Gated more
       narrowly on `tools_ran` — only a real search_web or
       fetch_url disarms it. Rationale (harness-78z): the model
       claims to have done a web search; only an actual web call
       legitimizes that claim. A search_memory miss followed by
       "I've searched the web" narration is still fabrication."""

    name: str = "fabricated_search"

    def check(self, ctx: BailContext) -> BailOutcome:
        web_ran = bool(ctx.tools_ran & _WEB_FETCH_TOOLS)
        if not web_ran and FABRICATED_WEB_CLAIM_RE.search(ctx.reply.content):
            return Nudge(_FABRICATED_SEARCH_NUDGE)
        if _content_tools_ran(ctx):
            return Continue()
        if FABRICATED_SEARCH_RE.search(ctx.reply.content):
            return Nudge(_FABRICATED_SEARCH_NUDGE)
        if FABRICATED_URL_LIST_RE.search(ctx.reply.content):
            return Nudge(_FABRICATED_SEARCH_NUDGE)
        return Continue()


_FABRICATED_ITEMIZATION_NUDGE = (
    "Your reply looks like a regenerated summary list ('here are the "
    "articles/stories/headlines…') but you did NOT call any tool this "
    "turn. When the user asks for more detail on a specific item from "
    "an earlier tool result, the ONLY correct response is to call "
    "`fetch_url` on THAT item's URL from the prior tool output. Do NOT "
    "paraphrase, re-list, or summarize from context. If you can't find "
    "the URL for the item the user named, tell them you need it pasted."
)


@dataclass(frozen=True)
class FabricatedItemizationHook:
    """Catches the follow-up-fabrication shape: user asks for more
    detail on item N from an earlier fetch, small model skips the
    index→URL lookup and regenerates a fake summary list. Gated on
    `_content_tools_ran(ctx)=False` so legitimate wrap-up lists after
    a real content tool ran never trip — but meta-tool execution
    (tool_search, load_tool) doesn't disarm us (harness-f5x, q7kn)."""

    name: str = "fabricated_itemization"

    def check(self, ctx: BailContext) -> BailOutcome:
        if _content_tools_ran(ctx):
            return Continue()
        if FABRICATED_ITEMIZATION_RE.search(ctx.reply.content):
            return Nudge(_FABRICATED_ITEMIZATION_NUDGE)
        return Continue()


_THIN_SOURCE_FABRICATION_NUDGE = (
    "The page you fetched returned thin / empty content for the "
    "data you're claiming. The body contained almost no numeric "
    "values, yet your reply emits specific numbers with units "
    "(temperatures, percentages, distances). That's fabrication. "
    "Reach for a different tool / source BEFORE giving up: "
    "(a) call search_web with a query specific to the data you "
    "need (e.g. 'Nairobi 4-day forecast wttr.in', 'population by "
    "gender KNBS census') to find a better source URL; "
    "(b) call fetch_url against a non-JS-rendered alternative — "
    "a real API endpoint (weather.gov forecast.json, wttr.in, "
    "aviationweather.gov, an official stats bureau JSON / CSV) "
    "usually carries the data the SPA page hid behind JavaScript; "
    "(c) only after both (a) and (b) have genuinely failed, tell "
    "the user plainly that you couldn't extract the data and what "
    "URL or paste they could provide instead. "
    "Do NOT invent specific numbers from a body that didn't carry "
    "them."
)


# Numeric-with-unit tokens — temperatures, percentages, speed,
# distance, currency. Counted on BOTH the reply (claims) and the most
# recent prior tool output (anchor). The detector fires when the
# reply has many and the body has few — i.e. the model produced
# specifics the source didn't carry.
THIN_SOURCE_NUMERIC_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:"
    r"°\s*[FC]\b"
    r"|"
    r"°(?=\s|$|[.,;)])"
    r"|"
    r"degrees?\b"
    r"|"
    r"%"
    r"|"
    r"(?:mph|kph|km/?h|knots?|mps)\b"
    r"|"
    r"(?:miles?|kilometers?|km|mi|ft|feet|m|meters?|inches?|in|mm|cm)\b"
    r"|"
    r"(?:hpa|mb|mbar|inhg|millibars?)\b"
    r")",
    re.IGNORECASE,
)


# Currency runs as a leading-symbol pattern, separately from the
# trailing-unit alternation above. Captured here so a price-claim
# fabrication off a thin body fires too.
THIN_SOURCE_CURRENCY_RE = re.compile(r"[$€£¥]\s*\d+(?:\.\d+)?")


def _thin_source_numeric_count(text: str) -> int:
    """Total numeric-with-unit tokens — temperatures + percentages +
    speeds + distances + pressures + currency. Sums the trailing-unit
    matches and the leading-symbol currency matches."""
    return len(THIN_SOURCE_NUMERIC_RE.findall(text)) + len(THIN_SOURCE_CURRENCY_RE.findall(text))


# Thresholds tuned biased toward false-negatives (harness-xszc). A
# real forecast / data page returns dozens of numeric-with-unit
# tokens; a JS-rendered SPA with no content extraction yields zero
# or a handful. The reply floor (>= 3) filters out the legitimate
# "I see one data point" case where overlap is incidental.
_THIN_SOURCE_BODY_FLOOR = 3
_THIN_SOURCE_REPLY_FLOOR = 3


@dataclass(frozen=True)
class ThinSourceFabricationHook:
    """Catch numeric-claim fabrication off a thin tool body
    (harness-xszc). Session repro 2026-05-19: model called fetch_url
    on weather.com (JS-rendered SPA), got back a body with no
    forecast data, fabricated four days of temperatures + humidity
    that drifted across three retry rounds.

    Existing fabrication catchers don't close this:
      - fabricated_search disarms once a real web tool ran.
      - fabricated_itemization gates on `_content_tools_ran=False`.
      - numeric_fabrication needs the body to CONTAIN the numbers
        for cross-row comparison; a thin body has no anchors so the
        catcher stays silent.
      - raw_results_dump gates on HIGH overlap — opposite of this
        catcher's case.

    Trigger conditions (ALL must hold):
      1. prior_tool_outputs non-empty (a tool ran and produced
         output we can inspect).
      2. Most recent tool output carries fewer than
         _THIN_SOURCE_BODY_FLOOR numeric-with-unit tokens — the
         body can't support specific numeric claims.
      3. Reply emits at least _THIN_SOURCE_REPLY_FLOOR specific
         numeric claims with units — short replies that don't claim
         numbers aren't fabrication candidates.

    Silent cases (by design):
      - Body has the numbers (≥ _THIN_SOURCE_BODY_FLOOR) →
        defer to numeric_fabrication / table_fabrication for the
        cross-row drift case.
      - Reply has no numeric claims → fabrication isn't the failure
        shape; other catchers handle prose drift.
      - No prior tool output → the no-tool case is fabricated_search
        / false_success / teaser territory.

    Lives in the universal bail roster — the failure mode is
    character-agnostic. Position after fabricated_itemization so
    the no-content-tool fabrication shapes get first pass."""

    name: str = "thin_source_fabrication"

    def check(self, ctx: BailContext) -> BailOutcome:
        if not ctx.prior_tool_outputs:
            return Continue()
        body = ctx.prior_tool_outputs[-1]
        if _thin_source_numeric_count(body) >= _THIN_SOURCE_BODY_FLOOR:
            return Continue()
        if _thin_source_numeric_count(ctx.reply.content) < _THIN_SOURCE_REPLY_FLOOR:
            return Continue()
        return Nudge(_THIN_SOURCE_FABRICATION_NUDGE)


_INCOMPLETE_MULTIPART_NUDGE = (
    "The user's request had multiple distinct asks (find X AND find Y / "
    "two questions in one turn). Your reply addressed part of it but "
    "signals you couldn't find the rest — phrases like 'we would need', "
    "'the source does not', 'not explicitly provided', 'for a precise "
    "breakdown'. Don't stop here. Issue ANOTHER tool call (search_web "
    "with a different query specific to the missing part, fetch_url "
    "against a different source, search_memory for prior context) "
    "BEFORE terminating. Only return a partial answer if multiple "
    "genuine attempts at the missing sub-ask have all failed."
)


# Asks-counter: imperative verbs at sentence start OR interrogatives, each
# preceded by a sentence boundary / conjunction / 'then' marker. Two or
# more matches in the user message = multi-part. Tuned to count the
# common multi-ask shapes ("Find X. What is Y?", "Find X and what's Y?",
# "X then list Y") without firing on every single-imperative prompt.
_MULTIPART_ASK_RE = re.compile(
    r"(?:^|[.!?]\s+|\band\s+|\bthen\s+|\balso\s+|\bplus\s+|;\s*)"
    r"(?:what|who|whom|when|where|why|how|which|whose|"
    r"find|list|show|tell|get|search|lookup|look\s+up|"
    r"calculate|compute|compare|describe|give|name|explain|"
    r"identify|determine|estimate|measure|rank|prioritize)\b",
    re.IGNORECASE,
)


# Give-up phrasing — the model signalling 'I tried, source didn't have
# it' in a way that should trigger a retry on the missing sub-ask, not
# a turn-ending refusal. Covers the dominant shapes from the 2026-05-19
# Nairobi gender-breakdown repro plus close cousins. Conservative: every
# phrase implies the model EXPLICITLY acknowledged it couldn't fulfill
# part of the request.
_INCOMPLETE_MULTIPART_GIVE_UP_RE = re.compile(
    r"(?:"
    r"\b(?:we|you|i)\s+would\s+need\b"
    r"|\bfor\s+a\s+(?:precise|complete|full|specific|more\s+detailed|detailed)\s+"
    r"(?:breakdown|answer|figure|number|view|picture|estimate|report)\b"
    r"|\b(?:the\s+)?source\s+(?:does\s+not|doesn't|does\s+n't)\b"
    r"|\b(?:does\s+not|doesn't|isn't|is\s+not|do\s+not|don't)\s+"
    r"(?:mention|provide|specify|include|cover|list|state|contain)\b"
    r"|\bnot\s+(?:explicitly\s+)?(?:provided|mentioned|specified|available|"
    r"listed|stated|covered|included|reported)\b"
    r"|\bwould\s+need\s+to\s+(?:look|consult|find|search|check|refer|access)\b"
    r"|\b(?:isn't|is\s+not)\s+(?:specified|provided|mentioned|available|"
    r"included|listed|stated)\b"
    r")",
    re.IGNORECASE,
)


def _multipart_ask_count(user_message: str) -> int:
    return len(_MULTIPART_ASK_RE.findall(user_message))


@dataclass(frozen=True)
class IncompleteMultipartHook:
    """Catch the multi-part task abandonment shape (harness-111v).
    Session repro 2026-05-19: user asked 'Find the population count
    of Nairobi, Kenya. What percent are female vs male?' Agent
    searched + answered the population (5,545,000) but gave up on
    the gender breakdown — 'the source does not mention any specific
    data on the percentage of males and females. For a precise
    breakdown, we would need to look at...'. No second search for the
    missing sub-ask.

    Distinct from raw_results_dump (which targets synthesis VERBS:
    rank/prioritize/compare/summarize). Here the user asked two
    parallel factual questions; the agent's tool budget allowed both
    but the agent treated 'one source didn't have it' as 'I'm done'.

    Trigger conditions (ALL must hold):
      1. A content tool ran this turn (_content_tools_ran). Pure
         meta-tool turns can't have done real lookups yet.
      2. User message contains at least two distinct 'asks' per
         _MULTIPART_ASK_RE — imperatives + interrogatives separated
         by sentence boundaries / 'and' / 'then'. Single-ask prompts
         that the agent legitimately couldn't answer don't trip.
      3. Reply contains give-up phrasing per
         _INCOMPLETE_MULTIPART_GIVE_UP_RE — explicit acknowledgement
         the model couldn't fulfill some part of the request. Silent
         when the model just answers cleanly (no give-up = nothing
         to nudge about).

    Position in the bail pipeline: AFTER the fabrication-shape
    catchers (raw_results_dump / fabricated_search /
    fabricated_itemization / thin_source_fabrication) — those address
    what the reply SAYS that's false; this addresses what the reply
    OMITS that it shouldn't have. Universal, not opt-in: the failure
    shape is character-agnostic."""

    name: str = "incomplete_multipart"

    def check(self, ctx: BailContext) -> BailOutcome:
        if not _content_tools_ran(ctx):
            return Continue()
        if not ctx.user_message:
            return Continue()
        if _multipart_ask_count(ctx.user_message) < 2:
            return Continue()
        if not _INCOMPLETE_MULTIPART_GIVE_UP_RE.search(ctx.reply.content):
            return Continue()
        return Nudge(_INCOMPLETE_MULTIPART_NUDGE)


_AB_FABRICATION_NUDGE = (
    "Your reply looks like fabricated tool output (ab_ops capture "
    "receipt / tiered plan / fake scope abbreviation). You did NOT "
    "call any tool this turn — you cannot produce a capture receipt "
    "or plan without actually calling `capture` / `plan`. Call the "
    "appropriate tool now, or tell the user plainly that you cannot."
)


@dataclass(frozen=True)
class AbFabricationHook:
    name: str = "ab_fabrication"

    def check(self, ctx: BailContext) -> BailOutcome:
        if _content_tools_ran(ctx):
            return Continue()
        if looks_like_ab_fabrication(ctx.reply.content):
            return Nudge(_AB_FABRICATION_NUDGE)
        return Continue()


_TOOL_INTENT_NUDGE = (
    "Your reply said you would do something ('I will search…', "
    "'let me check…', etc.) but you did NOT emit a tool_call. "
    "Stated intent is not action. To call a tool, emit EXACTLY "
    "this block (no wrapping text, no commentary) as part of "
    "your next reply:\n"
    '<tool_call>{"name": "<tool_name>", "arguments": {<args>}}</tool_call>\n'
    "Example for search_web:\n"
    '<tool_call>{"name": "search_web", "arguments": '
    '{"query": "ahwatukee bbq"}}</tool_call>\n'
    "If you cannot figure out the right tool/args, say so plainly."
)


@dataclass(frozen=True)
class ToolIntentHook:
    name: str = "tool_intent"

    def check(self, ctx: BailContext) -> BailOutcome:
        if _content_tools_ran(ctx):
            return Continue()
        if TOOL_INTENT_RE.search(ctx.reply.content):
            return Nudge(_TOOL_INTENT_NUDGE)
        return Continue()


# Replaced by `CitationGrammar.document_reference` per character
# (harness-jaqe) — `_ORDER_REFERENCE_RE` removed; the
# MissingCitationHook now reads its document-reference regex from the
# active character's grammar.


# Secondary in-scope signal: JO-normative phraseology. The order
# formats controller phraseology blocks in ALL CAPS (e.g.
# 'RADAR SERVICE TERMINATED', 'SQUAWK VFR'), but airton_c1's replies
# routinely narrate these in mixed case ('the correct phraseology is
# "Radar service terminated, squawk VFR"'). The regex is
# case-insensitive so both forms trigger — the phrase tokens are
# specific enough to ATC that a casual narrative use outside an
# airton_c1 reply is vanishingly unlikely. Plus meta-phraseology
# triggers ('correct phraseology', 'proper phraseology',
# 'phraseology for terminating/clearing/...') so a reply that DOESN'T
# quote the exact JO line but still teaches normative phraseology
# trips the in-scope check.
_JO_PHRASEOLOGY_MARKERS_RE = re.compile(
    r"\b(?:"
    r"SQUAWK\s+(?:VFR|IDENT|STANDBY|STOP|MAYDAY|\d{4}|"
    r"(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|ZERO)"
    r"(?:\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|ZERO))*)"
    r"|RADAR\s+(?:SERVICE\s+TERMINATED|CONTACT(?:\s+LOST)?)"
    r"|CLEARED\s+(?:TO|FOR)\s+[A-Z]"
    r"|CLEARED\s+FOR\s+(?:TAKEOFF|LANDING|THE\s+APPROACH)"
    r"|CONTACT\s+(?:TOWER|GROUND|DEPARTURE|APPROACH|CENTER|CLEARANCE)"
    r"|FREQUENCY\s+CHANGE\s+APPROVED"
    r"|CHANGE\s+TO\s+ADVISORY"
    r"|MAINTAIN\s+(?:VFR|ALTITUDE|FL|FLIGHT\s+LEVEL)"
    r"|HOLD\s+SHORT(?:\s+OF)?"
    r"|(?:DESCEND|CLIMB)\s+(?:AND\s+MAINTAIN|TO)"
    r"|TRAFFIC(?:\s+ALERT)?"
    r"|PROCEED\s+DIRECT"
    r"|CROSS\s+[A-Z]+\s+AT"
    r"|TURN\s+(?:LEFT|RIGHT)\s+HEADING"
    # Meta-phraseology signals (narrative references to normative
    # phraseology, not direct quotes):
    r"|correct\s+phraseology"
    r"|proper\s+phraseology"
    r"|phraseology\s+(?:for|is|should\s+be|would\s+be)"
    r"|the\s+phrase(?:ology)?\s+\""
    r")\b",
    re.IGNORECASE,
)


# Citation shapes we accept as 'the reply did cite a section':
#   §N-N-N                (primary anchor — JO chapter-section-paragraph)
#   TBL N-N-N / Table N-N-N (tables are always section-scoped; accept
#                            as a standalone cite when no § is present)
#   FIG N-N-N / Figure N-N-N (same rationale as tables)
#   Chapter N §N-N-N      (constitution's preferred form — §-anchor is
#                            what matches here, Chapter N is prose)
# Tolerant of the hyphen / en-dash / unicode-minus variants in the
# corpus. Reuses UNGROUNDED_SECTION_CITATION_RE's shape for the §-form
# so the two hooks can't disagree about what counts as a citation.
#
# This fallback covers the JO/AIM hyphen-form. Characters whose corpora
# use other shapes (e.g. airton_c_tfr citing 14 CFR §91.141 in dot-
# form) are handled by `_reply_has_citation`, which also consults the
# character's `CitationGrammar.surface_patterns` so each persona's
# native citation shape counts as 'cited' for the missing_citation
# check.
_CITATION_PRESENT_RE = re.compile(
    # §N-N-N (primary — chapter-section-paragraph) or §N-N (broader
    # chapter-section reference, e.g. '§9-6' for the entire Unmanned
    # Free Balloons section). Both count as a citation for the
    # missing_citation check even though UngroundedCitationHook
    # stays strict at 3-segment to avoid matching prices / anchors.
    r"§\s*\d+[-–−]\d+(?:[-–−]\d+)?"  # noqa: RUF001 — dash variants load-bearing
    r"|\b(?:TBL|Table|FIG|Figure)\s+\d+[-–−]\d+[-–−]\d+\b",  # noqa: RUF001
    re.IGNORECASE,
)


def _reply_has_citation(content: str, grammar: CitationGrammar | None) -> bool:
    """Return True when the reply contains a recognizable citation.

    Resolution order:
      1. The character's `surface_patterns` (when grammar is supplied)
         — these are the authoritative definition of what counts as a
         citation for this persona.
      2. The global JO/AIM hyphen-form fallback (`_CITATION_PRESENT_RE`)
         — applied only when grammar is None OR when the grammar opts
         in via `accept_faa_bare_anchor=True`. FAA-flavored characters
         (airton_c, airton_c1, airton_c_tfr) opt in so the model can
         use bare `§3-10-3` without the `JO 7110.65` prefix. Non-FAA
         characters (airton_f and future legal / RFC personas) leave
         the flag default-false — for them, the FAA hyphen-form is a
         false-positive (smoke 2026-05-15 airton_f repro: model wrote
         `§1-5 explicitly disclaims security`; without the gate, the
         global regex accepted it and let the reply through uncited)."""
    if grammar is not None:
        if any(pattern.search(content) for pattern in grammar.surface_patterns):
            return True
        if not grammar.accept_faa_bare_anchor:
            return False
    return bool(_CITATION_PRESENT_RE.search(content))


# A reply that's asking the user to clarify between variants isn't
# making a substantive claim — it's a question back, not an answer.
# No citation expected. MissingCitationHook and (later) any other
# 'you must cite' gate should exempt replies matching this shape so
# a valid clarifying question doesn't get nudged into a pointless
# retry loop (session 2026-04-24 repro: ambiguous_context fires;
# model asks 'manned or unmanned?'; missing_citation then fires on
# the clarifying question because it has 'JO 7110.65' but no §).
_CLARIFYING_QUESTION_RE = re.compile(
    r"(?:"
    r"could\s+you\s+(?:please\s+)?(?:clarify|specify)"
    r"|can\s+you\s+(?:please\s+)?(?:clarify|specify)"
    r"|do\s+you\s+mean"
    r"|are\s+you\s+(?:asking|referring)"
    r"|which\s+(?:one|variant|type|kind)\s+(?:of\b|do\s+you|are\s+you)"
    r"|which\s+type\s+of"
    r"|please\s+(?:specify|clarify)"
    r"|before\s+I\s+answer"
    r"|to\s+clarify[,:]"
    r")",
    re.IGNORECASE,
)


_MISSING_CITATION_NUDGE_TEMPLATE = (
    "Your reply references {document_name} but does not include a "
    "specific section citation (e.g. `{example_anchor}`). The character's "
    "constitution requires a citation on every substantive answer, and "
    "the tool output you just read contains explicit section anchors. "
    "Re-answer with the citation inline — 'per {document_name} <§>' — or, "
    "if the question is out of scope for {document_name}, say so plainly "
    "without the reference."
)


# Number-word -> integer lookup for count-claim parsing. Only cover
# 2..12 — higher numbers are rare in ATC prose and mostly appear as
# digits when they do. 'one' is excluded on purpose: 'one of the ...'
# is a different shape than a count claim and would false-positive
# on phrases like 'only one reason'.
_NUMBER_WORDS: dict[str, int] = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


# Matches a count claim followed by a list intro. Intended to fire on
# reply patterns like 'The four specific primary purposes of ATC are:',
# 'There are three reasons:', 'four main options include'. The span
# between the count word and the list-intro verb is permissive
# (non-newline chars up to 120 chars) so adjective runs with
# punctuation ('of Air Traffic Control (ATC)') match cleanly. Trailing
# 'are' / 'include' / 'comprise' pins it to a list intro — avoids
# matching narrative prose that happens to contain a number elsewhere.
_COUNT_CLAIM_RE = re.compile(
    # Digit-count is capped at 1-2 chars (≤ 99). Real chat-reply count
    # claims never exceed two digits — the `_NUMBER_WORDS` table caps
    # at 12 for a reason — and the cap also prevents phone numbers,
    # frequencies, timestamps, and other multi-digit runs from being
    # matched as a count claim. Observed 2026-05-15 with airton_c_tfr:
    # NOTAM echoed '406-444-4242 OR FREQ 123.725 JERICHO CREEK IS IN
    # CHARGE', and the old `\d+` matched '4242 or … is' as a count
    # claim of 4242 items, then fired list_count_mismatch against the
    # reply's actual 5-bullet geometry list. The word-boundary on the
    # digit run means '4242' (4 contiguous digits) is bounded only at
    # start/end, so `\b\d{1,2}\b` can't match an interior pair.
    r"\b(\d{1,2}|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
    # Require whitespace then a letter after the count word. Excludes
    # section numbers ('7110.65', '2-1-1'), list item markers ('1.'),
    # dates, phone numbers — anything where the digit is immediately
    # followed by punctuation or another digit.
    r"\s+[A-Za-z]"
    r"[^\n]{0,120}?"
    r"\b(?:are|is|include|comprise|consist\s+of)\b(?:\s*(?::|as\s+follows))?",
    re.IGNORECASE,
)


# Tokens that, when present inside a _COUNT_CLAIM_RE match span,
# disqualify the match as a real count claim. Legal/document
# citations like '14 CFR §99.7, ... operations are prohibited' fit
# the count-claim shape but '14' is a section prefix, not an item
# count. Observed 2026-05-15 with airton_c_tfr decoding a §99.7 UAS
# NOTAM — the post-cite gap to 'are prohibited' was inside the 120-
# char window. Adding § / CFR / USC / U.S.C. / AIM / JO / AC tokens
# as exclusions filters citation patterns without blocking real
# count claims like 'the 14 sections are listed below.'
_CITATION_TOKEN_IN_CLAIM_RE = re.compile(
    r"\b(?:CFR|USC|U\.S\.C\.|AIM|JO|AC)\b|§",
    re.IGNORECASE,
)


# Subset-language markers (harness-9gzk). When a count claim's match
# span contains one of these, the count is talking about a population
# total and the enumerated list is a labelled SUBSET — not the count
# itself. Example: 'Kenya is home to 50 genera. Among these, some
# species are unique... <list of 4 species>'. The 50 is the population;
# the 4 is a labelled subset ('species valuable for cut-flower
# production'). Without this carve-out, the catcher reads '50 genera'
# vs '4 items listed' as a self-inconsistency, which is the wrong
# reading.
#
# Markers are conservative: each phrase explicitly names a labelled
# subset (among-these / examples / partial / representative / for-X-
# purpose) rather than promising an exhaustive enumeration.
_SUBSET_MARKER_RE = re.compile(
    r"\b("
    r"among\s+(?:these|them|those|which)"
    r"|some\s+of\s+(?:these|them|those|which|the)"
    r"|a\s+few\s+(?:species|examples?|notable|of\s+(?:these|them|those))"
    r"|several\s+of\s+(?:these|them|those)"
    r"|several\s+(?:notable|examples?|species)"
    r"|for\s+(?:cut-flower|ornamental|specific|certain|particular|special|"
    r"medicinal|culinary|industrial|commercial)\s+(?:use|purpose|production|"
    r"value|interest|cultivation)"
    r"|including"
    r"|such\s+as"
    r"|examples?\s+include"
    r"|notable\s+(?:examples?|species)"
    r"|partial\s+list"
    r"|representative\s+(?:examples?|sample|selection)"
    r")\b",
    re.IGNORECASE,
)


# Matches the start of a numbered or bulleted list item. Requires
# punctuation after digits (`.`, `)`, or `]`) so stray numbers in
# prose don't count. Bullet characters (`-`, `*`, `•`) are standard
# markdown / CommonMark list markers. Multiline flag so we scan every
# line of the reply.
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:\d+[.\)\]]|[-*•])\s+\S",
    re.MULTILINE,
)


_LIST_COUNT_MISMATCH_NUDGE = (
    "[count mismatch — your reply stated a count of items (e.g. 'the "
    "four ... are') but the enumerated list has a different number of "
    "items. Re-answer with a matching count: either add the missing "
    "items if the source supports them, or restate the count to match "
    "what you actually listed. If the question asked for more items "
    "than the source lists, say so plainly — 'The source lists N, not "
    "what was asked for' — rather than padding or agreeing with the "
    "wrong count.]"
)


def _parse_count_word(token: str) -> int | None:
    """Turn '4' / 'four' into 4. Returns None on unparseable tokens
    so the caller can treat them as 'no count claim'."""
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


# ATC-domain vocabulary markers. Presence in a reply signals the
# content is ATC-focused — used by ScopeRedirectHook to confirm the
# model is discussing JO 7110.65 topics. Not used to decide the
# user's intent (too many legit in-scope lay questions lack explicit
# aviation acronyms — 'What document is required...', 'What is the
# purpose of the order?', etc.). Generic single words like 'flight'
# / 'plane' are omitted so travel/hobby prose doesn't false-positive.
_AVIATION_VOCAB_RE = re.compile(
    r"\b(?:"
    r"squawk|transponder|phraseology|radar|ATC|IFR|VFR|NAVAID|RNAV|"
    r"clearance|runway|taxiway|approach|departure|altitude|heading|"
    r"beacon|ADS[-\s]?B|pilot|controller|aircraft|airspace|vector|"
    r"separation|hijack|emergency|NORDO|MVA|MEA|hold\s+short|"
    r"flight\s+level|JO\s*7110|7110\.65|FAA\s+Order|"
    r"Air\s+Traffic\s+Control|RBN|VOR(?:TAC)?|TACAN|DME|ILS"
    r")\b",
    re.IGNORECASE,
)


# Strong out-of-scope markers. Presence in a user message is the
# positive trigger for ScopeRedirectHook — clearly non-aviation
# vocabulary (biology, programming, cooking, chat/greeting,
# supernatural, household, joke-meta) that no in-scope JO 7110.65
# question should contain. Narrow on purpose: prefer false negatives
# (missing an out-of-scope case) over false positives (nudging a
# valid in-scope lay query). Grow this list as new false-negative
# cases surface — the rule of discipline is 'only add terms that are
# almost never in a JO 7110.65 question.'
_CLEARLY_NON_AVIATION_RE = re.compile(
    r"\b(?:"
    # Biology / animals (farm + common pets). 'bird' is intentionally
    # NOT here: bird strikes / bird hazards / bird activity / bird
    # ingestion are real AIM §7-5 + 14 CFR §25.631/29.631/33.76/35.36
    # + JO 7110.65 §2-1-23 topics, and any thorough reply names
    # 'bird' too many times in too many shapes to mask cleanly. Joke
    # shapes ('why did the bird cross the road', 'if a bird, a fish,
    # and a horse') are caught structurally by `_JOKE_FRAME_RE`, not
    # by the bird token, so dropping it costs no joke-detection
    # power.
    r"rooster|chicken|hen|egg|eggs|cow|pig|horse|goat|sheep|dog|cat|fish|"
    r"plant|tree|flower|fungus|bacteria|virus|cell|"
    # More animals — wildlife / exotic, common in jokes & riddles
    r"turtle|tortoise|rabbit|bunny|mouse|rat|snake|lizard|frog|toad|"
    r"elephant|monkey|giraffe|lion|tiger|bear|wolf|fox|deer|moose|"
    r"raccoon|squirrel|hamster|kangaroo|penguin|whale|dolphin|shark|"
    r"octopus|spider|ant|bee|butterfly|"
    # Supernatural / fantasy — clear non-aviation joke vocabulary
    r"ghost|zombie|vampire|werewolf|dragon|unicorn|fairy|demon|angel|"
    r"elf|wizard|witch|troll|goblin|mermaid|santa|"
    # Household items / furniture / appliances
    r"refrigerator|fridge|microwave|oven|toaster|washing\s+machine|"
    r"dishwasher|couch|sofa|mattress|pillow|blanket|lamp|"
    # Food & cooking
    r"recipe|cook|bake|ingredient|breakfast|lunch|dinner|meal|"
    r"pizza|burger|sandwich|pasta|salad|soup|cake|cookie|"
    # Programming & software (not ATC software)
    r"python|javascript|typescript|ruby|rust|golang|react|vue|django|flask|"
    r"sql|bash|shell\s+script|variable|compile|debug|commit|git(?:hub)?|"
    # Consumer tech brands / devices. Bare 'mac', 'pc', 'computer'
    # are intentionally omitted — they collide with aviation
    # acronyms (MAC = Military Airlift Command / mean aerodynamic
    # chord; flight-data computer). These terms don't collide.
    r"iphone|ipad|ipod|android\s+phone|smartphone|laptop|"
    # Math / science (unrelated to ATC domain). 'geometry' is
    # intentionally omitted — it collides with TFR / chart / radar
    # vocabulary (e.g. airton_c_tfr's directive "Surface the geometry
    # (center, radius, floor, ceiling)" for NOTAM cylinders). 'history'
    # is kept because no in-scope FAA reply needs the bare word.
    r"equation|theorem|calculus|algebra|physics|chemistry|"
    r"astronomy|biology|history|literature|philosophy|"
    # Personal / chat / meta
    r"how\s+are\s+you|tell\s+me\s+about\s+yourself|what(?:'s|\sis)\s+your\s+name|"
    r"tell\s+me\s+a\s+joke|sing\s+(?:me\s+)?a\s+song|"
    # Joke-meta vocabulary
    r"joke|riddle|punchline|funny|hilarious|haha"
    # Entertainment
    r"|football|basketball|baseball|soccer|movie|film|song|album|book|novel"
    r")\b",
    re.IGNORECASE,
)


# Joke-frame structural regex. Common joke setups almost never ask
# real in-scope questions even when their specific nouns aren't in
# the non-aviation vocab list. Targeted patterns; add more as new
# shapes surface. 'If a X, a Y, and a Z...' is the lead-in from the
# observed 'ghost/turtle/refrigerator' repro. 'Why did the X...' and
# 'knock knock' are the other evergreen shapes.
_JOKE_FRAME_RE = re.compile(
    r"(?:"
    # 'If a X, a Y, and a Z ...' — multi-noun absurd setup, tolerant
    # of optional commas and 'and' connectors between items.
    r"\bif\s+a\s+\w+(?:\s*,\s*(?:and\s+)?(?:a|an)\s+\w+){2,}"
    # 'Why did the X cross/go/do ...' — classic joke opener
    r"|\bwhy\s+did\s+the\s+\w+\s+(?:cross|go|do|say|want|need)"
    # 'Knock knock' literal
    r"|\bknock[\s,-]+knock\b"
    # 'What do you call a X when ...' / 'What's the difference between X and Y'
    r"|\bwhat\s+do\s+you\s+(?:call|get)\s+(?:a|an|when)"
    r"|\bwhat(?:'s|\sis)\s+the\s+(?:difference|similarity)\s+between\s+\w+\s+and\s+\w+"
    # 'X walks into a bar' / 'three men walk into a bar' — the
    # evergreen bar-joke opener. Matches anything preceding 'walk(s)
    # into a/the bar'.
    r"|\bwalks?\s+(?:in)?to\s+(?:a|the)\s+bar\b"
    # 'Who won?' / 'Who wins?' riddle ending. ATC questions ask
    # procedural / definitional things ('who must issue X'); a
    # 'who won / who wins' tag is characteristic of race / contest
    # jokes ('Five Macs and PC join a computer race. Who won?').
    r"|\bwho\s+(?:won|wins)\b"
    r")",
    re.IGNORECASE,
)


# Phrases a reply uses when it IS correctly scope-redirecting.
# A reply that matches this pattern is already doing the right thing
# (declining, naming its own scope, pointing the user elsewhere) —
# even if it happens to mention JO 7110.65 / ATC by name to say 'that
# is NOT what I cover.' Exempting the reply from the scope mismatch
# catcher prevents a successful refusal from being nudged into a
# pointless retry loop.
_SCOPE_REDIRECT_MARKER_RE = re.compile(
    r"(?:"
    r"outside\s+(?:of\s+|the\s+)?(?:JO|FAA|the\s+order|scope|my\s+scope)"
    r"|outside\s+the\s+scope"
    r"|not\s+(?:in|within|part\s+of)\s+(?:scope|JO|the\s+order|my\s+scope)"
    r"|cannot\s+answer"
    r"|can(?:not|'t|\snot)\s+answer"
    r"|I\s+(?:am|'m)\s+(?:a\s+)?specialist"
    r"|that\s+(?:question\s+)?is\s+outside"
    r"|doesn(?:'t|\snot)\s+cover"
    r"|out\s+of\s+scope"
    r"|not\s+something\s+I\s+cover"
    r"|biological\s+query"
    r")",
    re.IGNORECASE,
)


# Threshold past which a "scope-redirect"-marked reply is no longer
# treated as a clean refusal. A real scope-redirect is short ("Outside
# JO 7110.65 — ask airton_c."). A reply that says "JO 7110.65 doesn't
# cover it" then writes three paragraphs of general-knowledge content
# is the redirect-then-answer fabrication shape (harness-zsbz repro,
# 2026-05-14). Both MissingCitationHook (bail) and
# UncitedSubstantiveReplyHook (finalize) consult this limit.
_SCOPE_REDIRECT_SOFT_LIMIT = 300


_SCOPE_REDIRECT_NUDGE_DEFAULT = (
    "[scope mismatch — the user's message is out of scope for this "
    "character, but your reply is discussing in-scope content. "
    "Respond with an explicit scope-redirect: name your scope and "
    "decline. Do NOT answer from priors, do NOT continue a prior "
    "turn's topic into this new unrelated question, and do NOT "
    "fabricate an in-scope interpretation.]"
)


def _build_scope_redirect_nudge(
    character_name: str | None,
    scope_redirect_template: str | None,
) -> str:
    """Compose the scope-redirect bail nudge.

    Threads the character's name and `scope_redirect_template` into the
    nudge so the model is told exactly how THIS character refuses out-
    of-scope prompts. Without parameterization the nudge would leak
    airton_c1's identity ('I'm a JO 7110.65 specialist') into every
    persona that ships scope_redirect — observed in airton_c_tfr
    sessions where the model parroted the example sentence verbatim.

    When either input is missing, falls back to the generic default.
    Strips the template aggressively so a multi-line refusal block
    stays readable inside the bracketed nudge."""
    if not scope_redirect_template or not scope_redirect_template.strip():
        return _SCOPE_REDIRECT_NUDGE_DEFAULT
    example = " ".join(scope_redirect_template.split())
    name = character_name or "this character"
    return (
        f"[scope mismatch — the user's message is out of scope for "
        f"{name}, but your reply is continuing in-scope content. "
        f"Respond with an explicit scope-redirect along the lines of: "
        f"'{example}' Do NOT answer from priors, do NOT continue a "
        f"prior turn's topic into this new unrelated question, and do "
        f"NOT fabricate an in-scope interpretation.]"
    )


# Ambiguous-term dictionary: maps a user-side term regex to a tuple
# of axes, where each axis is a tuple of mutually-exclusive qualifier
# patterns. Axes structure is load-bearing — a reply with multiple
# alternatives from THE SAME axis ('manned or unmanned') is offering
# a choice / asking for clarification; a reply with exactly one match
# per axis has committed to a specific variant.
#
# If the user's message contains the bare term without ANY qualifier
# (across all axes), and the reply picks exactly one qualifier in at
# least one axis without offering choice in any axis, the hook
# nudges.
#
# Rule of discipline: only add an entry when BOTH (a) the JO treats
# variants of this term with materially different rules AND (b) users
# plausibly ask the bare term without specifying the variant. Each
# entry is session-regression-backed.
_AMBIGUOUS_TERMS: tuple[tuple[re.Pattern[str], tuple[tuple[re.Pattern[str], ...], ...]], ...] = (
    # Balloons — unmanned free balloons fall under §9-6 (distinct
    # controller procedures: traffic advisory, no vertical separation
    # without verified altitude, derelict-balloon handling). Manned
    # balloons are handled as general aircraft. Session 2026-04-24
    # repro: user asked about 'a balloon' without specifying; model
    # answered for the unmanned-free variant silently.
    (
        re.compile(r"\bballoons?\b", re.IGNORECASE),
        (
            # Axis 1: crew
            (
                re.compile(r"\bmanned\b", re.IGNORECASE),
                re.compile(r"\bunmanned\b", re.IGNORECASE),
            ),
            # Axis 2: tether
            (
                re.compile(r"\bfree\s+balloon", re.IGNORECASE),
                re.compile(r"\btethered\b", re.IGNORECASE),
                re.compile(r"\bmoored\b", re.IGNORECASE),
            ),
            # Axis 3: lift medium / use
            (
                re.compile(r"\bhot\s+air\s+balloon", re.IGNORECASE),
                re.compile(r"\bgas\s+balloon", re.IGNORECASE),
                re.compile(r"\bweather\s+balloon", re.IGNORECASE),
            ),
        ),
    ),
)


_AMBIGUOUS_CONTEXT_NUDGE = (
    "[ambiguous context — the user's message contains a term whose "
    "correct handling depends on context they did not provide. For "
    "example, 'balloon' covers unmanned free balloons (JO 7110.65 "
    "§9-6, distinct controller procedures) AND manned balloons "
    "(treated under general aircraft rules) — the answer differs. "
    "Your reply assumed a specific variant instead of asking. "
    "Re-answer by asking the user to clarify the variant FIRST, then "
    "answer only after they provide it. Do not pick a variant and "
    "proceed.]"
)


@dataclass(frozen=True)
class AmbiguousContextHook:
    """Nudge replies that silently assume a specific variant of an
    ambiguous term instead of asking for clarification.

    Failure mode this catches: user asks about 'a balloon'. JO 7110.65
    treats unmanned free balloons (§9-6) with distinct rules from
    manned balloons (general aircraft rules). Model silently fills in
    'unmanned free balloon' and answers — robbing the student of the
    chance to learn that variant matters. Session 2026-04-24 repro:
    'a single prop squawking 1200 and a balloon are intersecting, who
    has the right of way?' — model inserted 'unmanned free' without
    asking.

    Trigger conditions (ALL must hold, checked per ambiguous term):
      1. `user_message` threaded through.
      2. User message contains the ambiguous term (bare form).
      3. User message does NOT contain any of the term's
         disambiguating qualifiers.
      4. Reply DOES contain at least one of the disambiguating
         qualifiers.

    Action: Nudge. Retry-able — the model's next round should ask a
    clarifying question ('manned or unmanned?') and wait for the
    answer instead of proceeding with an assumed variant.

    The `_AMBIGUOUS_TERMS` dictionary is the maintenance surface.
    Extend with new entries only when both (a) the JO treats variants
    differently and (b) a real user-visible miss occurs.

    Placed after ScopeRedirectHook in the bail list so scope
    mismatches (out-of-scope prompts) take priority over ambiguity
    nudges inside in-scope prompts."""

    name: str = "ambiguous_context"

    def check(self, ctx: BailContext) -> BailOutcome:
        user = ctx.user_message
        if user is None or not user.strip():
            return Continue()
        content = ctx.reply.content
        for term_re, axes in _AMBIGUOUS_TERMS:
            if not term_re.search(user):
                continue
            # User already specified any qualifier across any axis —
            # no ambiguity to challenge.
            all_qualifiers = [q for axis in axes for q in axis]
            if any(q.search(user) for q in all_qualifiers):
                continue
            # Per-axis match counts. Reply offers a choice if ANY
            # axis has >= 2 alternatives mentioned; reply commits if
            # at least one axis has exactly 1.
            axis_hits = [sum(1 for q in axis if q.search(content)) for axis in axes]
            if any(h >= 2 for h in axis_hits):
                # At least one axis shows multi-alternative mention —
                # reply is asking / comparing, not silently picking.
                continue
            if any(h == 1 for h in axis_hits):
                return Nudge(_AMBIGUOUS_CONTEXT_NUDGE)
        return Continue()


@dataclass(frozen=True)
class ScopeRedirectHook:
    """Nudge replies that continue JO 7110.65 / ATC content when the
    user's question has no aviation vocabulary at all.

    Failure mode this catches: user asks an out-of-scope question
    ('do roosters lay eggs', 'help me write Python'). Retrieval scores
    are uniformly weak — airton_c1's corpus is JO 7110.65 only, so
    there's nothing relevant to surface. The model, seeing weak
    retrieval + strong prior-turn context in its working history,
    latches onto the previous turn's topic instead of scope-redirecting.
    Session 2026-04-24 repro: 'do roosters lay eggs' got the previous
    turn's phraseology-correction content emitted back.

    Trigger conditions — ANY of the three positive signals trips
    the hook, provided the reply is not already a scope redirect:

      Signal A (user vocab): User message contains clearly
        non-aviation vocabulary (`_CLEARLY_NON_AVIATION_RE` —
        biology/cooking/programming/chat meta/entertainment/
        supernatural/household/joke-meta).
      Signal B (user shape): User message matches a joke-frame
        structural pattern (`_JOKE_FRAME_RE` — 'If a X, a Y, and a
        Z...', 'Why did the X...', 'knock knock', 'X walks into a
        bar'). Catches absurd setups even when the specific nouns
        aren't in vocab.
      Signal C (reply bleed): The REPLY contains BOTH aviation vocab
        AND clearly-non-aviation vocab. This is defense-in-depth for
        context-bleed (model pulled prior-turn non-aviation material
        into the current reply) — even if the user's latest message
        has no suspicious markers, a reply that mixes 'roosters don't
        lay eggs' with 'per §7-6-11' is confused.

    Signals A and B fire regardless of whether the reply has ATC
    vocabulary — a plain off-topic answer ('the bartender says hi')
    is just as much a scope failure as a context-bleed answer.
    airton_c1 must redirect off-topic prompts, not answer them in
    kind. Signal C keeps its reply-bleed gate (needs both aviation
    AND non-aviation vocab) because it exists to catch cases where
    the user's message gave no signal at all.

    Gate on substantiveness (user message >= 15 chars) so short
    follow-ups don't trip; exempt replies already shaped as scope
    redirects (`_SCOPE_REDIRECT_MARKER_RE`) so a correct refusal
    doesn't loop.

    Action: Nudge. Retry-able — on the next round the model should
    emit a scope-redirect sentence. Halt would be too brutal; a
    nudge lets the model actually produce a useful 'not in scope'
    message rather than a canned refusal.

    Placed last in the bail list so domain-specific shape catchers
    (fabrication / count / citation / reserved-code) run first on
    in-scope replies."""

    # Per-character nudge inputs. When the hook fires, the nudge text
    # is composed via `_build_scope_redirect_nudge` so the example
    # refusal sentence comes from THIS character's scope_redirect_
    # template, not airton_c1's hardcoded "I'm a JO 7110.65
    # specialist" line. `_DEFAULT_PIPELINE` and other call sites that
    # don't thread these through fall back to a generic nudge.
    character_name: str | None = None
    scope_redirect_template: str | None = None
    name: str = "scope_redirect"

    def check(self, ctx: BailContext) -> BailOutcome:
        user = ctx.user_message
        if user is None or len(user.strip()) < 15:
            return Continue()
        content = ctx.reply.content
        # Exempt replies already correctly scope-redirecting.
        if _SCOPE_REDIRECT_MARKER_RE.search(content):
            return Continue()
        # Signal A / B: user-side triggers. Fire regardless of
        # reply shape — an off-topic answer to an off-topic prompt
        # is still a scope failure.
        signal_user_vocab = bool(_CLEARLY_NON_AVIATION_RE.search(user))
        signal_joke_frame = bool(_JOKE_FRAME_RE.search(user))
        # Signal C: reply mixes aviation AND clearly-non-aviation
        # content. Only meaningful when BOTH appear — that's the
        # context-bleed shape.
        signal_reply_bleed = bool(
            _AVIATION_VOCAB_RE.search(content) and _CLEARLY_NON_AVIATION_RE.search(content)
        )
        if not (signal_user_vocab or signal_joke_frame or signal_reply_bleed):
            return Continue()
        return Nudge(_build_scope_redirect_nudge(self.character_name, self.scope_redirect_template))


# Reserved transponder codes are pilot-initiated emergency signals.
# Controllers OBSERVE them via 5-2-5 / 5-2-8 procedures; they never
# assign them as routine phraseology. Catcher guards against a model
# that uncritically reformats a user-provided reserved code into
# correct-looking ATC phraseology (session 2026-04-24 repro).
#
# 7400 (UAS lost link per §5-2-6) is intentionally excluded — it's
# UAS-specific and airtime is rare in controller phraseology; adding
# it would net-increase false-positive risk without matching the
# observed failure mode.
_RESERVED_SQUAWK_RE = re.compile(
    # Left-anchor: forbid a leading letter or digit (so 'resquawk' or
    # similar doesn't match). `\b` would block matches with leading
    # underscore because `_` is a word char — intentionally permissive
    # here to catch `_squawk ..._` markdown italic.
    r"(?<![A-Za-z0-9])squawk\s+"
    r"(?:"
    r"7500|7600|7700"
    # Digit-by-digit readback (JO phraseology convention):
    # 'seven five zero zero', etc.
    r"|seven\s+five\s+zero\s+zero"
    r"|seven\s+six\s+zero\s+zero"
    r"|seven\s+seven\s+zero\s+zero"
    # Grouped-hundreds (lay / student paraphrase):
    # 'seventy five hundred', 'seven five hundred', etc.
    r"|seven(?:ty)?\s+five\s+hundred"
    r"|seven(?:ty)?\s+six\s+hundred"
    r"|seven(?:ty)?\s+seven\s+hundred"
    r")"
    # Right-anchor: forbid a trailing digit or letter (so '75002' / 'seven
    # five hundredth' don't shadow the match). Underscore / punctuation /
    # whitespace / end-of-string all qualify — this matters because the
    # default `\b` treats `_` as a word char, which would have blocked
    # matches inside markdown italic like `_squawk 7500_`.
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


_RESERVED_SQUAWK_NUDGE = (
    "[reserved-squawk-code — your reply proposes assigning a reserved "
    "transponder code (7500 = hijack/unlawful interference per §5-2-5; "
    "7600 = comm failure; 7700 = emergency per §5-2-8). These are "
    "PILOT-INITIATED emergency signals — controllers observe them, "
    "never assign them as routine phraseology. For VFR radar service "
    "termination the correct code is 'squawk VFR' or 'squawk one two "
    "zero zero' per §5-2-7. Re-answer with the correct code; if the "
    "user's input contained a reserved code, call that out explicitly "
    "rather than silently reformatting it.]"
)


def _is_squawk_assignment_context(content: str, start: int, end: int) -> bool:
    """Decide whether a `squawk <reserved>` match is asserting an
    assignment vs. explaining the code's meaning.

    Assignment signals (any one is sufficient):
      - Match text starts with all-caps 'SQUAWK' (JO phraseology
        convention — actual phraseology lines in the order are
        capitalized).
      - Match is wrapped in paired double-quotes on the same line
        (a quoted phraseology line like
        `"Radar service terminated, squawk seven five hundred."`).
      - Match is wrapped in `_..._` (markdown italic — used in the
        corpus for PHRASEOLOGY-block emphasis).
      - Match is wrapped in `**...**` (markdown bold — reply-side
        formatting for proposed phraseology).

    Explanation / description contexts (none of the above present)
    pass through cleanly — so prose like
    'When you observe Code 7500, apply §10-2-6' or a warning like
    'squawk 7500 is the hijack code' doesn't trip the hook.
    """
    if content[start:end].startswith("SQUAWK"):
        return True
    line_start = content.rfind("\n", 0, start) + 1
    line_end = content.find("\n", end)
    if line_end == -1:
        line_end = len(content)
    prefix = content[line_start:start]
    suffix = content[end:line_end]
    # Paired quote/italic/bold anchors on the same line. We require
    # BOTH sides to carry the anchor so an unpaired `*` (common in
    # bullet lists) doesn't false-positive.
    if '"' in prefix and '"' in suffix:
        return True
    if "_" in prefix and "_" in suffix:
        return True
    return "**" in prefix and "**" in suffix


@dataclass(frozen=True)
class ReservedSquawkCodeHook:
    """Nudge a reply that assigns a reserved transponder code (7500 /
    7600 / 7700) in routine controller phraseology.

    Failure mode this catches: user transcribes incorrect phraseology
    containing a reserved code ('squawk seventy five hundred'). Model
    correctly fixes the phraseology shell ('Services stopped' ->
    'Radar service terminated') but blindly reformats the code value
    ('seventy five hundred' -> 'seven five hundred' i.e. 7500),
    propagating a hijack-code assignment into a 'correct' reply.
    Session 2026-04-24 repro — the corrected phraseology was still
    unsafe because 7500 is pilot-set when the aircraft is being
    hijacked; controllers do not assign it.

    Corpus anchors:
      §5-2-5 — HIJACK/UNLAWFUL INTERFERENCE (Code 7500 observation)
      §5-2-7 — VFR CODE ASSIGNMENTS (correct VFR code = 1200 / 'VFR')
      §5-2-8 — note on Code 7700 emergency activation

    Trigger conditions (ALL must hold):
      1. Reply contains `squawk <reserved code>` in any of its digit,
         digit-by-digit-readback, or grouped-hundreds forms.
      2. The match is in an ASSIGNMENT context — wrapped in quotes,
         italic, bold, or all-caps SQUAWK (see
         `_is_squawk_assignment_context`). Narrative descriptions
         ('when you observe Code 7500') pass through.

    Action: Nudge. Retry-able — model gets a round to swap in the
    correct code ('squawk VFR' / '1200') and, ideally, to flag that
    the user's input contained a reserved code. Do NOT Halt: the
    model's phraseology shell is usually correct; we want to fix the
    code, not lose the correction entirely."""

    name: str = "reserved_squawk_code"

    def check(self, ctx: BailContext) -> BailOutcome:
        content = ctx.reply.content
        # Lower-cased user_message for verbatim-echo detection. When
        # the model quotes the user's question to flag it as wrong
        # ('"Services stopped, squawk seventy five hundred" is
        # incorrect — the right form is ...'), the reserved-code match
        # lives inside an echo of the prompt, not a new assignment.
        # Compare lowercase since match text may differ in case
        # ('SQUAWK 7500' vs 'squawk 7500').
        user_lower = (ctx.user_message or "").lower()
        for match in _RESERVED_SQUAWK_RE.finditer(content):
            if not _is_squawk_assignment_context(content, match.start(), match.end()):
                continue
            if user_lower and match.group(0).lower() in user_lower:
                # Model echoed the user's exact reserved-code phrase —
                # not a new assignment proposal.
                continue
            return Nudge(_RESERVED_SQUAWK_NUDGE)
        return Continue()


@dataclass(frozen=True)
class ListCountMismatchHook:
    """Nudge replies whose stated count of items disagrees with the
    number of items the reply actually enumerates.

    Failure mode this catches: the user asks for N items, the model
    echoes 'The N ... are' and then emits an enumerated list of M != N
    items. Session 2026-04-24 repro: user asked for 4 primary purposes
    of ATC; model wrote 'The four specific primary purposes ... are as
    follows:' and listed 3. Self-falsifying — the reply contradicts
    itself within three sentences.

    Trigger conditions (ALL must hold):
      1. Exactly one count claim in the reply (matches
         `_COUNT_CLAIM_RE` and the number word / digit resolves to a
         known integer). Multiple claims are ambiguous — skip rather
         than pick the wrong one to check against.
      2. Reply contains an enumerated list of >= 2 items
         (`_LIST_ITEM_RE` multiline count). A single-item 'list' is
         prose, not an enumeration — skip.
      3. The claim count != the list count.

    Action: Nudge the model to re-answer with a consistent count.
    Retry-able — typical failure is 'model over-agreed with user's
    wrong premise'; retry lets the model push back on the premise
    instead of fabricating a missing item.

    Placed AFTER `missing_citation` in the bail list: once compliance
    (citation) is satisfied, internal consistency (count) is the next
    layer of quality.

    Silent cases (by design):
      - Zero count claims -> nothing to verify.
      - Multiple count claims -> ambiguous which is 'the' claim.
      - No enumerated list -> reply is narrative prose; count could
        be correct or the list implied but not formalized, so don't
        fire."""

    name: str = "list_count_mismatch"

    def check(self, ctx: BailContext) -> BailOutcome:
        content = ctx.reply.content
        # Fast-path: no digits/number-words -> no possible claim.
        if not re.search(
            r"\b(?:\d+|two|three|four|five|six|seven|eight|"
            r"nine|ten|eleven|twelve)\b",
            content,
            re.IGNORECASE,
        ):
            return Continue()
        # Walk every regex match and reject ones whose span contains
        # legal/document citation tokens (§ / CFR / USC / AIM / JO /
        # AC). Observed 2026-05-15 with airton_c_tfr: '14 CFR §99.7,
        # ... operations are prohibited' matched the count-claim
        # shape, with '14' treated as a 14-item count. Filtering
        # citation spans keeps real count claims ('the 14 sections
        # are ...') while killing false positives from cited rules.
        valid_claims: list[str] = []
        for match in _COUNT_CLAIM_RE.finditer(content):
            if _CITATION_TOKEN_IN_CLAIM_RE.search(match.group(0)):
                continue
            # Subset carve-out (harness-9gzk). When the match span
            # contains language that explicitly labels the enumerated
            # list as a subset of the counted population, skip — the
            # count is informational, not a promise of enumeration
            # length. Repro: 'Kenya is home to 50 genera. Among these,
            # some species are unique... <list of 4>' should not fire
            # 50-vs-4. See _SUBSET_MARKER_RE for the marker set.
            if _SUBSET_MARKER_RE.search(match.group(0)):
                continue
            valid_claims.append(match.group(1))
        if len(valid_claims) != 1:
            return Continue()
        claimed = _parse_count_word(valid_claims[0])
        if claimed is None:
            return Continue()
        list_count = len(_LIST_ITEM_RE.findall(content))
        if list_count < 2:
            return Continue()
        if claimed == list_count:
            return Continue()
        return Nudge(_LIST_COUNT_MISMATCH_NUDGE)


_SOURCE_COUNT_INFLATION_NUDGE = (
    "[source-count inflation — your reply enumerated {reply_count} items "
    "but the most recent tool result returned {tool_count}. Do NOT pad "
    "with sources the tool did not return. Re-answer using ONLY the "
    "items in the tool result; if you have additional knowledge about "
    "the topic, you may add prose (not numbered/bulleted entries) and "
    "label it explicitly as your own knowledge, not as a tool result.]"
)


# Tolerance: a reply may legitimately enumerate one more item than the
# tool (e.g. the tool returned 4 papers + the model adds a brief "also
# worth mentioning" line). > tool_count + 1 is the fabrication shape.
_SOURCE_COUNT_TOLERANCE = 1
# Below this floor the comparison is too noisy to act on — a 1-item
# reply enumerating a 0-item tool result is a "no matches" wrap-up, not
# fabrication. Pin reply_count at >= 3 so the catch shape ("padded to 5
# from 1") is unambiguous.
_SOURCE_COUNT_REPLY_FLOOR = 3


@dataclass(frozen=True)
class SourceCountInflationHook:
    """Nudge replies that enumerate substantially more items than the
    underlying tool actually returned.

    Failure mode this catches (harness-lynw): tool returns 1 search
    result; model writes a 5-item enumerated list, padding 4 entries
    with plausible-sounding names that never appeared in the tool
    output. The other fabrication catchers don't fire because:
      - FABRICATED_URL_LIST_RE requires per-entry URLs (this pattern
        has none — just `**Name**: description`).
      - The web-fetch tool ran legitimately, so the broad
        'fabricated_search' gate is disarmed.
      - ListCountMismatchHook compares the reply against a stated
        count claim in the SAME reply; here the model never claims
        a count, it just over-enumerates.

    Rule (ALL must hold):
      1. Reply enumerates >= _SOURCE_COUNT_REPLY_FLOOR items.
      2. Most recent prior tool output also enumerates >= 1 item.
      3. reply_count > tool_count + _SOURCE_COUNT_TOLERANCE.

    Silent cases (by design):
      - No prior tool output → no comparison.
      - Most recent tool output had no enumerated items (e.g. prose
        reply, no-matches sentinel) → no comparison.
      - Reply count is within tolerance → legitimate elaboration.

    Opt-in via the character's `catchers:` roster as
    `source_count_inflation`. Default-on for characters that do
    web research; off for chat-only profiles where the heuristic
    has no signal to gate on."""

    name: str = "source_count_inflation"

    def check(self, ctx: BailContext) -> BailOutcome:
        if not ctx.prior_tool_outputs:
            return Continue()
        reply_count = len(_LIST_ITEM_RE.findall(ctx.reply.content))
        if reply_count < _SOURCE_COUNT_REPLY_FLOOR:
            return Continue()
        # Walk priors in reverse to find the most recent enumerated
        # tool output. A tool that returned a no-matches blurb has 0
        # items and we keep walking; once we hit one with items, that
        # anchors the comparison. If none have items, we don't fire.
        tool_count: int | None = None
        for prior in reversed(ctx.prior_tool_outputs):
            count = len(_LIST_ITEM_RE.findall(prior))
            if count >= 1:
                tool_count = count
                break
        if tool_count is None:
            return Continue()
        if reply_count <= tool_count + _SOURCE_COUNT_TOLERANCE:
            return Continue()
        return Nudge(
            _SOURCE_COUNT_INFLATION_NUDGE.format(
                reply_count=reply_count,
                tool_count=tool_count,
            )
        )


@dataclass(frozen=True)
class MissingCitationHook:
    """Nudge a reply that references the corpus document substantively
    but doesn't include a specific `§N-N-N` / `TBL N-N-N` citation.
    Mirror image of `UngroundedCitationHook` — that one catches
    citation without grounding; this one catches grounding without
    citation.

    Failure mode this catches: a citation-disciplined character's
    constitution says 'Always cite at least one section when
    answering.' A grounding tool ran, the tool output surfaced a
    section, the reply paraphrases from that section correctly, but
    the model omits the §-anchor. Session 2026-04-24 reproducers
    (airton_c1 / JO 7110.65): 'What is the purpose of 7110.65?' →
    correct summary of §1-1-1, no citation. Both leave the student
    without a way to verify or locate the source.

    Trigger conditions (ALL must hold):
      0. The character ships a `citation_grammar` (harness-jaqe).
         Non-corpus characters auto-skip — there's no grammar to
         demand citations against.
      1. A grounding tool ran this turn (`ctx.tools_ran` intersects
         `_GROUNDING_TOOLS`). Without one there's nothing to cite
         from — that's `ungrounded_citation`'s domain.
      2. The reply contains EITHER a corpus-document reference
         (`grammar.document_reference`) OR a JO-normative phraseology
         marker (`_JO_PHRASEOLOGY_MARKERS_RE`) — domain vocabulary
         that makes the reply a corpus claim even without naming the
         document. (The phraseology-marker regex stays FAA-shaped
         pending the qvwq domain-catcher refactor.)
      3. The reply is substantive (>= 80 chars). Short replies are
         usually refusals or scope-redirects that don't need a cite.
      4. The reply does NOT contain a `§N-N-N` / `TBL N-N-N` anchor.

    Action: Nudge the model to re-answer with the citation inline.
    Retry-able — the model gets another round to add the citation.
    Do NOT Halt: losing a correct substantive answer to a canned
    refusal over a missing anchor is worse than the missing anchor
    itself.

    Placed AFTER `tool_intent` in the bail list: fabrication-shape
    catchers (teaser/false_success/meta_confirm/fabricated_*) all run
    first, so a fabricated-looking reply gets the fabrication-specific
    nudge rather than this one. Compliance comes after correctness."""

    grammar: CitationGrammar | None = None
    name: str = "missing_citation"

    def check(self, ctx: BailContext) -> BailOutcome:
        if self.grammar is None:
            return Continue()
        if not (ctx.tools_ran & _GROUNDING_TOOLS):
            return Continue()
        content = ctx.reply.content
        if len(content) < 80:
            return Continue()
        in_scope = bool(self.grammar.document_reference.search(content)) or bool(
            _JO_PHRASEOLOGY_MARKERS_RE.search(content)
        )
        if not in_scope:
            return Continue()
        if _reply_has_citation(content, self.grammar):
            return Continue()
        # URL citations (`[arxiv:..]`, `[scholar:..]`, `[doi:..]`,
        # `[wiki:..]`) count as grounding for scholar-style characters
        # that summarise search-tool results. Smoke 2026-05-15 (JEPA):
        # the reply was a clean `[arxiv:..] §Abstract: ...` summary of
        # search_scholar results. document_reference (`§\s*\S`) matched
        # the paper-internal `§Abstract:` headers, surface_patterns
        # required §-form paren-doc and didn't match, but the reply was
        # already richly cited via URL form. Accepting URL citations
        # here resolves the false positive without changing
        # surface_patterns (and so without affecting lead_with_citation,
        # which would otherwise hoist URL citations to the front of a
        # multi-paragraph summary). FAA-only characters don't emit URL
        # citations, so this branch is a no-op for them.
        if _URL_CITATION_RE.search(content):
            return Continue()
        # Scope-redirect replies name the order to explain what's
        # NOT covered ('That question is outside JO 7110.65'). Those
        # are declining to answer, not making a substantive claim —
        # don't demand a citation. BUT: a redirect marker followed by
        # a long body is the redirect-then-answer fabrication shape
        # (harness-zsbz, 2026-05-14 wedding-ring repro). When the
        # reply blows past `_SCOPE_REDIRECT_SOFT_LIMIT` chars, the
        # bypass is unsafe — let MissingCitation Nudge so the model
        # gets another shot at a clean refusal, and let the finalize
        # phase's UncitedSubstantiveReplyHook catch what survives.
        #
        # Additionally gated on no `§` in the reply (smoke 2026-05-15
        # airton_f). At this point in the check, `_reply_has_citation`
        # already returned False — surface_patterns didn't match and
        # the FAA bare-anchor fallback (if opted in) didn't either.
        # If the reply still contains a §, it's making a citation
        # ATTEMPT that the grammar rejected — i.e., a malformed cite.
        # A true scope-redirect doesn't combine "doesn't cover"
        # language with a § anchor. The §-coupling is acceptable
        # because every currently-shipped citation grammar uses §;
        # future non-§ citation forms would need their own gate.
        if (
            _SCOPE_REDIRECT_MARKER_RE.search(content)
            and len(content) <= _SCOPE_REDIRECT_SOFT_LIMIT
            and "§" not in content
        ):
            return Continue()
        # Clarifying-question replies ('Could you clarify manned or
        # unmanned?') are ASKING, not asserting. Exempt — a citation
        # in a question would be weirdly presumptive and looping the
        # retry would discard a perfectly good clarifying reply
        # (harness-5uq follow-up). Same §-gate as scope_redirect
        # above: a reply with bare-§ + clarifying question is still
        # a malformed-citation fabrication.
        if _CLARIFYING_QUESTION_RE.search(content) and "§" not in content:
            return Continue()
        return Nudge(
            _MISSING_CITATION_NUDGE_TEMPLATE.format(
                document_name=self.grammar.document_name,
                example_anchor=self.grammar.example_anchor,
            )
        )


@dataclass(frozen=True)
class FabricatedSectionHook:
    """Catch §-citations whose section number doesn't exist anywhere
    in the character's corpus (harness-aise).

    Different from `UngroundedCitationHook` (no grounding tool ran)
    and `LowConfidenceFallbackHook` (tool ran but scored weakly):
    this is a STRUCTURAL existence check. We don't ask whether the
    cited section was retrieved — we ask whether it's even a real
    section. A reply citing `§3-99-3` trips this hook regardless of
    retrieval state, because no chunk in the corpus carries that
    section number.

    The valid-anchor set is built once at startup from the chunks
    JSONL (see `section_index.collect_valid_anchors`) and includes
    both `§N-N-N` paragraph anchors and their `§N-N` parents — a
    reply that cites the parent section without paragraph passes.

    Trigger conditions (ALL must hold):
      1. `valid_anchors` is non-empty. Empty set = the character has
         no enumerable corpus, hook is silent (default behaviour for
         airton, airton_b, airton_c).
      2. Reply contains at least one §-style citation
         (`extract_citations` returns a member that starts with §).
      3. At least one cited §-anchor is NOT in `valid_anchors`.

    Action: Nudge — drop the reply and re-prompt with the invalid
    section name so the model can either pick a real anchor or
    admit it doesn't know. Bail-phase, not finalize, because a
    structural correction is recoverable: the model can call
    `search_memory` to find a real section, or scope-redirect.

    Placed AFTER `missing_citation` in the bail list: a no-citation
    reply gets the missing-cite nudge first; only a reply that DID
    cite something faces the existence check. TBL/FIG citations are
    deliberately not validated here — those need a different index
    (table-level enumeration) and are tracked separately."""

    valid_anchors: frozenset[str] = frozenset()
    name: str = "fabricated_section"

    def check(self, ctx: BailContext) -> BailOutcome:
        if not self.valid_anchors:
            return Continue()
        from harness.tools.citations import extract_citations

        cited = extract_citations(ctx.reply.content)
        if not cited:
            return Continue()
        # Only validate §-style cites here. TBL/FIG live alongside
        # sections but need their own enumeration (a TBL is implicit
        # in its parent section's body, not a top-level chunker
        # stamp), so we deliberately let those pass.
        section_cites = frozenset(c for c in cited if c.startswith("§"))
        if not section_cites:
            return Continue()
        invalid = section_cites - self.valid_anchors
        if not invalid:
            return Continue()
        if len(invalid) == 1:
            offender = next(iter(invalid))
            nudge = (
                f"Your reply cited {offender}, which is not a section "
                "in the corpus you have access to. Either cite a real "
                "section (call search_memory if you don't know which "
                "one covers the topic), or admit you don't know and "
                "redirect the user."
            )
        else:
            offenders = ", ".join(sorted(invalid))
            nudge = (
                f"Your reply cited {offenders} — none of these are "
                "sections in the corpus you have access to. Either "
                "cite real sections (call search_memory if you don't "
                "know which) or admit you don't know."
            )
        return Nudge(nudge)


# ---------- post-model hooks ----------


@dataclass(frozen=True)
class PairedMetaConfirmStripHook:
    """Strip meta-confirm narrative from a reply that ALSO carries a
    valid tool call ("Would you like me to …? <tool_call>…"). The tool
    call is real but the narrative is noise — clearing it keeps the
    wrap-up round from re-reading hallucinated confirmation prose."""

    name: str = "paired_meta_confirm_strip"

    def check(self, ctx: PostModelContext) -> PostModelOutcome:
        reply = ctx.reply
        if not reply.tool_calls:
            return Continue()
        if not META_CONFIRM_RE.search(reply.content):
            return Continue()
        return Replace(
            ModelReply(
                content="",
                tool_calls=reply.tool_calls,
                was_truncated=reply.was_truncated,
                had_unparseable_call=reply.had_unparseable_call,
            )
        )


# ---------- pre-tool hooks ----------


@dataclass(frozen=True)
class DuplicateCallHook:
    """Short-circuit an identical (name, arguments) re-invocation this
    turn. Small models sometimes wrap a real answer around a redundant
    re-call; re-issuing the prior result as the tool-role message lets
    the next round close out (harness-pun) without re-running the tool.

    Re-issues the EXACT prior ToolResult (output + success + error),
    prefixed with a duplicate annotation. If the prior call failed, the
    duplicate is also marked as failed — feeding back success=True for
    a duplicate of a failed call made the model hallucinate success
    (harness-v5w)."""

    name: str = "duplicate_call"

    def check(self, ctx: PreToolContext) -> PreToolOutcome:
        key = _call_key(ctx.call)
        prior = ctx.seen_calls.get(key)
        if prior is None:
            return Continue()
        return Skip(
            ToolResult(
                tool_name=ctx.call.name,
                output=_DUPLICATE_CALL_PREFIX + prior.output,
                success=prior.success,
                error=prior.error,
            )
        )


# Nudge fed back as the tool-role message when the grounding hook
# skips a call. Phrased so the next round knows exactly what failed
# (the arg that didn't trace back to the user) and what the remedy is
# (re-plan from what the user ACTUALLY said). Deliberately avoids
# imperatives like "call this tool instead" — the hook can't know the
# right replacement, only that the current one is off-prompt.
_ARG_GROUNDING_NUDGE = (
    "[grounded-args check failed — your tool arguments named entities "
    "({offenders}) that the user did not mention. The user's message "
    "was: {user_message!r}. Re-read that message and either call the "
    "tool with arguments derived from it, or tell the user you cannot "
    "answer. Do NOT reuse entities from memory / prior turns / your "
    "own prior replies.]"
)


@dataclass(frozen=True)
class ArgumentGroundingHook:
    """Reject tool calls whose argument entities don't trace back to the
    current user turn.

    Failure mode this catches: router or main model picks a specific
    domain / URL that appears nowhere in the user's message. The entity
    leaks in from retrieved episodic memory, a prior conversation, or
    the small router model's training bias. Stackoverflow-asked prompt
    becomes a `search_web(query='dailydrop.fm')` call — coherent-sounding
    but wrong. Without this hook, the orchestrator executes the call,
    the main model wraps a confident-looking summary around the
    irrelevant result, and the user gets a fabricated answer.

    Rule (narrow by design):
      1. Extract domain-like roots from the tool's string arguments.
      2. For each root, check whether it appears as a substring in the
         user message (case-insensitive).
      3. If ANY root has no match in the user message, Skip the call
         with a re-plan nudge.

    Skip conditions (never fire):
      - No user_message threaded through the context (rare; bootstrap
        cases / subagent calls).
      - No domain-like tokens in the args (prose queries pass; we only
        flag entity-specific arguments).
      - Every domain root is grounded in the user message.

    The hook runs AFTER `duplicate_call` so re-calls short-circuit first
    without a spurious grounding nudge. The attribution eval can disable
    it by name ('argument_grounding') to measure which failures it
    uniquely catches."""

    name: str = "argument_grounding"

    def check(self, ctx: PreToolContext) -> PreToolOutcome:
        if ctx.user_message is None:
            return Continue()
        arg_strings = _arg_string_values(ctx.call.arguments)
        if not arg_strings:
            return Continue()
        roots = _arg_domain_roots(arg_strings)
        if not roots:
            return Continue()
        # Ground against the user message AND any prior tool outputs
        # this turn (harness-jm9p) — a fetch_url targeting a domain
        # the prior search_web returned is legitimate even if the user
        # never spelled the domain out. Only "leaked from training /
        # parametric memory" domains stay flagged.
        grounded_corpus = ctx.user_message.lower()
        for prior in ctx.prior_tool_outputs:
            grounded_corpus += "\n" + prior.lower()
        ungrounded = sorted(root for root in roots if root not in grounded_corpus)
        if not ungrounded:
            return Continue()
        offenders = ", ".join(repr(r) for r in ungrounded)
        return Skip(
            ToolResult(
                tool_name=ctx.call.name,
                output=_ARG_GROUNDING_NUDGE.format(
                    offenders=offenders,
                    user_message=ctx.user_message,
                ),
                success=False,
                error="argument_grounding",
            )
        )


# Matches any http(s) URL inside a user message. Used by
# FetchUrlGuardHook to decide whether the user's turn carries an
# explicit URL the model should fetch. Greedy and case-insensitive;
# stops at whitespace because tool args / pasted bodies rarely contain
# URLs followed by other URL-fragment characters without whitespace.
_URL_IN_TEXT_RE = re.compile(r"https?://\S+", re.IGNORECASE)


# Skip-result body for the assemble_context-once guard. Phrased so the
# model knows the grounding bundle is already in its working thread —
# don't re-call, just answer from the existing context.
_ASSEMBLE_CONTEXT_ONCE_OUTPUT = (
    "assemble_context already ran successfully this turn — the context "
    "package is already in your working message thread. Do NOT call "
    "assemble_context again with a different role or different "
    "variables; the registered role is the ONE listed in the tool "
    "description's 'Available contracts' section. Answer from the "
    "grounding you already have."
)


@dataclass(frozen=True)
class AssembleContextOnceHook:
    """Skip model-issued `assemble_context` re-calls when the forced
    grounding already ran this turn.

    Failure mode this catches: characters with `require_assemble_context:
    true` get a forced `assemble_context(role=<default>)` call before
    round 0 — the result is already in the model's working thread. But
    small models sometimes treat the resulting context block as 'I
    should call this tool too' and issue their own `assemble_context`
    call, often with a hallucinated role name (observed 2026-05-15
    with airton_c_tfr: model invented `role='TFR_interpreter'` after
    the forced call with `role='airton_c_tfr'` succeeded). The unknown-
    role error then pollutes the working thread with a misleading
    'Available: airton_c_tfr' message the model may interpret as a
    correction it should retry.

    Rule:
      1. Only checks calls whose name is `assemble_context`.
      2. Inspects `seen_calls` for any prior entry with that name. The
         forced-grounding prelude adds its (name, args) key before
         round 0, so a model-issued call in round 1+ will see it.
      3. If yes → Skip with a result that names the rule and tells
         the model the bundle is already attached.

    Opt-in via the character's `catchers:` roster as
    `assemble_context_once`. Multi-contract characters that legitimately
    chain assemble_context across roles in one turn (none today, but
    not impossible) leave it off."""

    name: str = "assemble_context_once"

    def check(self, ctx: PreToolContext) -> PreToolOutcome:
        if ctx.call.name != "assemble_context":
            return Continue()
        already_ran = any(name == "assemble_context" for name, _args in ctx.seen_calls)
        if not already_ran:
            return Continue()
        return Skip(
            ToolResult(
                tool_name=ctx.call.name,
                output=_ASSEMBLE_CONTEXT_ONCE_OUTPUT,
                success=False,
                error="assemble_context_once",
            )
        )


# Skip-result body when fetch_url is gated. Phrased as a tool result
# (success=False) so the model sees a clean, actionable instruction
# instead of a halt. The message names the reason ('user did not paste
# a URL') and tells the model what to do instead (answer from the
# already-pasted content).
_FETCH_URL_GUARD_OUTPUT = (
    "fetch_url skipped — the user's message did not contain a URL. "
    "This character only fetches external pages when the user "
    "explicitly pastes a link to source material; otherwise it works "
    "from the text the user already provided. Re-plan: answer from "
    "the user's pasted content + the grounding context you already "
    "have. Do NOT retry fetch_url unless the user's NEXT message "
    "actually contains a URL."
)


@dataclass(frozen=True)
class FetchUrlGuardHook:
    """Skip speculative `fetch_url` calls when the user hasn't pasted a URL.

    Failure mode this catches: airton_c_tfr (and similar paste-only
    personas) ship `fetch_url` in their tool set so a pilot can hand
    over a `tfr.faa.gov` link instead of NOTAM text. But the model
    sometimes speculates — observed 2026-05-15 with a Palm Beach TFR
    where the model invented a guessed FAA listing URL
    (`/air_traffic/air_facts/notams/?state=fl&city=Palm+Beach`),
    got a 403, and prefixed the actual decode with an error-recovery
    preamble. The constitution says fetch_url is for the
    user-pasted-URL case only; this hook enforces it structurally.

    Rule:
      1. Only checks calls to `fetch_url` (or `fetch_url_text`,
         `fetch_url_json` — any tool whose name starts with
         `fetch_url`). Other tools pass through.
      2. Requires `user_message` to contain at least one http(s)://
         URL. Missing user_message OR no URL → Skip with the canned
         re-plan tool result.
      3. URL present → Continue; downstream grounding hook
         (argument_grounding) and the tool's own host allowlist
         decide whether the specific URL is allowed.

    Skip is preferred over Halt: the model gets the error-shaped
    tool result and naturally pivots in its next round. Halt would
    end the turn with a canned message, dropping the actual decode
    work the user wants.

    Opt-in via the character's `catchers:` roster (member name
    `fetch_url_guard`). Non-paste-only characters who DO want
    speculative web research (e.g. ab's open-ended planning) leave
    the catcher off and keep the freer behaviour."""

    name: str = "fetch_url_guard"

    def check(self, ctx: PreToolContext) -> PreToolOutcome:
        if not ctx.call.name.startswith("fetch_url"):
            return Continue()
        if ctx.user_message and _URL_IN_TEXT_RE.search(ctx.user_message):
            return Continue()
        return Skip(
            ToolResult(
                tool_name=ctx.call.name,
                output=_FETCH_URL_GUARD_OUTPUT,
                success=False,
                error="fetch_url_guard",
            )
        )


# Persist-body citation enforcement. The constitution forbids stripping
# URL citation tokens (`[arxiv:..]` / `[doi:..]` / etc) when persisting
# research summaries, but small-model drift keeps reducing them to
# parenthetical author descriptions ("(Choi et al, brain networks)").
# Three smoke cycles in a row demonstrated the constitution +
# nudge-text tightening were insufficient — needed a structural gate.
# This hook blocks `remember_event` PRE-execution when search_scholar
# ran this turn AND the body has fewer than this many URL tokens.
# Matches the post_research_persist threshold so the two gates align.
_PERSIST_BODY_MIN_URL_TOKENS: int = 2

_PERSIST_BODY_STRIP_NUDGE = (
    "Your `remember_event` body has fewer than 2 URL citation tokens "
    "(`[arxiv:..]` / `[doi:..]` / `[scholar:..]` / `[wiki:..]`), but "
    "`search_scholar` ran this turn — the persisted row MUST be re-"
    "fetchable from its body alone. Re-issue `remember_event` with "
    "the body rewritten so every paper you cite carries its full URL "
    "token verbatim, not just a parenthetical author description. "
    "If they don't all fit in 1-3 sentences, drop one or two papers "
    "rather than truncating the URL tokens — fewer fully-cited "
    "entries beats five tokenless ones. The row was NOT written; "
    "your previous call was rejected by the persist-body gate."
)


@dataclass(frozen=True)
class PersistBodyCitationsHook:
    """Block a `remember_event` call whose body is missing URL citation
    tokens, when `search_scholar` ran this turn.

    Failure mode this catches: airton_f's persist path strips URL
    tokens from the body, keeping only parenthetical author
    descriptions. Three smoke cycles repeated the pattern despite
    the constitution rule (458cf87) and the nudge-text tightening.
    Pre-tool enforcement so the bad write never reaches the user's
    write-tier approval dialog — Skip feeds an error result back to
    the model and it retries with a corrected body.

    Trigger conditions (ALL must hold):
      1. Call name is `remember_event`.
      2. `search_scholar` appears in `seen_calls` this turn — i.e.
         this is the new-research persist path, not some other
         memory-write flow. Other persist calls (e.g. user-asked
         "remember that …" chit-chat) don't carry the citation
         contract.
      3. `body` argument is a string with fewer than
         `_PERSIST_BODY_MIN_URL_TOKENS` URL citation tokens
         (matches the post_research_persist trigger threshold).

    Action: Skip with a `success=False` ToolResult carrying the
    re-issue instructions. The model sees the failure as a tool-
    role message and retries.

    Opt-in via the character's `catchers:` roster as
    `persist_body_citations`."""

    name: str = "persist_body_citations"

    def check(self, ctx: PreToolContext) -> PreToolOutcome:
        if ctx.call.name != "remember_event":
            return Continue()
        search_scholar_ran = any(n == "search_scholar" for n, _ in ctx.seen_calls)
        if not search_scholar_ran:
            return Continue()
        body = ctx.call.arguments.get("body", "")
        if not isinstance(body, str):
            return Continue()
        url_count = len(_URL_CITATION_RE.findall(body))
        if url_count >= _PERSIST_BODY_MIN_URL_TOKENS:
            return Continue()
        return Skip(
            ToolResult(
                tool_name=ctx.call.name,
                output=_PERSIST_BODY_STRIP_NUDGE,
                success=False,
                error="persist_body_citations",
            )
        )


# Trigger words that legitimize an `Opinion:` paragraph in the reply.
# When the user's most recent message contains none of these, a reply
# that nonetheless produces an Opinion: paragraph is constitution-
# violating and trips OpinionWithoutTriggerHook. Case-insensitive
# match against the user message; the literal "Opinion:" token in the
# reply stays case-sensitive (it's the canonical form the constitution
# documents).
_OPINION_TRIGGER_WORDS: tuple[str, ...] = (
    "opinion",
    "opinions",
    "thoughts",
    "what do you think",
    "your view",
    "your take",
)


_OPINION_NO_TRIGGER_NUDGE = (
    "Your reply contains an `Opinion:` paragraph but the user did "
    "NOT use any of these trigger phrases: opinion(s), thoughts, "
    "what do you think, your view, your take. DELETE the Opinion "
    "paragraph entirely and re-answer with the requested content "
    "only. Do NOT keep the Opinion content under a different label "
    "(no 'Note:', 'Analysis:', 'My view:'). The user asked for "
    "content; give content."
)


@dataclass(frozen=True)
class OpinionWithoutTriggerHook:
    """Nudge a reply that produced an `Opinion:` paragraph without
    the user explicitly asking for an opinion.

    Failure mode this catches: airton_f's constitution gates
    `Opinion:` paragraphs on explicit user request — the user has
    to use words like "opinion", "thoughts", "what do you think",
    or "your view". Smoke 2026-05-15: user asked "Search the web
    to find modern key exchange mechanisms and contrast them with
    diffie-hellman" — no trigger words. Model produced an `Opinion:`
    paragraph anyway, twice in a row, even after the constitution
    was tightened with the explicit trigger-word list. Constitution-
    only enforcement insufficient → structural hook.

    Trigger conditions (ALL must hold):
      1. The reply contains the literal token `Opinion:` (case-
         sensitive — that's the canonical form the constitution
         documents; "opinion" inside other prose is fine).
      2. `user_message` is non-None and contains NONE of
         `_OPINION_TRIGGER_WORDS` (case-insensitive substring).

    Action: Nudge the model to drop the opinion paragraph and
    re-answer with content only. Retry-able.

    Opt-in via the character's `catchers:` roster as
    `opinion_no_trigger`. Characters that allow unprompted opinions
    (most non-scholar personas) leave it off."""

    name: str = "opinion_no_trigger"

    def check(self, ctx: BailContext) -> BailOutcome:
        if "Opinion:" not in ctx.reply.content:
            return Continue()
        user_msg = (ctx.user_message or "").lower()
        if not user_msg:
            # No user message threaded through (bootstrap / subagent
            # contexts). Conservative: don't nudge — better to let a
            # legitimate opinion through than to false-positive on a
            # context the hook can't reason about.
            return Continue()
        for trigger in _OPINION_TRIGGER_WORDS:
            if trigger in user_msg:
                return Continue()
        return Nudge(text=_OPINION_NO_TRIGGER_NUDGE)


# URL citation forms airton_f and similar scholar personas use:
# [arxiv:2401.12345], [scholar:Author Year], [doi:10.1145/...],
# [wiki:Article_Name]. Matches the four tier-aware forms the
# constitution documents. The first capture group is the source
# kind; the bracket boundary anchors the match so prose mentions of
# "arxiv" alone don't trip it.
_URL_CITATION_RE = re.compile(
    r"\[(arxiv|scholar|doi|wiki):[^\]]+\]",
    re.IGNORECASE,
)


_POST_SEARCH_GROUNDING_NUDGE = (
    "You called a search tool (`search_web` or `search_scholar`) "
    "but your reply did not cite any of its results, fetch a URL, "
    "or refine the search. Take ONE of these actions NOW — by "
    "ISSUING the corresponding tool call OR by SUMMARIZING the "
    "search results inline with citations:\n"
    "\n"
    "ACTION 1 — Summarize the search results in prose with URL "
    "citations: `[scholar:<paperId>]`, `[arxiv:<id>]`, "
    "`[doi:<id>]`, or `[wiki:<article>]`. Use the URLs the tool "
    "returned. This is the right move when the user asked you to "
    "'review', 'summarize', 'explain', or 'find' — they want "
    "content, not just a list.\n"
    "ACTION 2 — Issue a `fetch_url` tool call with the URL of a "
    "search result. Prefer tier-1 hosts (scholar.google.com, "
    "arxiv.org).\n"
    "ACTION 3 — Issue a `search_web` or `search_scholar` tool "
    "call again with a refined query (e.g. add a `site:` operator "
    "or narrower terms).\n"
    "ACTION 4 — Reply with ONLY this sentence and NO other content: "
    "'No allowlisted source covered this query — want to broaden "
    "the allowlist?'\n"
    "\n"
    "DO NOT write 'you might want to search the web...' or 'Let's "
    "refine the search...' — that hands the work back to the user "
    "instead of doing it. DO NOT label content summaries as "
    "'Opinion:'; a summary of search results is content, not "
    "opinion."
)


# Tools that count as a "search ran" for the post-search grounding
# check. search_web (DDG general web) and search_scholar (Semantic
# Scholar + OpenAlex academic) both produce result lists the model is
# expected to ground its reply in. Adding a future search tool
# (PubMed, etc.) means adding its name here.
_SEARCH_TOOLS: frozenset[str] = frozenset({"search_web", "search_scholar"})


@dataclass(frozen=True)
class PostSearchGroundingHook:
    """Nudge a reply that called a search tool but didn't follow up
    with a fetch, a refined search, an inline summary with URL
    citations, or an explicit no-allowlisted-source statement.

    Failure mode this catches: scholar-style characters declare
    "after a search, ground or refine, never paraphrase" in their
    constitution. Smoke 2026-05-15 (search_web repro): search_web
    returned a Wikipedia hit; model didn't fetch_url it, didn't cite
    `[wiki:…]`, didn't refine — went straight to training-data prose.
    JEPA repro (search_scholar): search_scholar returned 5 real
    papers; model labeled its summary `Opinion:`; opinion_no_trigger
    nudged it away; the retry produced "you might want to search the
    web" with NO citations and NO substance. Both shapes have the
    same root: a search tool fired but the reply isn't grounded in
    its output.

    Trigger conditions (ALL must hold):
      1. At least one tool from `_SEARCH_TOOLS` (search_web,
         search_scholar) in `ctx.tools_ran` this turn.
      2. `fetch_url` NOT in `ctx.tools_ran` (no follow-up fetch).
      3. Reply is substantive (>= 80 chars). Short refusals like
         "no results, broaden?" don't need a URL.
      4. Reply contains NO URL citation pattern (`[arxiv:…]`,
         `[scholar:…]`, `[doi:…]`, `[wiki:…]`) AND NO raw http(s)://
         URL. Either form counts as grounding-after-search.

    Action: Nudge the model to take one of the four valid next moves
    (summarize-with-citations, fetch, refine, or no-source refusal).
    Retry-able.

    Opt-in via the character's `catchers:` roster as
    `post_search_grounding`. Characters that use search tools for
    their own sake (e.g. general-web-research personas without a
    citation discipline) leave it off."""

    name: str = "post_search_grounding"

    def check(self, ctx: BailContext) -> BailOutcome:
        if not (ctx.tools_ran & _SEARCH_TOOLS):
            return Continue()
        if "fetch_url" in ctx.tools_ran:
            return Continue()
        content = ctx.reply.content
        if len(content) < 80:
            return Continue()
        if _URL_CITATION_RE.search(content):
            return Continue()
        if _URL_IN_TEXT_RE.search(content):
            return Continue()
        return Nudge(text=_POST_SEARCH_GROUNDING_NUDGE)


# Persist research — scholar-style memory write enforcement. The
# contract has a `prior_discussion` episodic slot that recalls past
# research summaries on future turns, but only if something actually
# wrote them. PostResearchPersistHook nudges the model to commit a
# multi-source summary via `remember_event` before exiting the tool
# loop. Trigger is `search_scholar` ran (academic search — durable
# enough to persist) + reply has ≥2 URL citations (multi-source
# synthesis) + no `remember_event` call this turn.
_POST_RESEARCH_PERSIST_TOOL: str = "search_scholar"
_POST_RESEARCH_PERSIST_MIN_CITATIONS: int = 2

# Recall-lead marker: when the reply leads with one of these phrases,
# the model is rendering from `prior_discussion` (per the constitution's
# workflow Step 2 — harness-thhy). No new research was synthesized —
# the work is already in episodic memory — so PostResearchPersistHook
# must NOT nudge for a remember_event call. Persisting again would
# create a duplicate row. Constitution paired with this gate: airton_f
# explicitly tells the model to lead recall replies with "From prior
# discussion (captured <YYYY-MM-DD>):" — anchoring this regex on the
# documented form. Case-insensitive; accepts a leading whitespace
# prefix so model formatting variations (markdown bullet, blockquote)
# don't slip past.
_RECALL_LEAD_RE: re.Pattern[str] = re.compile(
    r"\s*(?:From\s+(?:a\s+)?prior\s+(?:discussion|session)|Found\s+in\s+prior\s+(?:discussion|session))",
    re.IGNORECASE,
)


def _build_post_research_persist_nudge(today: str) -> str:
    return (
        "You produced a multi-source research summary (2+ URL citations) "
        "but did NOT call `remember_event` to persist it. Future turns "
        "cannot recall what you found without this write — the contract's "
        "`prior_discussion` slot returns rows from episodic memory, and "
        "no row exists yet. Take this action NOW before the final reply:\n"
        "\n"
        "Issue a `remember_event` tool call with these arguments:\n"
        "  title: a short topic phrase (e.g. 'JEPA in predictive models')\n"
        f"  body: MUST start with the literal prefix `Captured: {today} — `, "
        "then a 1-3 sentence distillation. **EVERY paper you cite in the "
        "body MUST appear with its full URL citation token verbatim** — "
        "i.e. the literal `[arxiv:<id>]` / `[doi:<id>]` / `[scholar:<…>]` "
        "token from your final reply. A parenthetical author description "
        "like `(Choi et al, GNNs for brain networks)` is NOT a substitute "
        "for the citation token; the body must be re-fetchable months "
        "from now without re-searching. If the URL tokens don't all fit "
        "in 1-3 sentences, drop a paper or two — better fewer fully-cited "
        "entries than five truncated ones.\n"
        '  tags: ["research", "<topic-slug>"]  (lowercase, hyphenated)\n'
        "\n"
        "After the tool call returns, write your final reply (the same "
        "summary). DO NOT skip the persist step — without it, the scholar "
        "has no memory of this research between sessions."
    )


@dataclass(frozen=True)
class PostResearchPersistHook:
    """Nudge a reply that produced a multi-source research summary
    but didn't persist it via `remember_event`.

    Failure mode this catches: airton_f's contract has a
    `prior_discussion` episodic slot that surfaces past research
    summaries on follow-up turns ("what did we find on JEPA last
    week?"). But the slot can only return rows that something
    actually wrote — and the model's natural drift is to render a
    summary and stop, leaving no durable trace. Structural
    backstop for the constitution's "Persisting research" rule.

    Trigger conditions (ALL must hold):
      1. `search_scholar` (the primary academic-search tool) ran
         this turn. Generic `search_web` is excluded — its hits are
         too transient / non-academic to be worth a durable row.
      2. `remember_event` did NOT run this turn.
      3. Reply contains >= `_POST_RESEARCH_PERSIST_MIN_CITATIONS`
         URL citation tokens. Multi-source synthesis is worth
         persisting; single-paper drill-down stays in transcript.
      4. Reply is NOT a `prior_discussion` recall — `_RECALL_LEAD_RE`
         doesn't match at the start of the reply (harness-i5pk).
         Smoke 2026-05-15: constitution Step 2 worked — model led
         with "From prior discussion (captured 2026-05-15):" — but
         search_scholar had ALSO been called (router fronted) and
         the reply quoted 5 URL tokens from the recalled memory, so
         the hook nudged anyway. Result: model retried, conflated
         this-turn search results with the prior recall, persisted
         a duplicate memory. Skipping recall replies prevents that
         loop. The constitution paired with this gate forbids
         calling `remember_event` on a recall path.

    Action: Nudge the model to call `remember_event` with a stamped
    body (`Captured: <YYYY-MM-DD> — ...`). Today's date is injected
    into the nudge text so the model doesn't have to guess it.
    `today_provider` lets tests inject a fixed date; default reads
    today from `datetime.now(UTC).date().isoformat()` at check-time
    (NOT import-time — long-running sessions still get today's date).

    Opt-in via the character's `catchers:` roster as
    `post_research_persist`. airton_f ships it; other characters
    that adopt `search_scholar` should opt in too."""

    name: str = "post_research_persist"
    today_provider: Callable[[], str] = field(default=lambda: datetime.now(UTC).date().isoformat())

    def check(self, ctx: BailContext) -> BailOutcome:
        if _POST_RESEARCH_PERSIST_TOOL not in ctx.tools_ran:
            return Continue()
        if "remember_event" in ctx.tools_ran:
            return Continue()
        if _RECALL_LEAD_RE.match(ctx.reply.content):
            return Continue()
        citations = _URL_CITATION_RE.findall(ctx.reply.content)
        if len(citations) < _POST_RESEARCH_PERSIST_MIN_CITATIONS:
            return Continue()
        return Nudge(text=_build_post_research_persist_nudge(self.today_provider()))


# ---------- finalize hooks ----------


# JO-style section citation: `§ 12-1-2`, `§12-1-2`, tolerant of hyphen /
# en-dash / unicode-minus. Three numeric segments separated by dashes
# pins it to the JO 7110.65 shape (chapter-section-paragraph) without
# false-positiving on prices (`$19.99`), simple anchors (`§2`), or
# two-segment numbering (`§3-1`). Conservative by design — false
# positives here would replace good replies with a canned refusal.
UNGROUNDED_SECTION_CITATION_RE = re.compile(
    r"§\s*\d+[-–−]\d+[-–−]\d+",  # noqa: RUF001 — en-dash + minus are load-bearing variants seen in corpus
)


# The set of tools whose execution constitutes "grounding" for a
# section-citation reply. Intentionally narrow: only memory-backed
# retrieval counts. `fetch_url`, `search_web`, `read_file`, etc. do
# NOT count — a web-fetch result is not a substitute for having pulled
# the citation from character-owned memory. `assemble_context`
# (harness-j5cs) counts because it fans out into the same memory
# stores via the contract orchestrator — the agent sees a packaged
# bundle of episodic / tabular / tree hits with provenance.
_GROUNDING_TOOLS: frozenset[str] = frozenset({"search_memory", "fact_search", "assemble_context"})


# Canned refusal shown when the hook fires. Character-agnostic — the
# character's `cite_or_silent` / scope text lives in its constitution
# and drives the system prompt, but this hook runs after the reply is
# already generated and doesn't have a hook into per-character refusal
# templates. Keep the wording in the first person + short so it reads
# as the character's own voice regardless of who's speaking.
UNGROUNDED_CITATION_FALLBACK = (
    "I can't answer that from memory without fabricating a citation. "
    "Either that topic is outside what I cover, or I don't have the "
    "source material for it indexed. If you have the reference handy, "
    "paste it and I'll work from there."
)


LOW_CONFIDENCE_FALLBACK = (
    "I don't have a solid source for this — the best retrieval match I "
    "could find scored below my confidence threshold. Check the relevant "
    "section directly rather than relying on what I said."
)


# Default retrieval-score threshold below which citations that weren't
# explicitly grounded by a tool's structured return are treated as
# low-confidence and refused. 0.5 matches the passive-retrieval floor
# used in cli.py (`--memories-threshold`). Tunable per-hook at
# construction; the threshold-calibration follow-up (harness-5c0) will
# histogram the real distribution and move this to a character-config
# override.
_LOW_CONFIDENCE_THRESHOLD_DEFAULT = 0.5


@dataclass(frozen=True)
class LowConfidenceFallbackHook:
    """Catch section-citation replies where a grounding tool DID run
    but scored below the confidence threshold AND the cited section
    isn't in the tool's declared `citations_grounded` (harness-ywp.3).

    Complementary to UngroundedCitationHook:
      * UngroundedCitationHook fires when NO grounding tool ran.
      * LowConfidenceFallbackHook fires when a grounding tool ran but
        weakly (top hit below threshold) and the reply cites a
        section the tool didn't surface.

    Trigger conditions (ALL must hold):
      1. `ctx.retrieval_top_score` is set and < `threshold`.
      2. Reply contains at least one JO-style `§N-N-N` / `§N-N` /
         TBL/Table/FIG/Figure N-N-N citation (same extractor the
         audit log + tool-declared grounding use — harness-ywp.5).
      3. At least one of those citations is NOT in
         `ctx.citations_grounded` — i.e. the model cited something
         the tool didn't actually ground.

    Action: replace the reply with a character-agnostic fallback that
    tells the user the retrieval was weak and to check the source
    directly. Same reasoning as UngroundedCitationHook's refusal —
    a confident-looking answer riding on weak retrieval is worse than
    an honest 'I'm not sure, go look'.

    Runs BEFORE UngroundedCitationHook in the finalize phase. When
    THIS hook halts, the later hook never runs (its preconditions
    overlap but UngroundedCitationHook's tools_ran check will usually
    Continue here anyway — a tool did run)."""

    name: str = "low_confidence_fallback"
    threshold: float = _LOW_CONFIDENCE_THRESHOLD_DEFAULT

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        if ctx.retrieval_top_score is None:
            return Continue()
        if ctx.retrieval_top_score >= self.threshold:
            return Continue()
        from harness.tools.citations import extract_citations

        cited = extract_citations(ctx.reply.content)
        if not cited:
            return Continue()
        ungrounded = cited - ctx.citations_grounded
        if not ungrounded:
            return Continue()
        reply = ctx.reply
        return Halt(
            ModelReply(
                content=LOW_CONFIDENCE_FALLBACK,
                tool_calls=(),
                was_truncated=reply.was_truncated,
                had_unparseable_call=reply.had_unparseable_call,
            )
        )


@dataclass(frozen=True)
class UngroundedCitationHook:
    """Catch section-citation replies that never touched a grounding
    tool and had no memory block attached to the prompt (harness-oc8).

    Failure mode this catches: character is scoped to a specific source
    document (e.g. airton_c1 / JO 7110.65). User asks an out-of-scope
    or poorly-retrieved question. Retrieval scores all memory hits
    below the floor, so no memory block is attached. The main loop
    never fires a grounding tool. The model then fabricates a
    parametric-knowledge answer and closes with a real-looking section
    number — real `§N-N-N` string, fabricated attribution. Classic
    ungrounded citation.

    Trigger conditions (ALL must hold):
      1. Reply contains at least one JO-style `§N-N-N` citation.
      2. No memory block was attached for this turn
         (`ctx.memory_block_attached is False`).
      3. No grounding tool (`search_memory` / `fact_search`) ran this
         turn (`ctx.tools_ran` disjoint from `_GROUNDING_TOOLS`).

    Action: replace the reply with a character-agnostic refusal. We
    deliberately do NOT try to repair the reply by stripping just the
    citation — a confident fabricated body minus its source line is
    worse than a refusal.

    Runs BEFORE `fabrication_fallback` in the finalize phase: if this
    hook Halts, the fallback never runs (which is correct — our
    replacement is the terminal answer). If the reply has no section
    citation, we Continue and the fallback (if any) fires normally."""

    name: str = "ungrounded_citation"

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        if ctx.memory_block_attached:
            return Continue()
        if ctx.tools_ran & _GROUNDING_TOOLS:
            return Continue()
        if not UNGROUNDED_SECTION_CITATION_RE.search(ctx.reply.content):
            return Continue()
        reply = ctx.reply
        return Halt(
            ModelReply(
                content=UNGROUNDED_CITATION_FALLBACK,
                tool_calls=(),
                was_truncated=reply.was_truncated,
                had_unparseable_call=reply.had_unparseable_call,
            )
        )


@dataclass(frozen=True)
class UncitedSubstantiveReplyHook:
    """Finalize-phase mirror of MissingCitationHook (harness-zsbz).

    MissingCitationHook is bail-phase: when a substantive reply omits a
    §-anchor, it Nudges so the model can add one on retry. If retries
    are exhausted and the reply STILL touches the corpus document
    without anchoring it, that's the fabrication this persona exists
    to prevent — the model is paraphrasing from priors rather than
    from a section it actually saw. Halt and replace with the
    character's scope-redirect template instead of shipping it.

    Motivating failure (2026-05-14 chat session, mark): airton_c1
    asked an off-topic question ('wedding ring on which finger?'). The
    first draft answered from general knowledge; missing_citation
    fired; retry prefixed with 'JO 7110.65 doesn't cover it' (passing
    `_SCOPE_REDIRECT_MARKER_RE`) but then continued with three
    paragraphs of general-knowledge content. MissingCitationHook
    Continued; finalize had no hook for this shape; the fabricated
    answer shipped.

    Trigger conditions (ALL must hold):
      0. Character ships both `citation_grammar` and
         `scope_redirect_template`. Either alone → silent.
      1. A grounding tool ran this turn
         (`ctx.tools_ran` intersects `_GROUNDING_TOOLS`).
      2. Reply contains corpus document reference
         (`grammar.document_reference`) OR a JO phraseology marker
         (`_JO_PHRASEOLOGY_MARKERS_RE`).
      3. Reply is substantive (>= 80 chars).
      4. Reply contains NO §-anchor / TBL anchor.
      5. Reply is NOT a clarifying question.
      6. EITHER the scope-redirect marker is absent OR the reply is
         longer than `_SCOPE_REDIRECT_SOFT_LIMIT` chars. A short, clean
         scope-redirect (marker + concise body) is the right outcome;
         this hook stays silent. A long body with the marker is the
         'redirect-then-answer' shape and gets replaced.

    Action: Halt with the character's `scope_redirect_template` as
    the reply content. The bail-phase MissingCitationHook had two
    rounds to extract a citation; if the model still can't, the
    honest answer is 'outside my scope.'

    Runs AFTER UngroundedCitationHook in finalize so the no-tool-ran
    fabrication path is handled there first. The two hooks are
    complementary: ungrounded_citation catches 'cite, no tool';
    uncited_substantive_reply catches 'tool, no cite' after the
    retry budget has been spent."""

    grammar: CitationGrammar | None = None
    scope_redirect_template: str | None = None
    name: str = "uncited_substantive_reply"

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        if self.grammar is None:
            return Continue()
        template = self.scope_redirect_template
        if not template or not template.strip():
            return Continue()
        if not (ctx.tools_ran & _GROUNDING_TOOLS):
            return Continue()
        content = ctx.reply.content
        if len(content) < 80:
            return Continue()
        in_scope = bool(self.grammar.document_reference.search(content)) or bool(
            _JO_PHRASEOLOGY_MARKERS_RE.search(content)
        )
        if not in_scope:
            return Continue()
        if _reply_has_citation(content, self.grammar):
            return Continue()
        # Same URL-citation early-return as MissingCitationHook
        # (smoke 2026-05-15 JEPA): scholar-style replies grounded
        # via `[arxiv:..]` / `[scholar:..]` / `[doi:..]` / `[wiki:..]`
        # are cited even when no §-form anchor is present.
        if _URL_CITATION_RE.search(content):
            return Continue()
        # Same §-gate as MissingCitationHook above: exempt
        # clarifying-question replies only when no `§` is present.
        # bare-§ + clarifying question is a malformed-citation
        # fabrication, not a pure clarifier.
        if _CLARIFYING_QUESTION_RE.search(content) and "§" not in content:
            return Continue()
        # Short scope-redirects pass through. Longer marker-bearing
        # replies are the redirect-then-answer fabrication shape.
        # Same §-gate: don't exempt a reply that combines redirect
        # phrasing with a §-shaped citation attempt.
        if (
            _SCOPE_REDIRECT_MARKER_RE.search(content)
            and len(content) <= _SCOPE_REDIRECT_SOFT_LIMIT
            and "§" not in content
        ):
            return Continue()
        reply = ctx.reply
        return Halt(
            ModelReply(
                content=template.strip(),
                tool_calls=(),
                was_truncated=reply.was_truncated,
                had_unparseable_call=reply.had_unparseable_call,
            )
        )


# Matches a markdown pipe-table row: whitespace + `|` + at least two
# cells separated by more `|`s. Column count ≥3 pins it to "real table"
# shape — a stray `a | b` in prose doesn't trigger. We strip the row
# and its cells at check time, so tolerant of surrounding whitespace.
_PIPE_ROW_RE = re.compile(r"^\s*\|[^|\n]+\|[^|\n]+\|[^\n]*\|\s*$", re.MULTILINE)


# Markdown header separator row (e.g. `|---|---|---|`). These carry no
# data; excluded before we check rows against tool outputs so a missing
# separator never fires the catcher.
_PIPE_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$", re.MULTILINE)


# Pull the digit-bearing tokens we care about. Two-or-more-digit runs
# avoid coincidental single-digit matches ("3 miles" prose colliding
# with "3,000" in tool output). Accepts commas and decimals as internal
# separators so "1,000" / "1.25" survive as single tokens.
_NUMERIC_TOKEN_RE = re.compile(r"\d[\d,.]{1,}")


def _normalize_for_compare(s: str) -> str:
    """Collapse whitespace + strip commas so "|MH|Under 50|25|" matches
    "| MH | Under 50 | 25 |" and "1,000" matches "1000". We keep the
    `|` delimiter intact — the catcher's signal is cell-pair adjacency,
    and pipes are the row scaffolding we want to preserve."""
    return re.sub(r"\s+", "", s).replace(",", "")


# Canned refusal when the table-fabrication catcher fires. Stays
# character-agnostic — same reasoning as UNGROUNDED_CITATION_FALLBACK:
# by the time we're replacing the reply we don't have a hook into
# per-character refusal templates, and first-person keeps it reading
# as the character's own voice. Short-and-specific beats generic: the
# user gets a clear signal that the table (not the whole answer) was
# the suspect part.
TABLE_FABRICATION_FALLBACK = (
    "I can't reproduce that table from memory without risking a "
    "fabricated row. The retrieved source didn't include the full "
    "table, and I won't fill the gaps from priors. If you have the "
    "section's table in front of you, paste it and I'll work from "
    "there; otherwise ask me about a specific row and I'll try to "
    "retrieve it."
)


@dataclass(frozen=True)
class TableFabricationHook:
    """Catch replies whose pipe-table data rows don't appear verbatim
    in any tool output this turn (harness-5uq).

    Failure mode this catches: retrieval surfaces a section whose full
    numeric table was truncated by `SearchMemoryTool`'s body cap (or
    split across chunk boundaries by the chunker). The model correctly
    cites the section, sees the partial table fragment, and
    reconstructs the missing rows from parametric priors — a
    citation-real, table-fabricated reply. Reproducer (session
    2026-04-24, airton_c1): MH class RBN, cited §4-1-1 correctly,
    fabricated `|MH|50 - 1,999|50|` when the source row is
    `|MH|Under 50|25|`.

    Trigger conditions (ALL must hold):
      1. A grounding tool ran this turn (`ctx.tools_ran` intersects
         `_GROUNDING_TOOLS`). Without a grounding tool we defer to
         `ungrounded_citation` — a fabricated table with no grounding
         tool IS a fabricated citation, caught upstream.
      2. Reply contains ≥2 pipe-table rows (header + ≥1 data row).
         Non-separator, digit-bearing rows are the data rows.
      3. At least one data row's normalized pipe-signature does NOT
         appear in `ctx.tool_outputs` (also normalized).

    Action: Halt with the canned refusal. Deliberately does NOT attempt
    to strip just the offending row — leaving an explanation around a
    partial table is a subtler form of the same problem.

    Runs AFTER `ungrounded_citation` (which halts on the harder shape
    of fully-fabricated citations) and BEFORE `fabrication_fallback`.
    If no grounding tool ran but a table is still present, the earlier
    hook handles it; this one covers the narrower "grounded citation,
    fabricated row" case."""

    name: str = "table_fabrication"

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        if not (ctx.tools_ran & _GROUNDING_TOOLS):
            return Continue()
        rows = _PIPE_ROW_RE.findall(ctx.reply.content)
        if len(rows) < 2:
            return Continue()
        # Data rows: skip the markdown header separator (`|---|---|`)
        # and the header row itself (first match with no digits). A
        # row with zero numeric tokens can't be a numeric-fabrication
        # target, so we skip it rather than halt on label-only rows.
        data_rows = [
            r for r in rows if not _PIPE_SEPARATOR_RE.match(r) and _NUMERIC_TOKEN_RE.search(r)
        ]
        if not data_rows:
            return Continue()
        joined_outputs = "\n".join(ctx.tool_outputs)
        normalized_outputs = _normalize_for_compare(joined_outputs)
        for row in data_rows:
            if _normalize_for_compare(row) in normalized_outputs:
                continue
            # Row signature absent from tool outputs → fabricated.
            # Halt immediately; one bad row is enough to poison the
            # whole table from the user's perspective.
            reply = ctx.reply
            return Halt(
                ModelReply(
                    content=TABLE_FABRICATION_FALLBACK,
                    tool_calls=(),
                    was_truncated=reply.was_truncated,
                    had_unparseable_call=reply.had_unparseable_call,
                )
            )
        return Continue()


# Unit keywords we know how to recognize in a table header cell and in
# a prose claim. Kept narrow — only the ATC-corpus units that surface
# in JO 7110.65 tables (distances, altitudes, powers, bearings/speeds,
# times). `nm` and `mile` without trailing s are normalized by
# rstrip("s") at compare time so "miles" and "mile" canonicalize the
# same key.
_KNOWN_UNITS: frozenset[str] = frozenset(
    {
        "miles",
        "mile",
        "nm",
        "ft",
        "feet",
        "foot",
        "watts",
        "watt",
        "knots",
        "knot",
        "kt",
        "minutes",
        "minute",
        "min",
        "seconds",
        "second",
        "sec",
    }
)


# Pulls "<label> class ... <value> <unit>" claims out of prose. Label
# is 1-4 uppercase letters (matches atc class codes CL/MH/H/HH, airspace
# letters A/B/C/D/E/G, category codes CAT/III). `class` anchor word is
# required on one side so generic numbers like "50 miles apart" don't
# get picked up as labeled claims. Number allows commas + decimals.
# `DOTALL` + non-greedy `[^.|]*?` so we can span across a line without
# picking up pipe tables or sentence boundaries.
_LABELED_CLAIM_RE = re.compile(
    r"\b(?:([A-Z]{1,4})\s+class|class\s+([A-Z]{1,4}))\b"
    r"[^.|]*?"
    r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(miles?|NM|nm|ft|feet|foot|watts?|knots?|kt|minutes?|min|seconds?|sec)\b",
    re.IGNORECASE | re.DOTALL,
)


def _canon_unit(unit: str) -> str:
    """Lowercase + strip trailing s so 'Miles' / 'miles' / 'mile' all
    canonicalize to the same lookup key. `nm` stays as-is (not a plural
    form) — the rstrip is safe because `nm` has no trailing s to
    strip."""
    return unit.lower().rstrip("s")


def _extract_labeled_claims(reply: str) -> list[tuple[str, str, str]]:
    """Return (label, numeric_value, canonical_unit) triples from the
    reply's prose. Value is comma-stripped so '1,999' canonicalizes to
    '1999' for comparison against tool-output cells."""
    claims: list[tuple[str, str, str]] = []
    for match in _LABELED_CLAIM_RE.finditer(reply):
        label = (match.group(1) or match.group(2) or "").upper()
        value = match.group(3).replace(",", "")
        unit = _canon_unit(match.group(4))
        if label and unit in _KNOWN_UNITS:
            claims.append((label, value, unit))
    return claims


# Table-row shape after pipe-split: non-empty cell list. We strip
# surrounding whitespace + any emphasis markdown (`**bold**` / `*italic*`)
# from each cell so the header lookup finds "Distance" whether the source
# wrote `**Distance**` or just `Distance`.
def _strip_cell(cell: str) -> str:
    return cell.strip().strip("*").strip()


def _parse_pipe_tables(text: str) -> list[list[list[str]]]:
    """Extract pipe tables from `text`. Returns a list of tables;
    each table is a list of rows; each row is a list of stripped cells.
    A table ends when a non-pipe line breaks the run (blank line,
    prose, etc.) — same heuristic a markdown renderer uses."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("|") and line.count("|") >= 2:
            if _PIPE_SEPARATOR_RE.match(line):
                continue
            cells = [_strip_cell(c) for c in line.strip("|").split("|")]
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    return tables


def _unit_of_column(header_cell: str) -> str | None:
    """Best-effort unit-of-column parser. Looks for any known unit
    keyword as a substring of the header cell (case-insensitive),
    e.g. 'Distance (miles)' → 'mile'. None if the column isn't
    unit-tagged — we don't try to guess; unlabeled columns get skipped."""
    lower = header_cell.lower()
    for unit in _KNOWN_UNITS:
        if unit in lower:
            return _canon_unit(unit)
    return None


def _build_table_row_map(table: list[list[str]]) -> dict[str, dict[str, frozenset[str]]]:
    """Map {row_label: {unit: {acceptable_values}}} for a single pipe
    table. Label is the first cell of each row after the header.
    Acceptable_values is the set of numeric tokens found in that row's
    cell under the unit-tagged column — a single cell like 'Under 50'
    yields {'50'}; '14,500 - 17,999' yields {'14500', '17999'} so either
    endpoint matches (the model is free to cite the range's top or
    bottom as long as it doesn't invent a value)."""
    if len(table) < 2:
        return {}
    header = table[0]
    unit_columns: list[tuple[int, str]] = []
    for idx, cell in enumerate(header):
        unit = _unit_of_column(cell)
        if unit is not None:
            unit_columns.append((idx, unit))
    if not unit_columns:
        return {}
    out: dict[str, dict[str, frozenset[str]]] = {}
    for row in table[1:]:
        if not row:
            continue
        label = row[0].upper()
        if not label:
            continue
        for col_idx, unit in unit_columns:
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            nums = {
                _NUMERIC_TOKEN_RE.match(tok).group(0).replace(",", "")  # type: ignore[union-attr]
                for tok in _NUMERIC_TOKEN_RE.findall(cell)
            }
            if nums:
                existing = out.setdefault(label, {}).get(unit, frozenset())
                out[label][unit] = existing | nums
    return out


@dataclass(frozen=True)
class NumericFabricationHook:
    """Catch prose-form labeled numeric claims that disagree with the
    source table in the retrieved tool output (harness-5uq).

    Failure mode this catches: model retrieves a pipe table, reads the
    row labels, but swaps a different row's value onto the label the
    user asked about. Reproducer (session 2026-04-24 post-restart):
    tool output contained `|MH|Under 50|25|`, reply wrote
    'MH class RBN is 50 miles' — label MH pointed at 25 miles in the
    retrieved table, but the model chose 50 (the H row's value) from
    priors.

    Why this runs after `table_fabrication`: that hook only fires when
    the reply contains a literal pipe table; this one covers the
    structurally identical failure when the reply is prose ('MH is
    50 miles' instead of `| MH | … | 50 |`).

    Trigger conditions (ALL must hold):
      1. A grounding tool ran this turn (`ctx.tools_ran` intersects
         `_GROUNDING_TOOLS`). A prose numeric claim with no grounding
         is ungrounded-citation territory, handled upstream.
      2. Reply contains ≥1 labeled claim extracted by
         `_LABELED_CLAIM_RE` (e.g., 'MH class ... 50 miles').
      3. Some tool output contains a pipe table whose header has a
         column matching the claim's unit, AND whose first column has
         a row matching the claim's label.
      4. That row's unit-column value does NOT contain the claim's
         numeric value.

    Action: Halt with a canned refusal that names the fabrication
    shape (class label vs. cited value). Deliberately does not try to
    repair the reply by swapping in the true value — the model that
    fabricated once may re-ground wrong again next round; a refusal
    preserves trust.

    Silent cases (by design): reply has a labeled claim but no table
    in tool output matching the label → Continue. A claim the catcher
    can't verify against a structural source is out of scope."""

    name: str = "numeric_fabrication"

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        if not (ctx.tools_ran & _GROUNDING_TOOLS):
            return Continue()
        claims = _extract_labeled_claims(ctx.reply.content)
        if not claims:
            return Continue()
        tool_text = "\n".join(ctx.tool_outputs)
        tables = _parse_pipe_tables(tool_text)
        if not tables:
            return Continue()
        # Merge every table's row-map into one lookup. Overlap between
        # tables on the same label is rare in practice (one table per
        # topic) but we union values so a label appearing in two tables
        # passes if EITHER matches.
        merged: dict[str, dict[str, frozenset[str]]] = {}
        for table in tables:
            for label, units in _build_table_row_map(table).items():
                for unit, values in units.items():
                    existing = merged.setdefault(label, {}).get(unit, frozenset())
                    merged[label][unit] = existing | values
        for label, value, unit in claims:
            row_units = merged.get(label)
            if row_units is None:
                continue  # label unknown to any tool-output table
            allowed = row_units.get(unit)
            if allowed is None:
                continue  # unit-tagged column not present for this label
            if value in allowed:
                continue  # claim agrees with a known row value
            # Claim's label is in the table, unit-column exists for it,
            # but the value isn't one of the known cell values for that
            # (label, unit). Structural cross-row fabrication.
            reply = ctx.reply
            return Halt(
                ModelReply(
                    content=TABLE_FABRICATION_FALLBACK,
                    tool_calls=(),
                    was_truncated=reply.was_truncated,
                    had_unparseable_call=reply.had_unparseable_call,
                )
            )
        return Continue()


@dataclass(frozen=True)
class FabricationFallbackHook:
    """Substitute the canned refusal when bail retries are exhausted
    and the reply STILL trips a non-truncated bail outcome. Without
    this, the user would see the model's final hallucinated paragraph
    as the turn's answer (harness-24xj)."""

    name: str = "fabrication_fallback"

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        last = ctx.last_outcome
        # Only fabrication-shaped outcomes trigger the fallback. A
        # Continue means no catcher fired (genuine final reply), and a
        # Truncated means the partial reply is worth keeping.
        if not isinstance(last, Nudge):
            return Continue()
        reply = ctx.reply
        return Halt(
            ModelReply(
                content=EXHAUSTED_FABRICATION_FALLBACK,
                tool_calls=(),
                was_truncated=reply.was_truncated,
                had_unparseable_call=reply.had_unparseable_call,
            )
        )


# ---------- pipeline ----------


@dataclass(frozen=True)
class CatcherDoc:
    """One row in the hook registry — used by `HookPipeline.describe()`
    and by `scripts/gen_hook_docs.py` to render `docs/hooks.md`. Carries
    enough metadata to answer 'what does this catcher catch and where
    does it run' without re-reading the hook's docstring (harness-tm8t).
    """

    name: str
    phase: str  # "post_model" | "bail" | "pre_tool" | "post_tool" | "finalize"
    shape: str  # one-line failure-mode descriptor


# Maps each registered hook name to a one-line shape descriptor.
# Source of truth for `docs/hooks.md`. New hooks MUST add an entry here
# — the registry test (`tests/test_hook_registry.py`) enforces coverage.
HOOK_SHAPES: dict[str, str] = {
    # bail-phase catchers (read the model's content reply post-generation).
    "truncated": "Reply hit the token budget; auto-widen + retry.",
    "unparseable": "Reply had a malformed <tool_call> block.",
    "teaser": "Reply announced more work but emitted no tool call.",
    "false_success": "Reply claims a file edit without a write-tier tool call.",
    "meta_confirm": "Reply asks user to confirm in chat instead of calling the tool.",
    "raw_results_dump": (
        "Reply repeats tool output verbatim instead of synthesizing as the prompt asked."
    ),
    "fabricated_search": "Reply narrates web-search activity but no web tool ran.",
    "fabricated_itemization": "Reply fabricates additional list items beyond what was real.",
    "thin_source_fabrication": (
        "Reply makes specific numeric claims off a tool body that contained almost none."
    ),
    "incomplete_multipart": (
        "Multi-part prompt; reply gives up on missing sub-ask instead of another tool call."
    ),
    "ab_fabrication": "Reply imitates ab_ops output without a real tool call.",
    "tool_intent": "Reply restates a tool-call intent as prose, no actual call.",
    "ambiguous_context": "Reply silently picks one variant of an ambiguous term.",
    "scope_redirect": "Reply talks domain content for an out-of-scope question.",
    "reserved_squawk_code": "Reply assigns a reserved transponder code (7500/7600/7700).",
    "list_count_mismatch": "Reply's count claim disagrees with its enumerated list.",
    "source_count_inflation": "Reply enumerates more items than the most recent tool returned.",
    "missing_citation": "Reply references the corpus substantively without an anchor.",
    "fabricated_section": "Reply cites a §-anchor that doesn't exist in the corpus.",
    "paired_meta_confirm_strip": "Reply has both a tool call and meta-confirm prose; strip prose.",
    "opinion_no_trigger": "Reply emits 'Opinion:' without the user asking for one.",
    "post_search_grounding": "Reply followed a search but didn't cite or summarize results.",
    "post_research_persist": "Multi-source summary not persisted via remember_event.",
    "low_confidence_fallback": "Citation below confidence floor for the grounding run.",
    "table_fabrication": "Reply's pipe-table rows don't appear in any tool output.",
    "numeric_fabrication": "Reply's labeled numeric claim disagrees with the tool result.",
    # post_model-phase catchers (operate on the raw ModelReply).
    # (none today — phase exists for future use.)
    # pre_tool-phase catchers (gate tool execution).
    "duplicate_call": "Identical (name, args) call this turn; re-issues prior result.",
    "argument_grounding": "Tool args name domains not in user message or prior tool output.",
    "assemble_context_once": "Model re-calls assemble_context when forced-grounding already ran.",
    "fetch_url_guard": "fetch_url called speculatively when user pasted no URL.",
    "persist_body_citations": "remember_event body lacks URL tokens from research summary.",
    # post_tool-phase catchers (transform tool results before model sees them).
    "tool_result_summarizer": "Compress high-noise tool outputs before the working thread.",
    # finalize-phase catchers (last-chance fallbacks on final reply).
    "ungrounded_citation": "Reply cites a section without a grounding tool having run.",
    "uncited_substantive_reply": "Substantive reply with no citation at all (finalize mirror).",
    "fabrication_fallback": "Bail retries exhausted; substitute a canned refusal.",
}


@dataclass(frozen=True)
class HookPipeline:
    """Owns the four phase lists. Methods are pure functions of their
    context + a `disabled` frozenset — they never mutate module state
    — which makes them trivially testable and safe to reuse across
    concurrent turns. The tool loop constructs one pipeline at wiring
    time and re-uses it for every turn."""

    bail: list[BailHook] = field(default_factory=list)
    post_model: list[PostModelHook] = field(default_factory=list)
    pre_tool: list[PreToolHook] = field(default_factory=list)
    post_tool: list[PostToolHook] = field(default_factory=list)
    finalize: list[FinalizeHook] = field(default_factory=list)

    def describe(self) -> tuple[CatcherDoc, ...]:
        """Snapshot every installed hook as (name, phase, shape) tuples.
        Source for `docs/hooks.md` (rendered by `scripts/gen_hook_docs.py`)
        and the registry-coverage test. Hooks missing a HOOK_SHAPES entry
        surface here with shape='(no shape entry)' so the test fails loudly
        rather than silently emitting empty docs."""
        out: list[CatcherDoc] = []
        for phase_name, phase_list in (
            ("post_model", self.post_model),
            ("bail", self.bail),
            ("pre_tool", self.pre_tool),
            ("post_tool", self.post_tool),
            ("finalize", self.finalize),
        ):
            for hook in phase_list:
                shape = HOOK_SHAPES.get(hook.name, "(no shape entry)")
                out.append(CatcherDoc(name=hook.name, phase=phase_name, shape=shape))
        return tuple(out)

    def names(self) -> tuple[str, ...]:
        """Canonical ordering: bail → post_model → pre_tool →
        post_tool → finalize. The attribution eval enumerates these
        to disable one catcher at a time; stable order keeps
        diagnostic output reproducible."""
        out: list[str] = []
        for phase in (
            self.bail,
            self.post_model,
            self.pre_tool,
            self.post_tool,
            self.finalize,
        ):
            out.extend(h.name for h in phase)
        return tuple(out)

    def run_bail(self, ctx: BailContext, *, disabled: frozenset[str]) -> BailOutcome:
        for hook in self.bail:
            if hook.name in disabled:
                continue
            outcome = hook.check(ctx)
            if isinstance(outcome, Nudge) and not outcome.catcher:
                # Attach the hook's name so downstream observers (CLI
                # retry marker, tool-loop events, attribution eval)
                # can report WHICH rule fired without re-inspecting
                # the nudge text.
                return Nudge(text=outcome.text, catcher=hook.name)
            if not isinstance(outcome, Continue):
                return outcome
        return Continue()

    def run_post_model(
        self, ctx: PostModelContext, *, disabled: frozenset[str]
    ) -> PostModelOutcome:
        for hook in self.post_model:
            if hook.name in disabled:
                continue
            outcome = hook.check(ctx)
            if not isinstance(outcome, Continue):
                return outcome
        return Continue()

    def run_pre_tool(self, ctx: PreToolContext, *, disabled: frozenset[str]) -> PreToolOutcome:
        for hook in self.pre_tool:
            if hook.name in disabled:
                continue
            outcome = hook.check(ctx)
            if not isinstance(outcome, Continue):
                return outcome
        return Continue()

    def run_post_tool(self, ctx: PostToolContext, *, disabled: frozenset[str]) -> PostToolOutcome:
        for hook in self.post_tool:
            if hook.name in disabled:
                continue
            outcome = hook.check(ctx)
            if not isinstance(outcome, Continue):
                return outcome
        return Continue()

    def run_finalize(self, ctx: FinalizeContext, *, disabled: frozenset[str]) -> FinalizeOutcome:
        for hook in self.finalize:
            if hook.name in disabled:
                continue
            outcome = hook.check(ctx)
            if not isinstance(outcome, Continue):
                return outcome
        return Continue()


# Domain-specific catchers that ship in code but only register when a
# character explicitly opts in via `core.yaml: catchers:` (harness-qvwq).
# Each self-gates via regex anyway, but the opt-in moves the
# persona-coupling from "non-FAA replies happen never to match" to a
# declarative "this character runs these checks" — non-target
# characters provably skip them at construction time.
_OPT_IN_CATCHERS: frozenset[str] = frozenset(
    {
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
    }
)


def default_hook_pipeline(
    *,
    valid_section_anchors: frozenset[str] = frozenset(),
    citation_grammar: CitationGrammar | None = None,
    catchers: tuple[str, ...] = (),
    scope_redirect_template: str | None = None,
    character_name: str | None = None,
) -> HookPipeline:
    """Build the shipping pipeline. Order mirrors the pre-refactor
    `_diagnose_bail` branch order so first-match semantics stay
    identical — swapping two hooks could change which nudge text the
    user sees for a reply that trips both.

    `post_tool` ships empty by default: the only hook that currently
    targets this phase is `ToolResultSummarizerHook`, which requires
    a summarizer adapter and is registered opt-in by the CLI when
    --summarize-tool-results is set.

    `valid_section_anchors` is the structural §-anchor index for
    the FabricatedSectionHook (harness-aise). Empty (the default)
    leaves the hook silent — non-corpus characters never trip it.
    Built by `section_index.collect_valid_anchors` at startup from
    the chunks JSONL.

    `citation_grammar` is the active character's CitationGrammar
    (harness-jaqe). Non-citation-disciplined characters pass None;
    the citation hooks (MissingCitationHook today; more later)
    silently no-op.

    `catchers` is the per-character opt-in roster (harness-qvwq).
    Members must be drawn from `_OPT_IN_CATCHERS`. Each name installs
    its corresponding hook in the order documented below; absence
    skips registration entirely. airton_b ships `ab_fabrication`;
    airton_c{,1} ship `ambiguous_context`, `scope_redirect`,
    `reserved_squawk_code`. airton (default dev character) ships an
    empty roster — the codebase doesn't need ATC scope checks.

    Unknown names raise — typos in core.yaml surface immediately
    rather than silently dropping a catcher."""
    unknown = set(catchers) - _OPT_IN_CATCHERS
    if unknown:
        raise ValueError(
            f"Unknown opt-in catchers: {sorted(unknown)}. "
            f"Expected subset of {sorted(_OPT_IN_CATCHERS)}."
        )
    catchers_set = frozenset(catchers)

    bail: list[BailHook] = [
        TruncatedHook(),
        UnparseableHook(),
        TeaserHook(),
        FalseSuccessHook(),
        MetaConfirmHook(),
        # raw_results_dump (harness-s451) sits BETWEEN MetaConfirm and
        # FabricatedSearch. Gates on the OPPOSITE condition from the
        # fabrication catchers (content tool MUST have run) so they
        # can't both fire on the same turn; ordering here is cosmetic
        # for attribution-eval output. Placed inside the universal
        # block (not opt-in) — the failure shape is character-agnostic.
        RawResultsDumpHook(),
        FabricatedSearchHook(),
        FabricatedItemizationHook(),
        # thin_source_fabrication (harness-xszc): the inverse of
        # raw_results_dump. Fires when a tool ran but its body has
        # almost no numeric tokens AND the reply emits specific
        # numeric claims with units. Targets the JS-rendered SPA
        # fetch case (weather.com Nairobi repro 2026-05-19) where
        # fetch_url returns navigation + scripts and the model
        # fabricates coherent-looking numbers anyway. Universal,
        # not opt-in — failure mode is character-agnostic.
        ThinSourceFabricationHook(),
        # incomplete_multipart (harness-111v): user asks two parallel
        # questions in one turn ('find X. what is Y?'), agent answers
        # only one and gives up on the other with explicit refusal
        # phrasing ('the source doesn't mention', 'we would need').
        # Position AFTER the fabrication-shape catchers because
        # fabrication-of-content is a more specific failure than
        # omission-of-content; a reply that BOTH fabricates AND
        # gives-up should get the fabrication nudge first.
        IncompleteMultipartHook(),
    ]
    # ab_fabrication: ab_ops capture/plan/remember imitation shapes.
    # Self-gates on `tools_ran_this_turn=False`; non-ab characters
    # never produce these shapes anyway, but opt-in makes that
    # provable rather than empirical.
    if "ab_fabrication" in catchers_set:
        bail.append(AbFabricationHook())
    bail.append(ToolIntentHook())
    # Compliance check runs last — fabrication-shape catchers above
    # all get first pass at a malformed reply. Only a reply that
    # survived every fabrication gate gets asked the compliance
    # question 'did you cite your source?'.
    bail.append(MissingCitationHook(grammar=citation_grammar))
    # Structural existence check on whatever §-anchors the reply DID
    # cite (harness-aise). Runs immediately after MissingCitationHook
    # so the no-citation case is handled by the right catcher: missing
    # → MissingCitation; invented → FabricatedSection. Silent
    # (Continue-only) when valid_section_anchors is empty, which is
    # the default for non-corpus characters.
    bail.append(FabricatedSectionHook(valid_anchors=valid_section_anchors))
    # Internal-consistency check: the reply's own count claim vs. its
    # enumerated list. Runs after missing_citation so a reply that
    # ADDS a citation on retry doesn't get re-chained into a
    # count-mismatch nudge from its original pre-citation form.
    bail.append(ListCountMismatchHook())
    # source_count_inflation: reply enumerates substantially more
    # items than the most recent tool result produced (harness-lynw).
    # Opt-in for characters that do web research where the model can
    # pad with plausible-sounding source names; chat-only profiles
    # leave it off since the heuristic has no tool count to compare.
    if "source_count_inflation" in catchers_set:
        bail.append(SourceCountInflationHook())
    # reserved_squawk_code: ATC pilot-initiated reserved transponder
    # codes (7500/7600/7700). Runs after structural fabrication
    # checks so a reply that fails any earlier gate gets the
    # shape-specific nudge first; only a reply otherwise structurally
    # fine but proposing an unsafe code reaches this.
    if "reserved_squawk_code" in catchers_set:
        bail.append(ReservedSquawkCodeHook())
    # scope_redirect: user's question has no aviation vocabulary, but
    # the reply is talking ATC. Catches context-bleed and
    # out-of-scope fabrication ('do roosters lay eggs' getting
    # answered with phraseology content). character_name +
    # scope_redirect_template thread into the nudge so each persona
    # gets its OWN example refusal sentence instead of airton_c1's
    # hardcoded JO 7110.65 specialist line.
    if "scope_redirect" in catchers_set:
        bail.append(
            ScopeRedirectHook(
                character_name=character_name,
                scope_redirect_template=scope_redirect_template,
            )
        )
    # ambiguous_context: user asked about a term whose JO handling
    # depends on an unspecified variant ('balloon' = manned vs.
    # unmanned free), and the reply silently picked a variant.
    # Placed last — ambiguity is relevant only when the prompt is
    # otherwise in-scope.
    if "ambiguous_context" in catchers_set:
        bail.append(AmbiguousContextHook())
    # opinion_no_trigger: scholar-style characters gate `Opinion:`
    # paragraphs on explicit user request. Reply with an Opinion
    # paragraph but no trigger word in the user message → Nudge.
    # Self-gates on the literal "Opinion:" token plus the trigger-
    # word check, so non-scholar characters opting in are safe.
    if "opinion_no_trigger" in catchers_set:
        bail.append(OpinionWithoutTriggerHook())
    # post_search_grounding: scholar-style characters must follow
    # search_web with a fetch_url, a refined search, or an explicit
    # "no allowlisted source" reply. search_web ran but no fetch,
    # no URL citation, no raw URL → Nudge. Self-gates on
    # `search_web in tools_ran`, so non-search characters never fire.
    if "post_search_grounding" in catchers_set:
        bail.append(PostSearchGroundingHook())
    # post_research_persist: scholar-style characters must persist
    # multi-source research summaries via remember_event so the
    # contract's prior_discussion slot can recall them later. Runs
    # last among the scholar-flavored catchers — only a reply that
    # passed every fabrication / opinion / grounding check is
    # eligible to be a "real research summary" worth persisting.
    if "post_research_persist" in catchers_set:
        bail.append(PostResearchPersistHook())

    # Pre-tool catchers. Built mutably so opt-in characters can
    # tack on FetchUrlGuardHook without forcing every non-paste-only
    # persona to inherit it.
    pre_tool: list[PreToolHook] = [DuplicateCallHook(), ArgumentGroundingHook()]
    # fetch_url_guard: airton_c_tfr-style paste-only characters block
    # speculative fetch_url calls (those without a URL in the user's
    # message). Placed after argument_grounding so a real URL still
    # has its host validated by downstream wiring (the tool's own
    # allowlist + the grounding check).
    if "fetch_url_guard" in catchers_set:
        pre_tool.append(FetchUrlGuardHook())
    # assemble_context_once: characters with require_assemble_context
    # get a forced grounding prelude before round 0. Small models
    # sometimes re-issue their own assemble_context with a hallucinated
    # role; this hook Skips those redundant calls so the model answers
    # from the already-attached context bundle.
    if "assemble_context_once" in catchers_set:
        pre_tool.append(AssembleContextOnceHook())
    # persist_body_citations: scholar-style characters block a
    # `remember_event` call whose body is missing URL citation tokens
    # when search_scholar ran this turn. Three smoke cycles repeated
    # the strip-tokens drift; the constitution + nudge tightening were
    # insufficient — structural backstop fires pre-write so the bad
    # row never reaches the user's approval dialog.
    if "persist_body_citations" in catchers_set:
        pre_tool.append(PersistBodyCitationsHook())

    return HookPipeline(
        bail=bail,
        post_model=[PairedMetaConfirmStripHook()],
        # Order matters inside pre_tool: duplicate_call fires first so
        # a repeat call short-circuits before the grounding check
        # spends cycles analyzing it (and so the user sees the
        # "duplicate — skipped" nudge, not a grounding complaint,
        # when both would fire).
        pre_tool=pre_tool,
        post_tool=[],
        # Order matters inside finalize: ungrounded_citation runs first so
        # it can Halt with a scope-aware refusal for fabricated-citation
        # replies whose bail outcome was Continue (legitimate-looking
        # wrap-up that no bail hook caught). table_fabrication then
        # covers the narrower "grounded citation, fabricated row" shape
        # when the reply is itself a pipe table. numeric_fabrication
        # covers the same failure mode when the reply is prose
        # ('MH class is 50 miles' vs. the retrieved `|MH|…|25|`).
        # fabrication_fallback only fires when the bail loop itself left
        # a Nudge outcome on the table.
        finalize=[
            # Runs first: when a grounding tool ran but scored weakly
            # AND the reply cites a section the tool didn't ground,
            # halt with a low-confidence fallback. Complementary to
            # UngroundedCitationHook — this catches the 'tool ran but
            # I made up the §' shape; that catches the 'no tool ran at
            # all' shape.
            LowConfidenceFallbackHook(),
            UngroundedCitationHook(),
            # harness-zsbz: terminal mirror of MissingCitationHook —
            # after the bail-phase retry budget is spent, a still-uncited
            # substantive reply touching the corpus document is replaced
            # with the character's scope-redirect template. Auto-enabled
            # when the character ships BOTH citation_grammar and
            # scope_redirect_template; silent otherwise. Runs AFTER
            # UngroundedCitationHook so the no-tool-ran path is handled
            # by the older hook first.
            UncitedSubstantiveReplyHook(
                grammar=citation_grammar,
                scope_redirect_template=scope_redirect_template,
            ),
            TableFabricationHook(),
            NumericFabricationHook(),
            FabricationFallbackHook(),
        ],
    )


# ---------- post-tool summarizer ----------


_SUMMARIZER_SYSTEM_PROMPT = (
    "You compress tool output. Return a concise summary (<= 200 words) "
    "that preserves EVERY file path, line number, identifier, URL, "
    "error message, and proper noun verbatim. Do not add commentary, "
    "recommendations, or lists of next steps. Do not wrap the output "
    "in markdown fences. Do not start with 'Summary:' or similar "
    "preambles. Just the compressed content."
)

_SUMMARIZER_USER_TEMPLATE = (
    "Tool {tool_name!r} was called with arguments: {arguments}\n\n"
    "The raw output was:\n\n"
    "{output}\n\n"
    "Compress it per the rules above."
)


class _SummarizerAdapter(Protocol):
    """Narrow structural type for the summarizer. Only needs
    `complete` — any ModelAdapter satisfies this, as does the router's
    adapter. Reduces the coupling surface from hooks.py into the
    adapter hierarchy."""

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str: ...


@dataclass
class ToolResultSummarizerHook:
    """Compress high-noise tool outputs before they're appended to the
    model-visible message thread.

    Fires only when BOTH of:
    - `ctx.spec.high_noise` is True — the tool is flagged as prone to
      dumping bulk (grep / list_dir / search_web / etc).
    - `len(result.output) > threshold_chars` — there's enough text to
      be worth the summarization round-trip.

    Summarization failures (adapter raises, empty response) fall
    through as Continue — the original result reaches the model
    untouched, so context drift is the worst case, never a broken
    turn."""

    summarizer: _SummarizerAdapter
    threshold_chars: int = 1024
    max_summary_tokens: int = 256
    temperature: float = 0.0  # deterministic compression
    name: str = "tool_result_summarizer"

    def check(self, ctx: PostToolContext) -> PostToolOutcome:
        if not ctx.spec.high_noise:
            return Continue()
        if not ctx.result.success:
            # Errors are already small AND load-bearing — the model
            # needs the exact error text to recover. Never summarize.
            return Continue()
        if len(ctx.result.output) <= self.threshold_chars:
            return Continue()
        prompt = _SUMMARIZER_USER_TEMPLATE.format(
            tool_name=ctx.call.name,
            arguments=ctx.call.arguments,
            output=ctx.result.output,
        )
        messages = [
            ChatMessage(role="system", content=_SUMMARIZER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]
        try:
            summary = self.summarizer.complete(
                messages,
                max_tokens=self.max_summary_tokens,
                temperature=self.temperature,
            )
        except Exception:
            # Never let a summarization failure break the turn.
            return Continue()
        summary = summary.strip()
        if not summary:
            return Continue()
        # Tag the output so the main model knows what it's looking at
        # and can't accidentally quote it back as literal tool output
        # (fabrication catchers would flag a quoted summary as
        # fabricated-search, for example). Preserve success + name.
        annotated = f"[tool output summarized from {len(ctx.result.output)} chars]\n{summary}"
        return ReplaceResult(
            ToolResult(
                tool_name=ctx.result.tool_name,
                output=annotated,
                success=ctx.result.success,
                error=ctx.result.error,
            )
        )


def _call_key(call: ToolCall) -> tuple[str, str]:
    """Canonical (name, arguments-json) key for duplicate detection.
    Sorting keys means argument order doesn't create false-positive
    uniqueness ({'a':1,'b':2} == {'b':2,'a':1}); default=str keeps the
    key stable if a model emits exotic-but-JSON-stringifiable types
    (dates, Paths). We never decode the key back — only equality
    matters — so lossy coercion is fine."""
    import json

    return (call.name, json.dumps(call.arguments, sort_keys=True, default=str))


__all__ = [
    "AB_DATE_HEADER_RE",
    "AB_TIER_HEADER_RE",
    "ARG_DOMAIN_RE",
    "BARE_CLAIM_RE",
    "DUPLICATE_CALL_NUDGE",
    "EXHAUSTED_FABRICATION_FALLBACK",
    "FABRICATED_AB_CAPTURE_RE",
    "FABRICATED_AB_SCOPE_RE",
    "FABRICATED_BEAD_ID_RE",
    "FABRICATED_ITEMIZATION_RE",
    "FABRICATED_REMEMBER_RE",
    "FABRICATED_SEARCH_RE",
    "FALSE_SUCCESS_RE",
    "HOOK_SHAPES",
    "META_CONFIRM_RE",
    "TABLE_FABRICATION_FALLBACK",
    "TEASER_RE",
    "TOOL_INTENT_RE",
    "UNGROUNDED_CITATION_FALLBACK",
    "UNGROUNDED_SECTION_CITATION_RE",
    "AbFabricationHook",
    "AmbiguousContextHook",
    "ArgumentGroundingHook",
    "BailContext",
    "BailHook",
    "BailOutcome",
    "CatcherDoc",
    "Continue",
    "DuplicateCallHook",
    "FabricatedItemizationHook",
    "FabricatedSearchHook",
    "FabricatedSectionHook",
    "FabricationFallbackHook",
    "FalseSuccessHook",
    "FinalizeContext",
    "FinalizeHook",
    "FinalizeOutcome",
    "Halt",
    "HookPipeline",
    "ListCountMismatchHook",
    "MetaConfirmHook",
    "MissingCitationHook",
    "Nudge",
    "NumericFabricationHook",
    "PairedMetaConfirmStripHook",
    "PostModelContext",
    "PostModelHook",
    "PostModelOutcome",
    "PostToolContext",
    "PostToolHook",
    "PostToolOutcome",
    "PreToolContext",
    "PreToolHook",
    "PreToolOutcome",
    "Replace",
    "ReplaceResult",
    "ReservedSquawkCodeHook",
    "ScopeRedirectHook",
    "Skip",
    "SourceCountInflationHook",
    "TableFabricationHook",
    "TeaserHook",
    "ToolIntentHook",
    "ToolResultSummarizerHook",
    "Truncated",
    "TruncatedHook",
    "UncitedSubstantiveReplyHook",
    "UngroundedCitationHook",
    "UnparseableHook",
    "default_hook_pipeline",
    "looks_like_ab_fabrication",
]
