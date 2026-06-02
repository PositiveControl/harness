"""TurnService — the single grounded-turn implementation.

`run_turn` reproduces, step for step, the assembly the classic REPL used
to inline (retrieval → system prompt → history → model ± tool loop →
persona rewrite → persist → audit). The classic session now delegates to
it, so the existing CLI tests are the parity contract; web `/chat` calls
the same path when handed a `TurnContext`.

Everything UI lives behind `TurnIO`: a streaming renderer, a write-tier
confirm callback, a tool-event observer, and four notice hooks (warn /
header / announce / model-done). All default to no-ops, so a headless
caller (web, daemon, tests) gets the grounded turn with no console
coupling. `load_history` is injected as a callable rather than a
`ContextMeter` so the service doesn't depend on the REPL's context meter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from harness.model.adapter import ChatMessage
from harness.orchestrator import run_tool_loop
from harness.persona import build_rewriter_messages
from harness.store.audit import record_turn_audit

if TYPE_CHECKING:
    from pathlib import Path

    from harness.character import Character
    from harness.orchestrator import ToolLoopResult
    from harness.orchestrator.hooks import HookPipeline
    from harness.orchestrator.tool_loop import ConfirmFn, ObserverFn
    from harness.router import Router
    from harness.store.audit import AuditStore
    from harness.store.episodic import EpisodicStore
    from harness.store.semantic import SemanticStore
    from harness.store.transcript import Transcript

# A history loader returns (optional compaction-summary message, recent
# verbatim turns) — exactly what the REPL's ContextMeter.load_history
# yields. Decoupled so the service doesn't import the context meter.
HistoryLoader = Callable[[], "tuple[ChatMessage | None, list[ChatMessage]]"]


def _noop() -> None: ...


def _noop1(_msg: str) -> None: ...


@dataclass
class TurnIO:
    """Injected UI side effects for one turn. Defaults are no-ops, so a
    headless caller (web/daemon/tests) runs the grounded turn without any
    console coupling.

    - `confirm` / `observe` / `stream_renderer` thread straight into the
      tool loop (write-tier approval, event rendering, token streaming).
      When `stream_renderer` is None the service blocks on `complete`.
    - `warn` surfaces a degraded-retrieval-source notice.
    - `header` fires just before the model call (the REPL prints the
      reply banner there).
    - `announce` carries the dim voice-pass notices on the tool path.
    - `model_done` fires once the model has produced its draft, before
      persistence (the REPL stops its spinner there)."""

    confirm: ConfirmFn | None = None
    observe: ObserverFn | None = None
    stream_renderer: Any | None = None
    warn: Callable[[str], None] = field(default=_noop1)
    header: Callable[[], None] = field(default=_noop)
    announce: Callable[[str], None] = field(default=_noop1)
    model_done: Callable[[], None] = field(default=_noop)


@dataclass(frozen=True)
class TurnContext:
    """Long-lived session state the turn reads. Holds references (not
    copies) to mutable session objects — `retrieval_state` (muted by
    /clear) and any registry/store mutated across turns stay live."""

    character: Character
    adapter: Any  # ModelAdapter, possibly PersonaAdapter-wrapped
    transcript: Transcript
    load_history: HistoryLoader
    speaker: str
    session: str
    channel: str
    # retrieval
    retrieval_state: Any  # cli._RetrievalState
    retriever: Any | None = None  # VoiceRetriever
    memory_store: EpisodicStore | None = None
    semantic_store: SemanticStore | None = None
    top_k: int = 6
    memories: int = 0
    memories_threshold: float = 0.5
    facts: int = 0
    facts_threshold: float = 0.45
    allowed_sessions: tuple[str, ...] | None = None
    recency_ranks: dict[str, int] | None = None
    recency_weight: float = 0.0
    # tools / orchestration
    registry: Any | None = None  # ToolRegistry
    router: Router | None = None
    hooks: HookPipeline | None = None
    ab_adapter: Any | None = None  # BeadsAdapter
    banter_tracker: Any | None = None
    workspace_path: Path | None = None
    # persona (post-loop rewrite on the tool path)
    persona: bool = False
    rewrite_on_tools: bool = False
    chain_rewrites: bool = False
    # audit
    audit_store: AuditStore | None = None


@dataclass
class TurnResult:
    reply: str
    loop_result: ToolLoopResult | None
    streamed: bool


class TurnService:
    def __init__(self, ctx: TurnContext) -> None:
        self.ctx = ctx

    def _emit(
        self,
        messages: list[ChatMessage],
        io: TurnIO,
        *,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, bool]:
        """Stream through the renderer when one is wired, else block on
        complete(). Mirrors cli._stream_or_complete; reproduced here so a
        headless caller doesn't need a renderer."""
        if io.stream_renderer is not None:
            from harness.cli import _stream_or_complete

            return _stream_or_complete(
                self.ctx.adapter,
                messages,
                stream_renderer=io.stream_renderer,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        text = self.ctx.adapter.complete(messages, max_tokens=max_tokens, temperature=temperature)
        assert isinstance(text, str)
        return text, False

    def run_turn(self, user_input: str, io: TurnIO | None = None) -> TurnResult:
        """One grounded turn. Persists the user message, assembles the
        system prompt + history, runs the model (± tool loop), optionally
        rewrites for voice, persists the reply, and records the audit
        row. Returns the reply plus the tool-loop result (for callers
        that want it) and whether the model streamed."""
        from harness.cli import (
            _build_tool_grounding_block,
            _persist_tool_exchange,
            _render_ab_memories_block,
            _render_fact_block,
            _render_memory_block,
            _retrieve_turn_context,
            _topic_boundary_suffix,
        )

        ctx = self.ctx
        io = io or TurnIO()

        ctx.transcript.append(
            session=ctx.session,
            channel=ctx.channel,
            speaker=ctx.speaker,
            role="user",
            content=user_input,
        )

        examples, recalled, known_facts = _retrieve_turn_context(
            user_input=user_input,
            speaker=ctx.speaker,
            retriever=ctx.retriever,
            memory_store=ctx.memory_store,
            semantic_store=ctx.semantic_store,
            top_k=ctx.top_k,
            memories=ctx.memories,
            memories_threshold=ctx.memories_threshold,
            facts=ctx.facts,
            facts_threshold=ctx.facts_threshold,
            state=ctx.retrieval_state,
            warn=io.warn,
            allowed_sessions=ctx.allowed_sessions,
            recency_ranks=ctx.recency_ranks,
            recency_weight=ctx.recency_weight,
        )

        if examples:
            system_content = ctx.character.system_prompt(include_samples=examples, now=date.today())
        else:
            system_content = ctx.character.system_prompt(now=date.today())

        if recalled:
            system_content = f"{system_content}\n\n{_render_memory_block(recalled)}"

        if ctx.ab_adapter is not None:
            ab_mem_block = _render_ab_memories_block(ctx.ab_adapter)
            if ab_mem_block is not None:
                system_content = f"{system_content}\n\n{ab_mem_block}"

        if ctx.registry is not None:
            if ctx.workspace_path is None:
                raise ValueError("TurnContext.workspace_path is required when a registry is set")
            system_content = (
                f"{system_content}\n\n"
                f"{_build_tool_grounding_block(ctx.registry, ctx.workspace_path)}"
            )

        if known_facts:
            system_content = f"{system_content}\n\n{_render_fact_block(known_facts)}"

        system_content = (
            f"{system_content}{_topic_boundary_suffix(ctx.retrieval_state, ctx.allowed_sessions)}"
        )

        system = ChatMessage(role="system", content=system_content)

        summary_msg, history = ctx.load_history()
        history_messages: list[ChatMessage] = []
        if summary_msg is not None:
            history_messages.append(summary_msg)
        history_messages.extend(history)

        io.header()
        streamed = False
        loop_result: ToolLoopResult | None = None
        if ctx.registry is not None:
            initial_messages: list[ChatMessage] = [system, *history_messages]
            loop_result = run_tool_loop(
                ctx.adapter,
                initial_messages,
                ctx.registry,
                confirm=io.confirm,
                observe=io.observe,
                router=ctx.router,
                hooks=ctx.hooks,
                memory_block_attached=bool(recalled),
                force_search_memory=ctx.character.require_search_memory,
                force_assemble_context=(
                    ctx.character.default_contract_role
                    if ctx.character.require_assemble_context
                    else None
                ),
                banter_tracker=ctx.banter_tracker,
                scope_redirect_template=ctx.character.scope_redirect_template,
                scope_lexicon=ctx.character.scope_lexicon,
            )
            streamed = io.stream_renderer is not None
            _persist_tool_exchange(
                ctx.transcript,
                session=ctx.session,
                channel=ctx.channel,
                character_name=ctx.character.name,
                initial_count=len(initial_messages),
                loop_messages=loop_result.messages,
            )
            draft = loop_result.content
            # Small models occasionally bail after a tool result — empty
            # content and no further calls. Nudge once before falling back.
            if not draft.strip():
                nudge_msgs = [
                    *loop_result.messages,
                    ChatMessage(
                        role="user",
                        content=(
                            "Your last reply was empty. Give me a final answer "
                            "based on what the tools already returned. Restate "
                            "the key findings in prose. Do not return empty."
                        ),
                    ),
                ]
                retry = ctx.adapter.complete_with_tools(
                    nudge_msgs,
                    tools=ctx.registry.specs(),
                    max_tokens=2048,
                    temperature=0.3,
                )
                if retry.content.strip():
                    draft = retry.content
            # Skip the rewriter when rewrite-on-tools is off (it compresses
            # prose investigations need) or the loop left no draft.
            if ctx.persona and ctx.rewrite_on_tools and draft.strip():
                io.announce("voice pass")
                rewrite_msgs = build_rewriter_messages(ctx.character, draft, focus="style")
                reply, _ = self._emit(rewrite_msgs, io, temperature=0.2, max_tokens=2048)
                if ctx.chain_rewrites and reply.strip():
                    io.announce("concrete pass")
                    concrete_msgs = build_rewriter_messages(ctx.character, reply, focus="concrete")
                    reply, _ = self._emit(concrete_msgs, io, temperature=0.2, max_tokens=2048)
            else:
                reply = draft or "(no reply — model returned empty text after tool calls)"
        else:
            reply, streamed = self._emit(
                [system, *history_messages], io, max_tokens=512, temperature=0.7
            )

        io.model_done()
        ctx.transcript.append(
            session=ctx.session,
            channel=ctx.channel,
            speaker=ctx.character.name,
            role="assistant",
            content=reply,
        )
        record_turn_audit(
            ctx.audit_store,
            session=ctx.session,
            character=ctx.character.name,
            user_id=ctx.speaker,
            user_message=user_input,
            model_reply=reply,
            loop_result=loop_result,
        )
        return TurnResult(reply=reply, loop_result=loop_result, streamed=streamed)
