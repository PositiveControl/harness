from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import cast

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status
from rich.table import Table

import harness._quiet  # noqa: F401 — side-effect import: silences HF/transformers/sentence-transformers noise before they load
from harness.character import Character, VoiceSample, load_character
from harness.compaction import (
    CompactionOutcome,
    CompactionStore,
    run_compaction,
    should_compact,
)
from harness.config import settings
from harness.consolidate import run_consolidation
from harness.evals.voice import run_voice_eval
from harness.model import AdapterName, ChatMessage, ModelAdapter, make_adapter
from harness.model.adapter import Role, count_tokens
from harness.orchestrator import ToolLoopEvent, run_tool_loop
from harness.persona import PersonaAdapter
from harness.persona.rewriter import build_rewriter_messages
from harness.retrieval import VoiceRetriever
from harness.scribe import run_scribe
from harness.store import (
    EpisodicRecord,
    EpisodicStore,
    SemanticFact,
    SemanticStore,
    ensure_seeds_ingested,
)
from harness.store.transcript import Transcript, TranscriptMessage
from harness.tools import (
    DEFAULT_PROFILE,
    TOOL_PROFILES,
    ConsolidateMemoryTool,
    EditFileTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    RememberEventTool,
    RememberFactTool,
    ScribeSessionTool,
    SearchFactsTool,
    SearchMemoryTool,
    SearchWebTool,
    ShellTool,
    Tool,
    ToolCall,
    ToolRegistry,
    WriteFileTool,
    resolve_tool_names,
)

_EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit", ":q", ":quit"})

# Sentinel used to encode structured tool_calls onto an assistant turn's
# content when persisting to the transcript. Two-line format: human-readable
# content, then the sentinel, then a single JSON line with the tool_calls.
_TOOL_CALLS_SENTINEL = "\n__TOOL_CALLS_V1__\n"


def _encode_assistant_with_tool_calls(content: str, tool_calls: tuple[ToolCall, ...]) -> str:
    if not tool_calls:
        return content
    payload = json.dumps(
        {"tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls]}
    )
    return f"{content}{_TOOL_CALLS_SENTINEL}{payload}"


def _decode_transcript_message(m: TranscriptMessage) -> ChatMessage:
    """Convert a persisted transcript row back into a ChatMessage so the
    next turn's history reconstructs tool_calls on assistant turns and
    carries the tool name on tool-role turns."""
    role = cast(Role, m.role)
    content = m.content
    tool_calls: tuple[ToolCall, ...] = ()
    if role == "assistant" and _TOOL_CALLS_SENTINEL in content:
        head, _, tail_json = content.partition(_TOOL_CALLS_SENTINEL)
        try:
            payload = json.loads(tail_json)
            parsed = payload.get("tool_calls") or []
            tool_calls = tuple(
                ToolCall(name=p["name"], arguments=p.get("arguments", {}))
                for p in parsed
                if isinstance(p, dict) and isinstance(p.get("name"), str)
            )
            content = head
        except (json.JSONDecodeError, KeyError, TypeError):
            tool_calls = ()
    name = m.speaker if role == "tool" else None
    return ChatMessage(role=role, content=content, name=name, tool_calls=tool_calls)


def _persist_tool_exchange(
    transcript: Transcript,
    *,
    session: str,
    channel: str,
    character_name: str,
    initial_count: int,
    loop_messages: list[ChatMessage],
) -> None:
    """Append the assistant tool-call and tool-result turns from a tool
    loop to the transcript so the next user turn sees them in history.
    `initial_count` is the number of messages that were already in the
    working list before the loop added any (system + history length)."""
    for msg in loop_messages[initial_count:]:
        if msg.role == "assistant":
            transcript.append(
                session=session,
                channel=channel,
                speaker=character_name,
                role="assistant",
                content=_encode_assistant_with_tool_calls(msg.content, msg.tool_calls),
            )
        elif msg.role == "tool":
            transcript.append(
                session=session,
                channel=channel,
                speaker=msg.name or "tool",
                role="tool",
                content=msg.content,
            )


app = typer.Typer(add_completion=False, no_args_is_help=True)
eval_app = typer.Typer(help="Evaluations against the current character.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
memory_app = typer.Typer(help="Inspect and manage episodic memory.", no_args_is_help=True)
app.add_typer(memory_app, name="memory")
voice_app = typer.Typer(help="Voice suite — capture and manage samples.", no_args_is_help=True)
app.add_typer(voice_app, name="voice")
console = Console()


_EMBEDDER_SENTINEL: object = object()
_cached_embedder: object = _EMBEDDER_SENTINEL


def _load_embedder() -> object | None:
    """Lazy-import and memoize the default embedder. Returns None (with
    a warning) if the retrieval extra isn't installed.

    The result is cached process-wide — `cmd_chat` wires retriever +
    episodic store + semantic store from the same instance, so the 1.3
    GB embedder model loads once instead of three times."""
    global _cached_embedder
    if _cached_embedder is not _EMBEDDER_SENTINEL:
        return None if _cached_embedder is None else _cached_embedder
    try:
        from harness.retrieval.st_embedder import SentenceTransformersEmbedder
    except ImportError:
        console.print(
            "[yellow]retrieval extra not installed. "
            "Run `uv sync --extra retrieval` to enable retrieval + memory.[/yellow]"
        )
        _cached_embedder = None
        return None
    _cached_embedder = SentenceTransformersEmbedder()
    return _cached_embedder


def _maybe_retriever(character: Character, top_k: int) -> VoiceRetriever | None:
    """Build a VoiceRetriever if retrieval is requested and the optional
    sentence-transformers dep is installed. Returns None to signal the
    caller to fall back to full-set few-shot."""
    if top_k <= 0:
        return None
    embedder = _load_embedder()
    if embedder is None:
        return None
    with Status(f"warming embedder ({embedder.id})…", console=console):  # type: ignore[attr-defined]
        return VoiceRetriever(embedder=embedder, character=character)  # type: ignore[arg-type]


def _open_episodic_store(character: Character, *, ingest: bool = True) -> EpisodicStore | None:
    """Open the episodic store, ingesting seeds on first run. Returns
    None if the retrieval extra isn't installed."""
    embedder = _load_embedder()
    if embedder is None:
        return None
    store = EpisodicStore(settings.db_path, embedder=embedder)  # type: ignore[arg-type]
    if ingest:
        inserted = ensure_seeds_ingested(character, store)
        if inserted > 0:
            console.print(f"[dim]seeded {inserted} episodic memories from character.[/dim]")
    return store


def _open_semantic_store() -> SemanticStore | None:
    embedder = _load_embedder()
    if embedder is None:
        return None
    return SemanticStore(settings.db_path, embedder=embedder)  # type: ignore[arg-type]


def _format_ctx_meter(used: int, total: int) -> str:
    """Render 'ctx 4.2k / 32k (13%)' with color thresholds: dim under
    75%, yellow 75-90%, red above 90%. Only shown when `total > 0`."""
    if total <= 0:
        return ""
    pct = used / total
    if pct >= 0.9:
        color = "red"
    elif pct >= 0.75:
        color = "yellow"
    else:
        color = "dim"
    return f"[{color}]ctx {used / 1000:.1f}k / {total / 1000:.0f}k ({pct * 100:.0f}%)[/{color}]"


def _render_fact_block(facts: list[SemanticFact]) -> str:
    """Render retrieved semantic facts as a compact block for the system
    prompt. One line per fact — subject, predicate, object, confidence."""
    lines = ["Relevant facts I know:"]
    for f in facts:
        lines.append(f"- {f.subject} {f.predicate} {f.object} (conf={f.confidence:.2f})")
    return "\n".join(lines)


class _ThinkingSpinner:
    """Live 'thinking… Ns' spinner. Rich's Status animates the spinner
    glyph; a small daemon thread updates the elapsed-seconds suffix
    every 250 ms so the user sees the timer tick.

    Both start() and stop() are idempotent: start() is a no-op when
    already running, stop() a no-op when already stopped. This lets the
    chat loop kick the spinner on as soon as the user presses Enter
    (so the prompt isn't silent), have the tool-loop observer bounce it
    per model call, and stop it cleanly before any console.input() or
    final Markdown print — without any caller needing to track state."""

    def __init__(self, console: Console, label: str = "thinking") -> None:
        self._console = console
        self._label = label
        self._running = False
        self._status: Status | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._started_at = time.monotonic()
        self._status = self._console.status(
            f"[dim]⋯ {self._label}… 0s[/dim]",
            spinner="dots",
        )
        self._status.__enter__()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._status is not None:
            self._status.__exit__(None, None, None)
        self._status = None
        self._stop_event = None
        self._thread = None

    def _tick(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.wait(0.25):
            if self._status is None:
                return
            elapsed = int(time.monotonic() - self._started_at)
            self._status.update(f"[dim]⋯ {self._label}… {elapsed}s[/dim]")

    def __enter__(self) -> _ThinkingSpinner:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


class _StreamRenderer:
    """Plain-text streaming region for token deltas.

    Earlier versions wrapped a `rich.live.Live` around a re-rendered
    `Markdown` block. That repainted the full buffer at 10 Hz, and when
    the buffer exceeded terminal height Rich could not clear the prior
    frames — each tick leaked into scrollback as a growing-prefix
    duplicate. Streaming is now plain-text append: each delta is written
    directly with no repaint, so long replies render exactly once."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._buf = ""
        self._active = False

    def start(self) -> None:
        self._buf = ""
        self._active = True

    def append(self, delta: str) -> None:
        if not self._active:
            self.start()
        self._buf += delta
        self._console.print(delta, end="", markup=False, highlight=False, soft_wrap=True)

    def stop(self) -> str:
        out = self._buf
        if self._active and out:
            self._console.print()
        self._buf = ""
        self._active = False
        return out

    @property
    def active(self) -> bool:
        return self._active


def _stream_or_complete(
    adapter: object,
    messages: list[ChatMessage],
    *,
    stream_renderer: _StreamRenderer,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> tuple[str, bool]:
    """Stream via `adapter.stream(...)` when available, otherwise fall
    back to the blocking `adapter.complete(...)`. Returns the reply
    text and whether streaming actually happened — the caller uses the
    streamed flag to skip a duplicate final Markdown print (Live
    already rendered the content)."""
    stream_fn = getattr(adapter, "stream", None)
    if callable(stream_fn):
        stream_renderer.start()
        for delta in stream_fn(messages, max_tokens=max_tokens, temperature=temperature):
            stream_renderer.append(delta)
        text = stream_renderer.stop()
        return text, True
    complete_fn = adapter.complete  # type: ignore[attr-defined]
    text = complete_fn(messages, max_tokens=max_tokens, temperature=temperature)
    assert isinstance(text, str)
    return text, False


@dataclass
class _RetrievalState:
    """Per-chat-session health of the three retrieval sources. Once a
    source raises we disable it for the rest of the session so the user
    doesn't get a warning on every turn. The chat still works — just
    without that source's prompt context."""

    voice_ok: bool = True
    episodic_ok: bool = True
    semantic_ok: bool = True


def _retrieve_turn_context(
    *,
    user_input: str,
    speaker: str,
    retriever: VoiceRetriever | None,
    memory_store: EpisodicStore | None,
    semantic_store: SemanticStore | None,
    top_k: int,
    memories: int,
    memories_threshold: float,
    facts: int,
    facts_threshold: float,
    state: _RetrievalState,
    warn: Callable[[str], None],
) -> tuple[list[VoiceSample], list[EpisodicRecord], list[SemanticFact]]:
    """Run the three retrieval sources for one turn. Any that raise are
    disabled for the rest of the session (flagged on `state`) and a
    one-time `warn(msg)` fires. Returns the hits from the sources that
    are still healthy — empty lists for the ones that aren't."""
    examples: list[VoiceSample] = []
    if retriever is not None and state.voice_ok and top_k > 0:
        try:
            examples = retriever.top_k(user_input, k=top_k)
        except Exception as exc:
            state.voice_ok = False
            warn(f"voice retrieval disabled for this session: {exc}")

    recalled: list[EpisodicRecord] = []
    if memory_store is not None and state.episodic_ok and memories > 0:
        try:
            hits = memory_store.search(
                user_input,
                k=memories,
                min_score=memories_threshold,
                user_id=speaker,
            )
            recalled = [rec for rec, _score in hits]
        except Exception as exc:
            state.episodic_ok = False
            warn(f"episodic memory disabled for this session: {exc}")

    known_facts: list[SemanticFact] = []
    if semantic_store is not None and state.semantic_ok and facts > 0:
        try:
            fact_hits = semantic_store.search(
                user_input,
                k=facts,
                min_score=facts_threshold,
                user_id=speaker,
            )
            known_facts = [f for f, _score in fact_hits]
        except Exception as exc:
            state.semantic_ok = False
            warn(f"semantic facts disabled for this session: {exc}")

    return examples, recalled, known_facts


def _render_memory_block(memories: list[EpisodicRecord]) -> str:
    """Render retrieved memories as a section of the system prompt. One
    block per memory, title as heading, principle italicized, body as
    prose. Kept close to the on-disk seed format so the model sees
    familiar shape."""
    lines = ["Relevant past experience — things I remember from before:"]
    for m in memories:
        lines.append("")
        lines.append(f"## {m.title}")
        if m.principle:
            lines.append(f"*Lesson: {m.principle}*")
        lines.append("")
        lines.append(m.body)
    return "\n".join(lines)


def _resolve_adapter(
    name: str,
    *,
    persona: bool = False,
    character: Character | None = None,
    model_repo: str | None = None,
    lora_path: str | None = None,
) -> ModelAdapter:
    # Custom configs bypass the factory and instantiate the adapter
    # directly. --lora-path is MLX-only; --model-repo works for MLX
    # (HF repo) and Ollama (model tag like "gemma4:latest").
    if lora_path and name != "mlx":
        raise typer.BadParameter("--lora-path requires --model mlx.")

    adapter: ModelAdapter
    if model_repo or lora_path:
        if name == "mlx":
            from harness.model.mlx import MLXAdapter

            mlx_kwargs: dict[str, object] = {}
            if model_repo:
                mlx_kwargs["repo"] = model_repo
            if lora_path:
                mlx_kwargs["adapter_path"] = lora_path
            adapter = MLXAdapter(**mlx_kwargs)  # type: ignore[arg-type]
        elif name == "ollama":
            from harness.model.ollama import OllamaAdapter

            adapter = OllamaAdapter(model=model_repo) if model_repo else OllamaAdapter()
        else:
            raise typer.BadParameter(
                f"--model-repo not supported for --model {name}; use mlx or ollama."
            )
    else:
        try:
            adapter = make_adapter(cast(AdapterName, name))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if persona:
        if character is None:
            raise typer.BadParameter("persona=True requires a character")
        adapter = PersonaAdapter(adapter, character)

    # Honor an optional eager `.load()` method without making it part of
    # the ModelAdapter Protocol — only some adapters need it.
    loader = getattr(adapter, "load", None)
    if callable(loader):
        with Status(f"loading {adapter.id}…", console=console):
            loader()
    return adapter


@app.command()
def chat(
    session: str = typer.Option("local", help="Session identifier"),
    channel: str = typer.Option("cli", help="Channel name"),
    speaker: str = typer.Option("mark", help="Your handle"),
    model: str = typer.Option("echo", help="Adapter: echo | mlx | ollama"),
    model_repo: str | None = typer.Option(
        None,
        "--model-repo",
        help="Override the model identifier for the selected adapter. "
        "For mlx: HF repo (default mlx-community/Qwen2.5-7B-Instruct-4bit). "
        "For ollama: model tag (default gemma4:latest). Ignored for echo.",
    ),
    lora_path: str | None = typer.Option(
        None,
        "--lora-path",
        help="Path to a DIRECTORY produced by `mlx_lm.lora` training (contains "
        "adapter_config.json plus weight files). Applied on top of the base MLX "
        "model. Requires --model mlx.",
    ),
    persona: bool = typer.Option(
        False,
        "--persona/--no-persona",
        help="Wrap the model with a voice-rewrite post-pass (Airton's register).",
    ),
    top_k: int = typer.Option(
        6,
        help="Retrieve top-K voice samples by similarity to the user message "
        "(default 6). Set 0 to show every sample.",
    ),
    memories: int = typer.Option(
        3,
        "--memories",
        help="Max episodic memories to retrieve per user turn (default 3). "
        "Set 0 to disable memory retrieval.",
    ),
    memories_threshold: float = typer.Option(
        0.5,
        "--memories-threshold",
        help="Cosine-similarity floor for memory retrieval. Memories below "
        "this are dropped even if there are fewer than --memories of them. "
        "Prevents irrelevant memories from polluting the prompt.",
    ),
    facts: int = typer.Option(
        5,
        "--facts",
        help="Max semantic facts to retrieve per user turn (default 5). "
        "Set 0 to disable fact retrieval.",
    ),
    facts_threshold: float = typer.Option(
        0.45,
        "--facts-threshold",
        help="Cosine-similarity floor for fact retrieval. Lower than the "
        "memory floor because facts are much shorter strings and score lower.",
    ),
    tools: bool = typer.Option(
        False,
        "--tools/--no-tools",
        help="Enable tool use. Which tools are registered depends on "
        "--tool-set (default: 'core'). Write-tier tools prompt for "
        "confirmation the first time they're called each session.",
    ),
    tool_set: str = typer.Option(
        DEFAULT_PROFILE,
        "--tool-set",
        help=(
            f"Named profile of tools to enable with --tools. One of: "
            f"{sorted(TOOL_PROFILES)}. Schema cost targets kept under "
            f"~1500 tokens per profile."
        ),
    ),
    tools_add: str | None = typer.Option(
        None,
        "--tools-add",
        help="Comma-separated tool names to add on top of the --tool-set.",
    ),
    tools_drop: str | None = typer.Option(
        None,
        "--tools-drop",
        help="Comma-separated tool names to drop from the --tool-set.",
    ),
    rewrite_on_tools: bool = typer.Option(
        False,
        "--rewrite-on-tools/--no-rewrite-on-tools",
        help="When tools ran in a turn, also apply the persona rewriter to the "
        "final reply. Off by default — the rewriter is trained to compress, "
        "which is wrong for summarize / investigate tasks that need prose. Turn "
        "on for casual tooled chat where you want Airton-voice on every reply.",
    ),
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Directory read_file / write_file / shell operate inside. "
        "Default: the harness repo root. Only takes effect with --tools. "
        "Memory and transcripts still live under the harness data dir.",
    ),
    compact_at: float = typer.Option(
        0.8,
        "--compact-at",
        help="Fraction of the context window at which to auto-summarize "
        "older turns (0 to disable). When the context meter crosses this, "
        "every turn older than --compact-keep-recent is folded into a "
        "single session summary. The transcript is unchanged — only the "
        "prompt the model sees shrinks.",
    ),
    compact_keep_recent: int = typer.Option(
        10,
        "--compact-keep-recent",
        help="Number of most-recent turns to leave verbatim when "
        "compaction fires. Older turns become summary.",
    ),
) -> None:
    """CLI chat loop. Swap model runtimes with --model."""
    character = load_character(settings.character_path)
    workspace_path = Path(workspace).expanduser().resolve() if workspace else settings.root
    if tools and not workspace_path.is_dir():
        raise typer.BadParameter(f"workspace {workspace_path} is not a directory")
    # When tools are active we run persona manually *after* the tool loop,
    # so we resolve the base adapter unwrapped. Without tools, persona
    # wraps the base adapter as before.
    adapter = _resolve_adapter(
        model,
        persona=persona and not tools,
        character=character,
        model_repo=model_repo,
        lora_path=lora_path,
    )
    retriever = _maybe_retriever(character, top_k)
    memory_store = _open_episodic_store(character) if memories > 0 else None
    semantic_store = _open_semantic_store() if facts > 0 else None
    transcript = Transcript(settings.db_path)
    compaction_store = CompactionStore(settings.db_path) if compact_at > 0 else None

    registry: ToolRegistry | None = None
    approved_tools: set[str] = set()
    if tools:
        try:
            wanted_names = resolve_tool_names(
                tool_set,
                add=tuple((tools_add or "").split(",")),
                drop=tuple((tools_drop or "").split(",")),
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        # Map names → builders. Memory tools return None when their store
        # isn't available (--memories 0 / --facts 0). Unknown names fall
        # through to the warning path so future-tool profiles stay loadable.
        builders: dict[str, Callable[[], Tool | None]] = {
            "read_file": lambda: ReadFileTool(root=workspace_path),
            "edit_file": lambda: EditFileTool(root=workspace_path),
            "write_file": lambda: WriteFileTool(root=workspace_path),
            "shell": lambda: ShellTool(cwd=workspace_path),
            "list_dir": lambda: ListDirTool(root=workspace_path),
            "grep": lambda: GrepTool(root=workspace_path),
            "glob": lambda: GlobTool(root=workspace_path),
            "git_status": lambda: GitStatusTool(root=workspace_path),
            "git_diff": lambda: GitDiffTool(root=workspace_path),
            "git_log": lambda: GitLogTool(root=workspace_path),
            "search_memory": (
                lambda: (
                    SearchMemoryTool(store=memory_store, user_id=speaker)
                    if memory_store is not None
                    else None
                )
            ),
            "search_facts": (
                lambda: (
                    SearchFactsTool(store=semantic_store, user_id=speaker)
                    if semantic_store is not None
                    else None
                )
            ),
            "search_web": lambda: SearchWebTool(),
            "remember_fact": (
                lambda: (
                    RememberFactTool(store=semantic_store, user_id=speaker, session_id=session)
                    if semantic_store is not None
                    else None
                )
            ),
            "remember_event": (
                lambda: (
                    RememberEventTool(store=memory_store, user_id=speaker, session_id=session)
                    if memory_store is not None
                    else None
                )
            ),
            "scribe_session": (
                lambda: (
                    ScribeSessionTool(
                        adapter=adapter,
                        character=character,
                        transcript=transcript,
                        episodic_store=memory_store,
                        semantic_store=semantic_store,
                        default_user_id=speaker,
                    )
                    if memory_store is not None and semantic_store is not None
                    else None
                )
            ),
            "consolidate_memory": (
                lambda: (
                    ConsolidateMemoryTool(
                        episodic_store=memory_store,
                        semantic_store=semantic_store,
                    )
                    if memory_store is not None and semantic_store is not None
                    else None
                )
            ),
        }

        registry = ToolRegistry()
        for name in wanted_names:
            builder = builders.get(name)
            if builder is None:
                console.print(f"[yellow]⚠ tool {name!r} not yet implemented — skipping[/yellow]")
                continue
            tool = builder()
            if tool is None:
                console.print(
                    f"[yellow]⚠ tool {name!r} needs a store that isn't enabled "
                    f"(check --memories / --facts)[/yellow]"
                )
                continue
            registry.register(tool)

        if not registry.names():
            registry = None  # empty profile → same as --no-tools

    retrieval_state = _RetrievalState()
    thinking = _ThinkingSpinner(console)
    stream_renderer = _StreamRenderer(console)

    def _warn_once(msg: str) -> None:
        console.print(f"[yellow]⚠ {msg}[/yellow]")

    def _tool_label(name: str) -> str:
        if registry is not None and name in registry:
            return registry.get(name).spec.label
        return name

    def confirm_write_tool(call: ToolCall) -> bool:
        if call.name in approved_tools:
            return True
        label = _tool_label(call.name)
        console.print(f"[yellow]🔧 Airton wants to [bold]{label}[/bold][/yellow] {call.arguments}")
        answer = console.input("   approve? [y/N/always]: ").strip().lower()
        if answer == "always":
            approved_tools.add(call.name)
            return True
        return answer.startswith("y")

    def render_tool_event(event: ToolLoopEvent) -> None:
        if event.kind == "model_call_start":
            thinking.start()
        elif event.kind == "token_delta":
            # First token received — drop the spinner, open a Live region
            # (if not already) and append. Subsequent deltas just append.
            thinking.stop()
            if event.delta:
                stream_renderer.append(event.delta)
        elif event.kind == "model_call_end":
            thinking.stop()
            stream_renderer.stop()
        elif event.kind == "tool_call_start":
            call = event.call
            assert call is not None
            label = _tool_label(call.name)
            console.print(f"[cyan]🔧 {label}[/cyan] [dim]({call.arguments})[/dim]")
        elif event.kind in ("tool_call_end", "tool_call_failed"):
            result = event.result
            assert result is not None
            status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
            snippet = result.output[:120].replace("\n", " ")
            more = "…" if len(result.output) > 120 else ""
            console.print(f"   {status} [dim]{snippet}{more}[/dim]")
        elif event.kind == "tool_call_declined":
            console.print("   [yellow]✗ declined[/yellow]")

    console.print(
        f"[bold]{character.name}[/bold] loaded. "
        f"session={session} model={adapter.id} "
        f"top_k={top_k if retriever else 0} "
        f"memories={memories if memory_store else 0} "
        f"facts={facts if semantic_store else 0} "
        f"tools={'on' if registry else 'off'}"
        + (f" workspace={workspace_path}" if registry else "")
    )
    console.print("[dim](ctrl-c, /exit, /quit, or :q to exit)[/dim]\n")

    def _load_history() -> tuple[ChatMessage | None, list[ChatMessage]]:
        """Return (optional summary-system-message, turns-since-pointer).
        When a compaction summary exists, turns before the pointer are
        represented by the summary only; the raw rows stay in the
        transcript for audit but never hit the model."""
        record = compaction_store.latest_for_session(session) if compaction_store else None
        if record is not None:
            summary_msg = ChatMessage(
                role="system",
                content=(
                    "Earlier conversation in this session (summarized; "
                    f"{record.covered_turns} turns folded in):\n\n{record.summary}"
                ),
            )
            rows = transcript.fetch_after(session, after_id=record.up_to_turn_id)
            return summary_msg, [_decode_transcript_message(m) for m in rows]
        rows = transcript.tail(session, limit=50)
        return None, [_decode_transcript_message(m) for m in rows]

    def _measure_ctx() -> int:
        """Estimate tokens for what the NEXT turn will start with:
        character.system_prompt() (cheap fallback — no retrieval yet),
        plus any compaction summary, plus history since the pointer.
        Undercounts slightly because retrieved memories/facts add text
        per turn, but tracks transcript growth accurately."""
        baseline_system = ChatMessage(role="system", content=character.system_prompt())
        summary_msg, history_msgs = _load_history()
        msgs: list[ChatMessage] = [baseline_system]
        if summary_msg is not None:
            msgs.append(summary_msg)
        msgs.extend(history_msgs)
        return count_tokens(adapter, msgs)

    def _print_ctx_meter() -> None:
        used = _measure_ctx()
        meter = _format_ctx_meter(used, adapter.context_window)
        if meter:
            console.print(meter)

    def _maybe_compact() -> None:
        if compaction_store is None:
            return
        used = _measure_ctx()
        if not should_compact(
            used_tokens=used,
            context_window=adapter.context_window,
            threshold_pct=compact_at,
        ):
            return
        console.print(
            f"[dim]compacting history (ctx {used / 1000:.1f}k, threshold "
            f"{compact_at * 100:.0f}%)…[/dim]"
        )
        thinking.start()
        try:
            outcome: CompactionOutcome = run_compaction(
                adapter,
                transcript,
                compaction_store,
                session_id=session,
                keep_recent=compact_keep_recent,
            )
        finally:
            thinking.stop()
        if outcome.wrote:
            console.print(
                f"[dim]compacted {outcome.covered_turns} turns "
                f"(pointer → #{outcome.new_up_to_turn_id})[/dim]"
            )
        else:
            console.print(
                "[yellow]compaction skipped — nothing qualified "
                "(fewer turns than keep-recent, or model returned empty).[/yellow]"
            )

    try:
        while True:
            _maybe_compact()
            _print_ctx_meter()
            user_input = console.input("[bold cyan]you › [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in _EXIT_COMMANDS:
                break
            transcript.append(
                session=session,
                channel=channel,
                speaker=speaker,
                role="user",
                content=user_input,
            )
            # Start the spinner immediately so the user sees acknowledgement
            # of their submission, not a blank cursor, while retrieval warms
            # up and the model runs. The tool-loop observer drops/restarts it
            # as needed across rounds; we stop it unconditionally before any
            # interactive prompt or the final reply render.
            thinking.start()

            examples, recalled, known_facts = _retrieve_turn_context(
                user_input=user_input,
                speaker=speaker,
                retriever=retriever,
                memory_store=memory_store,
                semantic_store=semantic_store,
                top_k=top_k,
                memories=memories,
                memories_threshold=memories_threshold,
                facts=facts,
                facts_threshold=facts_threshold,
                state=retrieval_state,
                warn=_warn_once,
            )

            if examples:
                system_content = character.system_prompt(include_samples=examples)
            else:
                system_content = character.system_prompt()

            if recalled:
                system_content = f"{system_content}\n\n{_render_memory_block(recalled)}"

            if registry is not None:
                tool_names = ", ".join(registry.names())
                system_content = (
                    f"{system_content}\n\n"
                    f"Workspace grounding — you are a real process on Mark's Mac. "
                    f"The tool sandbox root is `{workspace_path}`. Available tools: "
                    f"{tool_names}. Paths passed to `read_file` / `write_file` / "
                    f"`edit_file` are relative to the sandbox root; `shell` runs "
                    f"with it as cwd.\n\n"
                    "RULES:\n"
                    "- Never describe the contents of the workspace from memory. If "
                    "the user asks what's in a directory, what a file contains, or "
                    "what this project does, you MUST call a tool first (`shell ls`, "
                    "`read_file`, etc.) and base your answer on the tool's output.\n"
                    "- NEVER claim you did something (added/updated/created/wrote/"
                    "edited/appended a file, ran a command, etc.) unless you actually "
                    "called the corresponding write-tier tool on this turn AND the "
                    "tool's result message says it succeeded. If you don't have a "
                    "tool for the action the user asked for, say so plainly.\n"
                    "- After tool results come back, respond with a substantive "
                    "reply that uses them. Never return an empty reply — the user "
                    "is waiting for your conclusion, not just the tool output.\n"
                    "- The user CANNOT see raw tool output — only your final reply. "
                    "Restate the key findings (names, numbers, quoted lines) in your "
                    "reply. Do not answer with meta-phrases like 'awaiting input' or "
                    "'the content is available'."
                )

            if known_facts:
                system_content = f"{system_content}\n\n{_render_fact_block(known_facts)}"

            system = ChatMessage(role="system", content=system_content)

            summary_msg, history = _load_history()
            history_messages: list[ChatMessage] = []
            if summary_msg is not None:
                history_messages.append(summary_msg)
            history_messages.extend(history)

            console.print(f"[bold green]{character.name} ›[/bold green]")
            streamed = False
            if registry is not None:
                # Tools active: drive the tool loop (observer handles live
                # rendering per model call), then optionally apply the
                # voice rewriter to the final text.
                initial_messages: list[ChatMessage] = [system, *history_messages]
                loop_result = run_tool_loop(
                    adapter,  # type: ignore[arg-type]
                    initial_messages,
                    registry,
                    confirm=confirm_write_tool,
                    observe=render_tool_event,
                )
                streamed = True
                # Persist the tool exchange (assistant tool-call turns +
                # tool-role result turns) so the next user turn can see
                # what was read / run. Without this, every turn is amnesia.
                _persist_tool_exchange(
                    transcript,
                    session=session,
                    channel=channel,
                    character_name=character.name,
                    initial_count=len(initial_messages),
                    loop_messages=loop_result.messages,
                )
                draft = loop_result.content
                # Small models (gemma4 8B) sometimes bail after a tool
                # result — empty content AND no further tool calls. Nudge
                # them once with an explicit follow-up asking for the
                # final answer before falling back to the sentinel.
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
                    retry = adapter.complete_with_tools(  # type: ignore[attr-defined]
                        nudge_msgs,
                        tools=registry.specs(),
                        max_tokens=2048,
                        temperature=0.3,
                    )
                    if retry.content.strip():
                        draft = retry.content
                # Skip the rewriter when (a) rewrite-on-tools is off (default)
                # because the rewriter compresses prose that summarize /
                # investigate tasks need, or (b) the tool loop left no
                # substantive draft — otherwise the rewriter sees an empty
                # "Draft:" block and hallucinates "paste the text."
                if persona and rewrite_on_tools and draft.strip():
                    console.print("\n[dim]*— voice pass —*[/dim]")
                    rewrite_msgs = build_rewriter_messages(character, draft)
                    reply, _ = _stream_or_complete(
                        adapter,
                        rewrite_msgs,
                        stream_renderer=stream_renderer,
                        temperature=0.2,
                        max_tokens=2048,
                    )
                else:
                    reply = draft or "(no reply — model returned empty text after tool calls)"
            else:
                reply, streamed = _stream_or_complete(
                    adapter,
                    [system, *history_messages],
                    stream_renderer=stream_renderer,
                )

            thinking.stop()
            stream_renderer.stop()
            transcript.append(
                session=session,
                channel=channel,
                speaker=character.name,
                role="assistant",
                content=reply,
            )
            if not streamed:
                console.print(Markdown(reply))
            console.print()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]bye.[/dim]")
    finally:
        thinking.stop()
        transcript.close()
        if compaction_store is not None:
            compaction_store.close()
        if memory_store is not None:
            memory_store.close()
        if semantic_store is not None:
            semantic_store.close()


@app.command()
def describe() -> None:
    """Print Airton's resolved character sheet (sanity check)."""
    character = load_character(settings.character_path)
    console.print(f"[bold]{character.name}[/bold] — {character.premise}\n")
    console.print("[bold]Values[/bold]")
    for v in character.values:
        console.print(f"  • {v.rule}")
    console.print("\n[bold]Taboos[/bold]")
    for t in character.taboos:
        console.print(f"  • {t}")
    console.print("\n[bold]Seed memories[/bold]")
    for s in character.seed_memories:
        console.print(f"  • {s.title} — {s.principle}")
    console.print("\n[bold]Voice samples[/bold]")
    for sample in character.voice_samples:
        console.print(f"  • {sample.id}")


@eval_app.command("voice")
def eval_voice(
    model: str = typer.Option("mlx", help="Adapter: echo | mlx | ollama"),
    model_repo: str | None = typer.Option(
        None,
        "--model-repo",
        help="Override the model id. MLX: HF repo. Ollama: model tag.",
    ),
    lora_path: str | None = typer.Option(
        None,
        "--lora-path",
        help="LoRA adapter directory (from `mlx_lm.lora` training). Requires --model mlx.",
    ),
    sample: list[str] | None = typer.Option(
        None, "--sample", help="Limit to a specific sample id (repeatable)"
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    temperature: float = typer.Option(0.5, help="Sampling temperature"),
    leave_one_out: bool = typer.Option(
        True,
        "--leave-one-out/--no-leave-one-out",
        help="Exclude each sample from its own few-shot examples (default on). "
        "Disable to measure the ceiling with the full example set in view.",
    ),
    persona: bool = typer.Option(
        False,
        "--persona/--no-persona",
        help="Run the voice-rewrite post-pass after the substance pass.",
    ),
    top_k: int = typer.Option(
        6,
        help="Retrieve top-K voice samples by similarity to each prompt "
        "(default 6). Set 0 to show every sample (Phase 1a.2 baseline).",
    ),
    chain_rewrites: bool = typer.Option(
        False,
        "--chain-rewrites/--no-chain-rewrites",
        help="Add a second 'concrete substitution' rewrite pass on top of the "
        "style pass. Requires --persona.",
    ),
    use_judge: bool = typer.Option(
        False,
        "--judge/--no-judge",
        help="After heuristic scoring, ask the adapter to rate each response "
        "1-10 against gold. Circular (same model) but catches register drift "
        "the regex scorer misses.",
    ),
) -> None:
    """Run the canonical voice prompts and show model-vs-gold side by side."""
    character = load_character(settings.character_path)
    # eval runs persona inline in run_voice_eval so both passes stay
    # leave-one-out-consistent — do not wrap adapter here.
    adapter = _resolve_adapter(model, model_repo=model_repo, lora_path=lora_path)
    retriever = _maybe_retriever(character, top_k)

    results = run_voice_eval(
        character,
        adapter,
        temperature=temperature,
        sample_ids=sample if sample else None,
        leave_one_out=leave_one_out,
        persona=persona,
        retriever=retriever,
        top_k=top_k,
        use_judge=use_judge,
        chain_rewrites=chain_rewrites,
    )

    if as_json:
        payload = [
            {
                "sample_id": r.sample_id,
                "prompt": r.prompt,
                "gold": r.gold,
                "actual": r.actual,
                "score": {
                    "aggregate": r.score.aggregate,
                    "length_match": r.score.length_match,
                    "no_banned_openers": r.score.no_banned_openers,
                    "bullet_discipline": r.score.bullet_discipline,
                    "bullet_density": r.score.bullet_density,
                    "filler_discipline": r.score.filler_discipline,
                    "judge_score": r.score.judge_score,
                    "notes": list(r.score.notes),
                },
                **({"draft": r.draft} if r.draft is not None else {}),
            }
            for r in results
        ]
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"Voice eval — {adapter.id}", show_lines=True)
    table.add_column("id", style="bold")
    table.add_column("prompt")
    table.add_column("gold", style="green")
    table.add_column("actual", style="yellow")
    table.add_column("score", style="cyan")
    for r in results:
        judge_line = f"\njudge={r.score.judge_score}/10" if r.score.judge_score is not None else ""
        score_cell = (
            f"{r.score.aggregate:.2f}\n"
            f"len={r.score.length_match:.2f}\n"
            f"open={r.score.no_banned_openers:.0f}\n"
            f"bul={r.score.bullet_discipline:.1f}\n"
            f"den={r.score.bullet_density:.2f}\n"
            f"fil={r.score.filler_discipline:.2f}"
            f"{judge_line}"
        )
        table.add_row(r.sample_id, r.prompt, r.gold.strip(), r.actual.strip(), score_cell)
    aggregate = sum(r.score.aggregate for r in results) / max(len(results), 1)
    console.print(table)
    console.print(
        f"[bold]aggregate voice score:[/bold] {aggregate:.3f} across {len(results)} sample(s)"
    )
    judge_scores = [r.score.judge_score for r in results if r.score.judge_score is not None]
    if judge_scores:
        judge_mean = sum(judge_scores) / len(judge_scores)
        console.print(
            f"[bold]judge mean:[/bold] {judge_mean:.2f}/10 across {len(judge_scores)} sample(s)"
        )


@memory_app.command("list")
def memory_list(
    tier: str | None = typer.Option(None, help="Filter by tier: seed | consolidated | working"),
) -> None:
    """List every record in the episodic store."""
    character = load_character(settings.character_path)
    store = _open_episodic_store(character)
    if store is None:
        raise typer.Exit(code=1)
    try:
        records = store.all(tier=tier)
        if not records:
            console.print("[dim](empty)[/dim]")
            return
        table = Table(title=f"Episodic memory ({len(records)} records)", show_lines=True)
        table.add_column("id", style="bold")
        table.add_column("tier")
        table.add_column("title")
        table.add_column("principle", style="dim")
        for r in records:
            table.add_row(str(r.id), r.tier, r.title, r.principle or "")
        console.print(table)
    finally:
        store.close()


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="What to search for"),
    k: int = typer.Option(3, help="How many matches to return"),
) -> None:
    """Search episodic memory by semantic similarity."""
    character = load_character(settings.character_path)
    store = _open_episodic_store(character)
    if store is None:
        raise typer.Exit(code=1)
    try:
        hits = store.search(query, k=k)
        if not hits:
            console.print("[dim](no matches — store empty?)[/dim]")
            return
        for record, score in hits:
            console.print(f"[bold cyan]{score:.3f}[/bold cyan]  [bold]{record.title}[/bold]")
            if record.principle:
                console.print(f"  [dim italic]{record.principle}[/dim italic]")
            snippet = record.body[:220]
            ellipsis = "…" if len(record.body) > 220 else ""
            console.print(f"  [dim]{snippet}{ellipsis}[/dim]")
            console.print()
    finally:
        store.close()


@memory_app.command("fact-list")
def memory_fact_list(
    tier: str | None = typer.Option(None, help="Filter by tier: seed | consolidated | working"),
    subject: str | None = typer.Option(None, help="Filter by subject"),
) -> None:
    """List facts in the semantic store."""
    store = _open_semantic_store()
    if store is None:
        raise typer.Exit(code=1)
    try:
        facts = store.all(tier=tier, subject=subject)
        if not facts:
            console.print("[dim](empty)[/dim]")
            return
        table = Table(title=f"Semantic facts ({len(facts)})", show_lines=False)
        table.add_column("id", style="bold")
        table.add_column("tier")
        table.add_column("subject")
        table.add_column("predicate")
        table.add_column("object")
        table.add_column("conf", style="cyan")
        for f in facts:
            table.add_row(
                str(f.id), f.tier, f.subject, f.predicate, f.object, f"{f.confidence:.2f}"
            )
        console.print(table)
    finally:
        store.close()


@memory_app.command("fact-search")
def memory_fact_search(
    query: str = typer.Argument(..., help="Query text"),
    k: int = typer.Option(5, help="How many matches to return"),
    min_confidence: float = typer.Option(0.0, help="Minimum confidence to include"),
) -> None:
    """Semantic-search the fact store."""
    store = _open_semantic_store()
    if store is None:
        raise typer.Exit(code=1)
    try:
        hits = store.search(query, k=k, min_confidence=min_confidence)
        if not hits:
            console.print("[dim](no matches)[/dim]")
            return
        for fact, score in hits:
            console.print(
                f"[bold cyan]{score:.3f}[/bold cyan]  "
                f"[bold]{fact.subject}[/bold] {fact.predicate} {fact.object} "
                f"[dim](conf={fact.confidence:.2f}, tier={fact.tier})[/dim]"
            )
    finally:
        store.close()


@memory_app.command("fact-add")
def memory_fact_add(
    subject: str = typer.Argument(..., help="Subject of the fact"),
    predicate: str = typer.Argument(..., help="Predicate (relation)"),
    object_: str = typer.Argument(..., metavar="OBJECT", help="Object / value"),
    confidence: float = typer.Option(0.9, help="Confidence 0-1"),
    tier: str = typer.Option("working", help="Tier: seed | consolidated | working"),
    source: str = typer.Option("user", help="Provenance label"),
    user: str | None = typer.Option(
        None,
        "--user",
        help="Relationship scope. Leave unset (or pass empty) to write a "
        "shared fact visible to everyone.",
    ),
) -> None:
    """Add a single fact to the semantic store."""
    store = _open_semantic_store()
    if store is None:
        raise typer.Exit(code=1)
    try:
        fact = store.add(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence,
            source=source,
            user_id=user if user else None,
            tier=tier,
        )
        console.print(
            f"[green]added[/green] id={fact.id}: "
            f"{fact.subject} {fact.predicate} {fact.object} (conf={fact.confidence:.2f})"
        )
    finally:
        store.close()


@memory_app.command("scribe")
def memory_scribe(
    session: str = typer.Option("local", help="Session id to scribe"),
    user: str = typer.Option(
        "mark",
        "--user",
        help="User to tag scribed memories with (relationship scope). "
        "Use --shared to write character-level shared memory instead.",
    ),
    shared: bool = typer.Option(
        False,
        "--shared",
        help="Write scribed memories as shared (user_id = NULL) rather "
        "than scoped to --user. Intended for character-level extractions.",
    ),
    model: str = typer.Option("mlx", help="Adapter for extraction: echo | mlx | ollama"),
    model_repo: str | None = typer.Option(
        None, "--model-repo", help="Override the model id. MLX: HF repo. Ollama: model tag."
    ),
    lora_path: str | None = typer.Option(
        None,
        "--lora-path",
        help="LoRA adapter directory (from `mlx_lm.lora` training). Requires --model mlx.",
    ),
    window_size: int = typer.Option(20, help="Turns per extraction window"),
) -> None:
    """Walk unprocessed transcript turns and extract candidate memories."""
    character = load_character(settings.character_path)
    adapter = _resolve_adapter(model, model_repo=model_repo, lora_path=lora_path)
    episodic = _open_episodic_store(character)
    semantic = _open_semantic_store()
    if episodic is None or semantic is None:
        raise typer.Exit(code=1)
    transcript = Transcript(settings.db_path)
    try:
        user_id = None if shared else user
        with Status(f"scribe running on session={session}…", console=console):
            summary = run_scribe(
                adapter,
                character,
                transcript,
                episodic,
                semantic,
                session_id=session,
                user_id=user_id,
                window_size=window_size,
            )
        console.print(
            f"processed [bold]{summary.turns_processed}[/bold] turns across "
            f"[bold]{summary.windows}[/bold] window(s). "
            f"wrote [bold]{summary.episodic_written}[/bold] episodic, "
            f"[bold]{summary.semantic_written}[/bold] semantic."
        )
        if summary.parse_errors:
            console.print(f"[yellow]{len(summary.parse_errors)} parse error(s):[/yellow]")
            for err in summary.parse_errors:
                console.print(f"  - {err}")
    finally:
        transcript.close()
        episodic.close()
        semantic.close()


@memory_app.command("consolidate")
def memory_consolidate(
    episodic_threshold: float = typer.Option(
        0.80,
        help="Cosine-similarity threshold for clustering near-duplicate episodes.",
    ),
) -> None:
    """Promote working-tier memories to consolidated. Clusters similar
    episodes; groups semantic facts by (subject, predicate); marks
    superseded rows so they drop out of retrieval. Idempotent."""
    character = load_character(settings.character_path)
    episodic = _open_episodic_store(character, ingest=False)
    semantic = _open_semantic_store()
    if episodic is None or semantic is None:
        raise typer.Exit(code=1)
    try:
        with Status("running consolidation…", console=console):
            summary = run_consolidation(episodic, semantic, episodic_threshold=episodic_threshold)
        console.print(
            f"episodic: considered [bold]{summary.episodic_considered}[/bold] working, "
            f"merged [bold]{summary.episodic_clusters_merged}[/bold] cluster(s), "
            f"promoted [bold]{summary.episodic_promoted}[/bold], "
            f"superseded [bold]{summary.episodic_superseded}[/bold]"
        )
        console.print(
            f"semantic: considered [bold]{summary.semantic_considered}[/bold] working, "
            f"merged [bold]{summary.semantic_groups_merged}[/bold] group(s), "
            f"promoted [bold]{summary.semantic_promoted}[/bold], "
            f"superseded [bold]{summary.semantic_superseded}[/bold]"
        )
    finally:
        episodic.close()
        semantic.close()


@memory_app.command("rebuild-embeddings")
def memory_rebuild_embeddings() -> None:
    """Re-embed every active episodic and semantic record with the
    current embedder. Use after switching embedder models so existing
    data participates in search again."""
    character = load_character(settings.character_path)
    episodic = _open_episodic_store(character, ingest=False)
    semantic = _open_semantic_store()
    if episodic is None or semantic is None:
        raise typer.Exit(code=1)
    try:
        ep_mismatched = episodic.count_mismatched_embeddings()
        sem_mismatched = semantic.count_mismatched_embeddings()
        console.print(
            f"episodic: {ep_mismatched} mismatched; semantic: {sem_mismatched} mismatched."
        )
        with Status("re-embedding episodic…", console=console):
            ep_updated, _ = episodic.rebuild_embeddings()
        with Status("re-embedding semantic…", console=console):
            sem_updated, _ = semantic.rebuild_embeddings()
        console.print(
            f"[green]rebuilt {ep_updated} episodic and {sem_updated} semantic "
            f"embeddings with {episodic.embedder.id}.[/green]"
        )
    finally:
        episodic.close()
        semantic.close()


@memory_app.command("wipe")
def memory_wipe(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Clear episodic, semantic, and scribe-watermark data. Transcripts
    and character data are preserved. Use after switching embedder
    dimensions, or when you want to re-ingest from scratch."""
    if not yes:
        typer.confirm(
            "This wipes all episodic, semantic, and scribe-watermark data. Proceed?",
            abort=True,
        )
    import contextlib
    import sqlite3

    # Hardcoded whitelist — not user input, so S608 is a false positive
    # for this interpolation, but we keep the table names fixed anyway.
    tables = ("episodic", "semantic", "scribe_watermark")
    conn = sqlite3.connect(settings.db_path)
    try:
        for table in tables:
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f"DELETE FROM {table}")  # noqa: S608
        conn.commit()
    finally:
        conn.close()
    console.print("[yellow]wiped episodic, semantic, and scribe_watermark.[/yellow]")


@memory_app.command("ingest")
def memory_ingest() -> None:
    """Force an ingestion pass of the character's seed memories.
    Idempotent — existing records with matching external_id are
    preserved."""
    character = load_character(settings.character_path)
    store = _open_episodic_store(character, ingest=False)
    if store is None:
        raise typer.Exit(code=1)
    try:
        inserted = ensure_seeds_ingested(character, store)
        total = len(store.all())
        console.print(f"[bold]{inserted}[/bold] new, [bold]{total}[/bold] total in episodic store.")
    finally:
        store.close()


@voice_app.command("capture")
def voice_capture(
    session: str = typer.Option("local", help="Session id to pull the exchange from."),
    gold: str = typer.Option(
        ...,
        "--gold",
        help="The corrected reply — what Airton should have said in response "
        "to the last user prompt in the session.",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Override the user prompt this sample is paired with. Defaults "
        "to the last user turn in the session.",
    ),
    sample_id: str | None = typer.Option(
        None,
        "--id",
        help="Custom sample id. Defaults to captured-<UTC timestamp>.",
    ),
) -> None:
    """Capture a user edit of Airton's reply as a new voice sample.

    The captured sample goes into `character/<name>/voice/captured.yaml`,
    a separate file from the curated canonical set, and will be loaded
    alongside canonical samples on the next character load. Over time
    this is how the voice corpus compounds from real use."""
    import yaml

    transcript = Transcript(settings.db_path)
    try:
        history = transcript.tail(session, limit=200)
    finally:
        transcript.close()

    if prompt is None:
        user_turns = [m for m in history if m.role == "user"]
        if not user_turns:
            console.print(
                f"[red]no user turns in session '{session}'. "
                "Pass --prompt to supply one explicitly.[/red]"
            )
            raise typer.Exit(code=1)
        prompt = user_turns[-1].content

    original: str | None = None
    assistant_turns = [m for m in history if m.role == "assistant"]
    if assistant_turns:
        original = assistant_turns[-1].content

    character_dir = settings.character_path
    captured_path = character_dir / "voice" / "captured.yaml"
    captured_path.parent.mkdir(parents=True, exist_ok=True)

    if captured_path.exists():
        doc = yaml.safe_load(captured_path.read_text()) or {"samples": []}
    else:
        doc = {"version": 1, "samples": []}

    if sample_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        sample_id = f"captured-{stamp}"

    new_sample: dict[str, object] = {
        "id": sample_id,
        "prompt": prompt,
        "gold": gold.strip(),
        "captured_at": datetime.now(UTC).isoformat(),
        "captured_from": f"session={session}",
    }
    if original is not None:
        new_sample["original"] = original

    doc.setdefault("samples", []).append(new_sample)
    captured_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))

    console.print(
        f"[green]captured[/green] id={sample_id!r} "
        f"→ {captured_path.relative_to(settings.root)} "
        f"(now {len(doc['samples'])} captured sample(s))"
    )


@voice_app.command("list-captured")
def voice_list_captured() -> None:
    """List every captured sample in the current character."""
    import yaml

    captured_path = settings.character_path / "voice" / "captured.yaml"
    if not captured_path.exists():
        console.print("[dim](no captured samples yet)[/dim]")
        return
    doc = yaml.safe_load(captured_path.read_text()) or {}
    samples = doc.get("samples", []) or []
    if not samples:
        console.print("[dim](no captured samples yet)[/dim]")
        return
    table = Table(title=f"Captured samples ({len(samples)})", show_lines=True)
    table.add_column("id", style="bold")
    table.add_column("captured_at")
    table.add_column("prompt")
    table.add_column("gold", style="green")
    for s in samples:
        table.add_row(
            str(s.get("id", "?")),
            str(s.get("captured_at", "?")),
            str(s.get("prompt", "?")),
            str(s.get("gold", "?")),
        )
    console.print(table)


if __name__ == "__main__":
    app()
