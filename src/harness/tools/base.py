from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from harness.tools.catalog import ToolCatalog


def tool_schema_from_model(model: type[BaseModel]) -> dict[str, Any]:
    """Render a tool's `parameters` JSON Schema from a pydantic args
    model (harness-5cjj9). Normalizes pydantic's `model_json_schema()`
    output to the flat `{type, properties, required}` shape the rest of
    the harness emits by hand: strips `title`/`default` noise and
    collapses `int | None` style `anyOf:[T, null]` unions back to the
    bare type `T` (optionality is already expressed by omission from
    `required`), so a generated schema reads the same as the
    hand-written ones the model and router already consume.

    Pilot models are flat (no nested BaseModel / `$ref`); extend this if
    a future tool nests models."""
    raw = model.model_json_schema()
    props = {name: _clean_schema_prop(sub) for name, sub in (raw.get("properties") or {}).items()}
    return {
        "type": "object",
        "properties": props,
        "required": list(raw.get("required", [])),
    }


def _clean_schema_prop(prop: dict[str, Any]) -> dict[str, Any]:
    p = dict(prop)
    description = p.get("description")
    p.pop("title", None)
    p.pop("default", None)
    any_of = p.get("anyOf")
    if any_of is not None:
        variants = [v for v in any_of if v.get("type") != "null"]
        if len(variants) == 1:
            merged = {k: v for k, v in variants[0].items() if k != "title"}
            if description is not None:
                merged["description"] = description
            return merged
        p["anyOf"] = variants
    return p


def format_validation_error(tool_name: str, exc: ValidationError) -> str:
    """Turn a pydantic ValidationError into a model-actionable message
    that names each bad field, what was wrong, and the value supplied —
    so the model can self-correct on the same round instead of repeating
    the call (harness-5cjj9, harness-ln7j)."""
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        line = f"  - {loc}: {err['msg']}"
        if "input" in err:
            line += f" (got {err['input']!r})"
        lines.append(line)
    body = "\n".join(lines) or "  - (no field detail)"
    return f"tool {tool_name!r} rejected arguments:\n{body}\nFix the field(s) and retry."


@dataclass(frozen=True)
class ToolSpec:
    """JSON-schema description of a tool, plus metadata the orchestrator
    needs (tier for authorization)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for arguments
    tier: str  # "read" | "write" — write-tier tools need user confirmation
    display_name: str | None = None  # human-readable label for UI; falls back to `name`
    # Tools flagged high_noise dump large, often-unhelpful bulk into
    # the message thread — grep / list_dir / search_web results are
    # classic offenders. When the tool-result summarizer hook is
    # registered (CLI --summarize-tool-results), outputs from these
    # tools above a size threshold get compressed before the model
    # sees them. Everything else passes through untouched.
    high_noise: bool = False
    # Argument-dependent write escalation (harness-qcukc). Some tools are
    # read-tier in the common case but overwrite files for certain
    # argument shapes — `stream_edit` / `python_stream` with
    # `in_place=True` are the canonical offenders. `tier` reflects the
    # static, common-case authorization; `write_when` is an optional
    # predicate over the call's arguments that escalates a single
    # invocation to write-tier. The orchestrator gates confirmation on
    # `effective_tier(arguments)`, never on `tier` alone, so an
    # argument-dependent write cannot slip through the read-tier path
    # (including the router prelude, which only auto-executes read-tier
    # calls). None = the tier is fixed.
    write_when: Callable[[Mapping[str, Any]], bool] | None = None
    # Optional typed argument model (harness-5cjj9). When set, the
    # registry validates + coerces the raw argument dict through it
    # before dispatch, returning a structured field error on mismatch
    # instead of relying on a post-hoc Python TypeError. `parameters`
    # is typically generated from the same model via
    # `tool_schema_from_model`, so the model-visible schema and the
    # runtime validation come from one source. None = legacy path
    # (raw dict forwarded to call(), TypeError recovery).
    args_model: type[BaseModel] | None = None

    @property
    def label(self) -> str:
        return self.display_name or self.name

    @property
    def can_write(self) -> bool:
        """True if this tool ever performs a write, statically — either
        it is write-tier outright, or some argument shape escalates it.
        Used where the decision must be made without a concrete call in
        hand (e.g. excluding escalatable tools from read-only subagents),
        as opposed to `effective_tier`, which needs the arguments."""
        return self.tier == "write" or self.write_when is not None

    def effective_tier(self, arguments: Mapping[str, Any]) -> str:
        """Authorization tier for a *specific* invocation. Returns
        ``"write"`` when the static tier is write, or when `write_when`
        fires for these arguments; otherwise the static tier. This is
        the value the orchestrator must gate confirmation on — gating on
        `tier` alone leaks argument-dependent writes (harness-qcukc)."""
        if self.tier == "write":
            return "write"
        if self.write_when is not None and self.write_when(arguments):
            return "write"
        return self.tier


@runtime_checkable
class Tool(Protocol):
    """A tool has a ToolSpec and a `call(**kwargs) -> str` method. The
    `call` signature is intentionally omitted from the protocol: each
    concrete tool has typed, per-tool keyword-only arguments (e.g.
    `call(self, *, path: str)`), which would fail strict structural
    typing against a generic `call(**kwargs: Any)` declaration. The
    registry invokes `call` with unpacked kwargs at runtime; tests
    cover the argument shape per tool."""

    @property
    def spec(self) -> ToolSpec: ...


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolHit:
    """A single retrieval hit surfaced by a grounding-tier tool. Decoupled
    from store types (EpisodicRecord, SemanticFact, WebHit) so hooks and
    audit consumers can reason about retrieval without importing the
    store layer.

    Introduced with the Tool.call() structured return type (harness-ywp.4).
    Populated today by search_memory; future tools (search_facts,
    fetch_url) will follow the same shape when their audit consumers
    land."""

    source: str  # "episodic" | "semantic" | "web" | etc.
    external_id: str | None
    title: str
    score: float
    principle: str | None = None


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    output: str
    success: bool = True
    error: str | None = None
    # Structured metadata from grounding-tier tools — consumed by the
    # per-turn audit log (harness-ywp.2) and the low-confidence fallback
    # hook (harness-ywp.3). Empty tuple / frozenset when the tool
    # doesn't expose retrieval signal (most tools). Plain-text tools
    # build these via `ToolResult.text(name, output)`.
    hits: tuple[ToolHit, ...] = ()
    citations_grounded: frozenset[str] = frozenset()

    @classmethod
    def text(cls, tool_name: str, output: str) -> ToolResult:
        """Factory for plain-text tool output. Equivalent to
        `ToolResult(tool_name, output)` — a call-site marker that the
        tool has no structured metadata to expose."""
        return cls(tool_name=tool_name, output=output)

    @property
    def top_score(self) -> float | None:
        """Highest score across `hits`, or None when no hits. Convenience
        for the confidence-fallback hook (harness-ywp.3)."""
        if not self.hits:
            return None
        return max(h.score for h in self.hits)


@dataclass(frozen=True)
class ModelReply:
    """Richer return type for adapters that support tool use. `content`
    may be empty if the model only wants to call tools this round.

    `was_truncated` and `had_unparseable_call` are diagnostic hints the
    orchestrator uses to recover from common failure modes (token-limit
    truncation, malformed `<tool_call>` blocks). Adapters that can't
    cheaply detect these leave them False — the loop falls back to
    teaser-regex detection."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    was_truncated: bool = False
    had_unparseable_call: bool = False

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True)
class StreamText:
    """Visible text delta emitted by a streaming adapter. The accumulation
    of every `StreamText.text` across a stream reconstructs the model's
    raw output (tool-call tags included — callers mask them for display)."""

    text: str


@dataclass(frozen=True)
class StreamComplete:
    """Terminal chunk of a tool-aware stream. Carries the parsed
    ModelReply so the tool loop can dispatch any tool calls the model
    emitted. Exactly one StreamComplete is yielded per stream, always
    last."""

    reply: ModelReply


StreamChunk = StreamText | StreamComplete


class ToolRegistry:
    """Holds tools by name, renders specs for the model, dispatches
    calls. Registry is mutable (CLI configures it per session); tool
    instances are read-only after construction."""

    def __init__(self, *, catalog: ToolCatalog | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        # Per-registration description overrides. Applied at specs()
        # emission so every downstream consumer (router, model schema,
        # introspect) sees the override without wrapping Tool instances.
        # Populated via override_description(), typically from
        # apply_profile_descriptions() in tools/profiles.py — which
        # layers BUILTIN_PROFILE_DESCRIPTIONS with the active
        # character's per-profile overrides loaded from
        # `character/<name>/tool_descriptions.yaml` (e.g. atc's
        # search_memory targets a rulebook, not "past events").
        self._description_overrides: dict[str, str] = {}
        # Working-set protocol (harness-fzvg). When None, specs()
        # renders every registered tool (existing behavior). When a
        # set, specs() renders only the named subset — the catalog
        # is the superset of *available* tools; the working set is
        # what the model sees this turn. Tools registered after
        # set_active() stay inactive until added explicitly. Reset
        # to all-active via clear_active().
        self._active_names: set[str] | None = None
        # Optional catalog handle (harness-yczi). When set, the
        # unknown-tool error path consults it: catalog-known names get
        # a load_tool-recovery hint; genuine misses point at
        # tool_search. Registries built without a catalog keep the
        # legacy single-shape error.
        self._catalog: ToolCatalog | None = catalog

    def set_catalog(self, catalog: ToolCatalog | None) -> None:
        """Wire (or unwire) a ToolCatalog for unknown-tool
        disambiguation. Optional — registries without a catalog use
        the legacy 'not found, here's the active list' error."""
        self._catalog = catalog

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        self._tools[name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"no tool named {name!r}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def override_description(self, name: str, description: str) -> None:
        """Replace the description surfaced in specs() for `name`. Raises
        KeyError if the tool isn't registered — overrides are meant to
        reframe real tools, not stub placeholders."""
        if name not in self._tools:
            raise KeyError(f"no tool named {name!r}")
        self._description_overrides[name] = description

    # --- working-set protocol (harness-fzvg) ----------------------------

    def set_active(self, names: Iterable[str]) -> None:
        """Restrict specs() to the named subset of registered tools.

        Names that aren't registered are silently dropped — the caller
        (typically `resolve_active` in profiles.py) may resolve a tag
        or profile to names that the current registry doesn't know
        about (e.g. a tag the catalog lists but no tool implements
        yet). Better than raising: lets the working set be expressed
        in catalog-level terms without coupling to which tools the
        session happened to register.
        """
        self._active_names = {n for n in names if n in self._tools}

    def clear_active(self) -> None:
        """Reset to all-active — every registered tool's spec renders.
        Same effect as never calling set_active. Idempotent."""
        self._active_names = None

    def active_names(self) -> tuple[str, ...]:
        """Names currently in the working set, sorted. When no working
        set is set, returns every registered tool (matches what
        specs() would render)."""
        if self._active_names is None:
            return tuple(sorted(self._tools))
        return tuple(sorted(self._active_names))

    def specs(self) -> list[ToolSpec]:
        """Tool specs the model sees this turn. When a working set is
        active, only names in it render — names registered later stay
        inactive until added (set_active again or clear_active to
        return to all-active)."""
        out: list[ToolSpec] = []
        for name, tool in self._tools.items():
            if self._active_names is not None and name not in self._active_names:
                continue
            spec = tool.spec
            override = self._description_overrides.get(spec.name)
            out.append(replace(spec, description=override) if override else spec)
        return out

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute the named tool. Failures are returned as ToolResult,
        not raised — the caller is the orchestrator, which wants to
        feed errors back to the model for it to recover from.

        Unknown-keyword TypeErrors get a structured rewrite (harness-d7e)
        that lists the tool's accepted properties so the model can
        retry without the offending field instead of repeating the
        same call. Generic Python TypeError messages don't enumerate
        valid kwargs, so the model has no way to know what to drop."""
        if name not in self._tools:
            active = list(self.active_names())
            if self._catalog is not None and self._catalog.get(name) is not None:
                # Catalog-known but not active this session: the model
                # called a tool whose schema it hadn't yet pulled in.
                # Hand back a load_tool-recovery hint instead of the
                # generic 'unknown' shape, which the model otherwise
                # reads as 'no such tool anywhere' and doom-loops on
                # tool_search (harness-yczi).
                msg = (
                    f"unknown tool: {name!r}. The tool exists in the catalog "
                    f"but is not active in this session. Call "
                    f"load_tool(name={name!r}) first to activate it, then "
                    f"retry. Currently active: {active}."
                )
            else:
                msg = (
                    f"unknown tool: {name!r}. Not in the catalog. "
                    f"Currently active: {active}. Use tool_search to find "
                    f"an available tool, or tell the user you cannot answer."
                )
            return ToolResult(
                tool_name=name,
                output=msg,
                success=False,
                error="unknown_tool",
            )
        tool = self._tools[name]
        # Typed-argument validation (harness-5cjj9). When the tool
        # declares an args_model, validate + coerce the raw dict through
        # it before dispatch — a malformed call is rejected with a
        # structured field error the model can act on, rather than
        # reaching tool code and failing on a Python TypeError. Coercion
        # also subsumes per-tool string→number / number→string fixups
        # (e.g. stream_edit int args, harness-ln7j).
        args_model = tool.spec.args_model
        if args_model is not None:
            try:
                validated = args_model.model_validate(arguments)
            except ValidationError as exc:
                errors = exc.errors()
                extras = [e for e in errors if e.get("type") == "extra_forbidden"]
                if extras and len(extras) == len(errors):
                    # Every error is an unknown argument — preserve the
                    # harness-d7e hint (lists what the tool DOES accept,
                    # keyed off the schema) so the model retries without
                    # the offending field instead of repeating the call.
                    unknown_field = str(extras[0]["loc"][-1])
                    accepted = sorted((tool.spec.parameters.get("properties") or {}).keys())
                    return ToolResult(
                        tool_name=name,
                        output=(
                            f"tool {name!r} rejected unknown argument {unknown_field!r}. "
                            f"Accepts: {', '.join(accepted) or '(none)'}. "
                            "Retry without the unknown field."
                        ),
                        success=False,
                        error=f"unknown_kwarg:{unknown_field}",
                    )
                return ToolResult(
                    tool_name=name,
                    output=format_validation_error(name, exc),
                    success=False,
                    error="validation_error",
                )
            # exclude_unset: forward only the arguments the caller
            # actually supplied (coerced), never injected defaults — the
            # dispatch shape stays identical to the legacy raw-dict path,
            # so a tool whose call() omits an optional kwarg isn't handed
            # an unexpected one.
            arguments = validated.model_dump(exclude_unset=True)
        try:
            # Tool's `call` is not on the Protocol (see Tool docstring); each
            # concrete implementation supplies it with typed kwargs.
            # Tools may return `str` (legacy) or `ToolResult` directly
            # (structured return, harness-ywp.4). The Registry normalises
            # both to a ToolResult below.
            out: str | ToolResult = tool.call(**arguments)  # type: ignore[attr-defined]
        except TypeError as exc:
            unknown = _unknown_kwarg_from(exc)
            if unknown is not None:
                accepted = sorted((tool.spec.parameters.get("properties") or {}).keys())
                msg = (
                    f"tool {name!r} rejected unknown argument {unknown!r}. "
                    f"Accepts: {', '.join(accepted) or '(none)'}. "
                    "Retry without the unknown field."
                )
                return ToolResult(
                    tool_name=name,
                    output=msg,
                    success=False,
                    error=f"unknown_kwarg:{unknown}",
                )
            return ToolResult(
                tool_name=name,
                output=f"error calling {name}: {exc}",
                success=False,
                error=f"TypeError: {exc}",
            )
        except Exception as exc:  # any failure goes back to the model, not up the stack
            return ToolResult(
                tool_name=name,
                output=f"error calling {name}: {exc}",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        if isinstance(out, ToolResult):
            # Structured return — trust the tool's `tool_name`, success flag,
            # and metadata, but re-stamp tool_name from the registry in case
            # of mismatch (defensive — keeps audit attribution honest).
            return out if out.tool_name == name else replace(out, tool_name=name)
        return ToolResult(tool_name=name, output=out, success=True)


_UNKNOWN_KWARG_RE = re.compile(r"got an unexpected keyword argument ['\"]([^'\"]+)['\"]")


def _unknown_kwarg_from(exc: TypeError) -> str | None:
    """Pluck the offending kwarg name from a TypeError raised by a
    `call(**arguments)` invocation. Returns None when the TypeError
    came from something else (shape mismatch, missing required arg)
    — those flow through the generic error path."""
    match = _UNKNOWN_KWARG_RE.search(str(exc))
    return match.group(1) if match else None
