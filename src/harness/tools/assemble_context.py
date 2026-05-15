"""assemble_context tool (harness-xysp / Phase 3 follow-up).

Wires the `assemble_package` orchestrator behind an agent-callable tool.
The Phase 3 modules ship the primitive; this tool makes it usable
inside a chat turn: the agent picks a role, supplies variables, and
gets a rendered context package back.

Why a tool instead of auto-loading at session start: contracts take
task-specific variables (customer_id, request_summary, ...) that
aren't known when chat boots. A tool defers the contract resolution
to the moment the agent has the variables in hand.

The tool's description interpolates every registered contract's role
+ template variables at spec time so the model can target the call
without a separate `list_contracts` round-trip — same pattern
`query_table` uses for table schemas.

Render shape (returned as the tool result):

    ```
    Context Package — <intent>
    Role: <role>   Variables: customer_id=C9148, request_summary=...
    Tokens used: 91 / 1500   (overflow: 0)

    ## <slot_name> (<n hits>)
    - [<store>/<method> score=<float>] <record_id>
      <body>
    ...

    ⚠️ Missing required slots: <names>   ← only when incomplete
    ```

Production tier=read: the contract only fans out to read-only store
methods. Errors (unknown role, missing variable, store failure)
surface as graceful error strings so the agent loop doesn't crash
mid-turn.
"""

from __future__ import annotations

import re
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from harness.retrieval.context_package import AccessPolicy, PackagedHit
from harness.retrieval.contract import (
    ContractBundle,
    StoreBundle,
    assemble_package,
    load_contract,
)
from harness.tools.base import ToolSpec

_TEMPLATE_VAR_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_BODY_CAP_CHARS = 1200


def _extract_template_variables(contract: ContractBundle) -> tuple[str, ...]:
    """Walk every slot's query_template / sql_template, pull out the
    {var} names. Deduped, sorted. Used in the tool description so the
    model knows what to pass."""
    seen: dict[str, None] = {}
    for slot in contract.slots:
        for tpl in (slot.query_template, slot.sql_template):
            if not tpl:
                continue
            for name in _TEMPLATE_VAR_RE.findall(tpl):
                seen.setdefault(name, None)
    return tuple(sorted(seen))


@dataclass
class _ContractEntry:
    """Cached parsed contract + its template variables. Built at tool
    construction so spec materialization is cheap."""

    contract: ContractBundle
    variables: tuple[str, ...]


def _load_contracts(contracts_dir: Path) -> dict[str, _ContractEntry]:
    """Eager-load every *.yaml under `contracts_dir`. Malformed
    contracts skip with a warning rather than crashing tool init —
    the missing role just isn't available for the agent to call."""
    out: dict[str, _ContractEntry] = {}
    if not contracts_dir.exists():
        return out
    for path in sorted(contracts_dir.glob("*.yaml")):
        try:
            contract = load_contract(path)
        except (ValueError, OSError):
            # Skip silently — bad contract is a fixture issue, not a
            # reason to break the agent loop. (Tests cover the
            # ValueError path of load_contract; production runs
            # surface the issue via observability, not by failing init.)
            continue
        out[contract.role] = _ContractEntry(
            contract=contract,
            variables=_extract_template_variables(contract),
        )
    return out


@dataclass
class AssembleContextTool:
    """Materialize a role contract into a rendered context package.

    The tool is wired with the concrete stores it should fan out to
    (via `StoreBundle`) and the directory holding contract YAMLs.
    Both are passed at construction; the tool doesn't reach into the
    filesystem or the model adapter beyond what the underlying
    `assemble_package` call already does.
    """

    stores: StoreBundle
    contracts_dir: Path
    user_id: str | None = None
    role_default: str | None = None
    _contracts: dict[str, _ContractEntry] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._contracts = _load_contracts(self.contracts_dir)

    @property
    def spec(self) -> ToolSpec:
        if not self._contracts:
            description = (
                "(no contracts registered — drop a YAML in the "
                f"contracts dir at {self.contracts_dir}.)"
            )
        else:
            sections: list[str] = []
            for role in sorted(self._contracts):
                entry = self._contracts[role]
                vars_blurb = ", ".join(entry.variables) or "(none)"
                sections.append(
                    f"- role={role}\n"
                    f"  intent: {entry.contract.intent}\n"
                    f"  variables: {vars_blurb}\n"
                    f"  slots: {len(entry.contract.slots)} "
                    f"(required: {sum(1 for s in entry.contract.slots if s.required)})"
                )
            description = (
                "Materialize a role contract into a packaged context "
                "bundle. Use this when starting a task that has a "
                "well-defined contract — the response carries every "
                "slot the role needs, with provenance and budget "
                "accounting. If a required slot can't be filled, the "
                "response says so up front rather than working with "
                "silently-incomplete context.\n\n"
                "Available contracts:\n\n" + "\n".join(sections)
            )
        return ToolSpec(
            name="assemble_context",
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": "Name of a registered contract role.",
                    },
                    "variables": {
                        "type": "object",
                        "description": (
                            "Template variables for the contract's slots. "
                            "Keys are variable names; values are strings."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["role"],
            },
            tier="read",
            display_name="Assemble context",
            high_noise=True,
        )

    def call(self, *, role: str, variables: Mapping[str, str] | None = None) -> str:
        entry = self._contracts.get(role)
        if entry is None:
            available = ", ".join(sorted(self._contracts)) or "(none)"
            return f"assemble_context error: unknown role {role!r}. Available: {available}"
        vars_dict = dict(variables or {})
        access = AccessPolicy(user_id=self.user_id, role=role)
        try:
            package = assemble_package(
                entry.contract,
                variables=vars_dict,
                access=access,
                stores=self.stores,
            )
        except ValueError as exc:
            # Expected ValueErrors from assemble_package describe contract
            # misconfiguration: missing template variable, slot's store
            # not in the bundle. Unexpected ValueErrors bubble up from
            # deeper layers (embedder, subprocess machinery during
            # tokenizer load, etc.) — observed 2026-05-14 chat session
            # firing 'bad value(s) in fds_to_keep' from inside the
            # embed/search pipeline with no stack visible because we
            # swallowed it to a one-line tool result.
            #
            # Always dump the traceback to stderr so the next intermittent
            # failure is localizable from the running session, while
            # keeping the model-facing tool result a clean one-liner.
            traceback.print_exc(file=sys.stderr)
            return f"assemble_context error: {exc}"
        return _render_package(
            role=role,
            contract=entry.contract,
            variables=vars_dict,
            package_intent=package.intent,
            tokens_used=package.tokens_used,
            tokens_max=package.budget.max_tokens,
            tokens_overflow=package.tokens_overflow,
            hits=package.hits,
            missing=package.missing_required_slots,
        )


def _render_package(
    *,
    role: str,
    contract: ContractBundle,
    variables: Mapping[str, str],
    package_intent: str,
    tokens_used: int,
    tokens_max: int,
    tokens_overflow: int,
    hits: tuple[PackagedHit, ...],
    missing: tuple[str, ...],
) -> str:
    """Render the package as the agent-facing tool result. Markdown-
    ish: section headers per slot, one bullet per hit, provenance
    inline so the model can quote source ids back to the user.
    Truncates each hit body at `_BODY_CAP_CHARS` so a single oversized
    row doesn't blow the token window."""
    vars_blurb = ", ".join(f"{k}={v}" for k, v in variables.items()) if variables else "(none)"
    header = [
        f"Context Package — {package_intent}",
        f"Role: {role}   Variables: {vars_blurb}",
        f"Tokens used: {tokens_used} / {tokens_max}   (overflow: {tokens_overflow})",
    ]
    if missing:
        header.append(f"⚠️ Missing required slots: {', '.join(missing)}")

    # Group hits by slot, preserving the contract's slot order.
    lines: list[str] = ["\n".join(header), ""]
    for slot in contract.slots:
        slot_hits = [h for h in hits if h.slot_name == slot.name]
        if not slot_hits and slot.name not in missing:
            # Optional slot returned nothing — skip rather than print
            # an empty section. Required slots that returned nothing
            # already showed up in the missing-slots line above.
            continue
        if not slot_hits:
            lines.append(f"## {slot.name} (required, no hits)\n")
            continue
        lines.append(f"## {slot.name} ({len(slot_hits)} hit{'' if len(slot_hits) == 1 else 's'})")
        for hit in slot_hits:
            body = hit.body
            if len(body) > _BODY_CAP_CHARS:
                body = body[: _BODY_CAP_CHARS - 1] + "…"
            lines.append(
                f"- [{hit.provenance.store}/{hit.provenance.method} "
                f"score={hit.provenance.score:.3f}] {hit.provenance.record_id}"
            )
            for body_line in body.splitlines():
                lines.append(f"  {body_line}")
        lines.append("")

    return "\n".join(lines).rstrip()
