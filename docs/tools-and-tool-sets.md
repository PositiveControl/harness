# Tools and tool-sets

How to add a new tool, register it under one or more tool-set profiles, and —
when needed — reframe its description for a specific character.

This is the practical authoring guide. For the authoring of *characters* that
USE these tools, see [`character-authoring.md`](character-authoring.md). For
the runtime architecture (orchestrator loop, hook pipeline, fabrication
catchers), see [`tool_loop_flow.md`](tool_loop_flow.md) and
[`../CLAUDE.md`](../CLAUDE.md).

---

## Anatomy of a tool

Every tool is a small dataclass conforming to the `Tool` protocol in
`src/harness/tools/base.py`:

```python
class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...    # name, description, JSON-schema params, tier
    def call(self, **kwargs) -> ToolResult | str: ...
```

### `ToolSpec` fields

| Field          | Purpose                                                                |
|----------------|------------------------------------------------------------------------|
| `name`         | Stable identifier the model calls (e.g. `read_file`).                  |
| `description`  | The single most load-bearing line — drives router intent.              |
| `parameters`   | JSON-schema dict; what the model emits as `arguments`.                 |
| `tier`         | `"read"` (free) or `"write"` (per-session approval modal).             |
| `display_name` | Optional human label for renderers (TUI, audit log).                   |

### `ToolResult`

`ToolResult.text(name, "string output")` for plain results;
`ToolResult.text_with_hits(name, body, hits=...)` when the tool grounds
specific citations (search_memory does this so fabrication catchers can ask
"did any tool actually ground §X?").

Pure failure modes return `ToolResult.error(...)`. Don't raise from `call()`
unless something genuinely catastrophic happened — the orchestrator catches
exceptions but loses the structured diagnostic.

---

## Where the file goes

`src/harness/tools/<name>.py` — one tool per file is the convention. The
existing 20-ish tools (read_file, edit_file, search_memory, fetch_url,
introspect, spawn_subagent, etc.) all follow this layout. Look at
`src/harness/tools/read_file.py` as the simplest template.

Re-export from `src/harness/tools/__init__.py` so `cli.py` and `cli_classic.py`
can import:

```python
from harness.tools.your_tool import YourTool
__all__ += ["YourTool"]
```

---

## Step 1 — Write the tool

Minimal example (a read-tier tool that reflects its input):

```python
# src/harness/tools/echo_tool.py
from dataclasses import dataclass

from harness.tools.base import ToolResult, ToolSpec


@dataclass
class EchoTool:
    """Returns its input verbatim. Smoke-test for tool wiring."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description=(
                "Return the input string verbatim. Useful for checking the "
                "tool loop is alive when no other read-tier tool is loaded."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Input to echo back.",
                    },
                },
                "required": ["text"],
            },
            tier="read",
            display_name="Echo",
        )

    def call(self, *, text: str) -> ToolResult:
        return ToolResult.text(self.spec.name, text)
```

Notes:

- **Description goes through the router.** The intent router reads this verbatim;
  ambiguous descriptions are the #1 cause of router misclassification. Be
  specific. If your tool overlaps with another (e.g. `search_memory` vs
  `search_facts`), name what each does NOT do.
- **`tier="write"` triggers the per-session confirmation modal.** Default to
  `"read"` unless your tool mutates filesystem / network / store state.
- **Workspace-sandboxed tools** (`read_file`, `write_file`, `shell`, `git_*`)
  take a `root: Path` field and clamp every path to that root. Use
  `Path.resolve()` + a `_clamp_to_root` helper. See `read_file.py:_clamp_to_root`.

---

## Step 2 — Register the tool in the builder map

Two CLI builders construct the registry — one for the TUI path, one for
classic REPL. Both live in `cli.py` / `cli_classic.py` as a `builders`
dict. Add an entry keyed by your tool name:

```python
# src/harness/cli.py — inside _build_tool_registry_for_tui()
builders: dict[str, Callable[[], Tool | None]] = {
    "read_file": lambda: ReadFileTool(root=workspace_path),
    # …
    "echo": lambda: EchoTool(),
    # …
}
```

The lambda is called only when the tool ends up in the resolved tool list
(via the active `--tool-set` plus `--tools-add` / `--tools-drop`). Returning
`None` from the lambda means "this tool can't be built in this context"
(e.g. `search_memory` returns None when the memory store isn't open) — the
builder warns and skips registration.

Mirror the entry in `cli_classic.py:build_classic_registry` (same dict
structure). The two paths are kept in sync by convention.

---

## Step 3 — Pick which tool-set profile(s) the tool belongs to

`src/harness/tools/profiles.py` defines `TOOL_PROFILES`: named tuples of
tool names that group commonly-used sets. Token budget per profile:
~1,500 tokens of schema overhead (measured via
`scripts/bench_tool_use.py --measure-tokens`).

Existing profiles:

| Profile      | Purpose                                                   |
|--------------|-----------------------------------------------------------|
| `minimal`    | Empty. Pure smoke test.                                   |
| `core`       | Default chat: read-tier fs + memory + introspect.         |
| `coding`     | Active code work: read + write + shell + git + memory.    |
| `memory`     | Memory curation: search + remember + scribe + consolidate. |
| `diagnostic` | Self-inspection.                                          |
| `research`   | Web-research: search + fetch + read + memory + subagent.  |
| `ops`        | ab's bd-routed personal-operations set (closed surface).  |
| `atc`        | airton_c's read-tier corpus + research + memory.          |
| `phraseology`| Single-purpose phraseology lint surface.                  |
| `full`       | Kitchen sink — every tool.                                |

To add `echo` to `core` and `coding`:

```python
TOOL_PROFILES["core"] = (..., "introspect", "echo")
TOOL_PROFILES["coding"] = (..., "introspect", "spawn_subagent", "echo")
```

Or to define a brand-new profile:

```python
TOOL_PROFILES["debug"] = (
    "read_file",
    "list_dir",
    "echo",
    "introspect",
)
```

After this, `--tool-set debug` resolves to those tools.

The `add` / `drop` flags compose: `--tool-set core --tools-add echo
--tools-drop search_facts`.

---

## Step 4 — Tests

Tests hit real stores when relevant; mock only the model adapter.
`tests/test_tools.py` covers the simple read tools.

Pattern for a new tool test:

```python
def test_echo_round_trips_input() -> None:
    tool = EchoTool()
    result = tool.call(text="hello")
    assert result.success
    assert result.output == "hello"


def test_echo_spec_shape() -> None:
    spec = EchoTool().spec
    assert spec.name == "echo"
    assert spec.tier == "read"
    assert "text" in spec.parameters["properties"]
    assert spec.parameters["required"] == ["text"]
```

Add a profile-membership test in `tests/test_tool_profiles.py`:

```python
def test_echo_in_core_profile() -> None:
    assert "echo" in resolve_tool_names("core")
```

---

## Step 5 — (Optional) Per-character description override

When a generic tool description doesn't fit a specific character — for
example, `search_memory`'s default ("past events and lessons") reads as
autobiographical for the airton_c1 character whose seeds ARE the FAA JO
7110.65 rulebook — declare a per-character override:

```yaml
# character/airton_c1/tool_descriptions.yaml
profiles:
  atc:
    search_memory: >-
      Search the FAA JO 7110.65 air-traffic control rulebook for sections,
      procedures, phraseology, separation minimums, and controller
      responsibilities. Use for ANY rule-shaped question…
```

Mechanism: `apply_profile_descriptions(registry, profile, character=...)` runs
at registry build time. It layers
`BUILTIN_PROFILE_DESCRIPTIONS[profile]` (currently empty back-stop) with
`character.tool_descriptions[profile]`; character entries win on key
collision. The override applies at `specs()` emission so every downstream
consumer (router, model schema, introspect) sees it.

See [`character-authoring.md`](character-authoring.md) — Step 5 — for the
full authoring pattern.

---

## Authoring patterns

### Read-tier tool with grounding declarations

If your tool returns content the model is meant to cite (e.g. corpus
sections, search results), declare the citations on the result so
fabrication catchers can ask structural questions:

```python
return ToolResult.text_with_hits(
    self.spec.name,
    body=rendered_text,
    hits=tuple(
        ToolHit(source="episodic", external_id=rec.external_id, ...)
        for rec, _score in matches
    ),
    citations_grounded=frozenset({"§3-9-10", "§3-10-5"}),
)
```

`UngroundedCitationHook` and `LowConfidenceFallbackHook` consume
`citations_grounded` to decide whether the model's reply hallucinated a
section the tool didn't actually surface.

### Write-tier tool with confirmation

```python
return ToolSpec(
    name="delete_file",
    description="…",
    parameters={...},
    tier="write",          # ← triggers the per-session approval modal
    display_name="Delete file",
)
```

The first `delete_file` call in a session pops the modal; user picks
APPROVE_ONCE / APPROVE_SESSION / DECLINE. APPROVE_SESSION pre-approves every
subsequent call. The tool itself doesn't need to know any of this — the
orchestrator gates dispatch.

### Workspace-sandboxed tool

Take a `root: Path` field, clamp every path argument with a helper:

```python
@dataclass
class WorkspaceTool:
    root: Path

    def _clamp(self, candidate: str) -> Path:
        path = (self.root / candidate).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise ValueError(f"path {candidate} escapes workspace {self.root}")
        return path
```

The CLI wires `root=workspace_path` from `--workspace`. Memory + transcripts
stay under the harness data dir regardless — only filesystem-tool access is
sandboxed.

### Tool that needs an adapter (small-model meta-tool)

`spawn_subagent` is the precedent. The orchestrator builds a depth-1
read-only sub-loop on demand:

```python
@dataclass
class SpawnSubagentTool:
    adapter: ModelAdapter
    registry: ToolRegistry
    hooks: HookPipeline
    router: Router | None
    # ...
```

The CLI builder constructs it lazily so the cost (instantiating a separate
hook pipeline) is paid only when the tool is in the active set.

---

## Wiring an opt-in domain catcher to a tool's behavior

Some tools depend on catchers that are themselves opt-in per character
(harness-qvwq). For example, `phraseology_lint` benefits from
`reserved_squawk_code` and `scope_redirect` running in the parent chat — but
nothing in the tool registers them. The character's `core.yaml: catchers:`
roster handles registration; the tool just trusts that what's wired is
wired.

Available opt-in catchers (today):

- `ab_fabrication` — ab_ops capture/plan/remember imitation.
- `ambiguous_context` — domain term with multiple meanings.
- `scope_redirect` — out-of-scope prompt drifting onto a corpus topic.
- `reserved_squawk_code` — domain-safety check (FAA-specific).

To add a new opt-in catcher:

1. Define the hook in `src/harness/orchestrator/hooks.py` as a frozen
   dataclass implementing the `BailHook` / `FinalizeHook` protocol.
2. Add its name to `_OPT_IN_CATCHERS` in the same file.
3. Append a conditional `bail.append(...)` block in
   `default_hook_pipeline()` keyed on `name in catchers_set`.
4. Document under which characters opt in via `core.yaml: catchers: [...]`.

The phase-2 follow-up bead `harness-yu1z` will move the catcher-specific
regex data (currently in `hooks.py`) into per-character config — until that
lands, only the registration is character-driven.

---

## Validating your tool

```bash
# 1. Spec is well-formed (the registry rejects malformed specs at build time)
uv run harness chat --model mlx --tools --tool-set core
# Look for "tool 'echo' registered" in the loading header.

# 2. Router sees it (when --router is on)
uv run harness eval router --tool-set core
# Add a fixture row that should pick `echo` to lock in router behavior.

# 3. Existing profile tests
uv run pytest tests/test_tool_profiles.py

# 4. Your new tool's tests
uv run pytest tests/test_your_tool.py

# 5. End-to-end smoke
uv run harness chat --model mlx --tools --tools-add echo
# > "echo this back: hello world"
```

---

## Reference

- Tool protocol + `ToolSpec` / `ToolResult` / `ToolHit`:
  `src/harness/tools/base.py`.
- Profile registry + description override mechanism:
  `src/harness/tools/profiles.py`.
- Runtime tool loop + hook pipeline: [`tool_loop_flow.md`](tool_loop_flow.md),
  `src/harness/orchestrator/tool_loop.py`, `src/harness/orchestrator/hooks.py`.
- Per-character description authoring: [`character-authoring.md`](character-authoring.md).
- Architectural rationale for the data-driven tool/catcher surface:
  [`architecture/character-as-data.md`](architecture/character-as-data.md).
