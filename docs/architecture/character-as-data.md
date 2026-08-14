# Character as data

The harness treats each persona as a directory of YAML/Markdown under
`character/<name>/`. This document is the architectural reference for what
that means in practice — every field on the `Character` dataclass, every
opt-in feature block, and the rationale behind the 2026-04 generalization
sweep that made all of this declarative.

For the practical "how do I add a character" guide, see
[`../character-authoring.md`](../character-authoring.md). For the runtime
tool / orchestrator semantics, see [`../tool_loop_flow.md`](../tool_loop_flow.md)
and [`../../CLAUDE.md`](../../CLAUDE.md).

---

## The invariant

> Character data is configuration, not code.
> Never hardcode persona rules into `src/`. The runtime reads `character/<name>/`.

This invariant predates the 2026-04 sweep — `core.yaml`, `voice/canonical.yaml`,
`seed_memories/` were always loaded as data. But several persona-coupled
features had crept into `src/` as `if character.name == "airton_b"` branches,
hardcoded constant sets (`_ATC_FAMILY_NAMES`, `_ATC_FETCH_URL_ALLOWED_HOSTS`),
or domain-specific regex literals embedded in `orchestrator/hooks.py`.

The umbrella epic `harness-uz1k` lifted those into per-character `core.yaml`
fields. Result: zero `if character.name ==` branches in `cli.py`, zero
`_ATC_FAMILY_NAMES`-style constants, and a clean opt-in registration model
for domain catchers.

---

## The `Character` dataclass surface

`src/harness/character.py:Character` is a frozen dataclass. Required fields
encode identity (name, premise, values, taboos, voice samples, seed memories);
optional fields encode opt-in features. Field-by-field:

### Identity (required, predates the sweep)

| Field                  | Source                              | Drives                              |
|------------------------|-------------------------------------|-------------------------------------|
| `name`                 | `core.yaml: name`                   | persona dispatch + paths            |
| `pronouns`, `era`      | `core.yaml`                         | system prompt rendering             |
| `relationship`         | `core.yaml: relationship:`          | first-among-equals framing in prompt |
| `premise`, `self_awareness`, `on_being_wrong` | `core.yaml`         | system prompt blocks                |
| `values`, `taboos`, `directives` | `core.yaml`               | system prompt blocks                |
| `deep_domains`, `shallow_domains` | `core.yaml`              | scope hints in system prompt        |
| `constitution`         | `constitution.md`                   | system prompt suffix                |
| `voice_samples`        | `voice/canonical.yaml` + `voice/captured.yaml` | top-K few-shot retrieval |
| `ablated_voice_samples`| `voice/ablation.yaml`               | held-out generalization eval        |
| `seed_memories`        | `seed_memories/*.md`                | episodic seed at `tier=seed`         |
| `thought_graph`        | `core.yaml: thought_graph:`         | bd-as-working-memory rules in prompt |

### Cite-discipline flags (existing, predates the sweep)

| Field                    | Default | Purpose                                                    |
|--------------------------|---------|------------------------------------------------------------|
| `require_search_memory`  | `False` | Force a `search_memory` call before round 0 (airton_c1).   |
| `lead_with_citation`     | `False` | Hoist first citation to opening of pass-1 draft.           |

### 2026-04 additions (harness-uz1k epic)

These are the fields the generalization sweep added. Each replaces a
hardcoded branch or constant in `src/`.

| Field                       | YAML block             | Default | Bead         | Purpose                                                       |
|-----------------------------|------------------------|---------|--------------|---------------------------------------------------------------|
| `tool_descriptions`         | `tool_descriptions.yaml: profiles:` | `{}` | harness-xo7v | Per-profile tool description overrides. Reframes generic tools (e.g. `search_memory`) for corpus-grounded characters. |
| `voice_rewriter`            | `core.yaml: voice.rewriter` | `"persona"` | harness-a2sa | Adapter wrap layer: `persona` (PersonaAdapter), `caveman` (CavemanRewriter), or `none`. |
| `bd_assignee`               | `core.yaml: bd.assignee` | `None` | harness-a2sa | bd-graph identity for budget-cap enforcement + scope-allowlist exemption. |
| `bd_exclude_assignee`       | `core.yaml: bd.exclude_assignee` | `None` | harness-a2sa | Default-hidden assignee (filtered from list/plan/drift). |
| `bd_scope_allowlist`        | `core.yaml: bd.scope_allowlist` | `()` | harness-a2sa | Read-side scope filter (e.g. `[professional, personal]`). |
| `fetch_url_allowed_hosts`   | `core.yaml: fetch_url.allowed_hosts` | `()` | harness-a2sa | FetchUrlTool host allowlist; empty = unrestricted. |
| `citation_grammar`          | `core.yaml: citation_grammar:` | `None` | harness-jaqe | Compiled regex set + nudge-text tokens for the corpus's citation shapes. |
| `catchers`                  | `core.yaml: catchers:` | `()` | harness-qvwq | Opt-in domain-catcher names (`ab_fabrication`, `ambiguous_context`, `scope_redirect`, `reserved_squawk_code`). |

---

## Design principle: opt-in beats self-gating

Pre-sweep, four orchestrator hooks were unconditionally registered in the
default pipeline:

- `AbFabricationHook` — ab_ops capture/plan/remember imitation.
- `AmbiguousContextHook` — FAA balloon manned/unmanned dictionary.
- `ScopeRedirectHook` — FAA aviation vocab + non-aviation vocab.
- `ReservedSquawkCodeHook` — ATC reserved transponder codes.

Each self-gated by regex: a non-FAA reply just never matched
`_AVIATION_VOCAB_RE`, a non-ab reply never produced `Captured.` shaped
prose. So in practice the hooks were inert for the wrong character — but
the *registration* was global and the persona-coupling was implicit.

The harness-qvwq refactor moved registration to a per-character `catchers:`
list. `default_hook_pipeline(catchers=character.catchers)` only appends the
hooks the character lists. Two wins:

1. **Provable absence.** "ab_fabrication" doesn't run for airton_c1 because
   it isn't in the roster — not because the regex happens not to match.
2. **No accidental cross-talk.** A future ab regex tweak can't accidentally
   start firing on airton replies, because the hook isn't even constructed
   in airton's pipeline.

Same principle for `citation_grammar`: previously the FAA citation regex
set lived as module constants and the citation-aware functions called
`extract_citations(text)` unconditionally. Non-FAA characters got `[]` back
because the regex didn't match. Post-sweep, `extract_citations(text,
grammar=None)` returns `[]` immediately — the absence of grammar IS the
gate, not "did the regex happen not to find anything."

---

## Design principle: data-driven branches at construction time

The pre-sweep `cli.py` had four hardcoded character-name branches:

```python
# 1. fetch_url allowlist
allowed_hosts = (
    _ATC_FETCH_URL_ALLOWED_HOSTS
    if character.name in _ATC_FAMILY_NAMES else None
)
# 2. rewriter pick
if character.name == "airton_b":
    adapter = CavemanRewriter(...)
else:
    adapter = PersonaAdapter(...)
# 3. bd scope allowlist
scope_allowlist = ("professional", "personal") if character.name == "airton_b" else None
# 4. bd ab_assignee
ab_assignee = "airton_b"
```

Each of these encoded a decision the character is in the best position to
declare for itself. Post-sweep:

```python
allowed_hosts = (
    frozenset(character.fetch_url_allowed_hosts) or None
)
if character.voice_rewriter == "caveman":
    adapter = CavemanRewriter(...)
elif character.voice_rewriter == "persona":
    adapter = PersonaAdapter(...)
# else "none": unwrapped
scope_allowlist = character.bd_scope_allowlist or None
ab_assignee = character.bd_assignee
```

Same wire shape, different source-of-truth. Adding a fifth character no
longer requires a `cli.py` edit.

---

## What's intentionally NOT lifted

Two surfaces stay as code constants by design:

### `config.py:bd_dir_for` / `db_path_for` — data-path back-compat

```python
if character_name == "airton_b" and self.ab_bd_dir is not None:
    return self.ab_bd_dir
if character_name in ("airton", "airton_b"):
    return self.root  # shared project bd dir (Path 2, harness-55y)
```

These resolve **on-disk data paths** laid down before the auto-silo path
landed. Lifting them would move the user's existing memory DB on disk —
destructive without a migration script. Documented in code; tracked as a
follow-up bead.

### `tools/citations.py:CITATION_RE` + `hooks.py:UNGROUNDED_SECTION_CITATION_RE`

These match the universal `§N-N-N` shape (chapter-section-paragraph
numerals — also valid for RFC, ISO, etc.) rather than FAA-specific surface
forms. They're not strictly persona-coupled. `tools/citations.py:
extract_citations` is also consumed by `audit.py` which lacks a `Character`
handle today; threading it through requires more plumbing than the current
work targets. Tracked as a future bead.

### Catcher regex DATA still lives in `hooks.py`

Phase 1 of harness-qvwq made registration opt-in. Phase 2 (tracked as
`harness-yu1z`) moves the regex data — `FABRICATED_AB_*`, `_AVIATION_VOCAB_RE`,
`_AMBIGUOUS_TERMS`, `_RESERVED_SQUAWK_RE` — into per-character config.
Multi-line regex literals don't survive YAML round-trips cleanly, so the
likely shape is per-character Python config modules
(`character/<name>/catchers.py`) loaded at character-load time. Until that
lands, the catchers self-gate via their built-in regex; the opt-in roster
controls whether they run at all.

---

## How CLI plumbing reads each field

| Site                                 | Field                          | Pre-sweep behavior                         |
|--------------------------------------|--------------------------------|--------------------------------------------|
| `cli.py:_resolve_adapter`             | `voice_rewriter`               | `if name == "airton_b": Caveman else Persona` |
| `cli.py:_maybe_bd_adapter`            | `bd_assignee`, `bd_exclude_assignee`, `bd_scope_allowlist` | three literal `"airton_b"` strings |
| `cli.py:_build_tool_registry_for_tui` (fetch_url builder) | `fetch_url_allowed_hosts` | `_ATC_FAMILY_NAMES` membership |
| `cli.py` phraseology_lint registration | `tool_descriptions["phraseology"]` | `_ATC_FAMILY_NAMES` membership |
| `cli.py:cmd_eval_phraseology` guard   | `tool_descriptions["phraseology"]` | `_ATC_FAMILY_NAMES` membership |
| `cli.py:cmd_phraseology_lint` guard   | `tool_descriptions["phraseology"]` | same |
| `cli.py:cmd_eval_atc_audio` guard     | `tool_descriptions["phraseology"]` | same |
| `default_hook_pipeline`               | `catchers`, `citation_grammar` | unconditional registration                 |
| `apply_profile_descriptions`          | `tool_descriptions[profile]`   | static `TOOL_PROFILE_DESCRIPTIONS[profile]` |

Every one of these is now data-driven. A new persona that wants different
behavior writes the YAML; nothing in `cli.py` changes.

---

## Loader contract

`src/harness/character.py:load_character(path)` is the single entry point.
It:

1. Reads `core.yaml` and parses every required field.
2. Loads `voice/canonical.yaml` + `voice/captured.yaml` + `voice/ablation.yaml`
   (latter two optional).
3. Loads `seed_memories/*.md` via `python-frontmatter`.
4. Loads each optional 2026-04 field via a small helper (`_load_voice_rewriter`,
   `_opt_str`, `_load_str_tuple`, `load_citation_grammar`).
5. Loads `tool_descriptions.yaml` if present (top-level `profiles:` key).
6. Returns a frozen `Character` instance.

Every malformed block raises a path-anchored `ValueError`. Examples:

- `airton_c1/core.yaml: voice.rewriter must be one of ['caveman', 'none', 'persona']; got 'nonsense'`
- `airton_c1/core.yaml: citation_grammar.surface_patterns[2] failed to compile: unbalanced parenthesis`
- `airton/tool_descriptions.yaml: profiles.atc must be a mapping, got list`

Loader tests in `tests/test_character.py` pin both the happy paths and the
malformed-input contracts. Adding a new optional field follows the same
template.

---

## Why this matters for the roadmap

The harness's identity invariant is:

> One identity, one memory, one orchestrator — many gateways.

Multiple gateways (web / Slack / Matrix) and multiple characters per gateway
were always part of the design. Pre-sweep, every new character cost a
`cli.py` PR. Post-sweep, a new character is a directory — which makes the
**multi-character** axis cheap independently of the multi-gateway axis. The
two grow on their own schedules.

The atc niche product lane (`harness-mw89` epic in the roadmap) depends on
this. Each ATC niche (phraseology lint, LOA drift, MOR drafting, deal
packet) is a corpus-grounded tutor variant — different scope, different
citation grammar, different catchers, possibly different voice register.
Without the data-driven character surface, each niche would require code
changes to the orchestrator. Now they're new directories.

---

## Reference

- Authoring guide: [`../character-authoring.md`](../character-authoring.md).
- Tool registration: [`../tools-and-tool-sets.md`](../tools-and-tool-sets.md).
- Runtime invariants + commands: [`../../CLAUDE.md`](../../CLAUDE.md).
- Per-bead detail in commit log: `git log --grep harness-uz1k`.
