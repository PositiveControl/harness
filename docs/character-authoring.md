# Authoring a character

How to add a new persona to the harness. Characters are **configuration, not
code** — adding one means dropping a directory under `character/<name>/` with
the right files, then running `HARNESS_CHARACTER_NAME=<name>` in the shell.

This document is the practical authoring guide. For the architectural
rationale (why persona-coupled features got lifted into core.yaml), see
[`architecture/character-as-data.md`](architecture/character-as-data.md). For
the runtime invariants that bound what's possible, see [`../CLAUDE.md`](../CLAUDE.md).

---

## When you might add a new character

- **You want a different voice** for the same domain (`airton_b` is the same
  data plane as `airton`, in a different register — it speaks ops/caveman).
- **You want a different domain** (FAA tutoring, legal analysis, RFC review),
  with its own corpus, its own citation discipline, its own catchers.
- **You want a sandboxed play character** to develop new behavior without
  contaminating the main character's memory or voice.

Each of these is a directory under `character/`. None requires a code change.

---

## Directory layout

A complete persona — say `character/legal_l1/` for a legal-doc assistant —
looks like:

```
character/legal_l1/
├── core.yaml                  # identity (required)
├── constitution.md            # principles + refusal stance (required)
├── voice/
│   ├── canonical.yaml         # curated voice samples (required, can start empty)
│   ├── captured.yaml          # live `/edit` captures (optional, auto-grown)
│   └── ablation.yaml          # held-out samples for generalization eval (optional)
├── seed_memories/             # formative narratives + corpus chunks
│   └── *.md                   # frontmatter-tagged
├── tool_descriptions.yaml     # per-profile tool description overrides (optional)
├── corpus/                    # optional, only for corpus-grounded characters
│   ├── synonyms.yaml          # ingest-side + retrieval lay-term map
│   ├── query_synonyms.yaml    # retrieval-only additive expansions
│   └── verb_anchors.yaml      # phraseology-lint verb → § map
├── router_eval.yaml           # tool-selection accuracy fixtures (optional)
├── session_resume_eval.yaml   # build_resume_summary contract (only if ops tools loaded)
├── atc_eval.yaml              # corpus-grounded eval (only if corpus-grounded)
├── phraseology_eval.yaml      # phraseology-lint eval (only if phraseology profile)
└── data/                      # auto-created — per-character SQLite store
```

Only `core.yaml`, `constitution.md`, `voice/canonical.yaml`, and
`seed_memories/` are required. Everything else is opt-in.

---

## Step 1 — `core.yaml` (identity + every opt-in feature)

This is where every persona-coupled feature is now declared. Field by field:

```yaml
# Identity (required)
name: legal_l1
pronouns: it
era: "2026 — current US legal corpus"
relationship:
  mark: legal-research client

premise: >
  legal_l1 is a careful research assistant for U.S. federal legal documents…

self_awareness: |
  legal_l1 knows it is software. When asked, it says so directly.

values:
  - id: cite_or_silent
    rule: Never claim a holding without citing the case or statute.
  - id: scope_aware
    rule: Refuses to give legal advice; only cites and summarizes the source.

taboos:
  - Never gives a recommendation a licensed attorney should give.

directives:
  - Lead every substantive answer with the citation.

deep_domains:
  - 14 USC, 18 USC subset
  - Federal Rules of Civil Procedure

shallow_domains:
  - state-specific case law

on_being_wrong: |
  Corrects directly, names the right cite, shows the source text…

# ───────────────────────────────────────────────────────────────────
# Optional feature blocks (harness-uz1k epic — generalization sweep)
# ───────────────────────────────────────────────────────────────────

# Voice rewrite layer (harness-a2sa). Default "persona" (Airton's two-pass
# voice rewrite). "caveman" wraps in CavemanRewriter (per-surface intensity
# map; ab uses this). "none" leaves the adapter unwrapped.
voice:
  rewriter: persona

# bd-graph identity (harness-a2sa). Set when the character ships ops tools.
# Dev-only characters can omit this entire block.
bd:
  assignee: legal_l1            # writes carry this assignee
  exclude_assignee: airton_b    # default-hide ab's beads from this character's reads
  scope_allowlist: []           # narrow reads to these scope:* labels (empty = no filter)

# fetch_url host allowlist (harness-a2sa). Empty = unrestricted (the default).
# Corpus-grounded characters typically want a tight allowlist.
fetch_url:
  allowed_hosts:
    - supremecourt.gov
    - www.supremecourt.gov
    - law.cornell.edu
    - www.law.cornell.edu

# Citation grammar (harness-jaqe). Required only for cite-disciplined
# characters. Drives extract_citations / lead_with_citation /
# preserve_citations / MissingCitationHook / cite_grounding /
# phraseology_lint. Omitted = the citation pipeline silently no-ops.
citation_grammar:
  surface_patterns:                       # full surface-form regexes
    - '\d+\s+U\.?S\.?C\.?\s+§\s*\d+'      # "18 USC §1001"
    - 'F\.?R\.?C\.?P\.?\s+\d+(?:[-(][a-z0-9]+)?'  # "FRCP 12(b)(6)"
    - '\b\d+\s+F\.\s*(?:Supp\.\s*)?\d+\s+\d+\b'   # case reporter cite
  strip_prefix: null                       # optional version-prefix scrubber
  anchor_pattern: '§\s*(\d+(?:[-(][a-z0-9]+)?)'   # bare anchor extractor
  document_reference: '\b(?:U\.?S\.?C\.?|FRCP)\b' # "did the reply mention the corpus"
  document_name: 'U.S. Code'                      # used in nudge text
  example_anchor: '§1001'                          # used in nudge text

# Opt-in domain catchers (harness-qvwq). Names of orchestrator catchers
# that ONLY register when this character lists them. Catchers self-gate
# via regex but the opt-in makes the persona-coupling explicit. Empty
# tuple = none. Available: ab_fabrication, ambiguous_context,
# scope_redirect, reserved_squawk_code.
catchers: []

# Cite-discipline flags (existing — predates the generalization sweep)
require_search_memory: false   # force a search_memory call at turn start
lead_with_citation: false      # hoist first citation to opening of pass-1 draft

# Thought-graph workflow (existing — only binds when bd ops tools are loaded)
# thought_graph:
#   query_first: |
#     Query bd before regenerating state…
#   budgets: |
#     Max 3 ab-owned beads per turn…
#   thought_labels: |
#     Free-form `thought:` prefix labels…
```

Every field above is documented in `src/harness/character.py` — read the
dataclass docstrings to see the load-time contract. `load_character(path)`
will raise a path-anchored `ValueError` if any field is malformed.

---

## Step 2 — `constitution.md`

Free-form prose. Whatever this character WILL DO and WILL NOT DO that doesn't
fit the structured `core.yaml` fields. Loaded into the system prompt verbatim.

Examples that live here rather than `core.yaml`:

- "Cite a case before paraphrasing it."
- "Refuse to opine on probability of success."
- "When the corpus has no answer, say so plainly."

The constitution rides every system prompt.

---

## Step 3 — `voice/canonical.yaml`

Curated voice samples. Top-K retrieval surfaces 6 of these per turn as
few-shot examples in the system prompt.

```yaml
samples:
  - id: lead_with_citation_basic
    prompt: "What's the elements rule for §1001?"
    gold: |
      18 USC §1001 — three elements: (1) a statement that is materially
      false, (2) made knowingly and willfully, (3) within the
      jurisdiction of a federal department or agency.

  - id: refuse_legal_advice
    prompt: "Should I plead guilty?"
    gold: |
      I don't give legal advice. The plea decision belongs to you and
      your attorney. The relevant procedural anchor is FRCP 11 — read it
      with counsel.
```

Start with whatever you have. The corpus grows through `/edit` captures and
`harness voice capture` (see [`usage.md`](usage.md) — voice corpus section).

---

## Step 4 — `seed_memories/*.md`

Formative episodic content — for relationship-shaped characters, this is
narrative ("the time I shipped X"); for corpus-grounded characters
(`airton_c1` / `airton_c`), this is rulebook chunks indexed for hybrid
retrieval.

Frontmatter format:

```markdown
---
id: seed-1001-elements
title: 18 USC §1001 — false statements
principle: USC §1001 (Federal — FALSE STATEMENTS)
tags: [federal, criminal, falsity]
era: "2026 codification"
---

Whoever, in any matter within the jurisdiction of the executive,
legislative, or judicial branch of the Government of the United States,
knowingly and willfully…
```

`harness memory ingest` reads these files and writes one episodic row each at
`tier=seed`, `user_id IS NULL` (shared across every speaker). Idempotent on
the frontmatter `id` — re-ingest is safe.

For large corpora, use `scripts/atc_ingest.py` or write a similar ingest
script that emits the same frontmatter shape. The `principle` field is
load-bearing — the citation grammar's `anchor_pattern` extracts the section
anchor from this string.

---

## Step 5 — `tool_descriptions.yaml` (only if you reframe tools)

Profile-scoped overrides for tool descriptions (harness-xo7v). Useful when
your corpus IS the search target, so the generic `search_memory` description
("past events and lessons") would mis-route the small intent router toward
`search_facts` on rule questions.

```yaml
profiles:
  legal:                    # the tool-set name — see tools-and-tool-sets.md
    search_memory: >-
      Search the U.S. Code corpus for sections, holdings, and elements.
      Use for ANY legal-shaped question. NOT for user-relationship facts —
      use search_facts for those. Empty-signal prompts ('test', 'ping')
      are NOT legal-shaped — return null.
    search_facts: >-
      Search atomic facts about the user. Returns triples. NOT for the
      U.S. Code — use search_memory for that.
```

The mechanism: `apply_profile_descriptions(registry, profile, character=...)`
runs at registry-build time and merges your overrides on top of the
shipping `BUILTIN_PROFILE_DESCRIPTIONS` (currently empty — back-stop only).
Character entries win on key collision.

If your character uses two profiles (e.g. `atc` for chat + `phraseology` for
the lint pipeline), declare both under `profiles:` and they coexist.

See [`tools-and-tool-sets.md`](tools-and-tool-sets.md) for how to wire a new
profile.

---

## Step 6 — Corpus support files (only for corpus-grounded characters)

These live under `character/<name>/corpus/` and are loaded at chat-time:

### `synonyms.yaml`

Lay-term ↔ jargon expansion applied to BOTH ingest text and incoming
queries. Drives the `QueryExpander` lifted by `cli.py:_make_query_expander`.

```yaml
"§1001":
  - false statement
  - lying to feds
"FRCP 12(b)(6)":
  - motion to dismiss
  - failure to state a claim
```

### `query_synonyms.yaml`

Retrieval-side additions only — never poison ingest precision with
lay-paraphrase entries.

```yaml
"§1001":
  - misleading statement to government
  - federal false statement
```

### `verb_anchors.yaml`

Section → distinctive-verb-or-phrase map (harness-ptya). The phraseology
lint pipeline uses this to:

1. Detect which section the user's utterance belongs to (verb-anchor match).
2. Promote that section's hits to the front of the candidate slate (rerank).
3. If the section isn't in the slate at all, text-mode-probe and inject one
   row (virtual-hit injection).

```yaml
"§1001":
  - LIE TO
  - KNOWINGLY FALSE
"FRCP 11":
  - SIGNED PLEADING
```

Verbs are uppercased on load; matching is case-insensitive word-boundary.

---

## Step 7 — Eval fixtures (optional but recommended)

### `router_eval.yaml`

YAML rows of `(user_message, expected_tool, expected_args)` that
`harness eval router` replays through the small-model router to measure
tool-selection accuracy. Locks in router quality before swapping models or
tweaking prompts.

### `session_resume_eval.yaml`

Only relevant for characters with ops tools. Pins `build_resume_summary`'s
contract — focus / in-progress / memories / drift.

### `atc_eval.yaml`, `phraseology_eval.yaml`

Corpus-grounded evals. Rows of `(question, expected_citations,
expected_keywords)` for `eval atc` and rows of `(utterance, scenario,
expected_verdict, expected_section, expected_phraseology)` for
`eval phraseology`. Both share a YAML envelope via `evals/_corpus.py`
(harness-p5lu).

---

## Step 8 — Wire it up

```bash
# Switch to the new character
export HARNESS_CHARACTER_NAME=legal_l1

# Verify the loader sees every block
uv run harness describe

# Seed episodic from seed_memories/
uv run harness memory ingest

# First chat
uv run harness chat --model mlx --persona --memories 3 --facts 5
```

If `core.yaml` has a malformed block, `harness describe` is the fastest place
to see the path-anchored error.

---

## Common authoring patterns

### "I want a generalist with no corpus" (like `airton`)

- `core.yaml`: identity + values + taboos. Skip `citation_grammar`,
  `catchers`, `tool_descriptions`, `fetch_url.allowed_hosts`.
- `voice/canonical.yaml`: curated samples.
- `seed_memories/`: a handful of formative narratives.

That's it. No corpus support files, no eval fixtures. Done.

### "I want an ops persona" (like `airton_b`)

- `core.yaml`: include `voice.rewriter: caveman` if you want compressed
  output, the full `bd:` block, the matching `thought_graph` block.
- Add `register_map.yaml` next to `core.yaml` if using the caveman rewriter
  (per-surface intensity map).
- Set `catchers: [ab_fabrication]` so ab_ops capture/plan imitation gets
  caught.
- `bd init` under your bd dir before first run.

### "I want a corpus-grounded tutor" (like `airton_c1`)

- `core.yaml`: the full kit — `citation_grammar`, `catchers:
  [ambiguous_context, scope_redirect, reserved_squawk_code]` (or your own),
  `tool_descriptions.yaml` reframing search_memory, `fetch_url.allowed_hosts`,
  `lead_with_citation: true`, `require_search_memory: true`.
- `seed_memories/`: the corpus chunks, one row per section.
- `corpus/synonyms.yaml`, `corpus/query_synonyms.yaml`,
  `corpus/verb_anchors.yaml` for retrieval quality.
- Eval fixtures: `atc_eval.yaml` (or your domain equivalent),
  `phraseology_eval.yaml` if you want a lint pipeline.

The atc archetype lives at `src/harness/character_templates/atc/` — copy it
as a starting point. `scripts/character_from_template.py` (planned;
`harness-ouo`) will automate this.

### "I want to fork a character for experiments"

```bash
cp -r character/airton character/airton_play
# edit core.yaml to set name: airton_play
export HARNESS_CHARACTER_NAME=airton_play
```

Third-plus characters auto-silo their memory + bd dirs under
`character/<name>/data/` and `character/<name>/bd/` (see `config.py:
db_path_for` / `bd_dir_for`). No data collision with airton.

---

## What you don't need to do

- **You don't need to edit `src/`** for a new character. Every persona-
  coupled feature is data-driven (harness-uz1k epic).
- **You don't need to register the character in a list anywhere** —
  `HARNESS_CHARACTER_NAME` is the only switch.
- **You don't need to run a migration** — the loader handles missing
  optional blocks gracefully (None / empty defaults).
- **You don't need to update the test suite** unless you're adding new
  catcher data or a new corpus-grounded eval. Existing tests cover the
  schema's structural contracts.

---

## Validating your character

```bash
# 1. Schema parses?
uv run harness describe

# 2. Memory ingests?
uv run harness memory ingest
uv run harness memory list --tier seed | head

# 3. Voice retrieval works?
uv run harness eval voice --model mlx --persona --top-k 6

# 4. Tool selection sane (if you wrote a router_eval.yaml)?
uv run harness eval router

# 5. Chat actually runs?
uv run harness chat --model mlx --persona

# 6. Test character-data invariants land
uv run pytest tests/test_character.py -v
```

If any step fails, the path-anchored error message points at the offending
file/line/key.

---

## Reference

- Architecture rationale + the generalization sweep that made all this
  data-driven: [`architecture/character-as-data.md`](architecture/character-as-data.md).
- Tool registration + tool-set wiring: [`tools-and-tool-sets.md`](tools-and-tool-sets.md).
- Daily-use commands once your character is alive: [`usage.md`](usage.md).
- Runtime invariants that bound character behavior: [`../CLAUDE.md`](../CLAUDE.md).
