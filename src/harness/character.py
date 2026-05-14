from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import frontmatter
import yaml

from harness.citation import CitationGrammar, load_citation_grammar


@dataclass(frozen=True)
class Value:
    id: str
    rule: str


@dataclass(frozen=True)
class VoiceSample:
    id: str
    prompt: str
    gold: str


@dataclass(frozen=True)
class SeedMemory:
    id: str
    title: str
    principle: str
    tags: tuple[str, ...]
    era: str
    body: str


@dataclass(frozen=True)
class TabularTableSpec:
    """One tabular table declared by a character in core.yaml under
    `tabular_tables:` (harness-kgpi). The CLI reads this list at
    session start and registers each table into a per-character
    `TabularStore` so the agent's `assemble_context` calls can fan
    out into SQL.

    `csv_path` is resolved against the character directory at load
    time. `columns` is a tuple of (name, type_hint, nl_description) —
    same shape `TabularStore.TableSchema.columns` consumes."""

    table_name: str
    description: str
    csv_path: Path
    columns: tuple[tuple[str, str, str], ...]


_DOCUMENT_TREE_FORMATS = frozenset({"markdown", "jsonl"})


@dataclass(frozen=True)
class DocumentTreeSpec:
    """One hierarchical document declared by a character in core.yaml
    under `document_trees:` (harness-ejxn + harness-k38k). The CLI reads
    this list at session start and builds a per-character
    `DocumentTreeStore` with one document ingested per spec.

    `name` is the document name that flows into
    `DocumentTreeStore.upsert_document` and shows up in tree-slot
    contract resolutions. `source_path` is resolved against the
    character directory at load time. `source_format` picks the
    `document_tree_ingest` adapter:
      - `markdown` — heading-derived hierarchy, positional path slugs.
        The canonical path for new prose-shaped corpora.
      - `jsonl` — pre-chunked rows with explicit hierarchy fields.
        Per-corpus knobs (which JSONL fields define depth, what to
        prepend to intermediate headings, which field holds the leaf
        title) live in the `jsonl_*` attributes below. Markdown specs
        ignore them.

    YAML shape under `document_trees:`:

        - name: jo_7110_65
          description: FAA Order JO 7110.65 — Air Traffic Control
          source: corpus/chunks/jo_7110_65.jsonl
          format: jsonl
          jsonl:
            depth_fields: [chapter, parent_section, section]
            heading_prefixes: ["Chapter ", "§", ""]
            leaf_heading_field: title
            # body_field / chunk_index_field default to "body" / "chunk_index"
    """

    name: str
    description: str
    source_path: Path
    source_format: str
    # JSONL-only config. Empty on markdown specs and ignored by the
    # markdown adapter. Required (non-empty depth_fields) when
    # source_format == "jsonl"; the parser raises with a clear pointer
    # when omitted, so the misconfig surfaces at YAML load rather than
    # at ingest time.
    jsonl_depth_fields: tuple[str, ...] = ()
    jsonl_heading_prefixes: tuple[str, ...] = ()
    jsonl_leaf_heading_field: str | None = None
    jsonl_body_field: str = "body"
    jsonl_chunk_index_field: str | None = "chunk_index"


@dataclass(frozen=True)
class ThoughtGraph:
    """Ab's thought-graph workflow rules (harness-9qw). These only bind
    when the matching bd ops tools are actually loaded — the runtime
    decides; the prompt just states the convention."""

    query_first: str
    budgets: str
    thought_labels: str


@dataclass(frozen=True)
class Character:
    name: str
    pronouns: str
    era: str
    relationship: dict[str, str]
    premise: str
    self_awareness: str
    values: tuple[Value, ...]
    taboos: tuple[str, ...]
    directives: tuple[str, ...]
    deep_domains: tuple[str, ...]
    shallow_domains: tuple[str, ...]
    on_being_wrong: str
    constitution: str
    thought_graph: ThoughtGraph | None
    voice_samples: tuple[VoiceSample, ...]
    # Canonical = curated samples in voice/canonical.yaml; captured =
    # live edits harvested from chat into voice/captured.yaml. `voice_samples`
    # is the merged tuple (canonical first, captured appended) that retrieval
    # uses; these counts let the introspect tool report the split without
    # re-reading disk. Invariant: canonical + captured = len(voice_samples).
    canonical_voice_count: int
    captured_voice_count: int
    # Ablated samples in voice/ablation.yaml — carved out of the canonical
    # set for generalization eval (airton_c1 harness-w49p; renamed from
    # "holdout" 2026-04-26 to avoid collision with ATC phraseology like
    # "hold short" / "holding pattern"). They are NOT added to
    # `voice_samples`, so the VoiceRetriever can't surface them; exposing
    # them separately lets an eval read back the prompts and measure
    # whether the model still answers correctly without the canonical
    # crutch. Empty for characters without an ablation file.
    ablated_voice_samples: tuple[VoiceSample, ...]
    seed_memories: tuple[SeedMemory, ...]
    # airton_c1 opt-in (harness-3uh): when true, the orchestrator injects
    # a forced `search_memory` tool call at the start of every user turn,
    # so the model always sees a grounding-tool result before its first
    # complete_with_tools call. Rationale: airton_c1's retrieval scores
    # lay-language queries below the 0.5 passive-retrieval floor, so no
    # memory block attaches and UngroundedCitationHook's grounding signal
    # never lights up. The forced call fires regardless of cosine score
    # and is mandatory character policy; passive retrieval (`--memories`
    # / `--facts`) still runs in addition, not in place. Default False;
    # every other character leaves behavior unchanged.
    require_search_memory: bool = False
    # Contract-flavored sibling of `require_search_memory` (harness-jkmk).
    # When True, the orchestrator injects a forced `assemble_context`
    # call at turn start, with role = `default_contract_role` and
    # variables = {request_summary: <user message>}. The package result
    # lands in `working` before the model's first round, so the model
    # sees a full contract resolution (episodic + tabular + tree hits,
    # whichever the contract names) without needing to ask. Used by
    # characters that move onto the contract primitive as their primary
    # retrieval path. Requires `default_contract_role` to be set;
    # otherwise the parser raises at YAML load time.
    require_assemble_context: bool = False
    # Role name passed to a forced assemble_context call (harness-jkmk).
    # Must be a contract role declared under
    # `character/<name>/contracts/<role>.yaml`. Validation deferred to
    # runtime (the AssembleContextTool surfaces an "unknown role" error
    # cleanly if the contract YAML is missing); the parser only enforces
    # presence when `require_assemble_context: true`. Convention: the
    # contract for that role takes a `{request_summary}` variable; the
    # orchestrator fills it from the latest user message.
    default_contract_role: str | None = None
    # Plan #7 / forward citation-discipline pass. When true,
    # PersonaAdapter.complete() runs `rewriter.lead_with_citation()` on
    # the pass-1 draft before the voice rewriter sees it — hoists the
    # first citation in the draft to the opening (`§X-Y-Z — body`)
    # when the draft has a citation but doesn't lead with it. Mirror
    # of `preserve_citations` (harness-cco, post-rewrite). Drafts
    # without any citation pass through unchanged. Default False;
    # set true on characters whose directive is "name the section
    # before answering" (currently airton_c1).
    lead_with_citation: bool = False
    # Profile-scoped tool-description overrides loaded from
    # `character/<name>/tool_descriptions.yaml` (harness-xo7v). Outer
    # key is a tool-set name (`atc`, `phraseology`, ...); inner map is
    # tool-name → reframed description. Layered onto the registry at
    # build time via `apply_profile_descriptions(registry, profile,
    # character=...)`; missing file → empty mapping (no override).
    # Lets a corpus-grounded persona reframe its own search tool
    # without editing src/harness/tools/profiles.py.
    tool_descriptions: dict[str, dict[str, str]] = field(default_factory=dict)
    # Tabular tables this character ships with (harness-kgpi). Empty
    # tuple for characters that don't ship table data. The CLI uses
    # this list to decide whether to spin up a per-character
    # TabularStore at session start.
    tabular_tables: tuple[TabularTableSpec, ...] = ()
    # Document trees this character ships with (harness-ejxn). Empty
    # tuple for characters that don't ship hierarchical document data.
    # The CLI uses this list to decide whether to spin up a per-
    # character DocumentTreeStore at session start, mirroring the
    # `tabular_tables` bootstrap. Each spec points at a markdown file
    # (the canonical new-character path) or — reserved for follow-up
    # — a JSONL chunk file.
    document_trees: tuple[DocumentTreeSpec, ...] = ()
    # ---- harness-a2sa: data-driven replacements for name-based branches ----
    # Voice rewrite layer. "persona" wraps the model adapter in
    # PersonaAdapter (Airton's two-pass voice rewrite); "caveman" wraps
    # in CavemanRewriter (ab's per-surface intensity map); "none" leaves
    # the adapter unwrapped (echo / dev). Default "persona" matches
    # the long-standing default for every character but airton_b.
    voice_rewriter: str = "persona"
    # bd-graph identity. When set, the BeadsAdapter constructed for
    # this character treats `assignee=<value>` as its own writes — the
    # in-flight + per-turn budget caps fire on creates with that
    # assignee, and `default_scope_allowlist` exempts this assignee
    # from scope filtering so internal thought-graph rows aren't
    # collateral-filtered. None = no budget enforcement (the default
    # for personas without an ops plane).
    bd_assignee: str | None = None
    # Default-exclude assignee for bd reads. When the active CLI run
    # has --include-internal off (the default), beads with this
    # assignee are filtered from list/plan/drift/search results so
    # the user isn't buried under another character's scratchpad.
    # Today airton + airton_b both set this to "airton_b" because they
    # share the project bd dir; airton_c characters that own a
    # private bd dir leave it None.
    bd_exclude_assignee: str | None = None
    # bd read-side scope allowlist. When non-empty, default reads
    # narrow to beads carrying any `scope:<value>` label in this
    # tuple (plus any beads owned by `bd_assignee` — those are
    # exempt from scope filtering so internal rows always flow).
    # ab uses ("professional", "personal") so dev/maintenance beads
    # are filtered out of its plan/list views. Empty = no filter.
    bd_scope_allowlist: tuple[str, ...] = ()
    # fetch_url host allowlist for the FetchUrlTool. When non-empty,
    # the tool refuses URLs whose netloc isn't in this set. Empty =
    # unrestricted (the default; matches FetchUrlTool's pre-existing
    # behaviour). atc-family characters ship a curated aviation-source
    # list so the tutoring surface stays bounded; non-corpus
    # characters leave it empty.
    fetch_url_allowed_hosts: tuple[str, ...] = ()
    # Citation grammar (harness-jaqe). Compiled regex set + nudge-text
    # tokens for the corpus this character speaks against. None when
    # the character has no citation discipline (Airton, ab, echo);
    # citation-aware functions in persona/rewriter, persona/
    # cite_grounding, and orchestrator/hooks short-circuit when
    # absent. Loaded from `core.yaml: citation_grammar:` via
    # persona/citation.py:load_citation_grammar.
    citation_grammar: CitationGrammar | None = None
    # Opt-in domain catchers (harness-qvwq). Names of orchestrator
    # hooks that ship in `default_hook_pipeline` ONLY when this
    # character lists them — currently `ab_fabrication`,
    # `ambiguous_context`, `scope_redirect`, `reserved_squawk_code`.
    # Each catcher self-gates via regex anyway, but moving the
    # registration to character data makes the persona-coupling
    # explicit instead of relying on "non-FAA replies happen never to
    # match this regex". Empty tuple ⇒ none installed (the default
    # for non-corpus characters: airton, echo). The catchers' regex
    # data still lives in hooks.py for now; a follow-up bead will
    # lift the data into per-character config.
    catchers: tuple[str, ...] = ()
    # Scope-gate (harness-8dop). When BOTH fields are set, the Hermes
    # router classifies each turn `in` / `out` / `unsure`; on `out` the
    # orchestrator short-circuits to `scope_redirect_template` instead
    # of calling the main model. `scope_hint` is the authored scope-
    # rule text injected into the router prompt — it tells the router
    # what's in/out for THIS persona (the router prompt is otherwise
    # persona-agnostic). `scope_redirect_template` is the canned reply
    # emitted on `out`. Both default None: characters without a
    # bounded corpus (airton, airton_b, airton_c) inherit the
    # always-`unsure` router prompt and never get short-circuited.
    scope_hint: str | None = None
    scope_redirect_template: str | None = None

    def system_prompt(
        self,
        *,
        exclude_example_ids: frozenset[str] | None = None,
        include_samples: Sequence[VoiceSample] | None = None,
        now: date | None = None,
    ) -> str:
        """Fallback system prompt for single-model ReAct and voice evals.
        Richer pipelines (multi-agent roles + critic) compose their own.

        Example selection (mutually exclusive):

        - `include_samples` (preferred when using retrieval): use exactly
          these samples in the order given. The caller is typically a
          retriever that picked top-K by similarity.
        - `exclude_example_ids`: use all voice samples except those with
          matching ids. The voice eval uses this for leave-one-out.

        If neither is given, all voice samples are shown."""
        excluded = exclude_example_ids or frozenset()
        values = "\n".join(f"  - {v.rule}" for v in self.values)
        taboos = "\n".join(f"  - {t}" for t in self.taboos)
        directives = "\n".join(f"  - {d}" for d in self.directives)
        deep = ", ".join(self.deep_domains)
        shallow = ", ".join(self.shallow_domains)

        style_rules = textwrap.dedent(
            """\
            How you speak:
              - Prose by default, not bullets. Use a short list only for
                two or three concrete alternatives.
              - 1-4 sentences is the usual length. A single paragraph
                is often enough.
              - When you see multiple correct paths, offer them as
                (a)/(b) or two dashes - never numbered steps.
              - When you don't know, say "Don't know" plainly, then
                list the paths you'd try.
              - No generic filler: do not open with "That's a solid
                approach", "Here are a few tips", "Would you like to
                discuss", or "I cannot comply". State the point.
              - If you refuse, give the concrete reason and the right
                alternative in the same breath.
              - First person, "I". Never hide that you are software;
                when asked, say so directly."""
        )

        if include_samples is not None:
            examples: list[VoiceSample] = list(include_samples)
        else:
            examples = [s for s in self.voice_samples if s.id not in excluded]
        examples_block = (
            "\n\n".join(f"User: {s.prompt}\nYou: {s.gold.strip()}" for s in examples)
            if examples
            else "(none shown for this turn)"
        )

        thought_block = ""
        if self.thought_graph is not None:
            tg = self.thought_graph
            thought_block = (
                "Thought-graph workflow (when bd ops tools are loaded):\n"
                f"  - Query-first: {tg.query_first.strip()}\n"
                f"  - Budgets: {tg.budgets.strip()}\n"
                f"  - Thought-labels: {tg.thought_labels.strip()}\n\n"
            )

        date_block = ""
        if now is not None:
            date_block = f"Today: {now.isoformat()} ({now.strftime('%A')}).\n\n"

        return (
            f"{date_block}"
            f"You are {self.name}. Pronoun: {self.pronouns}. "
            f"Era of origin: {self.era}.\n\n"
            f"Premise:\n{self.premise}\n\n"
            f"Self-awareness:\n{self.self_awareness}\n\n"
            f"Values (always defended):\n{values}\n\n"
            f"Taboos (always refused):\n{taboos}\n\n"
            f"Directives:\n{directives}\n\n"
            f"Deep domains: {deep}\n"
            f"Shallow domains: {shallow}\n\n"
            f"On being wrong:\n{self.on_being_wrong}\n\n"
            f"Constitution:\n{self.constitution}\n\n"
            f"{thought_block}"
            f"{style_rules}\n\n"
            "Voice examples - this is how you talk. "
            "Match this register, not a generic assistant's:\n\n"
            f"{examples_block}"
        )


def load_character(path: Path) -> Character:
    core = yaml.safe_load((path / "core.yaml").read_text())
    constitution = (path / "constitution.md").read_text().strip()

    voice_doc = yaml.safe_load((path / "voice" / "canonical.yaml").read_text())
    canonical_samples = [
        VoiceSample(id=s["id"], prompt=s["prompt"], gold=s["gold"].strip())
        for s in voice_doc["samples"]
    ]

    # Corpus growth: captured.yaml accrues samples harvested from live
    # chat edits. Same schema as canonical; loaded alongside so retrieval
    # treats them identically. Curated and captured stay separate on disk
    # so the canonical set remains reviewable/clean.
    captured_path = path / "voice" / "captured.yaml"
    captured_samples: list[VoiceSample] = []
    if captured_path.exists():
        captured_doc = yaml.safe_load(captured_path.read_text()) or {}
        for s in captured_doc.get("samples", []) or []:
            captured_samples.append(
                VoiceSample(id=s["id"], prompt=s["prompt"], gold=s["gold"].strip())
            )

    voice_samples = tuple(canonical_samples + captured_samples)

    # Ablation samples carved out of canonical for generalization eval.
    # Same schema; loaded separately and NOT merged into voice_samples,
    # so the VoiceRetriever can't see them. Absent file = no ablation.
    ablation_path = path / "voice" / "ablation.yaml"
    ablated_samples: list[VoiceSample] = []
    if ablation_path.exists():
        ablation_doc = yaml.safe_load(ablation_path.read_text()) or {}
        for s in ablation_doc.get("samples", []) or []:
            ablated_samples.append(
                VoiceSample(id=s["id"], prompt=s["prompt"], gold=s["gold"].strip())
            )

    seed_dir = path / "seed_memories"
    seeds: list[SeedMemory] = []
    for md_file in sorted(seed_dir.glob("*.md")):
        post = frontmatter.load(md_file)
        seeds.append(
            SeedMemory(
                id=str(post.get("id", md_file.stem)),
                title=str(post["title"]),
                principle=str(post["principle"]),
                tags=tuple(post.get("tags", [])),
                era=str(post.get("era", "")),
                body=post.content.strip(),
            )
        )

    # Optional per-character tool-description overrides
    # (harness-xo7v). Profile-scoped: top-level `profiles:` map, each
    # value is a tool-name → reframed-description map. Absent file →
    # empty dict (no override).
    tool_desc_path = path / "tool_descriptions.yaml"
    tool_descriptions: dict[str, dict[str, str]] = {}
    if tool_desc_path.exists():
        td_doc = yaml.safe_load(tool_desc_path.read_text()) or {}
        profiles_raw = td_doc.get("profiles") or {}
        if not isinstance(profiles_raw, dict):
            raise ValueError(
                f"{tool_desc_path}: top-level `profiles:` must be a mapping, "
                f"got {type(profiles_raw).__name__}"
            )
        for profile_name, overrides in profiles_raw.items():
            if not isinstance(overrides, dict):
                raise ValueError(
                    f"{tool_desc_path}: profiles.{profile_name} must be a mapping, "
                    f"got {type(overrides).__name__}"
                )
            tool_descriptions[str(profile_name)] = {
                str(tool): str(desc).strip() for tool, desc in overrides.items()
            }

    values = tuple(Value(id=v["id"], rule=v["rule"]) for v in core["values"])
    tabular_tables = _load_tabular_tables(core.get("tabular_tables"), path)
    document_trees = _load_document_trees(core.get("document_trees"), path)
    require_assemble_context = bool(core.get("require_assemble_context", False))
    default_contract_role = _opt_str(core, "default_contract_role", path=path, parent_name="core")
    if require_assemble_context and not default_contract_role:
        raise ValueError(
            f"{path}/core.yaml: `require_assemble_context: true` needs "
            "`default_contract_role` to name the contract to call (harness-jkmk)"
        )

    # Thought-graph block is optional: personas without bd ops tools
    # simply omit the section.
    tg_raw = core.get("thought_graph")
    thought_graph: ThoughtGraph | None = None
    if tg_raw:
        thought_graph = ThoughtGraph(
            query_first=str(tg_raw["query_first"]).strip(),
            budgets=str(tg_raw["budgets"]).strip(),
            thought_labels=str(tg_raw["thought_labels"]).strip(),
        )

    return Character(
        name=core["name"],
        pronouns=core["pronouns"],
        era=core["era"],
        relationship=dict(core["relationship"]),
        premise=core["premise"].strip(),
        self_awareness=core["self_awareness"].strip(),
        values=values,
        taboos=tuple(core["taboos"]),
        directives=tuple(core.get("directives", [])),
        deep_domains=tuple(core["deep_domains"]),
        shallow_domains=tuple(core["shallow_domains"]),
        on_being_wrong=core["on_being_wrong"].strip(),
        constitution=constitution,
        thought_graph=thought_graph,
        voice_samples=voice_samples,
        canonical_voice_count=len(canonical_samples),
        captured_voice_count=len(captured_samples),
        ablated_voice_samples=tuple(ablated_samples),
        seed_memories=tuple(seeds),
        require_search_memory=bool(core.get("require_search_memory", False)),
        require_assemble_context=require_assemble_context,
        default_contract_role=default_contract_role,
        lead_with_citation=bool(core.get("lead_with_citation", False)),
        tool_descriptions=tool_descriptions,
        tabular_tables=tabular_tables,
        document_trees=document_trees,
        voice_rewriter=_load_voice_rewriter(core, path),
        bd_assignee=_opt_str(core.get("bd"), "assignee", path=path),
        bd_exclude_assignee=_opt_str(core.get("bd"), "exclude_assignee", path=path),
        bd_scope_allowlist=_load_str_tuple(
            (core.get("bd") or {}).get("scope_allowlist"),
            path,
            "bd.scope_allowlist",
        ),
        fetch_url_allowed_hosts=_load_str_tuple(
            (core.get("fetch_url") or {}).get("allowed_hosts"),
            path,
            "fetch_url.allowed_hosts",
        ),
        citation_grammar=load_citation_grammar(core.get("citation_grammar"), path),
        catchers=_load_str_tuple(core.get("catchers"), path, "catchers"),
        scope_hint=_opt_str(core, "scope_hint", path=path, parent_name="core"),
        scope_redirect_template=_opt_str(
            core, "scope_redirect_template", path=path, parent_name="core"
        ),
    )


# ---- harness-a2sa: helpers for the new core.yaml fields ----


def _load_tabular_tables(raw: object, path: Path) -> tuple[TabularTableSpec, ...]:
    """Parse the optional `tabular_tables:` list from core.yaml
    (harness-kgpi). Each entry is a mapping with `table_name`,
    `description`, `csv_path` (relative to character_path), and
    `columns` (list of [name, type, desc] triples).

    Returns empty tuple when the key is absent — most characters
    don't ship tabular tables and the parser stays graceful for them.
    Malformed entries raise ValueError with a clear pointer at the
    offending row."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}/core.yaml: `tabular_tables` must be a list, got {type(raw).__name__}"
        )
    out: list[TabularTableSpec] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}/core.yaml: tabular_tables[{idx}] must be a mapping")
        table_name = str(entry.get("table_name") or "").strip()
        description = str(entry.get("description") or "").strip()
        csv_raw = str(entry.get("csv_path") or "").strip()
        if not table_name or not csv_raw:
            raise ValueError(
                f"{path}/core.yaml: tabular_tables[{idx}] needs `table_name` + `csv_path`"
            )
        cols_raw = entry.get("columns")
        if not isinstance(cols_raw, list) or not cols_raw:
            raise ValueError(
                f"{path}/core.yaml: tabular_tables[{idx}] needs a non-empty `columns` list"
            )
        columns: list[tuple[str, str, str]] = []
        for col_idx, col in enumerate(cols_raw):
            if not isinstance(col, list) or len(col) != 3:
                raise ValueError(
                    f"{path}/core.yaml: tabular_tables[{idx}].columns[{col_idx}] "
                    f"must be [name, type, description]"
                )
            columns.append((str(col[0]), str(col[1]), str(col[2])))
        out.append(
            TabularTableSpec(
                table_name=table_name,
                description=description,
                csv_path=(path / csv_raw).resolve(),
                columns=tuple(columns),
            )
        )
    return tuple(out)


def _load_document_trees(raw: object, path: Path) -> tuple[DocumentTreeSpec, ...]:
    """Parse the optional `document_trees:` list from core.yaml
    (harness-ejxn). Each entry is a mapping with `name`, `description`,
    `source` (relative path to the markdown or JSONL file), and an
    optional `format` (default `markdown`, also accepts `jsonl`).

    Returns empty tuple when the key is absent — most characters
    don't ship hierarchical doc data and the parser stays graceful.
    Malformed entries raise ValueError with a pointer at the offending
    row so a fixture author finds the typo without diving into stacks.

    Eager path resolution mirrors `_load_tabular_tables`: we resolve
    against the character directory so downstream callers (the CLI
    session bootstrap, eval fixtures) get an absolute path. We don't
    require the file to exist — that check belongs to the ingest
    driver, which has the only useful error context."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}/core.yaml: `document_trees` must be a list, got {type(raw).__name__}"
        )
    out: list[DocumentTreeSpec] = []
    seen_names: set[str] = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}/core.yaml: document_trees[{idx}] must be a mapping")
        name = str(entry.get("name") or "").strip()
        description = str(entry.get("description") or "").strip()
        source_raw = str(entry.get("source") or "").strip()
        if not name or not source_raw:
            raise ValueError(f"{path}/core.yaml: document_trees[{idx}] needs `name` + `source`")
        if name in seen_names:
            raise ValueError(f"{path}/core.yaml: document_trees[{idx}] duplicate name {name!r}")
        seen_names.add(name)
        fmt = str(entry.get("format") or "markdown").strip()
        if fmt not in _DOCUMENT_TREE_FORMATS:
            raise ValueError(
                f"{path}/core.yaml: document_trees[{idx}] format must be one of "
                f"{sorted(_DOCUMENT_TREE_FORMATS)}; got {fmt!r}"
            )
        jsonl_cfg = _parse_document_tree_jsonl(entry.get("jsonl"), path, idx=idx, fmt=fmt)
        out.append(
            DocumentTreeSpec(
                name=name,
                description=description,
                source_path=(path / source_raw).resolve(),
                source_format=fmt,
                **jsonl_cfg,
            )
        )
    return tuple(out)


def _parse_document_tree_jsonl(raw: object, path: Path, *, idx: int, fmt: str) -> dict[str, Any]:
    """Parse the optional `jsonl:` nested mapping on a document_trees
    entry. Returns kwargs ready to splat into DocumentTreeSpec.

    Schema (all keys optional when fmt != 'jsonl'):
      - depth_fields:        list[str] — required when fmt == 'jsonl'.
      - heading_prefixes:    list[str] — optional, defaults [].
      - leaf_heading_field:  str | null — optional.
      - body_field:          str — defaults 'body'.
      - chunk_index_field:   str | null — defaults 'chunk_index'.

    Catching depth_fields-missing at YAML load (rather than at builder
    time) means a fixture author sees the typo immediately instead of
    discovering it on the next chat session start."""
    location = f"{path}/core.yaml: document_trees[{idx}]"
    cfg: dict[str, Any] = {}
    if raw is None:
        if fmt == "jsonl":
            raise ValueError(
                f"{location}: jsonl format requires a `jsonl:` block with at least `depth_fields`"
            )
        return cfg
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: `jsonl` must be a mapping, got {type(raw).__name__}")

    df = raw.get("depth_fields")
    if df is None:
        if fmt == "jsonl":
            raise ValueError(f"{location}: jsonl format requires `jsonl.depth_fields`")
    else:
        if not isinstance(df, list) or not df or not all(isinstance(f, str) and f for f in df):
            raise ValueError(f"{location}: jsonl.depth_fields must be a non-empty list of strings")
        cfg["jsonl_depth_fields"] = tuple(df)

    hp = raw.get("heading_prefixes")
    if hp is not None:
        if not isinstance(hp, list) or not all(isinstance(p, str) for p in hp):
            raise ValueError(f"{location}: jsonl.heading_prefixes must be a list of strings")
        cfg["jsonl_heading_prefixes"] = tuple(hp)

    lhf = raw.get("leaf_heading_field")
    if lhf is not None:
        if not isinstance(lhf, str) or not lhf:
            raise ValueError(f"{location}: jsonl.leaf_heading_field must be a non-empty string")
        cfg["jsonl_leaf_heading_field"] = lhf

    bf = raw.get("body_field")
    if bf is not None:
        if not isinstance(bf, str) or not bf:
            raise ValueError(f"{location}: jsonl.body_field must be a non-empty string")
        cfg["jsonl_body_field"] = bf

    cif = raw.get("chunk_index_field")
    if cif is not None:
        if cif is False or (cif is not None and not isinstance(cif, str)):
            raise ValueError(f"{location}: jsonl.chunk_index_field must be a string or null")
        cfg["jsonl_chunk_index_field"] = cif if cif else None

    return cfg


_VOICE_REWRITERS = frozenset({"persona", "caveman", "none"})


def _load_voice_rewriter(core: dict[str, object], path: Path) -> str:
    """Read core['voice']['rewriter'] (or top-level 'voice_rewriter')
    and validate against the allowed set. Default 'persona' when
    absent so existing core.yaml files keep their current behavior."""
    voice = core.get("voice")
    raw: object | None = None
    if isinstance(voice, dict):
        raw = voice.get("rewriter")
    if raw is None:
        raw = core.get("voice_rewriter")
    if raw is None:
        return "persona"
    if not isinstance(raw, str) or raw not in _VOICE_REWRITERS:
        raise ValueError(
            f"{path}/core.yaml: voice.rewriter must be one of "
            f"{sorted(_VOICE_REWRITERS)}; got {raw!r}"
        )
    return raw


def _opt_str(parent: object, key: str, *, path: Path, parent_name: str = "bd") -> str | None:
    """Pull a string value from an optional nested mapping. Returns
    None when parent is missing/None/non-mapping or the key is absent
    or null. Used for the optional bd.* fields."""
    if not isinstance(parent, dict):
        return None
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{path}/core.yaml: {parent_name}.{key} must be a string or null; "
            f"got {type(value).__name__}"
        )
    return value


def _load_str_tuple(raw: object, path: Path, field_name: str) -> tuple[str, ...]:
    """Validate an optional list-of-strings core.yaml field. Missing /
    null → empty tuple (no filter). Wrong type or non-string entries
    raise with the field name so a fixture author can find it."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{path}/core.yaml: {field_name} must be a list; got {type(raw).__name__}")
    out: list[str] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(
                f"{path}/core.yaml: {field_name}[{idx}] must be a string; got {type(item).__name__}"
            )
        out.append(item)
    return tuple(out)
