from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import frontmatter
import yaml


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
        lead_with_citation=bool(core.get("lead_with_citation", False)),
        tool_descriptions=tool_descriptions,
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
    )


# ---- harness-a2sa: helpers for the new core.yaml fields ----

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
