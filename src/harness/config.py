from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HARNESS_",
        env_file=".env",
        extra="ignore",
    )

    root: Path = _ROOT
    character_name: str = "airton"
    log_level: str = "INFO"
    # Default models for pluggable components. Each value is the repo
    # string that the component would otherwise hardcode; centralizing
    # here lets any caller override via env (HARNESS_EMBEDDER_REPO /
    # HARNESS_ROUTER_REPO) or explicit ctor/CLI arg without touching
    # code.
    # Picked via bench: bge-small matches mxbai voice aggregate within
    # noise (0.9124 vs 0.9117) at 10% of the RAM (~130 MB vs ~1.3 GB).
    # Router stayed on Hermes-3 because every smaller generic-chat
    # candidate we tried under-routes: Qwen-1.5B 0.85, Llama-1B 0.65,
    # SmolLM2-1.7B 0.40 vs Hermes-3 1.00. Function-call-tuned small
    # routers don't exist yet in mlx-community — revisit when one
    # lands. See harness-e4m, harness-5b3 in beads for the full data.
    embedder_repo: str = "BAAI/bge-small-en-v1.5"
    router_repo: str = "mlx-community/Hermes-3-Llama-3.2-3B-4bit"
    # MLX free-cache cap in megabytes. None = no cap (MLX default).
    # Caps the pool of buffers that MLX has allocated but not yet
    # returned to the system allocator — the active model weights and
    # live KV cache are NOT affected, only the "would keep for reuse"
    # scratch pool. Lower = smaller peak RSS under memory pressure at
    # the cost of re-alloc on the next turn. Set via
    # HARNESS_MLX_CACHE_LIMIT_MB; benchmarked per tuning commit under
    # harness-0kw (see bench_results/mlx_cache_*.json).
    mlx_cache_limit_mb: int | None = None
    # MLX speculative-decoding draft model (HARNESS_MLX_DRAFT_MODEL_REPO).
    # When set, MLXAdapter loads this smaller same-vocab model alongside
    # the main model and lets mlx_lm.stream_generate use it as a draft
    # for speculative decoding. Distribution-preserving — output is
    # mathematically identical to non-speculative decoding; the only
    # cost is +~350 MB RAM for a 0.5B Qwen draft vs the default ~1-2x
    # single-stream throughput gain on 7B/32B targets. Recommended
    # pairing for the default Qwen2.5 family: mlx-community/Qwen2.5-
    # 0.5B-Instruct-4bit. None = disabled (MLX default).
    mlx_draft_model_repo: str | None = None
    # ab (airton_b) data-plane isolation. ab wraps bd as its backing
    # store; its beads DB lives outside the repo so personal tasks
    # don't leak into the harness git history, get a different backup
    # cadence, and can federate independently of dev work. Override
    # via HARNESS_AB_BD_DIR. The dir is treated as a bd working
    # directory — bd init / bd bootstrap must have been run there
    # once; the adapter only verifies, never auto-inits.
    ab_bd_dir: Path | None = None
    # ab's memory DB parent directory. Per-character isolation for
    # episodic + semantic + transcript + compaction stores so personal
    # turns and operating-style facts never cross into Airton's silo.
    # Defaults to <ab_bd_dir>/memory/ when unset — co-locates bd data
    # and memory under one ab dir for unified backup.
    ab_memory_dir: Path | None = None
    # ab caveman-rewriter register intensity (lite | full | ultra).
    # Consumed by src/harness/persona/caveman_rewriter.py (harness-inj.3).
    ab_register: str = "lite"
    # Whether to apply the caveman rewrite on turns where tools ran.
    # Off by default — same rationale as Airton's PersonaAdapter:
    # rewriter compresses, which is wrong for investigate/summarize
    # tool replies.
    ab_rewrite_on_tools: bool = False
    # ab thought-graph budget knobs (Phase 3.6). Consumed by
    # BeadsAdapter / ab_ops (C1-C4); centralized here so every knob is
    # tunable via HARNESS_AB_* env without hardcoding.
    ab_turn_cap: int = Field(default=3, ge=1, le=10)
    """Max ab orchestrator turns per user message."""
    ab_inflight_cap: int = Field(default=10, ge=5, le=50)
    """Max simultaneously in-flight ab issues before deferring new ones."""
    ab_stall_defers: int = Field(default=3, ge=1, le=10)
    """Consecutive defers before an ab issue is flagged as stalled."""
    ab_drift_days: int = Field(default=7, ge=1, le=90)
    """Days of inactivity before an ab issue is surfaced as drift."""

    @property
    def character_path(self) -> Path:
        return self.root / "character" / self.character_name

    @property
    def data_path(self) -> Path:
        path = self.root / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def db_path(self) -> Path:
        return self.data_path / "harness.sqlite"

    @property
    def ab_bd_dir_resolved(self) -> Path:
        """Back-compat alias — resolves airton_b's bd working directory.
        Kept so callers + env HARNESS_AB_BD_DIR continue to work while
        the rest of the code migrates to `bd_dir_for(character_name)`."""
        return self.bd_dir_for("airton_b")

    @property
    def ab_memory_dir_resolved(self) -> Path:
        """Resolve ab's memory dir, defaulting to <ab_bd_dir>/memory/
        so one ab directory holds both planes (bd + memory) for a
        clean backup boundary."""
        if self.ab_memory_dir is not None:
            return self.ab_memory_dir
        return self.ab_bd_dir_resolved / "memory"

    def bd_dir_for(self, character_name: str) -> Path:
        """Resolve the bd working directory for a character.

        airton (default) and airton_b share the project's bd dir
        (= `self.root`) — the project has one healthy Dolt instance
        and reusing it avoids the fresh-init bootstrap pain from
        harness-55y. Isolation between those two is enforced by
        `assignee` attribution, not directory separation. airton_b
        still honours HARNESS_AB_BD_DIR when explicitly set.

        Third-plus personas (airton_c and beyond) auto-silo under
        `character/<name>/bd/`. Their bead graphs are content-
        disjoint from harness-dev tracking (a tutoring persona's
        notes aren't relevant to project beads), so they should
        live in their own graph even though assignee filtering
        could keep them readable. The siloed dir needs a one-time
        `bd init` during character scaffolding; after that every
        bd op routes through the per-character adapter."""
        if character_name == "airton_b" and self.ab_bd_dir is not None:
            return self.ab_bd_dir
        if character_name in ("airton", "airton_b"):
            return self.root
        bd_dir = self.root / "character" / character_name / "bd"
        bd_dir.mkdir(parents=True, exist_ok=True)
        return bd_dir

    def db_path_for(self, character_name: str) -> Path:
        """Resolve the memory DB path for a character. airton uses
        the repo-relative default; airton_b is siloed under its own
        memory dir so personal turns + operating-style facts stay
        out of the dev store. Third-plus personas auto-silo under
        `character/<name>/data/` — keeps per-persona corpora (e.g.
        atc's FAA docs) from polluting retrieval across personas.
        Parent directories are created eagerly for isolated
        characters; airton's default is managed by `data_path`."""
        if character_name == "airton_b":
            memory_dir = self.ab_memory_dir_resolved
            memory_dir.mkdir(parents=True, exist_ok=True)
            return memory_dir / "harness.sqlite"
        if character_name == "airton":
            return self.db_path
        mem_dir = self.root / "character" / character_name / "data"
        mem_dir.mkdir(parents=True, exist_ok=True)
        return mem_dir / "harness.sqlite"

    @property
    def character_db_path(self) -> Path:
        """Shortcut to the current character's DB path, derived from
        `character_name`. Useful in CLI subcommands that don't already
        hold a Character instance."""
        return self.db_path_for(self.character_name)


settings = Settings()
