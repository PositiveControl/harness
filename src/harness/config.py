from __future__ import annotations

from pathlib import Path

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
    # ab (airton_b) data-plane isolation. ab wraps bd as its backing
    # store; its beads DB lives outside the repo so personal tasks
    # don't leak into the harness git history, get a different backup
    # cadence, and can federate independently of dev work. Override
    # via HARNESS_AB_BD_DIR. The dir is treated as a bd working
    # directory — bd init / bd bootstrap must have been run there
    # once; the adapter only verifies, never auto-inits.
    ab_bd_dir: Path | None = None
    # ab caveman-rewriter register intensity (lite | full | ultra).
    # Consumed by src/harness/persona/caveman_rewriter.py (harness-inj.3).
    ab_register: str = "lite"
    # Whether to apply the caveman rewrite on turns where tools ran.
    # Off by default — same rationale as Airton's PersonaAdapter:
    # rewriter compresses, which is wrong for investigate/summarize
    # tool replies.
    ab_rewrite_on_tools: bool = False

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
        """Resolve ab's bd working directory, falling back to
        ~/.harness/airton_b/ when unset. The dir itself is not created
        here — BeadsAdapter verifies on use."""
        if self.ab_bd_dir is not None:
            return self.ab_bd_dir
        return Path.home() / ".harness" / "airton_b"


settings = Settings()
