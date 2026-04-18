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


settings = Settings()
