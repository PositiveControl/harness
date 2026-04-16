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
