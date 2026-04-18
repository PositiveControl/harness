from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from harness.config import settings


class SentenceTransformersEmbedder:
    """Embedder backed by sentence-transformers. Default model is read
    from Settings.embedder_repo (currently BAAI/bge-small-en-v1.5 —
    384-dim, ~130 MB RAM; solid MTEB for English, tuned for the 32GB
    box). Override per-call with `model_name=...` or globally via the
    `HARNESS_EMBEDDER_REPO` env var. Previous default
    `mixedbread-ai/mxbai-embed-large-v1` (1024-dim, ~1.3 GB RAM) stays
    reachable the same way — the class doesn't care which repo.

    Uses Metal (MPS) on Apple silicon when available, CPU elsewhere.
    Model load is deferred until the first embed() call so imports stay
    cheap. `dimension` starts at 0 and is overwritten on load with the
    actual value from the model (no more wrong-dimension window between
    construction and first embed)."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name if model_name is not None else settings.embedder_repo
        self._device = device
        self._model: Any | None = None
        self.id = f"st:{self.model_name.split('/')[-1]}"
        # Updated after load. Start at 0 rather than 1024 so stores that
        # round-trip on embedder_id can't silently mismatch if the model
        # fails to load (would raise instead of storing wrong-dim data).
        self.dimension = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        device = self._device
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = SentenceTransformer(self.model_name, device=device)
        # sentence-transformers 3.3+ renamed `get_sentence_embedding_dimension`
        # to `get_embedding_dimension`. Prefer the new name; fall back on
        # older pinned versions.
        try:
            self.dimension = int(self._model.get_embedding_dimension())
        except AttributeError:
            self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        arr: np.ndarray = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return arr
