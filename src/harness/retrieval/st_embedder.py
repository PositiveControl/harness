from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


class SentenceTransformersEmbedder:
    """Embedder backed by sentence-transformers. Default model is
    mixedbread-ai/mxbai-embed-large-v1 — 1024-dim, ~1.3 GB RAM, near
    top-of-MTEB for English and Matryoshka-trained (truncatable to 512
    or 256 if we want cheaper search later).

    Uses Metal (MPS) on Apple silicon when available, CPU elsewhere.
    Model load is deferred until the first embed() call so imports stay
    cheap."""

    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        *,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._model: Any | None = None
        self.id = f"st:{model_name.split('/')[-1]}"
        self.dimension = 1024  # mxbai-embed-large default; updated after load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        device = self._device
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = SentenceTransformer(self.model_name, device=device)
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
