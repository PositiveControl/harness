from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding contract. Implementations MUST return L2-
    normalized embeddings so cosine similarity reduces to a dot
    product."""

    id: str
    dimension: int

    def embed(self, texts: Iterable[str]) -> np.ndarray: ...
