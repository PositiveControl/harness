from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from harness.character import Character, VoiceSample
    from harness.retrieval.embed import Embedder


@dataclass
class VoiceRetriever:
    """Retrieves the top-K voice samples most similar to a query. Sample
    embeddings are computed once at construction; queries embed per
    call. At 32 samples + a ~100 MB embedder, retrieval is ~milliseconds
    on a warm model.

    The caller is responsible for wiring the retrieval output into the
    system-prompt construction (`Character.system_prompt(include_samples=...)`
    or `persona.build_rewriter_messages(include_samples=...)`) so the
    rest of the pipeline stays ignorant of retrieval."""

    embedder: Embedder
    character: Character
    _sample_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        prompts = [s.prompt for s in self.character.voice_samples]
        if not prompts:
            self._sample_matrix = np.zeros((0, self.embedder.dimension), dtype=np.float32)
            return
        self._sample_matrix = self.embedder.embed(prompts)

    def top_k(
        self,
        query: str,
        *,
        k: int = 6,
        exclude_ids: frozenset[str] | None = None,
    ) -> list[VoiceSample]:
        excluded = exclude_ids or frozenset()
        if self._sample_matrix.shape[0] == 0:
            return []
        q = self.embedder.embed([query])[0]
        sims = self._sample_matrix @ q  # dot product == cosine (vectors normalized)
        scored: list[tuple[float, VoiceSample]] = [
            (float(sims[i]), sample)
            for i, sample in enumerate(self.character.voice_samples)
            if sample.id not in excluded
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [sample for _, sample in scored[:k]]
