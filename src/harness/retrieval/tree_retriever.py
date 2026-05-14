"""Document-tree retriever (harness-h5ly / Phase 1).

Thin retrieval layer over `DocumentTreeStore`. v1 just delegates to the
store's hybrid search — the value comes from the store, which embeds
at SECTION grain (aggregating chunks) instead of CHUNK grain. The class
exists so the bench (and any future caller) can swap the underlying
store without growing a hybrid-specific path; the auto-merge promotion
step lands here when it ships in a follow-up.

What this isn't (v1):
- An auto-merge retriever. The hooks are here (`top_k_with_promotion`,
  TBD) but rolling sibling leaves up to their parent is intentionally
  deferred until we know section-grained retrieval at least matches
  the Phase 0 chunk-grained baseline. The bd decision rule for
  harness-h5ly says: 'if tree retriever does not beat flat hybrid on
  recall@5, throw it out and commit the negative result.' Skipping
  auto-merge keeps the eval surface comparable to Phase 0 so the
  question stays answerable.

What this is:
- A clean wrapper that returns `(TreeNode, score)` pairs from a query,
  using the same hybrid (dense+BM25) signal the EpisodicStore uses.
  The score units are the store's, not normalized.

The retriever doesn't own the embedder or the store — both are passed
in. Same pattern as `voice_retriever.VoiceRetriever`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.store.document_tree import DocumentTreeStore, SearchMode, TreeNode


class TreeRetriever:
    """v1 = thin wrapper. The store does the actual ranking; this
    class exists so the bench (and future callers) program against a
    stable retriever interface even as the underlying ranker evolves."""

    def __init__(self, store: DocumentTreeStore) -> None:
        self._store = store

    def top_k(
        self,
        query: str,
        *,
        k: int = 10,
        mode: SearchMode = "hybrid",
        min_score: float = 0.0,
    ) -> list[tuple[TreeNode, float]]:
        """Return up to `k` embedded nodes ranked by `mode`. Pure
        delegation in v1 — the value vs. Phase 0 is the granularity
        the store ingested at, not retrieval cleverness here."""
        return self._store.search(query, k=k, mode=mode, min_score=min_score)
