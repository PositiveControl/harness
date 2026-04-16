"""Retrieval layer. Embeds content once at load, scores queries against
it, returns top-K. Used now for few-shot voice sample selection; will be
reused for episodic and semantic memory lookup in Phase 1b."""

from harness.retrieval.embed import Embedder
from harness.retrieval.voice_retriever import VoiceRetriever

__all__ = ["Embedder", "VoiceRetriever"]
