from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from harness.character import load_character

REPO = Path(__file__).resolve().parents[1]
AIRTON = REPO / "character" / "airton"


def _copy_character(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_character_loader_merges_captured_samples(tmp_path: Path) -> None:
    target = _copy_character(AIRTON, tmp_path / "airton")
    captured_path = target / "voice" / "captured.yaml"
    captured_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "samples": [
                    {
                        "id": "captured-test-1",
                        "prompt": "What should I do?",
                        "gold": "Check the logs first.",
                    }
                ],
            },
            sort_keys=False,
        )
    )

    character = load_character(target)
    ids = [s.id for s in character.voice_samples]
    assert "captured-test-1" in ids
    captured = next(s for s in character.voice_samples if s.id == "captured-test-1")
    assert captured.prompt == "What should I do?"
    assert captured.gold == "Check the logs first."


def test_character_loader_works_when_captured_file_missing(tmp_path: Path) -> None:
    target = _copy_character(AIRTON, tmp_path / "airton")
    captured_path = target / "voice" / "captured.yaml"
    # The source dir may already have captured samples from live use;
    # remove the copy so this test exercises the "file missing" path.
    captured_path.unlink(missing_ok=True)
    assert not captured_path.exists()

    character = load_character(target)
    canonical_doc = yaml.safe_load((target / "voice" / "canonical.yaml").read_text())
    assert len(character.voice_samples) == len(canonical_doc["samples"])


def test_character_loader_handles_empty_captured_file(tmp_path: Path) -> None:
    target = _copy_character(AIRTON, tmp_path / "airton")
    (target / "voice" / "captured.yaml").write_text("")

    character = load_character(target)
    # No crash; just no captured samples
    canonical_doc = yaml.safe_load((target / "voice" / "canonical.yaml").read_text())
    assert len(character.voice_samples) == len(canonical_doc["samples"])


def test_captured_samples_retrievable_via_voice_retriever(tmp_path: Path) -> None:
    """With the retrieval extra installed, captured samples should
    participate in similarity search just like canonical ones."""
    from collections.abc import Iterable

    import numpy as np

    from harness.retrieval import VoiceRetriever

    # Fake embedder so we don't load sentence-transformers here
    class _E:
        id = "e"
        dimension = 4

        def embed(self, texts: Iterable[str]) -> np.ndarray:
            out: list[np.ndarray] = []
            for t in texts:
                h = sum(ord(c) for c in t)
                v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
                n = float(np.linalg.norm(v))
                out.append(v / n if n > 0 else v)
            return np.stack(out)

    target = _copy_character(AIRTON, tmp_path / "airton")
    (target / "voice" / "captured.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "samples": [
                    {
                        "id": "captured-retrieval-test",
                        "prompt": "Unique phrase zzzz",
                        "gold": "Targeted gold.",
                    }
                ],
            },
            sort_keys=False,
        )
    )
    character = load_character(target)
    retriever = VoiceRetriever(embedder=_E(), character=character)

    # Captured sample is present in the overall suite
    ids = {s.id for s in character.voice_samples}
    assert "captured-retrieval-test" in ids

    # top_k can return it — exact ranking depends on fake embedder but
    # the sample must be reachable.
    all_returned = retriever.top_k("anything", k=len(character.voice_samples))
    assert any(s.id == "captured-retrieval-test" for s in all_returned)
