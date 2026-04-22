"""Tests for the bd → episodic skill harvester (sota punch #7,
harness-vu3)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from harness.skills import DEFAULT_HARVEST_LABELS, harvest_bd_skills
from harness.skills.harvester import _body_from_issue, _principle_from_labels
from harness.store._bd_types import BeadsIssue
from harness.store.episodic import EpisodicStore


@dataclass
class _Embedder:
    id: str = "fake"
    dimension: int = 4
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        batch = list(texts)
        self.calls.append(batch)
        return np.stack([np.ones(4, dtype=np.float32) / 2.0 for _ in batch])


@dataclass
class _StubAdapter:
    """Minimal stand-in for BeadsAdapter — only the subset of methods
    the harvester calls. Tests pass it positionally; `# type: ignore`
    on the adapter= kwarg keeps mypy quiet without coupling the stub
    to the full mixin MRO."""

    issues: list[BeadsIssue]
    calls: list[dict[str, object]] = field(default_factory=list)

    def list_issues(
        self,
        *,
        scope: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        issue_type: str | None = None,
        limit: int | None = None,
        assignee: str | None = None,
    ) -> list[BeadsIssue]:
        self.calls.append(
            {
                "scope": scope,
                "status": status,
                "priority": priority,
                "issue_type": issue_type,
                "limit": limit,
                "assignee": assignee,
            }
        )
        return [
            i
            for i in self.issues
            if status is None or status == "all" or i.status == status
        ]


def _issue(
    id_: str,
    *,
    title: str,
    labels: tuple[str, ...],
    status: str = "closed",
    description: str | None = None,
) -> BeadsIssue:
    raw: dict[str, object] = {}
    if description is not None:
        raw["description"] = description
    return BeadsIssue(
        id=id_,
        title=title,
        status=status,
        priority=2,
        issue_type="task",
        labels=labels,
        raw=raw,
        assignee="airton_b",
    )


@pytest.fixture
def episodic(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "h.sqlite", embedder=_Embedder())


# ---------- pure helpers ----------


def test_principle_from_labels_picks_first_thought_label() -> None:
    assert _principle_from_labels(("scope:professional", "thought:decision")) == "decision"
    assert (
        _principle_from_labels(("thought:observation", "thought:decision")) == "observation"
    )


def test_principle_from_labels_returns_none_when_no_thought_label() -> None:
    assert _principle_from_labels(("scope:professional", "urgency:high")) is None


def test_body_from_issue_prefixes_type_and_includes_description() -> None:
    issue = _issue(
        "harness-1",
        title="Prefer local models for CLI dev",
        labels=("thought:decision",),
        description="Long loop times with cloud models broke flow.",
    )
    body = _body_from_issue(issue)
    assert body.startswith("DECISION: Prefer local models for CLI dev")
    assert "broke flow" in body


def test_body_from_issue_skips_description_when_absent() -> None:
    issue = _issue("harness-2", title="Note the latency cliff", labels=("thought:observation",))
    body = _body_from_issue(issue)
    assert body == "OBSERVATION: Note the latency cliff"


# ---------- harvester integration ----------


def test_harvest_ingests_matching_thought_labels(episodic: EpisodicStore) -> None:
    adapter = _StubAdapter(
        issues=[
            _issue("harness-1", title="decision-a", labels=("thought:decision",)),
            _issue("harness-2", title="observation-a", labels=("thought:observation",)),
            _issue("harness-3", title="noise", labels=("scope:professional",)),
        ]
    )
    report = harvest_bd_skills(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]
    assert report.scanned == 2
    assert report.newly_ingested == 2
    assert report.already_present == 0
    assert set(report.ingested_ids) == {"harness-1", "harness-2"}


def test_harvest_skips_hypothesis_and_question_by_default(episodic: EpisodicStore) -> None:
    adapter = _StubAdapter(
        issues=[
            _issue("harness-1", title="hypothesis-a", labels=("thought:hypothesis",)),
            _issue("harness-2", title="question-a", labels=("thought:question",)),
            _issue("harness-3", title="decision-a", labels=("thought:decision",)),
        ]
    )
    report = harvest_bd_skills(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]
    assert report.newly_ingested == 1
    assert report.ingested_ids == ("harness-3",)


def test_harvest_is_idempotent(episodic: EpisodicStore) -> None:
    adapter = _StubAdapter(
        issues=[
            _issue("harness-1", title="decision-a", labels=("thought:decision",)),
        ]
    )
    first = harvest_bd_skills(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]
    second = harvest_bd_skills(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]
    assert first.newly_ingested == 1
    assert second.newly_ingested == 0
    assert second.already_present == 1


def test_harvest_filters_to_closed_by_default(episodic: EpisodicStore) -> None:
    adapter = _StubAdapter(
        issues=[
            _issue(
                "harness-1",
                title="in-progress decision",
                labels=("thought:decision",),
                status="in_progress",
            ),
            _issue(
                "harness-2",
                title="closed decision",
                labels=("thought:decision",),
                status="closed",
            ),
        ]
    )
    report = harvest_bd_skills(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]
    assert report.ingested_ids == ("harness-2",)
    # Confirm the adapter was actually asked to narrow by status.
    assert adapter.calls[0]["status"] == "closed"


def test_harvest_with_status_all_includes_open_thoughts(episodic: EpisodicStore) -> None:
    adapter = _StubAdapter(
        issues=[
            _issue(
                "harness-1",
                title="open",
                labels=("thought:decision",),
                status="open",
            ),
            _issue(
                "harness-2",
                title="closed",
                labels=("thought:decision",),
                status="closed",
            ),
        ]
    )
    report = harvest_bd_skills(ab_adapter=adapter, episodic=episodic, status="all")  # type: ignore[arg-type]
    assert set(report.ingested_ids) == {"harness-1", "harness-2"}


def test_harvest_custom_label_set(episodic: EpisodicStore) -> None:
    adapter = _StubAdapter(
        issues=[
            _issue("harness-1", title="decision", labels=("thought:decision",)),
            _issue("harness-2", title="hypothesis", labels=("thought:hypothesis",)),
        ]
    )
    # Caller opts into harvesting hypotheses too — useful for a
    # diagnostic sweep over recent exploration.
    report = harvest_bd_skills(
        ab_adapter=adapter,  # type: ignore[arg-type]
        episodic=episodic,
        labels=("thought:hypothesis",),
    )
    assert report.ingested_ids == ("harness-2",)


# ---------- retrieval integration ----------


def test_harvested_rows_participate_in_episodic_search(
    tmp_path: Path,
) -> None:
    """Smoke test: after harvest, procedural-tier rows show up in
    normal `search()` calls. The chat flow already injects episodic
    hits into the system prompt, so this is what closes the loop."""
    # Use a deterministic embedder that produces distinct vectors per text
    # so we can verify the procedural row ranks near a related query.

    @dataclass
    class _KeywordEmbedder:
        id: str = "kw"
        dimension: int = 4

        def embed(self, texts: Iterable[str]) -> np.ndarray:
            vecs: list[np.ndarray] = []
            for text in texts:
                low = text.lower()
                # Trivial topic vectors: boost dim[0] for 'decision', dim[1]
                # for the query keyword, etc. Good enough to prove ordering.
                v = np.array(
                    [
                        1.0 if "decision" in low else 0.0,
                        1.0 if "latency" in low else 0.0,
                        1.0 if "bd" in low else 0.0,
                        0.1,
                    ],
                    dtype=np.float32,
                )
                n = float(np.linalg.norm(v))
                vecs.append(v / n if n > 0 else v)
            return np.stack(vecs)

    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_KeywordEmbedder())
    # Non-procedural seed that mentions an unrelated topic.
    store.ingest(
        external_id="seed-1",
        title="Voice sample",
        body="Something about weather.",
        tier="seed",
        source="yaml",
    )
    adapter = _StubAdapter(
        issues=[
            _issue(
                "harness-1",
                title="Prefer 7B over 32B when latency matters",
                labels=("thought:decision",),
                description="We decided: 7B is the default. 32B only when depth > speed.",
            ),
        ]
    )
    harvest_bd_skills(ab_adapter=adapter, episodic=store)  # type: ignore[arg-type]

    # A query that lexically overlaps the procedural row should find it.
    hits = store.search("latency decision", k=3, mode="hybrid")
    ids = {r.external_id for r, _ in hits}
    assert "harness-1" in ids


def test_procedural_rows_are_tier_procedural(episodic: EpisodicStore) -> None:
    adapter = _StubAdapter(
        issues=[
            _issue("harness-1", title="t", labels=("thought:decision",)),
        ]
    )
    harvest_bd_skills(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]
    rows = episodic.all()
    assert len(rows) == 1
    assert rows[0].tier == "procedural"
    assert rows[0].source == "bd"
    assert rows[0].external_id == "harness-1"
    assert rows[0].principle == "decision"


# ---------- constants ----------


def test_default_harvest_labels_covers_decision_and_observation() -> None:
    assert set(DEFAULT_HARVEST_LABELS) == {"thought:decision", "thought:observation"}
