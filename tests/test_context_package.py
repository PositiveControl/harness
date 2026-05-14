"""Tests for harness.retrieval.context_package (harness-s3f7 / Phase 3).

Pure value types — these tests pin the shape, the budget arithmetic,
and the slot-grouping accessors. No store, no orchestrator."""

from __future__ import annotations

import pytest

from harness.retrieval.context_package import (
    AccessPolicy,
    PackagedHit,
    Provenance,
    RetrievalContextPackage,
    TokenBudget,
    estimate_tokens,
)


def _hit(slot: str, body: str, *, score: float = 1.0) -> PackagedHit:
    return PackagedHit(
        slot_name=slot,
        body=body,
        provenance=Provenance(
            store="episodic", record_id=f"r:{slot}", method="hybrid", score=score
        ),
        est_tokens=estimate_tokens(body),
    )


# ---------- estimate_tokens ----------


def test_estimate_tokens_floors_at_one() -> None:
    """Empty or single-char text still counts as 1 token — prevents
    zero-token edge cases that would let the budget enforcer believe a
    hit was 'free' and pack unlimited empty hits."""
    assert estimate_tokens("") == 1
    assert estimate_tokens("x") == 1


def test_estimate_tokens_uses_char_count_over_4() -> None:
    assert estimate_tokens("abcdefgh") == 2  # 8 chars / 4
    assert estimate_tokens("a" * 100) == 25


# ---------- TokenBudget ----------


def test_token_budget_rejects_non_positive_max() -> None:
    with pytest.raises(ValueError, match="max_tokens must be >= 1"):
        TokenBudget(max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens must be >= 1"):
        TokenBudget(max_tokens=-5)


def test_token_budget_accepts_custom_estimator() -> None:
    """Callers with a real tokenizer should be able to swap the
    estimator in via the dataclass field, no monkey-patching needed."""
    budget = TokenBudget(max_tokens=100, token_estimator=lambda s: len(s.split()))
    assert budget.token_estimator("hello world how are you") == 5


# ---------- AccessPolicy ----------


def test_access_policy_carries_user_id_and_role() -> None:
    access = AccessPolicy(user_id="C9148", role="returns-handler")
    assert access.user_id == "C9148"
    assert access.role == "returns-handler"


def test_access_policy_role_defaults_to_none() -> None:
    access = AccessPolicy(user_id="C9148")
    assert access.role is None


# ---------- RetrievalContextPackage ----------


def test_package_tokens_used_sums_packaged_hits() -> None:
    hits = (
        _hit("customer_history", "x" * 40),  # 10 tokens
        _hit("refund_policy", "y" * 80),  # 20 tokens
    )
    pkg = RetrievalContextPackage(
        intent="returns",
        access=AccessPolicy(user_id="u"),
        budget=TokenBudget(max_tokens=500),
        hits=hits,
        missing_required_slots=(),
    )
    assert pkg.tokens_used == 30


def test_package_tokens_overflow_sums_overflow_only() -> None:
    pkg = RetrievalContextPackage(
        intent="returns",
        access=AccessPolicy(user_id="u"),
        budget=TokenBudget(max_tokens=10),
        hits=(_hit("a", "x" * 20),),  # 5 tokens
        missing_required_slots=(),
        overflow_hits=(_hit("a", "y" * 100),),  # 25 tokens
    )
    assert pkg.tokens_used == 5
    assert pkg.tokens_overflow == 25


def test_package_is_complete_when_no_missing_required_slots() -> None:
    pkg = RetrievalContextPackage(
        intent="returns",
        access=AccessPolicy(user_id="u"),
        budget=TokenBudget(max_tokens=100),
        hits=(_hit("a", "ok"),),
        missing_required_slots=(),
    )
    assert pkg.is_complete is True


def test_package_is_incomplete_when_required_slot_missing() -> None:
    pkg = RetrievalContextPackage(
        intent="returns",
        access=AccessPolicy(user_id="u"),
        budget=TokenBudget(max_tokens=100),
        hits=(),
        missing_required_slots=("refund_policy",),
    )
    assert pkg.is_complete is False


def test_hits_for_slot_filters_by_slot_name() -> None:
    hits = (
        _hit("customer_history", "ch1"),
        _hit("refund_policy", "rp1"),
        _hit("customer_history", "ch2"),
    )
    pkg = RetrievalContextPackage(
        intent="returns",
        access=AccessPolicy(user_id="u"),
        budget=TokenBudget(max_tokens=100),
        hits=hits,
        missing_required_slots=(),
    )
    history = pkg.hits_for_slot("customer_history")
    assert tuple(h.body for h in history) == ("ch1", "ch2")


def test_provenance_for_slot_returns_audit_records_in_order() -> None:
    hits = (
        _hit("a", "1", score=0.9),
        _hit("b", "2", score=0.8),
        _hit("a", "3", score=0.7),
    )
    pkg = RetrievalContextPackage(
        intent="x",
        access=AccessPolicy(user_id="u"),
        budget=TokenBudget(max_tokens=100),
        hits=hits,
        missing_required_slots=(),
    )
    a_prov = pkg.provenance_for_slot("a")
    assert tuple(p.score for p in a_prov) == (0.9, 0.7)
    assert all(p.store == "episodic" for p in a_prov)


def test_package_is_frozen() -> None:
    pkg = RetrievalContextPackage(
        intent="x",
        access=AccessPolicy(user_id="u"),
        budget=TokenBudget(max_tokens=100),
        hits=(),
        missing_required_slots=(),
    )
    with pytest.raises(AttributeError):
        pkg.intent = "y"  # type: ignore[misc]
