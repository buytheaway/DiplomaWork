"""Tests for search threshold / decision logic."""

from __future__ import annotations

import os

os.environ.setdefault("TESTING", "true")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from app.api.schemas.search import SearchResponse


def test_search_response_defaults():
    """New threshold fields have safe defaults."""
    resp = SearchResponse(k=5, model="dummy", results=[])
    assert resp.decision == "unknown"
    assert resp.best_score is None
    assert resp.threshold_used == 0.0
    assert resp.best_match_above_threshold is False


def test_search_response_match():
    resp = SearchResponse(
        k=1,
        model="test",
        results=[],
        best_score=0.85,
        threshold_used=0.4,
        best_match_above_threshold=True,
        decision="match",
    )
    assert resp.decision == "match"
    assert resp.best_match_above_threshold is True


def test_search_response_unknown():
    resp = SearchResponse(
        k=1,
        model="test",
        results=[],
        best_score=0.2,
        threshold_used=0.4,
        best_match_above_threshold=False,
        decision="unknown",
    )
    assert resp.decision == "unknown"
    assert resp.best_match_above_threshold is False


def test_search_response_threshold_boundary():
    """Score exactly at threshold counts as match."""
    threshold = 0.4
    score = 0.4
    above = score >= threshold
    decision = "match" if above else "unknown"
    resp = SearchResponse(
        k=1,
        model="test",
        results=[],
        best_score=score,
        threshold_used=threshold,
        best_match_above_threshold=above,
        decision=decision,
    )
    assert resp.decision == "match"
    assert resp.best_match_above_threshold is True
