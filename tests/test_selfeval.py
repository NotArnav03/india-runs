"""Tests for the judgment set, self-eval harness, domain-title rule, and the
cross-encoder integration — all model-free."""

import datetime as dt

import pytest

from evaluation.judgment_archetypes import build_judgment_set, REF
from evaluation.selfeval import evaluate, composite_from_labels, tune_weights
from ir.adapters import candidate_from_raw
from ir.features_library import build_feature_library
from ir.honeypot import consistency_report
from ir.jd_requirements import REDROB_SENIOR_AI_ENGINEER as REQS


# ─── judgment set integrity ─────────────────────────────────────────────────

def test_judgment_set_spans_all_tiers():
    js = build_judgment_set()
    tiers = {tier for _, tier in js}
    assert tiers == {0, 1, 2, 3, 4, 5}
    assert len(js) >= 15  # enough for NDCG@10 / P@10 to be meaningful


def test_only_the_honeypot_is_inconsistent():
    """Every authored archetype is internally consistent except the honeypot,
    so the consistency detector's precision is validated here."""
    suspects = [raw["candidate_id"] for raw, _ in build_judgment_set()
                if consistency_report(raw, REF).is_suspect]
    assert suspects == ["CAND_9000053"]  # the one impossible profile


# ─── metric wiring ──────────────────────────────────────────────────────────

def test_composite_perfect_ordering_maxes_ndcg_and_map():
    # Scores == tiers → the ranking matches the ideal tier order, so the
    # graded NDCG and MAP both hit 1.0.  (P@10 is capped at 0.6 here because
    # only 6 of the 12 items are tier ≥ 3, so the composite maxes at ~0.98 —
    # a useful reminder that P@10 is bounded by the relevant-item count.)
    y_true = [5, 4, 3, 2, 1, 0] * 2
    rep = composite_from_labels(y_true, list(y_true))
    assert rep.ndcg10 == pytest.approx(1.0, abs=1e-6)
    assert rep.map_ == pytest.approx(1.0, abs=1e-6)
    assert rep.composite > 0.95


def test_composite_inverted_ordering_is_low():
    y_true = [5, 4, 3, 2, 1, 0] * 2
    y_scores = list(range(len(y_true)))  # ascending == worst
    rep = composite_from_labels(y_true, y_scores)
    assert rep.composite < 0.5


# ─── end-to-end self-eval ───────────────────────────────────────────────────

def test_selfeval_ranks_archetypes_well():
    rep = evaluate(embedder_name="hashing")
    assert rep.composite >= 0.85          # strong even with the model-free embedder
    assert rep.honeypot_rate_top10 == 0.0  # Stage-3 guard: no honeypots up top


def test_tuner_does_not_regress_below_baseline():
    from rank import RANK_WEIGHTS
    base = evaluate(weights=RANK_WEIGHTS).composite
    _, tuned = tune_weights(RANK_WEIGHTS, rounds=1, step=0.05)
    assert tuned >= base - 1e-9


# ─── domain-title soft penalty ──────────────────────────────────────────────

def _role_coherence(raw):
    lib = {f.name: f for f in build_feature_library(REQS, REF)}
    return lib["role_coherence"].compute(REQS.build_job(), candidate_from_raw(raw))


def test_cv_title_without_nlp_is_soft_penalized():
    from tests.test_pipeline import mk_raw
    cv = mk_raw(1, profile={"current_title": "Computer Vision Engineer",
                            "headline": "Computer Vision | detection"})
    coherent = mk_raw(2, profile={"current_title": "ML Engineer",
                                  "headline": "ML Engineer | retrieval, ranking"})
    assert _role_coherence(cv) == pytest.approx(0.55)
    assert _role_coherence(coherent) == pytest.approx(1.0)


# ─── cross-encoder integration (mocked model) ───────────────────────────────

def test_cross_encoder_blend_runs_without_model(monkeypatch):
    import rank as rankmod
    from tests.test_pipeline import mk_raw

    cands = [candidate_from_raw(mk_raw(i)) for i in range(1, 6)]

    def fake_rerank(reqs, ranked, raw_by_id, k):
        # deterministic "cross-encoder" head scores, no model needed
        return {r.id: 0.5 for r in ranked}

    monkeypatch.setattr(rankmod, "_cross_encoder_rerank", fake_rerank)
    final, guard = rankmod.score_candidates(
        cands, embedder_name="hashing", use_cross_encoder=True,
        cross_encoder_k=10, reference_date=REF,
    )
    assert len(final) == 5
    assert all(v >= 0.0 for v in final.values())
