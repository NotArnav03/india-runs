"""Tests for the multi-signal inference entrypoint."""

import numpy as np
import pytest

from ir.features import Candidate, FeatureExtractor, Job, StructuredFeature
from ir.ranker import MultiSignalRanker, TrainingExample

SKILL_VOCAB = ["python", "sql", "machine learning", "react", "marketing", "docker"]


def _job():
    return Job(
        id="jd1",
        text="Senior Python engineer with machine learning and sql experience.",
        required_skills={"python", "machine learning", "sql"},
    )


def _candidates():
    return [
        Candidate(id="strong", text="Python developer, machine learning, sql, docker.",
                  skills={"python", "machine learning", "sql", "docker"}),
        Candidate(id="weak", text="Marketing manager skilled in branding and react.",
                  skills={"marketing", "react"}),
        Candidate(id="mid", text="Python analyst with sql reporting background.",
                  skills={"python", "sql"}),
    ]


def test_extractor_feature_schema(fake_embedder):
    fx = FeatureExtractor(embedder=fake_embedder, skill_vocab=SKILL_VOCAB)
    names, X, ids = fx.extract(_job(), _candidates())
    assert names[:7] == [
        "sbert_similarity", "tfidf_similarity", "skill_coverage",
        "num_matched_skills", "keyword_overlap",
        "candidate_word_count", "jd_word_count",
    ]
    assert X.shape == (3, 7)
    assert ids == ["strong", "weak", "mid"]
    # Strong candidate covers all 3 required skills.
    cov = dict(zip(ids, X[:, 2]))
    assert cov["strong"] == pytest.approx(1.0)
    assert cov["mid"] == pytest.approx(2 / 3)
    assert cov["weak"] == pytest.approx(0.0)


def test_unsupervised_ranking_orders_by_relevance(fake_embedder):
    fx = FeatureExtractor(embedder=fake_embedder, skill_vocab=SKILL_VOCAB)
    ranker = MultiSignalRanker(feature_extractor=fx)
    ranked = ranker.rank_candidates(_job(), _candidates())
    assert [r.id for r in ranked] == ["strong", "mid", "weak"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    # Per-signal trail is attached for explainability.
    assert "skill_coverage" in ranked[0].signals


def test_top_k_truncates(fake_embedder):
    fx = FeatureExtractor(embedder=fake_embedder, skill_vocab=SKILL_VOCAB)
    ranker = MultiSignalRanker(feature_extractor=fx)
    ranked = ranker.rank_candidates(_job(), _candidates(), top_k=2)
    assert len(ranked) == 2
    assert ranked[0].id == "strong"


def test_empty_candidates(fake_embedder):
    fx = FeatureExtractor(embedder=fake_embedder, skill_vocab=SKILL_VOCAB)
    ranker = MultiSignalRanker(feature_extractor=fx)
    assert ranker.rank_candidates(_job(), []) == []


def test_single_candidate(fake_embedder):
    fx = FeatureExtractor(embedder=fake_embedder, skill_vocab=SKILL_VOCAB)
    ranker = MultiSignalRanker(feature_extractor=fx)
    ranked = ranker.rank_candidates(_job(), [_candidates()[0]])
    assert len(ranked) == 1 and ranked[0].rank == 1


def test_structured_feature_hook_changes_ranking(fake_embedder):
    # A behavioral signal: recency (lower last_active_days = better).
    recency = StructuredFeature(
        name="recency",
        fn=lambda job, c: 1.0 / (1.0 + c.metadata.get("last_active_days", 999)),
    )
    fx = FeatureExtractor(
        embedder=fake_embedder, skill_vocab=SKILL_VOCAB,
        structured_features=[recency],
    )
    assert fx.feature_names[-1] == "recency"
    cands = _candidates()
    cands[1].metadata["last_active_days"] = 0  # weak candidate is very active
    names, X, ids = fx.extract(_job(), cands)
    assert X.shape == (3, 8)
    recency_col = dict(zip(ids, X[:, 7]))
    assert recency_col["weak"] == pytest.approx(1.0)


def test_structured_feature_missing_field_uses_default(fake_embedder):
    sf = StructuredFeature(name="tenure", fn=lambda job, c: c.metadata["years"], default=-1.0)
    fx = FeatureExtractor(embedder=fake_embedder, skill_vocab=SKILL_VOCAB,
                          structured_features=[sf])
    names, X, ids = fx.extract(_job(), [_candidates()[0]])  # no "years" key
    assert X[0, -1] == pytest.approx(-1.0)


def test_supervised_fit_save_load_roundtrip(tmp_path, fake_embedder):
    pytest.importorskip("xgboost")
    fx = FeatureExtractor(embedder=fake_embedder, skill_vocab=SKILL_VOCAB)
    ranker = MultiSignalRanker(feature_extractor=fx)
    job = _job()
    cands = _candidates()
    examples = [
        TrainingExample(job, cands[0], 1.0),
        TrainingExample(job, cands[1], 0.0),
        TrainingExample(job, cands[2], 1.0),
    ]
    info = ranker.fit(examples)
    assert info["n_examples"] == 3 and info["n_jobs"] == 1
    assert ranker.model is not None

    path = ranker.save(tmp_path / "ranker.joblib")
    fx2 = FeatureExtractor(embedder=fake_embedder, skill_vocab=SKILL_VOCAB)
    loaded = MultiSignalRanker.load(path, feature_extractor=fx2)
    assert loaded.feature_names == ranker.feature_names

    r1 = ranker.rank_candidates(job, cands)
    r2 = loaded.rank_candidates(job, cands)
    assert [c.id for c in r1] == [c.id for c in r2]


def test_load_rejects_schema_mismatch(tmp_path, fake_embedder):
    pytest.importorskip("xgboost")
    fx = FeatureExtractor(embedder=fake_embedder, skill_vocab=SKILL_VOCAB)
    ranker = MultiSignalRanker(feature_extractor=fx)
    job = _job()
    labels = [1.0, 0.0, 1.0]
    ranker.fit([TrainingExample(job, c, y) for c, y in zip(_candidates(), labels)])
    path = ranker.save(tmp_path / "r.joblib")
    # Extractor with an extra structured feature => different schema.
    bad = FeatureExtractor(
        embedder=fake_embedder, skill_vocab=SKILL_VOCAB,
        structured_features=[StructuredFeature("x", lambda j, c: 1.0)],
    )
    with pytest.raises(ValueError, match="feature schema mismatch"):
        MultiSignalRanker.load(path, feature_extractor=bad)
