"""MultiSignalRanker — the single inference entrypoint for Track 1.

This is the keystone the gap analysis identified as missing: a callable
``rank_candidates(job, candidates) -> ranked list`` that runs the *real*
multi-signal stack on arbitrary in-memory input (closing Gap A), with a
learned model that can be persisted and reloaded (closing Gap B).

Two scoring modes:
  * Unsupervised (default): min-max normalize each signal across the
    candidate set, then take a weighted blend.  Works with zero training
    data — useful before Redrob's labels land.
  * Supervised (after ``fit``): a gradient-boosted ranker (XGBoost) learns
    the optimal signal weights from labeled (job, candidate, relevance)
    examples.  ``save`` / ``load`` persist the trained model + the exact
    feature schema so inference is reproducible.

Every ranked result carries its full per-signal trail for explainability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_logger
from ir.features import Candidate, FeatureExtractor, Job

logger = get_logger("ir.ranker")


# Default blend weights for the text signals (unsupervised mode).  Word
# counts get ~0 weight: they're scale features the *learned* model can use,
# but as a raw blend they only add noise.  Tunable via MultiSignalRanker.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "sbert_similarity": 0.45,
    "tfidf_similarity": 0.15,
    "skill_coverage": 0.25,
    "num_matched_skills": 0.05,
    "keyword_overlap": 0.10,
    "candidate_word_count": 0.0,
    "jd_word_count": 0.0,
}


@dataclass
class RankedCandidate:
    """One ranked result with its score and per-signal trail."""

    id: str
    score: float
    rank: int
    signals: dict = field(default_factory=dict)


@dataclass
class TrainingExample:
    """A labeled (job, candidate, relevance) triple for supervised fit."""

    job: Job
    candidate: Candidate
    label: float  # binary {0,1} or graded relevance


class MultiSignalRanker:
    """Schema-independent multi-signal candidate ranker."""

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        self.extractor = feature_extractor or FeatureExtractor()
        self.weights = dict(weights) if weights is not None else dict(_DEFAULT_WEIGHTS)
        self.model = None  # set by fit(); None => unsupervised blend
        self.feature_names: list[str] = self.extractor.feature_names

    # ── inference ────────────────────────────────────────────────────
    def rank_candidates(
        self,
        job: Job,
        candidates: list[Candidate],
        top_k: Optional[int] = None,
    ) -> list[RankedCandidate]:
        """Rank ``candidates`` against ``job``; return sorted RankedCandidates."""
        names, X, ids = self.extractor.extract(job, candidates)
        if len(ids) == 0:
            return []

        if self.model is not None:
            scores = self._supervised_scores(X)
        else:
            scores = self._blend_scores(names, X)

        order = np.argsort(scores)[::-1]
        ranked: list[RankedCandidate] = []
        for new_rank, idx in enumerate(order, start=1):
            ranked.append(
                RankedCandidate(
                    id=ids[idx],
                    score=float(scores[idx]),
                    rank=new_rank,
                    signals={n: float(X[idx, j]) for j, n in enumerate(names)},
                )
            )
        return ranked[:top_k] if top_k is not None else ranked

    def _blend_scores(self, names: list[str], X: np.ndarray) -> np.ndarray:
        """Min-max normalize each column, then weighted sum."""
        if X.shape[0] == 1:
            # Single candidate: normalization is degenerate; use raw weighted
            # sum so the lone score is still meaningful and deterministic.
            w = np.array([self.weights.get(n, 0.0) for n in names])
            return X @ w
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        span = np.where(col_max > col_min, col_max - col_min, 1.0)
        norm = (X - col_min) / span
        w = np.array([self.weights.get(n, 0.0) for n in names])
        return norm @ w

    def _supervised_scores(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        # Binary classifier: P(relevant). Multi-class graded: expected grade.
        if proba.shape[1] == 2:
            return proba[:, 1]
        grades = np.arange(proba.shape[1])
        return proba @ grades

    # ── training ─────────────────────────────────────────────────────
    def fit(
        self,
        examples: list[TrainingExample],
        model_type: str = "xgboost",
        **model_kwargs,
    ) -> dict:
        """Train a supervised ranker on labeled examples.

        Groups examples by job, extracts features per job (so batch SBERT /
        TF-IDF semantics match inference exactly), then fits a gradient
        boosted classifier.  Stores the model and freezes ``feature_names``.
        """
        from collections import defaultdict

        by_job: dict[str, list[TrainingExample]] = defaultdict(list)
        jobs: dict[str, Job] = {}
        for ex in examples:
            by_job[ex.job.id].append(ex)
            jobs[ex.job.id] = ex.job

        feature_rows, labels = [], []
        names: list[str] = self.extractor.feature_names
        for job_id, exs in by_job.items():
            cand_list = [ex.candidate for ex in exs]
            names, X, ids = self.extractor.extract(jobs[job_id], cand_list)
            id_to_label = {ex.candidate.id: ex.label for ex in exs}
            for j, cid in enumerate(ids):
                feature_rows.append(X[j])
                labels.append(id_to_label[cid])

        X_all = np.asarray(feature_rows, dtype=float)
        y_all = np.asarray(labels)
        self.feature_names = names

        if model_type == "xgboost":
            import xgboost as xgb
            params = dict(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric="logloss",
            )
            params.update(model_kwargs)
            self.model = xgb.XGBClassifier(**params)
        elif model_type == "lightgbm":
            import lightgbm as lgb
            params = dict(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
            )
            params.update(model_kwargs)
            self.model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"unknown model_type {model_type!r}")

        self.model.fit(X_all, y_all)
        importance = dict(
            zip(self.feature_names, self.model.feature_importances_.tolist())
        )
        logger.info("Trained %s ranker on %d examples across %d jobs",
                    model_type, len(y_all), len(by_job))
        return {
            "n_examples": int(len(y_all)),
            "n_jobs": len(by_job),
            "feature_importance": importance,
        }

    # ── persistence (closes Gap B) ───────────────────────────────────
    def save(self, path: str | Path) -> Path:
        """Persist the trained model + feature schema + blend weights."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "weights": self.weights,
            },
            path,
        )
        logger.info("Saved ranker -> %s", path)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> "MultiSignalRanker":
        """Load a persisted ranker.

        A ``feature_extractor`` must be supplied to reconstruct the exact
        signals (embedder + skill vocab + structured features) used at
        train time — its ``feature_names`` are checked against the saved
        schema to catch silent drift.
        """
        import joblib
        blob = joblib.load(Path(path))
        ranker = cls(feature_extractor=feature_extractor, weights=blob["weights"])
        ranker.model = blob["model"]
        ranker.feature_names = blob["feature_names"]
        if feature_extractor is not None:
            if feature_extractor.feature_names != ranker.feature_names:
                raise ValueError(
                    "feature schema mismatch between saved model and the "
                    f"provided extractor:\n  saved:    {ranker.feature_names}\n"
                    f"  extractor: {feature_extractor.feature_names}"
                )
        return ranker
