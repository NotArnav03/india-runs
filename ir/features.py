"""Multi-signal feature extraction for candidate ranking.

Produces a named feature vector for every (job, candidate) pair.  Text
signals reuse FAIMR's vetted primitives (SBERT semantics, TF-IDF, the
look-around-anchored skill matcher).  Structured / behavioral signals
(career metadata, tenure, seniority, engagement, ...) are *pluggable*
via :class:`StructuredFeature`, so the Redrob profile schema can be wired
in without touching the ranker core.

This directly addresses two gaps from the gap analysis:
  * Gap C — JD requirements are no longer assumed to come from a lookup
    table; ``Job.required_skills`` is provided by the caller (or derived
    from JD text via the shared skill vocabulary).
  * Gap D — structured/behavioral features are first-class, not text-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_logger
from ranking.ranking_utils import extract_skills_in_text

logger = get_logger("ir.features")


# ─── Schema-independent input types ──────────────────────────────────────

@dataclass
class Job:
    """A job opening to rank candidates against.

    Attributes:
        id: stable identifier.
        text: the full job description (free text).
        required_skills: optional explicit required-skill set.  When None,
            skills are derived from ``text`` using the feature extractor's
            skill vocabulary (see :class:`FeatureExtractor`).
        metadata: arbitrary structured fields (seniority, location, ...)
            consumed by registered structured features.
    """

    id: str
    text: str
    required_skills: Optional[set[str]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Candidate:
    """A candidate profile.

    Attributes:
        id: stable identifier.
        text: resume / profile free text.
        skills: optional explicit candidate-skill set.  When None, skills
            are derived from ``text``.
        metadata: arbitrary structured/behavioral fields (years_experience,
            current_title, last_active_days, ...) consumed by registered
            structured features.
    """

    id: str
    text: str
    skills: Optional[set[str]] = None
    metadata: dict = field(default_factory=dict)


# ─── Embedder protocol (injectable for testing) ──────────────────────────

class Embedder(Protocol):
    """Minimal embedding interface the extractor depends on.

    The production implementation wraps FAIMR's ``EmbeddingManager``
    (SBERT).  Tests inject a lightweight fake so the suite never has to
    download a transformer model.
    """

    def encode(self, texts: dict[str, str]) -> dict[str, np.ndarray]: ...

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float: ...


class SbertEmbedder:
    """Default embedder backed by FAIMR's cached SBERT EmbeddingManager."""

    def __init__(self, manager=None):
        self._manager = manager

    @property
    def manager(self):
        if self._manager is None:
            from embeddings.embedding_manager import EmbeddingManager
            self._manager = EmbeddingManager()
        return self._manager

    def encode(self, texts: dict[str, str]) -> dict[str, np.ndarray]:
        return self.manager.encode_sbert(texts, use_cache=True)

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        from embeddings.embedding_manager import EmbeddingManager
        return EmbeddingManager.cosine_similarity(a, b)


# ─── Structured / behavioral feature plug-in ─────────────────────────────

@dataclass
class StructuredFeature:
    """A named, pluggable feature computed from structured metadata.

    ``fn`` receives the :class:`Job` and :class:`Candidate` and returns a
    float.  Missing fields should be handled inside ``fn`` (return
    ``default``); the extractor never inspects metadata itself, so the
    Redrob schema lives entirely in these functions.
    """

    name: str
    fn: Callable[[Job, Candidate], float]
    default: float = 0.0

    def compute(self, job: Job, candidate: Candidate) -> float:
        try:
            value = self.fn(job, candidate)
        except (KeyError, TypeError, ValueError):
            return self.default
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return self.default
        return float(value)


_TEXT_FEATURE_NAMES = [
    "sbert_similarity",
    "tfidf_similarity",
    "skill_coverage",
    "num_matched_skills",
    "keyword_overlap",
    "candidate_word_count",
    "jd_word_count",
]

_WORD_RE = re.compile(r"\b\w{3,}\b")


class FeatureExtractor:
    """Builds the (n_candidates, n_features) matrix for a single job.

    Args:
        embedder: an :class:`Embedder`; defaults to SBERT via EmbeddingManager.
        skill_vocab: global skill vocabulary used to derive skills from text
            when a Job/Candidate doesn't carry an explicit skill set.
        structured_features: ordered list of :class:`StructuredFeature`
            appended after the text features.  This is the extension point
            for career-metadata / behavioral signals.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        skill_vocab: Optional[Sequence[str]] = None,
        structured_features: Optional[list[StructuredFeature]] = None,
    ):
        self.embedder: Embedder = embedder or SbertEmbedder()
        self.skill_vocab = [s.lower() for s in (skill_vocab or [])]
        self.structured_features = list(structured_features or [])

    @property
    def feature_names(self) -> list[str]:
        return _TEXT_FEATURE_NAMES + [sf.name for sf in self.structured_features]

    # ── skill helpers ────────────────────────────────────────────────
    def _job_skills(self, job: Job) -> set[str]:
        if job.required_skills is not None:
            return {s.lower() for s in job.required_skills}
        if self.skill_vocab:
            return extract_skills_in_text(self.skill_vocab, job.text)
        return set()

    def _candidate_skills(self, candidate: Candidate) -> set[str]:
        if candidate.skills is not None:
            return {s.lower() for s in candidate.skills}
        if self.skill_vocab:
            return extract_skills_in_text(self.skill_vocab, candidate.text)
        return set()

    # ── main entry point ─────────────────────────────────────────────
    def extract(
        self, job: Job, candidates: list[Candidate],
    ) -> tuple[list[str], np.ndarray, list[str]]:
        """Return (feature_names, X, candidate_ids).

        Embeddings and TF-IDF are computed in batch for the whole candidate
        set, so this is a single SBERT pass and a single TF-IDF fit per job.
        """
        if not candidates:
            return self.feature_names, np.empty((0, len(self.feature_names))), []

        ids = [c.id for c in candidates]

        # SBERT: one batch encode for the JD + all candidates, then a single
        # vectorized cosine over the whole pool.  A per-candidate Python loop
        # here (one cosine call each) does not scale — at 100K candidates the
        # call overhead alone blows the rank-step budget.  cosine_similarity
        # handles both dense (SBERT) and sparse (hashing) rows in one shot.
        to_encode = {"__jd__": job.text}
        to_encode.update({c.id: c.text for c in candidates})
        embeddings = self.embedder.encode(to_encode)
        jd_emb = embeddings["__jd__"]
        sbert_sims = self._batch_cosine(jd_emb, [embeddings[c.id] for c in candidates], ids)

        tfidf_sims = self._tfidf_similarities(job, candidates)

        jd_skills = self._job_skills(job)
        jd_words = set(_WORD_RE.findall(job.text.lower()))
        jd_wc = len(job.text.split())

        rows = []
        for c in candidates:
            cand_skills = self._candidate_skills(c)
            matched = jd_skills & cand_skills if jd_skills else set()
            coverage = len(matched) / len(jd_skills) if jd_skills else 0.0

            cand_words = set(_WORD_RE.findall(c.text.lower()))
            kw_overlap = len(jd_words & cand_words) / len(jd_words) if jd_words else 0.0

            row = [
                sbert_sims[c.id],
                tfidf_sims[c.id],
                coverage,
                float(len(matched)),
                kw_overlap,
                float(len(c.text.split())),
                float(jd_wc),
            ]
            row.extend(sf.compute(job, c) for sf in self.structured_features)
            rows.append(row)

        return self.feature_names, np.asarray(rows, dtype=float), ids

    @staticmethod
    def _batch_cosine(jd_emb, cand_embs: list, ids: list[str]) -> dict[str, float]:
        """Vectorized cosine of the JD against every candidate embedding.

        Stacks the candidate vectors once and calls sklearn
        ``cosine_similarity`` a single time.  Works for dense numpy arrays
        (SBERT / fake embedder) and sparse rows (hashing embedder).
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if not cand_embs:
            return {}
        if hasattr(cand_embs[0], "toarray"):  # sparse rows (HashingVectorizer)
            from scipy.sparse import vstack as sparse_vstack
            matrix = sparse_vstack(cand_embs)
            jd = jd_emb if hasattr(jd_emb, "toarray") else np.asarray(jd_emb).reshape(1, -1)
        else:
            matrix = np.vstack([np.asarray(e).ravel() for e in cand_embs])
            jd = np.asarray(jd_emb).reshape(1, -1)
        sims = cosine_similarity(jd, matrix).ravel()
        return {cid: float(s) for cid, s in zip(ids, sims)}

    def _tfidf_similarities(
        self, job: Job, candidates: list[Candidate],
    ) -> dict[str, float]:
        """Cosine TF-IDF similarity of each candidate to the JD.

        Fits a fresh vectorizer on [JD] + candidate texts (no disk cache;
        cheap and keeps the extractor independent of the SBERT backend).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [job.text] + [c.text for c in candidates]
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        try:
            matrix = vec.fit_transform(corpus)
        except ValueError:
            # Empty vocabulary (e.g. all stop-words) — degrade gracefully.
            return {c.id: 0.0 for c in candidates}
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return {c.id: float(s) for c, s in zip(candidates, sims)}
