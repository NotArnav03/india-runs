"""India Runs — intelligent candidate discovery (Track 1).

Hackathon-specific code built on top of the vendored FAIMR core
(``embeddings/``, ``ranking/``, ``evaluation/``, ``preprocessing/``).

Public surface:
    - features.FeatureExtractor   multi-signal feature extraction with a
                                  pluggable hook for structured/behavioral
                                  profile signals.
    - features.Job, features.Candidate   schema-independent input types.
    - ranker.MultiSignalRanker    the single inference entrypoint:
                                  fit / save / load / rank_candidates.
"""

from ir.features import Candidate, FeatureExtractor, Job, StructuredFeature
from ir.ranker import MultiSignalRanker, RankedCandidate
from ir.adapters import load_candidates, write_submission, candidate_from_raw
from ir.jd_requirements import JDRequirements, REDROB_SENIOR_AI_ENGINEER
from ir.features_library import build_feature_library, hard_disqualifier_penalty
from ir.honeypot import consistency_report, ConsistencyReport
from ir.reasoning import build_reasoning, derive_concerns

__all__ = [
    "Candidate",
    "FeatureExtractor",
    "Job",
    "StructuredFeature",
    "MultiSignalRanker",
    "RankedCandidate",
    "load_candidates",
    "write_submission",
    "candidate_from_raw",
    "JDRequirements",
    "REDROB_SENIOR_AI_ENGINEER",
    "build_feature_library",
    "hard_disqualifier_penalty",
    "consistency_report",
    "ConsistencyReport",
    "build_reasoning",
    "derive_concerns",
]
