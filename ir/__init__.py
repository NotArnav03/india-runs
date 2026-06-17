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

__all__ = [
    "Candidate",
    "FeatureExtractor",
    "Job",
    "StructuredFeature",
    "MultiSignalRanker",
    "RankedCandidate",
]
