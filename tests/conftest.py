"""Shared test fixtures."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

# Repo root on sys.path so `config`, `ir`, `ranking` resolve like in prod.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEmbedder:
    """Deterministic bag-of-words embedder — cosine reflects token overlap.

    Lets the suite exercise the full ranking path without downloading a
    transformer model.  Builds a vocabulary on the fly per ``encode`` call
    and returns L2-normalizable term-frequency vectors over the union vocab.
    """

    _WORD = re.compile(r"\b\w{2,}\b")

    def encode(self, texts: dict[str, str]) -> dict[str, np.ndarray]:
        toks = {k: self._WORD.findall(v.lower()) for k, v in texts.items()}
        vocab = sorted({t for ts in toks.values() for t in ts})
        index = {t: i for i, t in enumerate(vocab)}
        out = {}
        for k, ts in toks.items():
            vec = np.zeros(len(vocab), dtype=float)
            for t in ts:
                vec[index[t]] += 1.0
            out[k] = vec
        return out

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
