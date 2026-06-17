"""Pre-computation step: download models and warm the embedding cache.

The submission spec allows pre-computation to exceed the 5-minute window, but
the ranking step that produces the CSV must run offline and within budget.
This script does the one-time heavy lifting so `rank.py` is fast and
network-free:

  1. Downloads the SBERT model (and, with --cross-encoder, the cross-encoder)
     into the local HuggingFace cache.
  2. Optionally warms the FAIMR on-disk embedding cache for the candidate pool,
     so the subsequent `rank.py` run reads embeddings from disk instead of
     re-encoding.

Usage:
    python prepare.py                                  # just fetch the model(s)
    python prepare.py --candidates ./candidates.jsonl  # also warm the pool cache
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_logger
from ir.adapters import load_candidates
from ir.features import SbertEmbedder
from ir.jd_requirements import REDROB_SENIOR_AI_ENGINEER

logger = get_logger("prepare")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download models + warm the embedding cache.")
    ap.add_argument("--candidates", default=None, help="optional candidates.jsonl(.gz) to pre-encode")
    ap.add_argument("--cross-encoder", action="store_true", help="also fetch the cross-encoder model")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    embedder = SbertEmbedder()
    logger.info("Downloading / loading SBERT model ...")
    embedder.encode({"__probe__": REDROB_SENIOR_AI_ENGINEER.semantic_target})  # triggers download
    logger.info("SBERT ready (%.0fs)", time.time() - t0)

    if args.cross_encoder:
        from ranking.cross_encoder_ranker import CrossEncoderRanker
        logger.info("Downloading / loading cross-encoder model ...")
        _ = CrossEncoderRanker().model
        logger.info("Cross-encoder ready (%.0fs)", time.time() - t0)

    if args.candidates:
        logger.info("Warming embedding cache for the candidate pool ...")
        candidates = list(load_candidates(args.candidates, limit=args.limit))
        to_encode = {"__jd__": REDROB_SENIOR_AI_ENGINEER.semantic_target}
        to_encode.update({c.id: c.text for c in candidates})
        embedder.encode(to_encode)  # cached to data/embeddings_cache/
        logger.info("Encoded + cached %d profiles (%.0fs total)", len(candidates), time.time() - t0)

    print(f"Pre-computation done in {time.time() - t0:.0f}s. rank.py will now run offline.")


if __name__ == "__main__":
    main()
