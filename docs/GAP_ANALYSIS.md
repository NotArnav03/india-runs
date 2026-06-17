# Track 1 Gap Analysis & Work Plan

Maps the India Runs Track 1 problem statement ("Intelligent Candidate
Discovery") to concrete code, and tracks what's built vs. outstanding.

## Judging axes (inferred from the problem statement)

1. **Deep Job Understanding** — interpret complex JDs
2. **Contextual Relevance** — semantic fit beyond keywords
3. **Signal Integration** — profile attributes, career metadata, behavioral signals
4. **Fast + accurate + expertly ranked shortlist**
5. **Deliverables** — GitHub repo + README + ranked output in a predefined format

## Status

| # | Item | Severity | Status | Where |
|---|------|----------|--------|-------|
| A | Single multi-signal inference entrypoint (`rank(job, candidates)`) | CRITICAL | ✅ done | `ir/ranker.py` |
| B | Trained ranker persistence (save/load) | HIGH | ✅ done | `ir/ranker.py:save/load` |
| D | Structured/behavioral feature hook | CRITICAL | ✅ scaffolded | `ir/features.py:StructuredFeature` |
| C | JD requirement/skill extraction from text | CRITICAL | ⬜ todo | `ir/jd_understanding.py` (new) |
| E | Redrob schema in/out adapter | HIGH | ⬜ todo (schema-blocked) | `ir/adapters.py` (new) |
| — | Ranked-shortlist exporter (their format) | HIGH | ⬜ todo (schema-blocked) | `ir/adapters.py` |
| F | Graded-relevance eval harness on held-out shortlist | MEDIUM | ⬜ todo | reuse `evaluation/metrics.py` |
| — | Retrieve-then-rerank vector index (FAISS) for scale | MEDIUM | ⬜ todo | `ir/retrieval.py` (new) |
| — | Concrete career/behavioral feature library | CRITICAL | ⬜ todo (schema-blocked) | `ir/features_library.py` (new) |
| G | Fairness audit + EEOC re-ranker (differentiator) | LOW | ⬜ todo | vendor FAIMR `fairness/` + corpora |

## Done this iteration

- `ir/features.py`: batch SBERT + TF-IDF + skill-coverage + keyword-overlap
  features, with a pluggable `StructuredFeature` list as the extension point
  for career/behavioral signals (Gap D). Embedder is injectable.
- `ir/ranker.py`: `MultiSignalRanker` with unsupervised weighted blend and
  supervised XGBoost mode; `fit` / `save` / `load`; per-signal explainability
  trail on every result (Gaps A + B).
- `tests/`: full path tested with a deterministic fake embedder — no model
  download needed.

## Blocked until the Redrob dataset drops

Items E, the exporter, the concrete feature library, and G's calibration all
depend on the actual profile/JD schema and whether ground-truth relevance is
provided. Build order once data lands: adapter → JD extractor → feature
library → eval harness → (retrieval if corpus is large) → fairness audit.
