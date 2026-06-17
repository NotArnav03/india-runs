# Track 1 Gap Analysis & Work Plan

Maps the India Runs Track 1 problem statement ("Intelligent Candidate
Discovery") to concrete code, and tracks what's built vs. outstanding.
See `docs/HACKATHON_BRIEF.md` for the source problem statement, deadlines,
and prizes this plan is derived from.

## Judging axes (inferred from the problem statement)

1. **Deep Job Understanding** — interpret complex JDs
2. **Contextual Relevance** — semantic fit beyond keywords
3. **Signal Integration** — profile attributes, career metadata, behavioral signals
4. **Fast + accurate + expertly ranked shortlist**
5. **Deliverables** — GitHub repo + README + ranked output in a predefined format

## Where these gaps came from

The gaps below were found by auditing the upstream FAIMR repo
(`C:\Projects\faimr`) against the Track 1 ask. The headline finding:

> The "multi-signal ranking" that makes FAIMR impressive lives only in
> offline batch-evaluation scripts that read pre-built CSVs and print
> metrics. The one path that ranks arbitrary input — the `/rank` API
> endpoint — was **plain SBERT cosine similarity and nothing else**
> (`faimr/api/server.py:538-547`). So the distance between "what FAIMR
> demos" and "what Track 1 needs shipped" was larger than its README
> suggested.

`file:line` citations in the rationale column point into the upstream
FAIMR tree, not this repo.

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

## Rationale & evidence per gap

Citations point into the upstream FAIMR tree (`C:\Projects\faimr`).

- **A — No inference entrypoint for the multi-signal stack.** Hybrid /
  cross-encoder / LTR were batch scripts that read labeled-pair CSVs and
  print metrics, not a callable `score(jd, candidates)`. The live `/rank`
  endpoint was SBERT-cosine only (`api/server.py:523-557`), so none of the
  sophistication was reachable on arbitrary input. **Fixed** by
  `ir/ranker.py:MultiSignalRanker.rank_candidates`.
- **B — Trained LTR was never persisted.** `LearningToRank.train_and_evaluate`
  fit in-memory and `predict()` required the same process; no ranker
  `model.pkl` existed (`ranking/learning_to_rank.py:218,244-248`). **Fixed**
  by `MultiSignalRanker.save/load` (joblib, with feature-schema check).
- **C — JD skills came from a lookup table, not the JD text.** JD skills were
  joined from LinkedIn's `job_skills.csv` keyed by `job_id`
  (`ranking/ranking_utils.py:134-146`); resume skills *were* extracted from
  text (`ranking_utils.py:74-83`) but a JD given as raw text had no
  skill/requirement extractor. Half the "job understanding" signal silently
  evaluated to empty on novel JDs. **Mitigated**: `Job.required_skills` is now
  caller-provided or derived from text via the skill vocab; a real extractor
  (`ir/jd_understanding.py`) is still todo.
- **D — Features were 100% text-derived.** The 7 LTR features were all text
  (SBERT/TF-IDF sim, skill coverage, matched count, keyword overlap, two word
  counts) — `ranking/learning_to_rank.py:53-61`. Zero handling of tenure,
  seniority, trajectory, recency, or behavioral/engagement signals, which
  Track 1 names explicitly. **Scaffolded** via `ir/features.py:StructuredFeature`
  (pluggable); the concrete feature library is schema-blocked.
- **E — Input/output schema mismatch.** FAIMR expects resumes as `.txt` files
  on disk, JDs in `postings_balanced.csv`, skills via two mapping CSVs, pairs
  as labeled CSVs (`ranking/ranking_utils.py:104-159`;
  `ingestion/resume_ingestion.py`). Redrob will hand a different schema, and
  there is **no ranked-shortlist exporter** today (outputs are console prints
  / API JSON). Both schema-blocked until the dataset drops.
- **F — Eval is binary-label pair classification.** Metrics exist
  (`evaluation/metrics.py` has NDCG/MRR/MAP) but the harness is built around
  synthetic 0/1 pair labels (`ranking/ranking_utils.py:189-247`), not graded
  relevance on a held-out shortlist. Wire in Redrob's ground truth when known.
- **Scale — brute-force cosine.** Ranking scores all candidates by brute-force
  cosine (`api/server.py:541-547`); fine for thousands, not "oceans of
  profiles." Add precomputed embeddings + an ANN index (FAISS) for
  retrieve-then-rerank.
- **G — Fairness is FAIMR's strength but NOT what Track 1 asks for.** The
  calibrated audit (dual AIR, Wilson CIs, drift, counterfactual robustness)
  is a *differentiator*, not a requirement — lead the pitch with ranking
  accuracy and position fairness as the enterprise/EEOC-defensible moat. Its
  subtree drags GFDL-licensed name corpora + `model.pkl`, deliberately not
  vendored here yet.

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
