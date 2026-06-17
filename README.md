# India Runs — Intelligent Candidate Discovery (Track 1)

Hackathon entry for **India Runs by Redrob AI**, Track 1 (The Data & AI
Challenge): rank candidates against a job description by deep job
understanding, contextual relevance beyond keywords, and integration of
profile/career/behavioral signals — returning a fast, accurate, ranked
shortlist.

Built on a vendored, license-clean subset of the FAIMR ranking core, with a
new schema-independent inference layer on top.

## Layout

```
india-runs/
├── config.py              # shared config (vendored from FAIMR)
├── embeddings/            # SBERT + TF-IDF manager with disk cache (vendored)
├── evaluation/           # P@K, R@K, NDCG, MRR, MAP, ROC-AUC (vendored)
├── preprocessing/        # text normalizer + section parser (vendored)
├── ranking/              # fairness re-ranker + skill matcher + LTR ref (vendored)
├── ir/                   # NEW hackathon code
│   ├── features.py       #   multi-signal feature extraction + structured hook
│   └── ranker.py         #   MultiSignalRanker: fit / save / load / rank_candidates
├── tests/                # runs without downloading any model (fake embedder)
└── docs/GAP_ANALYSIS.md  # what's done, what's left, mapped to code
```

## Quick start

```bash
pip install -r requirements.txt
pytest                       # full suite, no model download required
```

```python
from ir import Job, Candidate, FeatureExtractor, MultiSignalRanker

job = Job(id="jd1", text="Senior Python + ML engineer...",
          required_skills={"python", "machine learning", "sql"})
candidates = [Candidate(id="c1", text="..."), Candidate(id="c2", text="...")]

ranker = MultiSignalRanker(FeatureExtractor(skill_vocab=[...]))
for r in ranker.rank_candidates(job, candidates, top_k=10):
    print(r.rank, r.id, round(r.score, 4), r.signals)
```

The ranker works **unsupervised** (weighted blend of normalized signals) out
of the box, and switches to a **learned XGBoost ranker** once labeled data is
available via `ranker.fit(examples)` + `ranker.save(...)` / `.load(...)`.

## Status

This repo currently delivers the keystone inference entrypoint (the piece the
gap analysis flagged as missing). Remaining work — Redrob schema adapters, a
JD-requirement extractor, the structured/behavioral feature library, vector
retrieval for scale, and the optional fairness audit — is tracked in
`docs/GAP_ANALYSIS.md`.

Provenance: vendored core derives from FAIMR (MIT). The GFDL-licensed name
corpora and the fairness-audit subtree are intentionally **not** vendored here.
