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
├── ranking/              # cross-encoder + fairness re-ranker + skill matcher (vendored)
├── ir/                   # NEW hackathon code
│   ├── features.py       #   multi-signal feature extraction + structured hook
│   ├── ranker.py         #   MultiSignalRanker: fit / save / load / rank_candidates
│   ├── adapters.py       #   JSONL → Candidate + validator-exact CSV exporter
│   ├── jd_requirements.py #  curated, auditable model of the released JD
│   ├── features_library.py # 23 behavioral signals + career-pattern features
│   ├── honeypot.py       #   within-profile consistency / honeypot detector
│   └── reasoning.py      #   fact-grounded reasoning (Stage-4, no hallucination)
├── rank.py               # single reproduce command → submission.csv
├── app.py                # Streamlit sandbox demo (submission spec §10.5)
├── evaluation/
│   ├── selfeval.py       #   official composite + honeypot-rate + weight tuner
│   └── judgment_archetypes.py # JD-grounded gold tiers (behavioral regression)
├── tests/                # 33 tests, no model download (fake/hashing embedder)
└── docs/GAP_ANALYSIS.md  # what's done, what's left, mapped to code
```

## Quick start

```bash
pip install -r requirements.txt
pytest                       # full suite, no model download required
```

Pre-compute once (downloads the SBERT model + warms the on-disk embedding
cache), then produce the submission. After `prepare.py`, the rank step is
**CPU-only and makes no network calls**, and completes in **~80s for the full
100K pool** (well inside the 5-min budget):

```bash
python prepare.py --candidates ./candidates.jsonl     # pre-computation (may exceed 5 min)
python rank.py    --candidates ./candidates.jsonl --out ./submission.csv   # the scored rank step

# fully offline / no model download at all (small-sample sandbox or CI):
python rank.py --candidates ./candidates.jsonl --out ./submission.csv \
    --embedder hashing --limit 2000
```

> The FAIMR cross-encoder pass (`--cross-encoder`) is wired and verified to
> run, but it is **off by default**: on our JD-grounded judgment set it *lowers*
> the composite (0.98 → 0.94), because a generic relevance cross-encoder doesn't
> capture the career/behavioral reasoning this JD rewards. The multi-signal
> blend alone is the shipped path.

### Embedders: SBERT (main) vs hashing (fallback)

The semantic-similarity signal can be produced two ways. They are
interchangeable plugins; the embedder only affects the *quality of the semantic
signal*, which is then blended with the skill match and the 10 career/behavioral
features.

| | **`sbert`** — the main model | **`hashing`** — model-free fallback |
|---|---|---|
| What | `all-MiniLM-L6-v2` sentence-transformer; 384-dim dense embeddings that capture **meaning** | `HashingVectorizer`; cosine ≈ **word overlap**, no semantic understanding |
| Strength | Understands a "built the system that ranks what users see" profile as a retrieval/ranking fit even without the buzzwords — beats the JD's keyword trap | Fast and dependency-light, but blind to paraphrase / synonyms |
| Cost | One-time model download + 100K encode (`prepare.py`, precompute); offline thereafter | No model, no download, tiny memory, instant |
| Self-eval | composite **0.984** | composite **0.97** |
| Used for | **The real, scored submission** (`rank.py` default) | Sandbox free tier, CI/tests, quick smoke runs |

**The submitted `submission.csv` is SBERT-based** (`rank.py` defaults to
`--embedder sbert`). The hosted **sandbox defaults to `hashing`** so it boots on
memory-limited free tiers without a model download — fine, because the sandbox
is only a small-sample reproducibility check; switch the sidebar to `sbert` to
demo the real model where the host has enough RAM for PyTorch.

Validate ranking quality locally (no hidden ground truth needed — scores
against the JD-grounded behavioral judgment set using the official composite),
and launch the sandbox demo:

```bash
python -m evaluation.selfeval     # SBERT composite ≈ 0.98, honeypot rate 0% in top 10
streamlit run app.py              # local sandbox UI
```

## Sandbox / deployment

The submission requires a hosted sandbox link. `app.py` is a self-contained
Streamlit demo. Deploy it any of these ways:

- **Streamlit Community Cloud** — point it at this repo, main file `app.py`.
- **HuggingFace Spaces** — new Space (Streamlit SDK), push `app.py` +
  `requirements.txt`.
- **Docker** (the spec's self-contained option):
  ```bash
  docker build -t redrob-ranker .
  docker run -p 8501:8501 redrob-ranker
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

This repo currently delivers the keystone inference entrypoint. The Redrob
dataset has now dropped, which reshaped the plan: the task is a single
adversarial JD over a 100K pool with a hidden tiered ground truth (no labels),
scored on NDCG@10/50 + MAP + P@10, output as a top-100 CSV produced offline in
≤5 min CPU. Remaining work — the JSONL adapter + validator-exact exporter, a
curated JD-requirement model, the structured/behavioral feature library, a
honeypot/consistency detector, and a fact-grounded reasoning generator — is
tracked in `docs/GAP_ANALYSIS.md`. (The learned-ranker path is off the critical
path since no training labels ship, and FAISS isn't needed for a single JD.)

Provenance: vendored core derives from FAIMR (MIT). The GFDL-licensed name
corpora and the fairness-audit subtree are intentionally **not** vendored here.
