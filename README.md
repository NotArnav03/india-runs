# India Runs — Intelligent Candidate Discovery (Track 1)

Hackathon entry for **India Runs by Redrob AI**, Track 1 (The Data & AI
Challenge). Given one job description and a pool of 100,000 candidate profiles,
we return a **ranked top-100 shortlist** — judging fit by deep job
understanding, contextual relevance beyond keywords, and the integration of
profile, career, and behavioral signals.

Built on a vendored, license-clean subset of the **FAIMR** ranking core (SBERT,
cross-encoder, anchored skill matcher, evaluation metrics), with a new,
schema-independent inference and reasoning layer on top.

---

## TL;DR

- **One adversarial JD, 100K candidates, hidden tiered ground truth.** Scored
  on `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`; output is a
  validator-exact top-100 CSV.
- **Multi-signal ranking, not keyword/embedding matching.** Semantic fit is
  deliberately *down-weighted* — the JD says keyword matching is a trap. Career
  evidence, demonstrated skill, and behavioral availability carry the weight.
- **Trap-aware.** Hard JD disqualifiers and an internal profile-consistency
  check act as multiplicative guards, so keyword-stuffers, off-target roles,
  consulting-only careers, and the dataset's ~80 planted honeypots sink.
- **Fits the budget.** Rank step is **~80 s for the full 100K on CPU, offline**,
  ≤ 0.3 GB heap — well inside the 5-min / 16-GB limit.
- **Defensible reasoning.** Every shortlist row carries a 1–2 sentence
  justification built only from facts in the profile (no hallucination), with
  honest concerns surfaced and tone matched to rank.

Reproduce the submission:

```bash
pip install -r requirements.txt
python prepare.py --candidates ./candidates.jsonl                       # one-time precompute
python rank.py    --candidates ./candidates.jsonl --out ./submission.csv  # ≤5 min, CPU, offline
```

---

## How it works (methodology)

The ranker scores every candidate against the JD by blending complementary
signals, then applies hard guards. All signals are normalized so weights stay
interpretable; **raw semantic/keyword similarity is intentionally light** so the
ranker can't be gamed by buzzword-stuffed skill lists (an explicit trap in the
JD and dataset).

1. **JD requirement model** (`ir/jd_requirements.py`) — the single JD is
   modeled explicitly and auditably: must-have skills, nice-to-haves, hard
   disqualifiers, "shipped-systems" evidence phrases, target locations, and
   notice preference. Every field is traceable to `job_description.docx`.

2. **Semantic fit** — FAIMR's SBERT bi-encoder (`all-MiniLM-L6-v2`) matches the
   JD against each profile's *career narrative* (summary + role descriptions),
   so a candidate who "built the system that ranks what users see" reads as a
   strong fit even without writing "RAG" or "Pinecone".

3. **Anchored skill match** — FAIMR's look-around-anchored matcher (so "java"
   ≠ "javascript"), down-weighted because presence ≠ proficiency.

4. **Career + behavioral feature library** (`ir/features_library.py`) — 10
   features over the 23 Redrob signals and career history: shipped-systems
   evidence, product-vs-services trajectory, role coherence, skill credibility
   (proficiency × endorsements × usage × assessment scores), experience-band
   fit, tenure stability, availability composite, recruiter engagement, GitHub
   activity (with `-1` "no GitHub" sentinel handled), and location fit.

5. **Hard guards** (`ir/honeypot.py`, `ir/features_library.py`) — JD
   disqualifiers (consulting-only, pure research, CV/speech/robotics without
   NLP-IR, off-target roles) and a within-profile consistency check (e.g.
   "expert" skill with 0 months of use; tenure exceeding the role's date span)
   are applied as **multiplicative penalties**, so a keyword-perfect but
   disqualified or impossible profile cannot be rescued by similarity.

6. **Fact-grounded reasoning** (`ir/reasoning.py`) — a 1–2 sentence
   justification composed only from facts in the profile, with valid JD-fit
   concerns surfaced and tone consistent with the rank. Honeypot detection is
   never revealed in the output — traps are silently down-ranked.

7. **Validator-exact export** (`ir/adapters.py`) — emits the top-100 CSV under
   the official rules (unique ranks, non-increasing score, tie-break by
   `candidate_id`); the official `validate_submission.py` accepts it.

The whole pipeline is the single entrypoint `MultiSignalRanker`
(`ir/ranker.py`), runnable on arbitrary in-memory input.

---

## Repository layout

```
india-runs/
├── rank.py                 # single reproduce command → submission.csv
├── prepare.py              # one-time precompute: model download + embedding cache
├── app.py                  # Streamlit sandbox / demo (submission spec §10.5)
├── submission_metadata.yaml
├── Dockerfile              # self-contained sandbox image
├── ir/                     # hackathon inference + reasoning layer
│   ├── ranker.py           #   MultiSignalRanker: rank_candidates / fit / save / load
│   ├── features.py         #   feature extraction + pluggable structured-signal hook
│   ├── features_library.py #   10 career/behavioral features + disqualifier guards
│   ├── jd_requirements.py  #   curated, auditable model of the released JD
│   ├── honeypot.py         #   within-profile consistency / honeypot detector
│   ├── reasoning.py        #   fact-grounded reasoning (no hallucination)
│   └── adapters.py         #   JSONL → Candidate + validator-exact CSV exporter
├── evaluation/             # FAIMR metrics + our local self-eval
│   ├── metrics.py          #   P@K, R@K, NDCG, MRR, MAP, ROC-AUC (vendored)
│   ├── selfeval.py         #   official composite + honeypot-rate + weight tuner
│   └── judgment_archetypes.py  # JD-grounded gold tiers (behavioral regression)
├── embeddings/             # SBERT + TF-IDF manager with disk cache (vendored)
├── ranking/                # cross-encoder + skill matcher + fairness ref (vendored)
├── preprocessing/          # text normalizer + section parser (vendored)
├── config.py               # shared config (vendored)
├── tests/                  # 34 tests, no model download (fake/hashing embedder)
└── docs/GAP_ANALYSIS.md    # the dataset-grounded gap analysis + build log
```

---

## Reproducing the submission

Pre-computation (model download + warming the on-disk embedding cache) is a
one-time step and may exceed 5 minutes. After it, the **ranking step is
CPU-only, makes no network calls, and finishes in ~80 s for the full 100K**:

```bash
python prepare.py --candidates ./candidates.jsonl                        # precompute (one-time)
python rank.py    --candidates ./candidates.jsonl --out ./submission.csv   # the scored rank step
python validate_submission.py submission.csv                             # → "Submission is valid."

# fully offline, no model download at all (small-sample sandbox / CI):
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --embedder hashing --limit 2000
```

> An optional FAIMR cross-encoder pass (`--cross-encoder`) is wired and verified
> to run, but **off by default**: on our judgment set it *lowers* the composite
> (0.98 → 0.94), because a generic relevance cross-encoder doesn't capture the
> career/behavioral reasoning this JD rewards.

### Embedders: SBERT (main) vs hashing (fallback)

The semantic signal is a swappable plugin; it only affects the *quality of the
semantic component*, which is blended with the skill match and the 10
career/behavioral features.

| | **`sbert`** — main model | **`hashing`** — model-free fallback |
|---|---|---|
| What | `all-MiniLM-L6-v2`; 384-dim dense embeddings that capture **meaning** | `HashingVectorizer`; cosine ≈ **word overlap** |
| Strength | Reads paraphrased fits even without buzzwords — beats the keyword trap | Instant, dependency-light; blind to synonyms |
| Cost | One-time download + 100K encode (precompute), offline after | None |
| Self-eval | composite **0.984** | composite **0.97** |
| Used for | **the scored submission** (`rank.py` default) | sandbox free tier, CI/tests |

The submitted `submission.csv` is **SBERT-based**. The hosted **sandbox defaults
to `hashing`** so it boots on memory-limited free tiers; flip the sidebar to
`sbert` to demo the real model where there's enough RAM for PyTorch.

---

## Validation & results

The competition ground truth is hidden, so the true score is only known after
the deadline. Locally we validate **behavior** against a JD-grounded judgment
set (`evaluation/judgment_archetypes.py`) — archetypes labeled from the JD's own
examples (ideal fit, keyword-stuffer, plain-language fit, inactive,
consulting-only, pure-research, honeypot) — using the **official composite**:

```bash
python -m evaluation.selfeval     # SBERT: composite ≈ 0.984, NDCG@10 0.984, honeypot rate 0% in top 10
```

On the real `candidates.jsonl`, the produced `submission.csv`:
- passes the official `validate_submission.py`;
- has **0 honeypot-suspect and 0 hard-disqualified profiles in the top 100**
  (the Stage-3 honeypot DQ threshold is >10%);
- ranks coherent retrieval/ranking/recsys engineers at the top with honest
  concerns (notice period, relocation, sub-band experience) surfaced lower down.

The full suite (`pytest`, **34 tests**) runs without any model download.

---

## Sandbox / deployment

`app.py` is a self-contained Streamlit demo (light "talent-intelligence"
dashboard: ranked cards with a fit-score gauge, behavioral badges, and an
expandable per-signal breakdown). Deploy options:

- **Streamlit Community Cloud** — point at this repo, main file `app.py`.
- **HuggingFace Spaces** — new Streamlit Space; push `app.py` + `requirements.txt`.
- **Docker** — `docker build -t redrob-ranker . && docker run -p 8501:8501 redrob-ranker`.

---

## Library usage

`MultiSignalRanker` is schema-independent and works on arbitrary input:

```python
from ir import Job, Candidate, FeatureExtractor, MultiSignalRanker

job = Job(id="jd1", text="Senior Python + ML engineer...",
          required_skills={"python", "machine learning", "sql"})
candidates = [Candidate(id="c1", text="..."), Candidate(id="c2", text="...")]

ranker = MultiSignalRanker(FeatureExtractor(skill_vocab=[...]))
for r in ranker.rank_candidates(job, candidates, top_k=10):
    print(r.rank, r.id, round(r.score, 4), r.signals)
```

It ranks **unsupervised** (a weighted blend of normalized signals) out of the
box — the mode we ship, since no training labels accompany the dataset — and
can switch to a learned XGBoost ranker via `fit` / `save` / `load` when labels
exist.

---

## Provenance & license

The vendored core derives from **FAIMR (MIT)**. The dataset is **not** committed
(fetch it from the hackathon portal). The GFDL-licensed name corpora and the
full fairness-audit subtree are intentionally **not** vendored here.
