# Track 1 Gap Analysis & Work Plan

Maps the India Runs Track 1 problem statement ("Intelligent Candidate
Discovery") to concrete code, and tracks what's built vs. outstanding.
See `docs/HACKATHON_BRIEF.md` for the event-level facts.

> **Status of this document — updated 2026-06-17.** The Redrob dataset has
> dropped. Everything previously marked "schema-blocked until the dataset
> drops" is now unblocked, and the real data **invalidates several
> assumptions** the earlier plan was built on. This revision re-grounds the
> whole analysis in the released bundle. Dataset citations below point into
> `data/raw/.../India_runs_data_and_ai_challenge/` (gitignored); repo
> citations point into this tree.

## The task, exactly as the data defines it

The released bundle (`candidate_schema.json`, `job_description.docx`,
`submission_spec.docx`, `redrob_signals_doc.docx`, `validate_submission.py`,
`candidates.jsonl`, `sample_candidates.json`) pins down the task much more
tightly than the public problem statement did:

- **One job, not many.** There is a single JD (`job_description.docx`): a
  "Senior AI Engineer — Founding Team" role at Redrob. We rank against this
  one JD only. There is no stream of arbitrary JDs to generalize over.
- **100,000 candidates** (`wc -l candidates.jsonl` = 100000), rich structured
  profiles (`candidate_schema.json`): `profile`, `career_history[]`,
  `education[]`, `skills[]` (with `proficiency`/`endorsements`/
  `duration_months`), `certifications[]`, `languages[]`, and a 23-field
  `redrob_signals` object (`redrob_signals_doc.docx`).
- **Output**: a CSV of the **top 100** candidates only — columns
  `candidate_id,rank,score,reasoning`; ranks 1–100 each used once; `score`
  non-increasing with rank; ties broken by `candidate_id` ascending
  (`validate_submission.py`, `submission_spec.docx` §2–3).
- **Scoring** (hidden ground truth, revealed only after close,
  `submission_spec.docx` §4):
  `composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`.
  Ground truth is a **graded relevance tier** per candidate; `P@10` counts
  tier ≥ 3 as relevant. **The top 10 dominate** (NDCG@10 alone is half the score).
- **Compute budget for the ranking step** (`submission_spec.docx` §3):
  ≤ 5 min wall-clock, ≤ 16 GB RAM, **CPU only, no network** (no hosted LLM
  calls), ≤ 5 GB disk. Pre-computation (e.g. embedding the pool) may exceed
  5 min, but the step that emits the CSV must fit the budget.

## What the data changes about the old plan — four headline findings

These supersede the previous "schema-blocked" framing.

1. **No labels ship with the data → the learned XGBoost path is not the
   competition path.** The bundle contains *no* ground-truth/relevance file;
   scoring happens once, server-side, against a hidden truth
   (`submission_spec.docx` §4, §8). So `MultiSignalRanker.fit()`
   (`ir/ranker.py:135`) has nothing to train on. The shippable system must be
   a **principled unsupervised / heuristic scoring function** over the
   signals, not a supervised ranker. `fit/save/load` remain useful only for
   optional *weak-supervision* experiments (e.g. pseudo-labels from a
   curated mini-judgment set) — they are no longer on the critical path.

2. **One JD, fully readable → hand-model its requirements; don't build a
   generic extractor.** The earlier Gap C ("extract requirements from
   arbitrary JD text") is the wrong shape for a single, known, deliberately
   adversarial JD. The JD *spells out* its must-haves, nice-to-haves, and —
   unusually — explicit **disqualifiers** and a "read between the lines"
   ideal profile. The high-value work is a curated, auditable JD-requirement
   model (must/nice/negative + career patterns + location + notice), not NLP
   extraction.

3. **The dataset is adversarial; keyword/embedding match alone is a trap.**
   The JD says so directly: *"The right answer is not 'find candidates whose
   skills section contains the most AI keywords' — that's a trap we've
   explicitly built into the dataset."* Three trap classes to defeat:
   - **Keyword stuffers / wrong-role**: e.g. a "Marketing Manager" with every
     AI skill listed — not a fit (JD, final section).
   - **Plain-language true fits ("Tier 5")**: someone who built a recsys at a
     product company but never writes "RAG"/"Pinecone" — *is* a fit. Reward
     **career-history evidence** over skill-list keywords.
   - **~80 honeypots** with subtly impossible profiles (e.g. 8 yrs at a
     company founded 3 yrs ago; "expert" in 10 skills with 0 months used),
     forced to relevance tier 0. **Honeypot rate > 10% in the top 100
     disqualifies at Stage 3** (`submission_spec.docx` §7). This makes
     profile-consistency checking a *correctness* requirement, not a nicety.

4. **The `reasoning` column is graded, and must be produced with no network.**
   Stage 4 samples 10 rows and checks reasoning for: specific profile facts,
   explicit JD connection, honest acknowledgement of concerns, **no
   hallucination**, variation across rows, and tone consistent with the rank
   (`submission_spec.docx` §3, §5). Since ranking runs offline (no hosted
   LLM), reasoning must be **generated from real extracted facts** (templated
   with genuine variation, or a small local model) — never invented.

A fifth, quieter finding: **scale is a non-issue here.** We score one query
vector against 100K candidate vectors — a single brute-force cosine pass over
a `(100000, d)` matrix is well within 5 min / 16 GB. The previously-planned
FAISS/ANN retrieve-then-rerank layer is **not needed** for this submission
(it would matter for a multi-JD production system, which is a pitch point, not
a deliverable). Embedding the pool is the only heavy step and is *pre*-compute.

## Judging axes (now precise)

1. **Deep Job Understanding** — model *this* JD's must/nice/disqualifier
   structure and its "between the lines" ideal.
2. **Contextual Relevance** — semantic fit from `summary` + `career_history`
   descriptions, beyond the `skills[]` list (defeats keyword stuffers, rewards
   plain-language fits).
3. **Signal Integration** — fold the 23 `redrob_signals` (availability,
   engagement, verification) into the score as the JD instructs
   ("down-weight a perfect-on-paper candidate who hasn't logged in for 6
   months with a 5% response rate").
4. **Fast + accurate ranked shortlist** — top-100 CSV, reproducible in budget.
5. **Deliverables** — repo + README + `submission_metadata.yaml` + a working
   sandbox link + a single reproduce command (`submission_spec.docx` §10).

## Status

| # | Item | Severity | Status | Where |
|---|------|----------|--------|-------|
| A | Single multi-signal inference entrypoint (`rank(job, candidates)`) | CRITICAL | ✅ done | `ir/ranker.py` |
| E | Redrob JSONL → `Candidate` adapter (real `candidate_schema.json`) | CRITICAL | ✅ done | `ir/adapters.py` |
| E2 | Top-100 CSV exporter matching the validator exactly | CRITICAL | ✅ done (passes official `validate_submission.py`) | `ir/adapters.py` |
| C | JD requirement model for *this* JD (must/nice/disqualify) | CRITICAL | ✅ done | `ir/jd_requirements.py` |
| D2 | Concrete career/behavioral feature library (23 signals + career patterns) | CRITICAL | ✅ done | `ir/features_library.py` |
| H | Honeypot / profile-consistency detector (Stage 3 disqualifier) | CRITICAL | ✅ done | `ir/honeypot.py` |
| R | Fact-grounded reasoning generator (Stage 4, no-network) | HIGH | ✅ done | `ir/reasoning.py` |
| CLI | `rank.py --candidates … --out …` reproducible in ≤5 min CPU | HIGH | ✅ done | `rank.py` |
| CE | Cross-encoder precision pass on top-K (FAIMR differentiator) | MEDIUM | ✅ wired (opt-in `--cross-encoder`) | `rank.py` + `ranking/cross_encoder_ranker.py` |
| D | Structured/behavioral feature *hook* | CRITICAL | ✅ done (10 features wired) | `ir/features.py:StructuredFeature` |
| B | Learned-ranker persistence (save/load) | LOW (no labels) | ✅ built, off critical path | `ir/ranker.py:save/load` |
| F | Self-eval harness (no official GT): composite + honeypot-rate + weight tuner | MEDIUM | ✅ done | `evaluation/selfeval.py` + `evaluation/judgment_archetypes.py` |
| S | Hosted sandbox demo (spec §10.5) | HIGH (Stage-1 flag) | ✅ done (Streamlit app) | `app.py` |
| — | FAISS/ANN retrieve-then-rerank | NOT NEEDED | ⬜ dropped for single-JD | (pitch only) |
| G | Fairness audit + EEOC re-ranker | LOW / tension | ⬜ optional, blocked (no group data in schema) | vendor FAIMR `fairness/` |

## Rationale & evidence per item

Dataset citations point into the released bundle; repo citations into this tree.

- **A — Inference entrypoint.** Built: `MultiSignalRanker.rank_candidates`
  (`ir/ranker.py:83`) runs the multi-signal stack on in-memory candidates with
  a per-signal explainability trail. Unchanged by the data drop, but its
  default text-only blend (`_DEFAULT_WEIGHTS`, `ir/ranker.py:39`) is
  insufficient alone — it must be fed the structured/behavioral features and
  JD-requirement signals below, and the unsupervised blend is the mode we
  actually ship (see finding #1).

- **E — JSONL adapter (now unblocked).** `candidate_schema.json` is the real
  schema (top keys: `candidate_id, profile, career_history, education, skills,
  certifications, languages, redrob_signals`). Need `load_candidates(path)`
  that streams `candidates.jsonl` (100K lines; stream, don't slurp — ~465 MB)
  into `ir.features.Candidate`, mapping: free-text `text` =
  headline+summary+career descriptions (the semantic surface), structured
  fields → `Candidate.metadata`, and `skills[].name` → `Candidate.skills`.
  Handle the `.jsonl` / `.jsonl.gz` duality (`README.docx`).

- **E2 — CSV exporter (now unblocked).** `validate_submission.py` is the exact
  contract: header `candidate_id,rank,score,reasoning`; **exactly 100** data
  rows; `candidate_id` matches `^CAND_[0-9]{7}$`; ranks 1–100 unique;
  `score` non-increasing by rank; **equal scores must tie-break by
  `candidate_id` ascending**; UTF-8. The exporter must run
  `validate_submission.py` semantics in-process so we never ship a rejectable
  file. (Run the official validator in CI.)

- **C — JD requirement model (reshaped, not an extractor).** For the single
  JD, encode an auditable model with: **must-haves** (production
  embeddings/retrieval, vector DB / hybrid search, strong Python, ranking-eval
  experience — NDCG/MRR/MAP); **nice-to-haves** (LoRA/QLoRA/PEFT, LTR,
  HR-tech, distributed systems, OSS); **hard disqualifiers** (pure-research /
  no production; "AI = LangChain→OpenAI in last <12 mo" without prior ML; no
  production code in 18 mo; **consulting-only career** at TCS/Infosys/Wipro/
  Accenture/Cognizant/Capgemini; CV/speech/robotics without NLP/IR; 5+ yrs
  entirely closed-source with no external validation); **soft negatives**
  (title-chasing / job-hopping < ~1.5 yr cadence); **logistics** (Noida/Pune
  or willing to relocate; sub-30-day notice preferred). All quotes traceable
  to `job_description.docx`. This file is the single source of truth a Stage-5
  interviewer can be walked through.

- **D2 — Concrete feature library (now unblocked; the core of the work).**
  `redrob_signals_doc.docx` enumerates 23 signals; `candidate_schema.json`
  gives ranges. Concrete features to implement as `StructuredFeature`s
  (`ir/features.py:114`):
  - *Experience-band fit*: distance of `years_of_experience` to the 5–9 band
    (soft, JD says it's a range not a gate).
  - *Product-vs-services trajectory*: classify `career_history[].company` /
    `industry` / `company_size`; penalize consulting-only careers
    (disqualifier), reward product-company applied-ML roles.
  - *Evidence-of-shipped-systems*: semantic match of `career_history[].
    description` against "built/shipped ranking/search/recommendation/retrieval
    at scale" — this is how plain-language Tier-5 fits are surfaced.
  - *Role/title coherence*: `current_title` + headline vs. AI-engineering
    relevance (defeats the "Marketing Manager with all the keywords" trap).
  - *Skill credibility*: combine `skills[].proficiency`, `endorsements`,
    `duration_months`, and `redrob_signals.skill_assessment_scores` — reward
    *demonstrated* skill over mere listing.
  - *Tenure/stability*: per-stint `duration_months` cadence (job-hopping
    soft-negative).
  - *Availability composite* (the JD's explicit ask): recency from
    `last_active_date`, `recruiter_response_rate`, `open_to_work_flag`,
    `interview_completion_rate`, `notice_period_days`.
  - *Engagement / market signal*: `saved_by_recruiters_30d`,
    `profile_views_received_30d`, `search_appearance_30d`.
  - *External validation*: `github_activity_score` (note: **`-1` = no GitHub**,
    a sentinel, not a zero — mildly negative, not disqualifying).
  - *Location fit*: `location`/`country` + `willing_to_relocate` vs Noida/Pune.
  **Correctness traps to honor:** `github_activity_score` and
  `offer_acceptance_rate` use `-1` sentinels (`candidate_schema.json:207,229`)
  — never average them in as 0. The availability signals act as a *multiplier
  / modifier* on fit (per `redrob_signals_doc.docx`), not as additive skill
  points.

- **H — Honeypot / consistency detector (NEW, CRITICAL).** ~80 honeypots are
  forced to tier 0 and a >10% rate in the top 100 is an automatic Stage-3
  disqualification (`submission_spec.docx` §7). Implement cross-field
  consistency checks, e.g.: stint `duration_months` / tenure exceeding the
  employer's plausible age; `skills[]` with `proficiency:"expert"` but
  `duration_months:0`; `years_of_experience` inconsistent with summed
  `career_history`; assessment scores contradicting claimed proficiency;
  impossible date ranges (`end_date` < `start_date`, future dates). The JD/
  spec say a good ranker *naturally* avoids these — so prefer integrating
  consistency as a **score penalty** (and as a safety filter on the final
  100) rather than special-casing only the known examples.

- **R — Reasoning generator (NEW, HIGH).** Per `submission_spec.docx` §3/§5
  the 1–2 sentence `reasoning` must cite concrete profile facts (yrs, title,
  named skills, signal values), connect to a *specific* JD requirement,
  acknowledge real concerns, vary across candidates, and **never reference a
  skill/employer not in the profile**. Generate it from the same extracted
  feature facts that drive the score (so tone tracks rank automatically), with
  enough templating variety to pass the "variation" check. No hosted-LLM calls
  at rank time (network is off) — a deterministic fact-to-text composer, or a
  small local model, only.

- **CLI / reproducibility (NEW, HIGH).** Stage 3 reproduces the ranking step
  in a sandboxed 5 min / 16 GB / CPU / no-network container
  (`submission_spec.docx` §3, §10.3). Provide `rank.py --candidates
  ./candidates.jsonl --out ./submission.csv`, with embedding **pre-compute**
  separated from the in-budget ranking step, all model weights vendored
  locally (no download at rank time), and a documented single reproduce
  command + a sandbox link (HF Spaces/Streamlit/Colab/Docker, §10.5).

- **B — Learned ranker (downgraded).** `save/load` + XGBoost fit
  (`ir/ranker.py:135,200`) are built and tested, but with no labels in the
  bundle there is nothing to train for the actual submission (finding #1).
  Keep for optional weak-supervision experiments; do not put it on the
  critical path or imply in the pitch that a model was trained on Redrob
  ground truth (it wasn't, and can't be).

- **F — Self-eval (reshaped).** We cannot compute the official NDCG/MAP
  locally (ground truth is hidden). `evaluation/metrics.py` still helps for:
  (a) a small **hand-built judgment set** (a few dozen candidates we manually
  tier) to sanity-check ordering and run ablations; (b) **honeypot-rate** on
  our own top-100; (c) **distribution / sanity** checks (score monotonicity,
  no disqualified profiles in the top, role coherence of the top 10). Lead
  local validation with methodology and spot-checks, as §8 advises.

- **FAISS — dropped.** Single query vs 100K vectors is one brute-force cosine
  pass; ANN indexing adds complexity and approximation error for no benefit
  inside this budget (finding #5). Mention it only as the production-scale
  story in the write-up.

- **G — Fairness (optional, with a real tension).** Track 1 is judged on
  ranking quality, not fairness (`HACKATHON_BRIEF.md`). Note the tension: this
  JD *requires* location filtering (Noida/Pune) and company-type signals,
  which a naive demographic-parity re-ranker would fight. Keep fairness as an
  enterprise-pitch differentiator, not a scored component; do not vendor the
  GFDL name corpora.

## Built this iteration (the end-to-end ranker)

The full submission pipeline now exists on top of the FAIMR core and runs
end-to-end, CPU-only and offline at rank time:

- `ir/adapters.py` — streaming JSONL→`Candidate` adapter + a top-100 CSV
  exporter that re-implements the official validator's checks in-process.
  **Verified**: a real-data run emits a CSV that `validate_submission.py`
  accepts ("Submission is valid.").
- `ir/jd_requirements.py` — the curated, auditable model of the single JD
  (must/nice skills, hard disqualifiers, shipped-systems phrases, locations,
  notice), every field traceable to `job_description.docx`.
- `ir/features_library.py` — 10 career/behavioral `StructuredFeature`s over
  the 23 signals (experience-band fit, product-vs-services, shipped-systems
  evidence, role coherence, skill credibility, tenure, availability composite,
  engagement, github with `-1` sentinel handling, location) + a separate
  `hard_disqualifier_penalty` applied as a multiplicative guard.
- `ir/honeypot.py` — within-profile consistency audit (expert-skill-with-0-
  months, experience-vs-summed-stints, tenure-exceeds-date-span, impossible
  dates, assessment-vs-proficiency), calibrated against the real sample.
- `ir/reasoning.py` — fact-grounded, tone-by-rank, deterministically-varied
  reasoning with no hallucinated skills. Concerns are derived from genuine
  JD-fit signals (`derive_concerns`: limited shipped-systems evidence, out-of-
  band experience, low engagement, off-domain background, notice/location),
  so a low placement is always explained validly. Honeypot/consistency flags
  are **deliberately never surfaced** — planted traps are silently down-ranked
  out of the top 100 via their scoring penalty, not announced in the output.
- `rank.py` — orchestrator: SBERT bi-encoder + anchored skill match + feature
  blend → honeypot/disqualifier guards → optional FAIMR cross-encoder rerank
  → top-100 CSV. `--embedder hashing` gives a model-free, fully-offline run
  for the sandbox/CI. Default weights deliberately down-weight raw keyword/
  embedding similarity so the JD's anti-keyword trap is respected.
- `tests/` — 33 tests (adapter, validator semantics, features, honeypot
  real-vs-synthetic, reasoning, self-eval metrics, judgment-set integrity,
  domain-title rule, cross-encoder integration, and an end-to-end run
  asserting the honeypot/consulting-only/off-target plants stay out of the
  top 10). Whole suite is green with the fake/hashing embedder — no download.

### Validation (gap F)

- `evaluation/judgment_archetypes.py` — 19 archetype profiles with gold
  relevance tiers authored *from the JD's own examples* (ideal fit, keyword-
  stuffer marketing manager, plain-language Tier-5, perfect-on-paper-but-
  inactive, consulting-only, pure-research, CV-without-NLP, recent-LangChain-
  only, and the impossible honeypot). Each non-honeypot archetype is internally
  consistent, so the honeypot detector's precision is testable (only the
  honeypot trips it).
- `evaluation/selfeval.py` — computes the **official composite**
  (`0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`, FAIMR metrics) and the
  top-10 honeypot rate, plus an offline coordinate-ascent weight tuner.
  **Current result (model-free hashing embedder): composite ≈ 0.97, NDCG@10
  0.96, honeypot rate 0%.** The tuner finds only ~+0.013 over the hand-set
  weights, so we keep the hand-set weights — they're near-optimal *and*
  explainable for the Stage-5 defend-your-work interview. SBERT is expected to
  raise the semantic archetypes (esp. the plain-language Tier-5) further.

> **Why this isn't circular:** the golds come from the JD's stated examples,
> not from our scoring function, so the harness measures whether the ranker
> exhibits the behaviors the JD demands (avoid keyword stuffers, reward plain-
> language fits, down-weight the unavailable, never surface honeypots).

## Recommended build order

1. **E + E2** — adapter + validator-exact exporter. Unblocks an
   end-to-end skeleton (even a trivial ranker now produces a *valid* CSV).
2. **C** — curated JD requirement model (the spec a human can defend).
3. **D2** — feature library (text-semantic + 23 signals + career patterns),
   wired as `StructuredFeature`s into `FeatureExtractor`.
4. **H** — honeypot/consistency penalty + final-100 safety filter.
5. **Score blend** — tune unsupervised weights so availability modifies fit
   and disqualifiers/honeypots are pushed down; verify top-10 coherence.
6. **R** — fact-grounded reasoning generator.
7. **CLI + sandbox** — `rank.py`, vendored weights, reproduce command, hosted
   sandbox; confirm ≤5 min CPU / 16 GB / offline.
8. **F** — self-eval (judgment set + honeypot-rate + ablations).
9. *(optional)* **B** weak-supervision experiment; **G** fairness pitch layer.

## Top risks to engineer against

- **Honeypots in the top 100 > 10%** → auto-DQ. Treat consistency as
  correctness (item H), not polish.
- **Keyword-embedding-only ranking** → ranks stuffers high, misses
  plain-language Tier-5s, and Stage-3/4 reviewers can see it. Career-history
  evidence and behavioral signals must materially move the order.
- **Sentinel mishandling** (`-1` for `github_activity_score` /
  `offer_acceptance_rate`) silently corrupting scores.
- **Reproduction failure at Stage 3** (network download at rank time, >5 min,
  GPU assumption) → DQ regardless of score. Vendor weights; separate
  pre-compute; test on a 16 GB CPU box.
- **Hallucinated or templated reasoning** → Stage-4 penalty. Ground every
  clause in an extracted fact.
```
