"""JD-grounded behavioral judgment set (gap analysis item F).

No ground truth ships with the dataset, so we cannot compute the official
score locally.  What we *can* do — and what Stage 4/5 reviewers effectively
check — is whether the ranker exhibits the behaviors the JD explicitly
demands.  This module encodes the JD's own stated examples as labeled
archetype profiles with gold relevance *tiers* (0..5, matching the hidden
ground-truth's tiering described in ``submission_spec.docx``).

These golds come from the JD, not from our scoring function, so tuning or
gating against them is meaningful rather than circular.  Each archetype is
internally consistent (years ≈ summed tenure, stint spans match
``duration_months``) so the honeypot detector only fires on the one archetype
that is *meant* to be impossible.

Reference for the golds (``job_description.docx``):
  * tier 5 — the "ideal candidate": 6-8 yrs, product company, shipped a
    ranking/retrieval/recsys system, embeddings + vector DB, eval frameworks,
    in/near Noida-Pune, active and responsive, sub-30-day notice.
  * tier 4 — strong, incl. the "plain-language Tier-5" who built a recsys at a
    product company but never writes "RAG"/"Pinecone".
  * tier 3 — relevant (the P@10 cutoff): adjacent but credible.
  * tier 2 — weak: e.g. "perfect on paper but hasn't logged in for 6 months
    with a 5% response rate" → "not actually available", down-weight.
  * tier 1 — poor: CV/speech specialist w/o NLP-IR; recent-LangChain-only.
  * tier 0 — not a fit / trap: keyword-stuffer in a non-AI role, consulting-
    only career, pure-research-only, and the impossible honeypot.
"""

from __future__ import annotations

import datetime as dt

REF = dt.date(2026, 6, 4)


def _iso(d: dt.date) -> str:
    return d.isoformat()


def _history(stints: list[tuple], end: dt.date = REF) -> list[dict]:
    """Lay stints back-to-back ending at ``end`` so dates and durations agree.

    ``stints`` is newest-first: (company, title, months, industry, size, desc).
    """
    out = []
    cursor = end
    for i, (company, title, months, industry, size, desc) in enumerate(stints):
        # approximate month arithmetic by 30-day steps (good enough; the
        # consistency checker tolerates ±6 months and uses calendar months).
        start = cursor - dt.timedelta(days=int(months * 30.44))
        out.append({
            "company": company, "title": title,
            "start_date": _iso(start),
            "end_date": None if i == 0 else _iso(cursor),
            "duration_months": months, "is_current": i == 0,
            "industry": industry, "company_size": size, "description": desc,
        })
        cursor = start
    return out


def _signals(**over) -> dict:
    s = {
        "profile_completeness_score": 85, "signup_date": "2023-01-01",
        "last_active_date": "2026-06-01", "open_to_work_flag": True,
        "profile_views_received_30d": 30, "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.75, "avg_response_time_hours": 6.0,
        "skill_assessment_scores": {}, "connection_count": 250,
        "endorsements_received": 40, "notice_period_days": 20,
        "expected_salary_range_inr_lpa": {"min": 25, "max": 40},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 55, "search_appearance_30d": 40,
        "saved_by_recruiters_30d": 6, "interview_completion_rate": 0.85,
        "offer_acceptance_rate": 0.5, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    }
    s.update(over)
    return s


def _skills(*specs) -> list[dict]:
    """specs: (name, proficiency, duration_months[, endorsements])."""
    out = []
    for spec in specs:
        name, prof, dur = spec[0], spec[1], spec[2]
        end = spec[3] if len(spec) > 3 else 10
        out.append({"name": name, "proficiency": prof, "endorsements": end,
                    "duration_months": dur})
    return out


def _profile(years, title, headline, summary, location="Pune", country="India",
             company="ProductCo", size="201-500", industry="Software") -> dict:
    return {
        "anonymized_name": "Candidate", "headline": headline, "summary": summary,
        "location": location, "country": country, "years_of_experience": years,
        "current_title": title, "current_company": company,
        "current_company_size": size, "current_industry": industry,
    }


def _c(cid, tier, profile, skills, history, signals, education=None) -> tuple[dict, int]:
    raw = {
        "candidate_id": cid, "profile": profile, "career_history": history,
        "education": education or [{"institution": "State University", "degree": "BTech",
                                    "field_of_study": "CS", "start_year": 2013,
                                    "end_year": 2017, "grade": None, "tier": "tier_2"}],
        "skills": skills, "certifications": [], "languages": [],
        "redrob_signals": signals,
    }
    return raw, tier


# ─── archetypes ───────────────────────────────────────────────────────────

def build_judgment_set() -> list[tuple[dict, int]]:
    """Return [(raw_profile, gold_tier)] spanning all six tiers."""
    A: list[tuple[dict, int]] = []

    # ── tier 5: ideal ──────────────────────────────────────────────────
    A.append(_c(
        "CAND_9000001", 5,
        _profile(7.0, "Senior ML Engineer", "ML Engineer | retrieval, ranking, embeddings",
                 "Owned the ranking and retrieval stack at a product company; built hybrid "
                 "search with FAISS and learning-to-rank, ran NDCG/MAP offline evals and A/B tests."),
        _skills(("Python", "expert", 84, 50), ("FAISS", "advanced", 40), ("Embeddings", "advanced", 44),
                ("Learning to Rank", "advanced", 30), ("NDCG", "intermediate", 24)),
        _history([("ProductCo", "Senior ML Engineer", 40, "Software", "201-500",
                   "Built and shipped the candidate ranking + retrieval system: embeddings, "
                   "FAISS vector search, hybrid retrieval, NDCG/MAP evaluation and online A/B tests."),
                  ("ScaleApp", "ML Engineer", 44, "Software", "501-1000",
                   "Recommendation and search relevance; embedding pipelines, index refresh.")]),
        _signals(recruiter_response_rate=0.85, last_active_date="2026-06-02",
                 notice_period_days=15, github_activity_score=80, saved_by_recruiters_30d=20)))

    A.append(_c(
        "CAND_9000002", 5,
        _profile(8.0, "Staff ML Engineer", "Search & ranking | embeddings, vector DBs",
                 "Built semantic search and recommendation at scale; Pinecone + hybrid retrieval, "
                 "rigorous offline-to-online eval correlation.", location="Noida"),
        _skills(("Python", "expert", 96), ("Pinecone", "advanced", 36), ("Semantic Search", "advanced", 40),
                ("Recommendation", "advanced", 48), ("MRR", "intermediate", 20)),
        _history([("RetailTech", "Staff ML Engineer", 50, "E-commerce", "1001-5000",
                   "Owned search relevance and recsys; vector retrieval, ranking models, A/B testing."),
                  ("Marketplace", "ML Engineer", 46, "Software", "201-500",
                   "Candidate generation and ranking for a two-sided marketplace.")]),
        _signals(recruiter_response_rate=0.9, last_active_date="2026-05-30",
                 notice_period_days=0, github_activity_score=70)))

    A.append(_c(
        "CAND_9000003", 5,
        _profile(6.0, "Applied Scientist", "Applied ML | retrieval, RAG, ranking",
                 "Production retrieval and RAG systems; handled embedding drift and index refresh; "
                 "designed the ranking eval harness.", location="Hyderabad"),
        _skills(("Python", "expert", 72), ("RAG", "advanced", 24), ("Retrieval", "advanced", 30),
                ("Elasticsearch", "advanced", 36), ("XGBoost", "intermediate", 24)),
        _history([("SaaSCo", "Applied Scientist", 36, "Software", "201-500",
                   "Built RAG + hybrid retrieval; ranking with learning-to-rank; NDCG/MAP evals."),
                  ("DataCo", "ML Engineer", 36, "Software", "51-200",
                   "Search ranking and relevance tuning.")]),
        _signals(recruiter_response_rate=0.8, last_active_date="2026-06-01", notice_period_days=25)))

    A.append(_c(
        "CAND_9000004", 5,
        _profile(7.5, "Senior Search Engineer", "Search relevance | ranking, embeddings",
                 "Shipped search ranking to millions of users; embeddings + Qdrant; eval frameworks.",
                 location="Pune"),
        _skills(("Python", "expert", 90), ("Qdrant", "advanced", 30), ("Ranking", "expert", 50),
                ("Embeddings", "advanced", 44), ("Information Retrieval", "advanced", 40)),
        _history([("SearchCo", "Senior Search Engineer", 54, "Software", "1001-5000",
                   "Owned search ranking and relevance; vector retrieval; NDCG-driven iteration."),
                  ("WebCo", "Backend/ML Engineer", 36, "Software", "201-500",
                   "Built recommendation and retrieval features.")]),
        _signals(recruiter_response_rate=0.78, last_active_date="2026-06-03", github_activity_score=65)))

    # ── tier 4: strong, incl. plain-language Tier-5 (no buzzwords) ──────
    A.append(_c(
        "CAND_9000010", 4,
        _profile(6.5, "Software Engineer", "Backend & ML | building product features",
                 "I built the system that decides which items to show users — we generate candidates, "
                 "score them, and order them, and I obsess over measuring whether the ordering is "
                 "actually good. Did this end-to-end at a product company. I don't use a lot of the "
                 "trendy vocabulary.",  # the JD's 'Tier 5' who never says RAG/Pinecone
                 location="Mumbai"),
        _skills(("Python", "expert", 78), ("Spark", "advanced", 40), ("SQL", "advanced", 60)),
        _history([("ConsumerApp", "Software Engineer", 42, "Software", "201-500",
                   "Built the system that picks and orders what users see: candidate generation, "
                   "scoring, ordering, and offline + online measurement of result quality."),
                  ("StartupCo", "Engineer", 36, "Software", "51-200",
                   "Built a system to recommend items to users and measured click-through improvements.")]),
        _signals(recruiter_response_rate=0.7, last_active_date="2026-05-28", notice_period_days=30)))

    A.append(_c(
        "CAND_9000011", 4,
        _profile(5.0, "ML Engineer", "ML Engineer | retrieval, embeddings",
                 "Built embedding retrieval and ranking at a product company; some eval experience.",
                 location="Bangalore"),
        _skills(("Python", "expert", 60), ("FAISS", "intermediate", 18), ("Embeddings", "advanced", 30),
                ("Recommendation", "intermediate", 24)),
        _history([("ProductCo", "ML Engineer", 30, "Software", "201-500",
                   "Embedding-based retrieval and ranking; offline NDCG evaluation."),
                  ("EarlyCo", "Junior ML Engineer", 30, "Software", "11-50",
                   "Recommendation prototypes and relevance experiments.")]),
        _signals(recruiter_response_rate=0.65, last_active_date="2026-05-20", notice_period_days=45)))

    A.append(_c(
        "CAND_9000012", 4,
        _profile(9.0, "Principal Engineer", "Search & ML systems | ranking, retrieval",
                 "Deep retrieval/ranking background; built search and recsys at product companies.",
                 location="Delhi"),
        _skills(("Python", "expert", 110), ("Milvus", "advanced", 30), ("Ranking", "expert", 60),
                ("Information Retrieval", "expert", 70)),
        _history([("BigProduct", "Principal Engineer", 60, "Software", "5001-10000",
                   "Architected ranking and retrieval platform; mentored teams; A/B framework."),
                  ("MidProduct", "Senior Engineer", 48, "Software", "501-1000",
                   "Search relevance and recommendation systems.")]),
        _signals(recruiter_response_rate=0.6, last_active_date="2026-05-15", notice_period_days=60)))

    # ── tier 3: relevant (P@10 cutoff) ──────────────────────────────────
    A.append(_c(
        "CAND_9000020", 3,
        _profile(6.0, "Data Scientist", "Data Scientist | ML, NLP, some retrieval",
                 "ML and NLP background; some exposure to recommendation and ranking; strong Python.",
                 location="Pune"),
        _skills(("Python", "expert", 72), ("NLP", "advanced", 40), ("Recommendation", "intermediate", 18)),
        _history([("AnalyticsCo", "Data Scientist", 40, "Software", "201-500",
                   "Built ML models incl. a recommendation prototype; NLP feature pipelines."),
                  ("ConsultProduct", "Data Analyst", 32, "Software", "51-200",
                   "Analytics and modeling; some search relevance work.")]),
        _signals(recruiter_response_rate=0.55, last_active_date="2026-05-25")))

    A.append(_c(
        "CAND_9000021", 3,
        _profile(7.0, "ML Engineer", "ML Engineer | modeling, pipelines",
                 "General ML engineering; built data and model pipelines, limited retrieval depth.",
                 location="Gurgaon"),
        _skills(("Python", "expert", 84), ("XGBoost", "advanced", 36), ("SQL", "advanced", 60)),
        _history([("ProductCo", "ML Engineer", 44, "Software", "201-500",
                   "Modeling and feature pipelines; one ranking experiment."),
                  ("SmallCo", "Engineer", 40, "Software", "51-200", "Backend and ML support.")]),
        _signals(recruiter_response_rate=0.5, last_active_date="2026-05-18")))

    A.append(_c(
        "CAND_9000022", 3,
        _profile(5.5, "Backend Engineer", "Backend | search infra adjacent",
                 "Backend engineer who operated Elasticsearch and search infra at a product company.",
                 location="Noida"),
        _skills(("Python", "advanced", 60), ("Elasticsearch", "advanced", 40), ("SQL", "expert", 66)),
        _history([("WebProduct", "Backend Engineer", 36, "Software", "201-500",
                   "Operated and tuned Elasticsearch-based search; relevance config."),
                  ("StartCo", "Engineer", 30, "Software", "11-50", "Backend services.")]),
        _signals(recruiter_response_rate=0.6, last_active_date="2026-05-29")))

    # ── tier 2: weak — perfect-on-paper but unavailable, or off-band ────
    A.append(_c(
        "CAND_9000030", 2,  # JD: hasn't logged in for months + low response = not available
        _profile(7.0, "Senior ML Engineer", "ML Engineer | retrieval, ranking, embeddings",
                 "Strong retrieval/ranking background — on paper a great match — but disengaged.",
                 location="Pune"),
        _skills(("Python", "expert", 84), ("FAISS", "advanced", 40), ("Embeddings", "advanced", 44),
                ("Ranking", "advanced", 36)),
        _history([("ProductCo", "Senior ML Engineer", 44, "Software", "201-500",
                   "Built ranking and retrieval systems with embeddings and FAISS; NDCG evals."),
                  ("ScaleApp", "ML Engineer", 40, "Software", "501-1000", "Search relevance.")]),
        _signals(recruiter_response_rate=0.05, last_active_date="2025-12-01",
                 open_to_work_flag=False, interview_completion_rate=0.2, saved_by_recruiters_30d=0)))

    A.append(_c(
        "CAND_9000031", 2,
        _profile(13.0, "Engineering Manager", "Eng Manager | ML org leadership",
                 "Mostly management for the last several years; limited recent hands-on coding.",
                 location="Bangalore"),  # JD: no production code in 18 months = probably not
        _skills(("Python", "intermediate", 130), ("Leadership", "expert", 60)),
        _history([("BigCo", "Engineering Manager", 60, "Software", "5001-10000",
                   "Led ML teams; roadmap and people management, little hands-on code."),
                  ("MidCo", "Senior Engineer", 96, "Software", "501-1000", "Earlier IC work.")]),
        _signals(recruiter_response_rate=0.4, last_active_date="2026-04-01", notice_period_days=90)))

    A.append(_c(
        "CAND_9000032", 2,
        _profile(4.0, "ML Engineer", "ML Engineer | learning retrieval",
                 "Junior-ish; building retrieval/ranking competence but limited production depth.",
                 location="Pune"),
        _skills(("Python", "advanced", 48), ("Embeddings", "intermediate", 12)),
        _history([("StartCo", "ML Engineer", 24, "Software", "11-50",
                   "Built a first retrieval prototype; learning ranking evaluation."),
                  ("Intern", "ML Intern", 24, "Software", "11-50", "ML internship projects.")]),
        _signals(recruiter_response_rate=0.7, last_active_date="2026-06-01")))

    # ── tier 1: poor — out-of-domain w/o NLP-IR; recent-LangChain-only ──
    A.append(_c(
        "CAND_9000040", 1,  # JD: CV/speech/robotics without NLP/IR
        _profile(6.0, "Computer Vision Engineer", "Computer Vision | detection, segmentation",
                 "Computer vision specialist: object detection, image segmentation, video analytics. "
                 "No NLP or information-retrieval work.", location="Pune"),
        _skills(("Python", "expert", 72), ("Image Classification", "expert", 50),
                ("Object Detection", "advanced", 40)),
        _history([("VisionCo", "Computer Vision Engineer", 40, "Software", "201-500",
                   "Object detection and segmentation pipelines for video analytics."),
                  ("CamCo", "CV Engineer", 32, "Hardware", "51-200", "Image classification models.")]),
        _signals(recruiter_response_rate=0.6, last_active_date="2026-05-20")))

    A.append(_c(
        "CAND_9000041", 1,  # JD: 'AI experience' = recent LangChain→OpenAI only
        _profile(5.0, "AI Engineer", "AI Engineer | LangChain, LLM apps",
                 "Built LLM apps over the last 10 months using LangChain to call OpenAI. "
                 "Before that, web development; no pre-LLM ML/retrieval background.", location="Pune"),
        _skills(("Python", "advanced", 24), ("LangChain", "advanced", 10), ("Prompt Engineering", "advanced", 10)),
        _history([("AppCo", "AI Engineer", 10, "Software", "11-50",
                   "LangChain + OpenAI chatbots and demos."),
                  ("WebCo", "Web Developer", 50, "Software", "51-200",
                   "Frontend and backend web development; no ML.")]),
        _signals(recruiter_response_rate=0.65, last_active_date="2026-06-01", github_activity_score=30)))

    # ── tier 0: not a fit / traps ───────────────────────────────────────
    A.append(_c(
        "CAND_9000050", 0,  # JD: keyword-stuffer in a non-AI role
        _profile(6.0, "Marketing Manager", "Marketing Manager | AI, ML, RAG, Pinecone, FAISS, NLP",
                 "Marketing manager. My skills section lists every AI keyword but my work is campaigns.",
                 location="Pune"),
        _skills(("RAG", "expert", 36), ("Pinecone", "expert", 36), ("FAISS", "expert", 36),
                ("NLP", "expert", 36), ("Embeddings", "expert", 36), ("Ranking", "expert", 36)),
        _history([("BrandCo", "Marketing Manager", 40, "Consumer", "201-500",
                   "Ran marketing campaigns, brand strategy, and ad spend; no engineering."),
                  ("AgencyCo", "Marketing Specialist", 32, "Advertising", "51-200",
                   "Social media and content marketing.")]),
        _signals(recruiter_response_rate=0.8, last_active_date="2026-06-01")))

    A.append(_c(
        "CAND_9000051", 0,  # JD: consulting-only career
        _profile(8.0, "Technology Lead", "Tech Lead | enterprise delivery",
                 "Entire career in IT services delivery across client projects.", location="Pune",
                 company="Infosys", size="10001+", industry="IT Services"),
        _skills(("Python", "advanced", 60), ("Java", "advanced", 80), ("SQL", "expert", 90)),
        _history([("Infosys", "Technology Lead", 50, "IT Services", "10001+",
                   "Led client delivery teams on enterprise integration projects."),
                  ("Wipro", "Senior Developer", 46, "IT Services", "10001+",
                   "Enterprise application development for clients.")]),
        _signals(recruiter_response_rate=0.7, last_active_date="2026-05-30")))

    A.append(_c(
        "CAND_9000052", 0,  # JD: pure research, no production deployment
        _profile(7.0, "Research Scientist", "Research Scientist | ML theory",
                 "Academic research-only career; publications but no production deployment.",
                 location="Pune", company="University Lab", size="1001-5000", industry="Education"),
        _skills(("Python", "advanced", 84), ("PyTorch", "advanced", 60), ("NLP", "advanced", 50)),
        _history([("University Lab", "Research Scientist", 48, "Education", "1001-5000",
                   "Academic research on ML methods; papers and prototypes, no production systems."),
                  ("Research Institute", "Postdoc", 36, "Education", "1001-5000",
                   "Postdoctoral research; no deployment.")]),
        _signals(recruiter_response_rate=0.5, last_active_date="2026-05-22")))

    # The impossible honeypot (forced tier 0; must trip the consistency check).
    hp_profile = _profile(8.0, "ML Engineer", "ML Engineer | retrieval, ranking",
                          "Senior engineer with deep retrieval and ranking experience.", location="Pune")
    hp = {
        "candidate_id": "CAND_9000053", "profile": hp_profile,
        "career_history": [{
            "company": "NewStartup", "title": "ML Engineer",
            "start_date": "2023-09-01", "end_date": None,
            "duration_months": 96,  # 8 yrs claimed at a company that can't be that old
            "is_current": True, "industry": "Software", "company_size": "11-50",
            "description": "Built ranking and retrieval systems.",
        }],
        "education": [{"institution": "Uni", "degree": "BTech", "field_of_study": "CS",
                       "start_year": 2014, "end_year": 2018, "grade": None, "tier": "tier_2"}],
        "skills": _skills(("RAG", "expert", 0), ("FAISS", "expert", 0), ("Embeddings", "expert", 0),
                          ("Ranking", "expert", 0)),  # expert with 0 months used
        "certifications": [], "languages": [],
        "redrob_signals": _signals(recruiter_response_rate=0.8, last_active_date="2026-06-01"),
    }
    A.append((hp, 0))

    return A
