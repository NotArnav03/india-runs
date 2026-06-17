"""Tests for the Redrob pipeline: adapter, exporter, features, honeypot,
reasoning, and an end-to-end rank — all runnable without a model download."""

import csv
import datetime as dt
import json

import pytest

from ir.adapters import (
    candidate_from_raw, load_candidates, build_submission_rows,
    validate_rows, write_submission, SubmissionError,
)
from ir.jd_requirements import REDROB_SENIOR_AI_ENGINEER as REQS
from ir.features_library import build_feature_library, hard_disqualifier_penalty
from ir.honeypot import consistency_report
from ir.reasoning import build_reasoning

REF = dt.date(2026, 6, 4)


# ─── fixtures ─────────────────────────────────────────────────────────────

def mk_raw(i, **over):
    """A realistic, internally-consistent profile; override fields per test."""
    raw = {
        "candidate_id": f"CAND_{i:07d}",
        "profile": {
            "anonymized_name": f"Person {i}",
            "headline": "ML Engineer | retrieval, ranking",
            "summary": "Applied ML engineer who shipped a recommendation and search "
                       "system with embeddings and vector retrieval at a product company.",
            "location": "Pune", "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior ML Engineer",
            "current_company": "ProductCo", "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {"company": "ProductCo", "title": "Senior ML Engineer",
             "start_date": "2021-01-01", "end_date": None, "duration_months": 65,
             "is_current": True, "industry": "Software", "company_size": "201-500",
             "description": "Built ranking and retrieval systems with embeddings, "
                            "FAISS vector search and learning-to-rank; ran NDCG/MAP evals."},
        ],
        "education": [{"institution": "IIT", "degree": "BTech", "field_of_study": "CS",
                       "start_year": 2012, "end_year": 2016, "grade": None, "tier": "tier_1"}],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 80},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 10, "duration_months": 36},
            {"name": "Retrieval", "proficiency": "advanced", "endorsements": 8, "duration_months": 40},
        ],
        "certifications": [], "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 90, "signup_date": "2023-01-01",
            "last_active_date": "2026-06-01", "open_to_work_flag": True,
            "profile_views_received_30d": 40, "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.8, "avg_response_time_hours": 5.0,
            "skill_assessment_scores": {"Python": 88, "FAISS": 75}, "connection_count": 300,
            "endorsements_received": 58, "notice_period_days": 15,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 70, "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5, "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.5, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }
    # shallow-merge overrides into nested dicts
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k].update(v)
        else:
            raw[k] = v
    return raw


# ─── adapter ──────────────────────────────────────────────────────────────

def test_candidate_from_raw_maps_text_and_skills():
    c = candidate_from_raw(mk_raw(1))
    assert c.id == "CAND_0000001"
    assert "python" in c.skills and "faiss" in c.skills
    assert "ranking and retrieval" in c.text.lower()
    assert c.metadata["raw"]["candidate_id"] == "CAND_0000001"


def test_load_candidates_streams(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(mk_raw(i)) for i in range(1, 6)) + "\n", encoding="utf-8")
    cands = list(load_candidates(p, limit=3))
    assert [c.id for c in cands] == ["CAND_0000001", "CAND_0000002", "CAND_0000003"]


# ─── exporter / validator semantics ────────────────────────────────────────

def test_build_rows_orders_by_score_then_id():
    rows = build_submission_rows([
        ("CAND_0000002", 0.5, "b"),
        ("CAND_0000001", 0.5, "a"),   # tie -> id ascending
        ("CAND_0000003", 0.9, "c"),
    ])
    assert [r["candidate_id"] for r in rows] == ["CAND_0000003", "CAND_0000001", "CAND_0000002"]
    assert [r["rank"] for r in rows] == [1, 2, 3]


def test_write_submission_roundtrip_and_validates(tmp_path):
    triples = [(f"CAND_{i:07d}", 1.0 - i / 1000.0, f"reason {i}") for i in range(100)]
    out = write_submission(triples, tmp_path / "team.csv")
    with open(out, encoding="utf-8", newline="") as f:
        rdr = list(csv.reader(f))
    assert rdr[0] == ["candidate_id", "rank", "score", "reasoning"]
    assert len(rdr) == 101  # header + 100
    scores = [float(r[2]) for r in rdr[1:]]
    assert scores == sorted(scores, reverse=True)  # non-increasing


def test_write_submission_rejects_wrong_count(tmp_path):
    with pytest.raises(SubmissionError):
        write_submission([("CAND_0000001", 0.9, "x")], tmp_path / "bad.csv")


def test_validate_rows_catches_increasing_score():
    rows = build_submission_rows([(f"CAND_{i:07d}", i, "x") for i in range(100)])
    rows[0]["score"], rows[1]["score"] = 0.1, 0.2  # force an inversion at the top
    errs = validate_rows(rows)
    assert any("non-increasing" in e for e in errs)


# ─── honeypot / consistency ─────────────────────────────────────────────────

def test_clean_profile_has_no_penalty():
    assert consistency_report(mk_raw(1), REF).penalty == 0.0


def test_honeypot_expert_skill_zero_months():
    raw = mk_raw(2, skills=[{"name": "RAG", "proficiency": "expert",
                             "endorsements": 5, "duration_months": 0}])
    rep = consistency_report(raw, REF)
    assert rep.is_suspect and any("0 months" in f for f in rep.flags)


def test_honeypot_tenure_exceeds_date_span():
    # 96 months claimed on a stint that started ~3 years ago.
    raw = mk_raw(3, career_history=[{
        "company": "NewCo", "title": "Engineer", "start_date": "2023-06-01",
        "end_date": None, "duration_months": 96, "is_current": True,
        "industry": "Software", "company_size": "11-50", "description": "work",
    }])
    rep = consistency_report(raw, REF)
    assert rep.penalty > 0 and any("spans only" in f for f in rep.flags)


# ─── disqualifiers ──────────────────────────────────────────────────────────

def test_consulting_only_career_disqualified():
    raw = mk_raw(4, career_history=[{
        "company": "Infosys", "title": "Engineer", "start_date": "2018-01-01",
        "end_date": None, "duration_months": 90, "is_current": True,
        "industry": "IT Services", "company_size": "10001+", "description": "delivery work",
    }])
    pen, reasons = hard_disqualifier_penalty(raw, REQS)
    assert pen >= 0.8 and any("consulting-only" in r for r in reasons)


def test_off_target_title_disqualified():
    raw = mk_raw(5, profile={"current_title": "Marketing Manager"})
    pen, reasons = hard_disqualifier_penalty(raw, REQS)
    assert pen >= 0.8


# ─── feature library ────────────────────────────────────────────────────────

def test_github_minus_one_sentinel_is_neutral_low():
    lib = {f.name: f for f in build_feature_library(REQS, REF)}
    raw = mk_raw(6, redrob_signals={"github_activity_score": -1})
    c = candidate_from_raw(raw)
    assert lib["github_signal"].compute(REQS.build_job(), c) == pytest.approx(0.3)


def test_experience_band_fit_peaks_in_band():
    lib = {f.name: f for f in build_feature_library(REQS, REF)}
    job = REQS.build_job()
    in_band = lib["experience_band_fit"].compute(job, candidate_from_raw(mk_raw(7)))
    junior = lib["experience_band_fit"].compute(
        job, candidate_from_raw(mk_raw(8, profile={"years_of_experience": 1.0})))
    assert in_band == pytest.approx(1.0) and junior < in_band


# ─── reasoning ──────────────────────────────────────────────────────────────

def test_reasoning_grounded_and_no_hallucination():
    raw = mk_raw(9)
    text = build_reasoning(raw, rank=3, reqs=REQS, concerns=[], reference_date=REF)
    assert "7.0 yrs" in text and "Senior ML Engineer" in text
    assert "Strong fit" in text  # tone matches a top rank
    # Only skills actually in the profile may appear.
    assert "tensorflow" not in text.lower()


def test_reasoning_surfaces_concern():
    raw = mk_raw(10)
    text = build_reasoning(raw, rank=70, reqs=REQS,
                           concerns=["last active ~6 months ago"], reference_date=REF)
    assert "concern" in text.lower()


# ─── end-to-end ─────────────────────────────────────────────────────────────

def test_honeypot_is_down_ranked_but_never_mentioned_in_reasoning():
    # A planted honeypot trap (expert skill, 0 months used) must be penalized
    # in scoring, but its consistency flags must NOT leak into the reasoning
    # concerns — we silently avoid the trap rather than announce we caught it.
    from rank import score_candidates

    raw = mk_raw(1, skills=[{"name": "RAG", "proficiency": "expert",
                             "endorsements": 1, "duration_months": 0}])
    c = candidate_from_raw(raw)
    final, guard = score_candidates([c], embedder_name="hashing", reference_date=REF)
    _, penalty, concerns = guard[c.id]
    assert penalty > 0.0  # still down-ranked
    leaks = ("0 months", "spans only", "inconsistent", "future", "precedes",
             "assessment score", "honeypot")
    assert all(not any(p in concern for p in leaks) for concern in concerns)
    text = build_reasoning(raw, rank=99, reqs=REQS, concerns=concerns, reference_date=REF)
    assert all(p not in text for p in leaks)


def test_end_to_end_rank_produces_valid_submission(tmp_path):
    from rank import rank

    rows = []
    # 97 plausible candidates with varied strength...
    for i in range(1, 98):
        rows.append(mk_raw(i, profile={"years_of_experience": 4.0 + (i % 6)}))
    # ...one consulting-only, one off-target, one honeypot — must sink.
    hp = mk_raw(998, skills=[{"name": "RAG", "proficiency": "expert",
                              "endorsements": 1, "duration_months": 0}])
    consulting = mk_raw(999, career_history=[{
        "company": "Wipro", "title": "Engineer", "start_date": "2017-01-01",
        "end_date": None, "duration_months": 100, "is_current": True,
        "industry": "IT Services", "company_size": "10001+", "description": "x"}])
    offtarget = mk_raw(1000, profile={"current_title": "Sales Manager"})
    pool = rows + [hp, consulting, offtarget]

    src = tmp_path / "cands.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in pool) + "\n", encoding="utf-8")

    out = rank(src, tmp_path / "submission.csv", embedder_name="hashing",
               top_n=100, reference_date=REF)

    with open(out, encoding="utf-8", newline="") as f:
        ranked = list(csv.DictReader(f))
    assert len(ranked) == 100
    top10 = {r["candidate_id"] for r in ranked[:10]}
    assert "CAND_0000998" not in top10  # honeypot
    assert "CAND_0000999" not in top10  # consulting-only
    assert "CAND_0001000" not in top10  # off-target title
    # validator-exact: ranks unique 1..100, scores non-increasing
    assert [int(r["rank"]) for r in ranked] == list(range(1, 101))
    scores = [float(r["score"]) for r in ranked]
    assert scores == sorted(scores, reverse=True)
