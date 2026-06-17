"""Concrete career + behavioral feature library (gap analysis item D2).

Turns the Redrob profile schema (``candidate_schema.json`` +
``redrob_signals_doc.docx``) into a list of :class:`ir.features.StructuredFeature`
plug-ins for the ranker.  Every feature reads from ``candidate.metadata["raw"]``
and returns a float in roughly ``[0, 1]`` where **higher is always better** —
so the unsupervised blend in :class:`ir.ranker.MultiSignalRanker` behaves
predictably and weights stay interpretable.

These are the signals the JD says actually matter — career trajectory and
behavioral availability — *not* keyword presence.  The hard kill-switches
(consulting-only career, pure research, out-of-domain, LangChain-only) are
kept separate in :func:`hard_disqualifier_penalty`, applied as a multiplicative
guard rather than blended, so a disqualified profile cannot be rescued by a
strong embedding score.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Optional

from ir.features import Candidate, Job, StructuredFeature
from ir.jd_requirements import JDRequirements

_PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
_DEFAULT_REFERENCE_DATE = dt.date(2026, 6, 4)  # dataset snapshot (see README.docx)


# ─── small parsing helpers ────────────────────────────────────────────────

def _raw(c: Candidate) -> dict:
    return c.metadata.get("raw", {}) if c.metadata else {}


def _profile(c: Candidate) -> dict:
    return _raw(c).get("profile", {}) or {}


def _signals(c: Candidate) -> dict:
    return _raw(c).get("redrob_signals", {}) or {}


def _parse_date(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _career_text(c: Candidate) -> str:
    raw = _raw(c)
    parts = [str(_profile(c).get("headline", "")), str(_profile(c).get("summary", ""))]
    for stint in raw.get("career_history", []):
        parts.append(f"{stint.get('title', '')} {stint.get('description', '')}")
    return " ".join(parts).lower()


def _company_terms(c: Candidate) -> list[tuple[str, int]]:
    """(`lowercased company name`, duration_months) per stint."""
    return [
        (str(s.get("company", "")).lower(), int(s.get("duration_months", 0) or 0))
        for s in _raw(c).get("career_history", [])
    ]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ─── graded features (higher = better) ────────────────────────────────────

def _experience_band_fit(reqs: JDRequirements):
    lo, hi = reqs.experience_band

    def fn(job: Job, c: Candidate) -> float:
        y = float(_profile(c).get("years_of_experience", 0.0) or 0.0)
        if lo <= y <= hi:
            return 1.0
        if y < lo:
            # Linear from hard_floor..lo; the JD says it'll consider strong
            # signals outside the band, so we decay rather than zero out.
            floor = reqs.experience_hard_floor
            return _clamp01((y - floor) / (lo - floor)) * 0.8 if lo > floor else 0.4
        # Above band: gentle decay (over-experienced is a mild concern only).
        return _clamp01(1.0 - (y - hi) / 10.0)

    return StructuredFeature("experience_band_fit", fn, default=0.3)


def _product_vs_services(reqs: JDRequirements):
    def fn(job: Job, c: Candidate) -> float:
        terms = _company_terms(c)
        total = sum(d for _, d in terms) or 1
        services = sum(
            d for name, d in terms
            if any(firm in name for firm in reqs.consulting_only_firms)
        )
        return _clamp01(1.0 - services / total)

    return StructuredFeature("product_vs_services", fn, default=0.6)


def _shipped_systems_evidence(reqs: JDRequirements):
    phrases = reqs.shipped_systems_phrases

    def fn(job: Job, c: Candidate) -> float:
        text = _career_text(c)
        hits = sum(1 for p in phrases if p in text)
        # 3+ distinct relevance phrases in the career narrative is strong
        # evidence of a shipped ranking/retrieval/recsys system.
        return _clamp01(hits / 3.0)

    return StructuredFeature("shipped_systems_evidence", fn, default=0.0)


def _role_coherence(reqs: JDRequirements):
    eng_terms = {"engineer", "scientist", "ml", "machine learning", "ai",
                 "developer", "data", "research", "architect", "nlp", "applied"}

    nlp_ir_terms = {"nlp", "retrieval", "ranking", "search", "recommend",
                    "information retrieval", "rag", "relevance", "embedding"}

    def fn(job: Job, c: Candidate) -> float:
        title = str(_profile(c).get("current_title", "")).lower()
        headline = str(_profile(c).get("headline", "")).lower()
        blob = f"{title} {headline}"
        off = any(t in title for t in reqs.off_target_title_terms)
        on = any(t in blob for t in eng_terms)
        # JD soft-negative: CV/speech/robotics specialists are a fit only with
        # NLP/IR exposure.  An out-of-domain *title* without any NLP/IR cue in
        # the title/headline is a coherence concern even though it's an
        # engineering role (so it's softer than the off-target penalty).
        out_of_domain_title = any(t in blob for t in reqs.out_of_domain_terms)
        nlp_ir_cue = any(t in blob for t in nlp_ir_terms)
        if on and not off:
            if out_of_domain_title and not nlp_ir_cue:
                return 0.55
            return 1.0
        if on and off:
            return 0.5
        if off:
            return 0.1
        return 0.4

    return StructuredFeature("role_coherence", fn, default=0.4)


def _skill_credibility(reqs: JDRequirements):
    relevant = reqs.must_have_skills | reqs.nice_to_have_skills

    def fn(job: Job, c: Candidate) -> float:
        skills = _raw(c).get("skills", [])
        assess = _signals(c).get("skill_assessment_scores", {}) or {}
        scored = 0.0
        matched = 0
        for s in skills:
            name = str(s.get("name", "")).lower()
            if name not in relevant:
                continue
            matched += 1
            w = _PROFICIENCY_WEIGHT.get(str(s.get("proficiency", "")), 0.25)
            months = float(s.get("duration_months", 0) or 0)
            depth = _clamp01(months / 36.0)  # 3 yrs of use ≈ full depth
            # If an assessment exists, it grounds the self-reported proficiency.
            a = assess.get(s.get("name"))
            verify = (a / 100.0) if isinstance(a, (int, float)) else w
            scored += (0.5 * w + 0.3 * depth + 0.2 * verify)
        if matched == 0:
            return 0.0
        # Reward breadth up to ~5 credible relevant skills, then saturate.
        return _clamp01(scored / 5.0)

    return StructuredFeature("skill_credibility", fn, default=0.0)


def _tenure_stability(reqs: JDRequirements):
    def fn(job: Job, c: Candidate) -> float:
        durations = [int(s.get("duration_months", 0) or 0)
                     for s in _raw(c).get("career_history", [])]
        durations = [d for d in durations if d > 0]
        if not durations:
            return 0.4
        avg = sum(durations) / len(durations)
        # JD penalizes job-hopping (<~1.5 yr cadence); ~2 yr avg ≈ stable.
        return _clamp01(avg / 24.0)

    return StructuredFeature("tenure_stability", fn, default=0.4)


def _availability_composite(reqs: JDRequirements, reference_date: dt.date):
    def fn(job: Job, c: Candidate) -> float:
        s = _signals(c)
        last_active = _parse_date(s.get("last_active_date"))
        if last_active is not None:
            days = max(0, (reference_date - last_active).days)
            recency = math.exp(-days / 60.0)  # ~2-month half-life-ish decay
        else:
            recency = 0.3
        response = float(s.get("recruiter_response_rate", 0.0) or 0.0)
        interview = float(s.get("interview_completion_rate", 0.0) or 0.0)
        otw = 1.0 if s.get("open_to_work_flag") else 0.4
        notice = s.get("notice_period_days")
        notice_fit = 1.0 if (isinstance(notice, (int, float)) and notice <= reqs.preferred_notice_max_days) \
            else _clamp01(1.0 - (float(notice or 90) - reqs.preferred_notice_max_days) / 150.0)
        # The JD frames availability as a modifier: an unreachable candidate
        # is "for hiring purposes, not actually available".
        return _clamp01(
            0.35 * recency + 0.30 * response + 0.15 * interview
            + 0.10 * otw + 0.10 * notice_fit
        )

    return StructuredFeature("availability_composite", fn, default=0.3)


def _engagement_market():
    def _log_norm(x: float, scale: float) -> float:
        return _clamp01(math.log1p(max(0.0, x)) / math.log1p(scale))

    def fn(job: Job, c: Candidate) -> float:
        s = _signals(c)
        saved = _log_norm(float(s.get("saved_by_recruiters_30d", 0) or 0), 50)
        views = _log_norm(float(s.get("profile_views_received_30d", 0) or 0), 500)
        appear = _log_norm(float(s.get("search_appearance_30d", 0) or 0), 500)
        return _clamp01(0.5 * saved + 0.25 * views + 0.25 * appear)

    return StructuredFeature("engagement_market", fn, default=0.2)


def _github_signal():
    def fn(job: Job, c: Candidate) -> float:
        g = _signals(c).get("github_activity_score", -1)
        # -1 is a sentinel ("no GitHub linked"), NOT a zero score.  The JD
        # values external validation but doesn't require GitHub, so map the
        # sentinel to a neutral-low value instead of the bottom of the scale.
        if g is None or g < 0:
            return 0.3
        return _clamp01(float(g) / 100.0)

    return StructuredFeature("github_signal", fn, default=0.3)


def _location_fit(reqs: JDRequirements):
    def fn(job: Job, c: Candidate) -> float:
        p = _profile(c)
        loc = str(p.get("location", "")).lower()
        country = str(p.get("country", "")).lower()
        in_target = any(t in loc for t in reqs.target_locations)
        willing = bool(_signals(c).get("willing_to_relocate"))
        if in_target:
            return 1.0
        if country in ("india", "in"):
            return 0.7 if willing else 0.55
        # Outside India: JD is "case-by-case, no visa sponsorship".
        return 0.4 if willing else 0.2

    return StructuredFeature("location_fit", fn, default=0.5)


def build_feature_library(
    reqs: JDRequirements,
    reference_date: dt.date = _DEFAULT_REFERENCE_DATE,
) -> list[StructuredFeature]:
    """Assemble the ordered list of structured/behavioral features."""
    return [
        _experience_band_fit(reqs),
        _product_vs_services(reqs),
        _shipped_systems_evidence(reqs),
        _role_coherence(reqs),
        _skill_credibility(reqs),
        _tenure_stability(reqs),
        _availability_composite(reqs, reference_date),
        _engagement_market(),
        _github_signal(),
        _location_fit(reqs),
    ]


# ─── hard disqualifiers (multiplicative guard, not a blended feature) ─────

def hard_disqualifier_penalty(
    raw: dict,
    reqs: JDRequirements,
) -> tuple[float, list[str]]:
    """Return ``(penalty in [0,1], reasons)`` for JD hard-DQ patterns.

    These are the JD's explicit "we will not / will probably not move
    forward" cases.  Used as a multiplicative guard in ``rank.py`` so a
    disqualified candidate is pushed out of contention regardless of skill
    keywords — the central anti-trap of this challenge.
    """
    reasons: list[str] = []
    penalty = 0.0
    profile = raw.get("profile", {}) or {}
    history = raw.get("career_history", []) or []
    title = str(profile.get("current_title", "")).lower()
    blob = " ".join(
        f"{s.get('title', '')} {s.get('description', '')}" for s in history
    ).lower() + " " + str(profile.get("summary", "")).lower()

    # Consulting-only career: every stint at a services firm.
    companies = [str(s.get("company", "")).lower() for s in history]
    if companies and all(
        any(firm in name for firm in reqs.consulting_only_firms) for name in companies
    ):
        penalty = max(penalty, 0.85)
        reasons.append("consulting-only career (no product-company experience)")

    # Off-target current role (e.g. Marketing Manager with AI keywords).
    eng_terms = ("engineer", "scientist", "developer", "ml", "machine learning",
                 "ai", "data", "nlp", "research", "architect")
    if any(t in title for t in reqs.off_target_title_terms) and not any(t in title for t in eng_terms):
        penalty = max(penalty, 0.8)
        reasons.append(f"current role is off-target for an AI-engineering hire ({title})")

    # Pure-research career without production deployment (JD: "pure research
    # environments ... without any production deployment — we will not move
    # forward").  Fires only when *every* stint looks research/academic.
    def _is_research(stint: dict) -> bool:
        text = f"{stint.get('title', '')} {stint.get('industry', '')}".lower()
        return any(t in text for t in reqs.research_only_terms) or "education" in text
    if history and all(_is_research(s) for s in history):
        has_shipped = any(p in blob for p in reqs.shipped_systems_phrases)
        if not has_shipped:
            penalty = max(penalty, 0.7)
            reasons.append("pure-research career without production deployment")

    # Out-of-domain (CV/speech/robotics) without NLP/IR exposure.
    if any(t in blob for t in reqs.out_of_domain_terms):
        has_nlp_ir = any(t in blob for t in ("nlp", "retrieval", "ranking", "search",
                                             "recommendation", "information retrieval", "rag"))
        if not has_nlp_ir:
            penalty = max(penalty, 0.7)
            reasons.append("primary domain is CV/speech/robotics without NLP/IR exposure")

    # Below the experience hard floor.
    yexp = float(profile.get("years_of_experience", 0.0) or 0.0)
    if yexp < reqs.experience_hard_floor:
        penalty = max(penalty, 0.6)
        reasons.append(f"{yexp:.1f} yrs experience is below the role's floor")

    return penalty, reasons
