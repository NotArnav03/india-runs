"""Honeypot / profile-consistency detector (gap analysis item H).

The dataset embeds ~80 honeypot candidates with *subtly impossible* profiles,
forced to relevance tier 0 in the ground truth.  Ranking >10% of them into the
top 100 is an automatic Stage-3 disqualification (``submission_spec.docx`` §7).
So consistency is a **correctness** requirement, not polish.

We don't special-case the known honeypots — the spec says a good ranker should
avoid them naturally.  Instead we score *within-profile* contradictions, the
kind a profile-reading system would notice but a keyword-embedding system would
miss.  Each check is calibrated against the real sample, where: years_of_exp ≈
Σ(stint duration)/12, stint date spans match ``duration_months`` (±3mo), and no
advanced/expert skill has ``duration_months == 0``.

The returned penalty feeds a multiplicative guard in ``rank.py`` and the
acknowledged flags feed the reasoning generator.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

_DEFAULT_REFERENCE_DATE = dt.date(2026, 6, 4)
_ADV = {"advanced", "expert"}


@dataclass
class ConsistencyReport:
    """Outcome of the consistency audit for one profile."""

    penalty: float = 0.0           # in [0, 1]; higher = more likely a honeypot
    flags: list[str] = field(default_factory=list)

    @property
    def is_suspect(self) -> bool:
        return self.penalty >= 0.5


def _parse_date(s) -> Optional[dt.date]:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s))
    except (ValueError, TypeError):
        return None


def _months_between(start: dt.date, end: dt.date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def consistency_report(
    raw: dict,
    reference_date: dt.date = _DEFAULT_REFERENCE_DATE,
) -> ConsistencyReport:
    """Audit one raw profile for internal contradictions."""
    report = ConsistencyReport()
    profile = raw.get("profile", {}) or {}
    history = raw.get("career_history", []) or []
    skills = raw.get("skills", []) or []
    education = raw.get("education", []) or []

    def flag(weight: float, msg: str) -> None:
        report.penalty = min(1.0, report.penalty + weight)
        report.flags.append(msg)

    # 1. Expert/advanced proficiency with zero months of use.
    #    (Spec example: "'expert' proficiency in 10 skills with 0 years used".)
    zero_expert = [
        s.get("name") for s in skills
        if str(s.get("proficiency", "")) in _ADV and int(s.get("duration_months", 1) or 0) == 0
    ]
    if zero_expert:
        flag(0.6, f"{len(zero_expert)} advanced/expert skill(s) claimed with 0 months of use")

    # 2. Stated experience vs. summed career duration.
    summed_years = sum(int(s.get("duration_months", 0) or 0) for s in history) / 12.0
    yexp = float(profile.get("years_of_experience", 0.0) or 0.0)
    if summed_years - yexp > 3.0 or yexp - summed_years > 5.0:
        flag(0.5, f"years_of_experience ({yexp:.1f}) inconsistent with career history (~{summed_years:.1f} yrs)")

    # 3. Per-stint: duration_months must match the date span; dates must be sane.
    #    (Spec example: "8 years of experience at a company founded 3 years ago"
    #    surfaces here as duration_months >> (end - start).)
    for s in history:
        start = _parse_date(s.get("start_date"))
        end = _parse_date(s.get("end_date")) or (reference_date if s.get("is_current") else None)
        dur = int(s.get("duration_months", 0) or 0)
        if start and end:
            if end < start:
                flag(0.7, f"stint '{s.get('title', '')}' ends before it starts")
                continue
            span = _months_between(start, end)
            if dur - span > 6:  # claims far more tenure than the dates allow
                flag(0.6, f"stint '{s.get('title', '')}' claims {dur} months but spans only ~{span}")
        if start and start > reference_date:
            flag(0.5, f"stint '{s.get('title', '')}' starts in the future")

    # 4. Assessment score flatly contradicts a top proficiency claim.
    assess = (raw.get("redrob_signals", {}) or {}).get("skill_assessment_scores", {}) or {}
    for s in skills:
        if str(s.get("proficiency", "")) == "expert":
            score = assess.get(s.get("name"))
            if isinstance(score, (int, float)) and score < 30:
                flag(0.3, f"'expert' in {s.get('name')} but assessment score is {score:.0f}/100")
                break  # one such contradiction is enough signal

    # 5. Education timeline impossible.
    for e in education:
        sy, ey = e.get("start_year"), e.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            flag(0.4, "education end year precedes start year")
            break

    return report
