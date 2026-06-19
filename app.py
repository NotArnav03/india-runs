"""Streamlit sandbox + demo for the Redrob Track-1 ranker (submission spec §10.5).

Upload a small candidate sample (JSONL, ≤100 rows) or use the built-in
JD-grounded archetypes, run the ranking system end-to-end on CPU, and download
the ranked CSV. The UI also exposes *why* each candidate ranks where it does —
the per-signal breakdown that drives the score — which is the story we defend
at Stage 5. The ranking logic is identical to `rank.py` (shared
`score_candidates`); this module only presents it.

Run locally:   streamlit run app.py
Deploy free:   Streamlit Community Cloud / HuggingFace Spaces (CPU tier).
"""

from __future__ import annotations

import datetime as dt
import io
import json

import streamlit as st

from ir.adapters import candidate_from_raw, build_submission_rows
from ir.features import Candidate
from ir.features_library import build_feature_library
from ir.honeypot import consistency_report
from ir.jd_requirements import REDROB_SENIOR_AI_ENGINEER as REQS
from ir.reasoning import build_reasoning
from rank import score_candidates

REF = dt.date(2026, 6, 4)

# Friendly labels + plain-language explanations for the 10 structured/behavioral
# signals (all in [0,1], higher = better).
SIGNAL_LABELS = {
    "shipped_systems_evidence": "Shipped retrieval / ranking",
    "skill_credibility": "Skill credibility",
    "role_coherence": "Role relevance",
    "product_vs_services": "Product vs. services",
    "experience_band_fit": "Experience fit (5–9 yrs)",
    "availability_composite": "Availability",
    "engagement_market": "Recruiter engagement",
    "tenure_stability": "Tenure stability",
    "github_signal": "GitHub activity",
    "location_fit": "Location fit",
}
SIGNAL_DESCRIPTIONS = {
    "shipped_systems_evidence": "Evidence in the career history of actually building/shipping "
        "retrieval, ranking, search or recommendation systems — the JD's real target.",
    "skill_credibility": "Depth of JD-relevant skills, weighted by proficiency, endorsements, "
        "months of use and assessment scores — rewards demonstrated skill, not just a listed keyword.",
    "role_coherence": "How well the current title/headline fits an AI-engineering role; "
        "penalizes off-target roles (e.g. Marketing) and CV/speech/robotics titles without NLP/IR.",
    "product_vs_services": "Share of career spent at product companies vs. IT-services / "
        "consulting firms — the JD strongly prefers product experience.",
    "experience_band_fit": "Closeness to the role's 5–9 year target band (a soft preference, not a gate).",
    "availability_composite": "How reachable they actually are: recent activity, recruiter response "
        "rate, open-to-work, interview completion, and notice period.",
    "engagement_market": "Recruiter-side demand: profile saves, views and search appearances in the last 30 days.",
    "tenure_stability": "Average time per role — low values flag job-hopping the JD warns against.",
    "github_signal": "GitHub activity as external validation of public work (neutral if no GitHub is linked).",
    "location_fit": "Fit to Noida/Pune (or willingness to relocate, or at least an India base).",
}

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide", page_icon="🎯")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem;}
      .hero {background: linear-gradient(135deg,#0f172a 0%,#1e3a8a 100%);
             padding: 1.4rem 1.6rem; border-radius: 14px; color:#fff; margin-bottom:1rem;}
      .hero h1 {margin:0; font-size:1.6rem;}
      .hero p {margin:.4rem 0 0; opacity:.85; font-size:.95rem;}
      .rankbadge {display:inline-flex; align-items:center; justify-content:center;
                  width:42px; height:42px; border-radius:50%; font-weight:700;
                  color:#fff; font-size:1.05rem;}
      .pill {display:inline-block; padding:2px 10px; border-radius:999px;
             font-size:.72rem; font-weight:600; margin:2px 4px 2px 0;}
      .bar-wrap {background:#e5e7eb; border-radius:6px; height:9px; width:100%; margin:2px 0 3px;}
      .bar-fill {height:9px; border-radius:6px; background:linear-gradient(90deg,#3b82f6,#22c55e);}
      .siglabel {font-size:.78rem; color:#374151; display:flex; justify-content:space-between; font-weight:600;}
      .sigdesc {font-size:.72rem; color:#6b7280; margin:0 0 12px; line-height:1.25;}
      .muted {color:#6b7280; font-size:.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎯 Redrob — Intelligent Candidate Discovery</h1>
      <p>Multi-signal ranker on the FAIMR core: SBERT semantics + anchored skill match
      + 23 behavioral signals + career-pattern features, with honeypot and
      JD-disqualifier guards. CPU-only, no network at rank time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("📋 What this role wants (the JD model the ranker uses)"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Must-have signals**")
        st.write("Production embeddings/retrieval, vector DB / hybrid search, strong "
                 "Python, rigorous ranking evaluation (NDCG/MRR/MAP), shipped a "
                 "ranking/search/recsys system at a product company. Target band 5–9 yrs.")
    with c2:
        st.markdown("**Hard disqualifiers (down-ranked)**")
        st.write("Consulting-only careers, pure research without production, "
                 "CV/speech/robotics without NLP/IR, off-target roles (e.g. Marketing), "
                 "and the dataset's planted honeypots (impossible profiles).")

with st.sidebar:
    st.header("⚙️ Controls")
    embedder = st.selectbox(
        "Embedder", ["hashing", "sbert"], index=0,
        help="'hashing' is model-free and instant (best for the free sandbox tier); "
             "'sbert' loads the transformer for higher semantic quality.",
    )
    top_n = st.slider("How many to show", 5, 100, 10)
    uploaded = st.file_uploader("Candidates JSONL (≤100 rows)", type=["jsonl", "json"])
    st.caption("No upload? It ranks the built-in JD archetypes so you can see it work.")
    run = st.button("🚀 Rank candidates", type="primary", use_container_width=True)


def _load_pool():
    if uploaded is not None:
        raws = []
        for line in io.TextIOWrapper(uploaded, encoding="utf-8"):
            line = line.strip()
            if line:
                raws.append(json.loads(line))
        return raws, f"{len(raws)} uploaded candidates"
    from evaluation.judgment_archetypes import build_judgment_set
    raws = [raw for raw, _ in build_judgment_set()]
    return raws, f"{len(raws)} built-in JD archetypes"


def _signal_bars(values: dict[str, float]) -> str:
    """HTML for the labeled [0,1] signal bars, strongest first."""
    items = sorted(values.items(), key=lambda kv: -kv[1])
    html = ""
    for name, val in items:
        label = SIGNAL_LABELS.get(name, name)
        desc = SIGNAL_DESCRIPTIONS.get(name, "")
        pct = max(0, min(100, int(round(val * 100))))
        html += (
            f"<div class='siglabel'><span>{label}</span><span>{pct}%</span></div>"
            f"<div class='bar-wrap'><div class='bar-fill' style='width:{pct}%'></div></div>"
            f"<div class='sigdesc'>{desc}</div>"
        )
    return html


def _badges(raw: dict) -> str:
    s = raw.get("redrob_signals", {}) or {}
    out = []

    def pill(text, color):
        out.append(f"<span class='pill' style='background:{color}22;color:{color}'>{text}</span>")

    resp = s.get("recruiter_response_rate")
    if isinstance(resp, (int, float)):
        pill(f"{resp:.0%} response", "#16a34a" if resp >= 0.5 else "#d97706" if resp >= 0.2 else "#dc2626")
    last = s.get("last_active_date")
    if last:
        try:
            d = (REF - dt.date.fromisoformat(str(last))).days
            pill("active <2wks" if d <= 14 else f"active ~{d//30}mo ago",
                 "#16a34a" if d <= 30 else "#d97706" if d <= 120 else "#dc2626")
        except (ValueError, TypeError):
            pass
    if s.get("open_to_work_flag"):
        pill("open to work", "#2563eb")
    notice = s.get("notice_period_days")
    if isinstance(notice, (int, float)):
        pill(f"{int(notice)}d notice", "#16a34a" if notice <= 30 else "#d97706")
    g = s.get("github_activity_score")
    if isinstance(g, (int, float)) and g >= 0:
        pill(f"GitHub {int(g)}", "#7c3aed")
    if s.get("verified_email") and s.get("verified_phone"):
        pill("verified", "#0891b2")
    return "".join(out)


if run:
    raws, src = _load_pool()
    raw_by_id = {r["candidate_id"]: r for r in raws}
    with st.spinner(f"Ranking {src} with the '{embedder}' embedder…"):
        candidates = [candidate_from_raw(r) for r in raws]
        final, guard = score_candidates(candidates, embedder_name=embedder)
        order = sorted(final, key=lambda i: (-final[i], i))[:top_n]

        # Per-candidate structured signals for the shown set (display only).
        job = REQS.build_job()
        lib = build_feature_library(REQS, REF)
        flagged_pool = sum(1 for r in raws if consistency_report(r, REF).is_suspect)

    # ── summary metrics ───────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates ranked", len(raws))
    m2.metric("Shown", len(order))
    m3.metric("Top score", f"{final[order[0]]:.3f}")
    m4.metric("Flagged & down-ranked", flagged_pool,
              help="Profiles failing internal consistency checks (likely honeypots); "
                   "penalized so they sink out of contention.")

    table_rows = []
    shortlist_tab, table_tab = st.tabs(["🏅 Ranked shortlist", "📄 Table / CSV"])

    top_score = max(final[i] for i in order) or 1.0
    with shortlist_tab:
        for rank_pos, cid in enumerate(order, start=1):
            raw = raw_by_id[cid]
            prof = raw.get("profile", {}) or {}
            _, penalty, concerns = guard[cid]
            reasoning = build_reasoning(raw, rank_pos, REQS, concerns, REF)
            table_rows.append({"rank": rank_pos, "candidate_id": cid,
                               "score": round(final[cid], 4), "reasoning": reasoning})
            cand = Candidate(id=cid, text="", skills=set(), metadata={"raw": raw})
            sigvals = {f.name: f.compute(job, cand) for f in lib}

            color = "#16a34a" if rank_pos <= len(order) * 0.3 else \
                    "#2563eb" if rank_pos <= len(order) * 0.7 else "#6b7280"
            with st.container(border=True):
                head, score_col = st.columns([4, 1])
                with head:
                    st.markdown(
                        f"<span class='rankbadge' style='background:{color}'>{rank_pos}</span> "
                        f"&nbsp;<b>{prof.get('current_title','—')}</b> · "
                        f"{prof.get('years_of_experience','?')} yrs · "
                        f"{prof.get('location','—')} "
                        f"<span class='muted'>({cid})</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(_badges(raw), unsafe_allow_html=True)
                with score_col:
                    st.markdown(f"<div class='muted'>score</div><h3 style='margin:0'>{final[cid]:.3f}</h3>",
                                unsafe_allow_html=True)
                    st.markdown(f"<div class='bar-wrap'><div class='bar-fill' "
                                f"style='width:{int(final[cid]/top_score*100)}%'></div></div>",
                                unsafe_allow_html=True)
                st.write(reasoning)
                with st.expander("Why ranked here — signal breakdown"):
                    st.markdown(_signal_bars(sigvals), unsafe_allow_html=True)
                    if penalty > 0:
                        st.caption(f"Applied a {penalty:.0%} guard penalty (JD-fit / consistency).")

    with table_tab:
        st.dataframe(table_rows, use_container_width=True, hide_index=True,
                     column_config={"score": st.column_config.NumberColumn(format="%.4f")})
        rows = build_submission_rows((r["candidate_id"], r["score"], r["reasoning"]) for r in table_rows)
        csv_buf = io.StringIO()
        csv_buf.write("candidate_id,rank,score,reasoning\n")
        for r in rows:
            csv_buf.write(f'{r["candidate_id"]},{r["rank"]},{r["score"]:.6f},'
                          f'"{r["reasoning"].replace(chr(34), chr(34)*2)}"\n')
        st.download_button("⬇️ Download ranked CSV", csv_buf.getvalue(),
                           file_name="submission_sample.csv", mime="text/csv",
                           use_container_width=True)
else:
    st.info("Set your options in the sidebar and press **🚀 Rank candidates**. "
            "With no upload it ranks the built-in JD archetypes so you can see the "
            "system separate true fits from keyword-stuffers and honeypots.")
