"""Streamlit sandbox for the Redrob Track-1 ranker (submission spec §10.5).

A minimal hosted demo: upload a small candidate sample (JSONL, ≤100 rows) or
use the built-in JD-grounded archetypes, run the ranking system end-to-end on
CPU, and download the ranked CSV.  This is the "small-sample reproducibility"
check the organizers ask for — it does not need to handle the full 100K pool.

Run locally:   streamlit run app.py
Deploy free:   HuggingFace Spaces / Streamlit Community Cloud (CPU tier).
"""

from __future__ import annotations

import io
import json

import streamlit as st

from ir.adapters import candidate_from_raw, build_submission_rows
from ir.jd_requirements import REDROB_SENIOR_AI_ENGINEER as REQS
from ir.reasoning import build_reasoning
from rank import score_candidates


st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob — Intelligent Candidate Discovery (Track 1)")
st.caption(
    "Multi-signal ranker on the FAIMR core: SBERT semantics + anchored skill "
    "match + 23 behavioral signals + career-pattern features, with honeypot "
    "and JD-disqualifier guards. CPU-only, no network at rank time."
)

with st.sidebar:
    st.header("Input")
    embedder = st.selectbox("Embedder", ["hashing", "sbert"], index=0,
                            help="'hashing' is model-free (instant); 'sbert' needs the model cached.")
    top_n = st.slider("Top N to show", 5, 100, 10)
    uploaded = st.file_uploader("Candidates JSONL (≤100 rows)", type=["jsonl", "json"])


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
    return raws, f"{len(raws)} built-in JD archetypes (no upload provided)"


if st.button("Rank candidates", type="primary"):
    raws, src = _load_pool()
    st.info(f"Ranking {src} with the '{embedder}' embedder…")
    candidates = [candidate_from_raw(r) for r in raws]
    raw_by_id = {r["candidate_id"]: r for r in raws}

    final, guard = score_candidates(candidates, embedder_name=embedder)
    order = sorted(final, key=lambda i: (-final[i], i))[:top_n]

    table = []
    for rank_pos, cid in enumerate(order, start=1):
        _, _, concerns = guard[cid]
        table.append({
            "rank": rank_pos,
            "candidate_id": cid,
            "score": round(final[cid], 4),
            "reasoning": build_reasoning(raw_by_id[cid], rank_pos, REQS, concerns),
        })
    st.dataframe(table, use_container_width=True, hide_index=True)

    rows = build_submission_rows((r["candidate_id"], r["score"], r["reasoning"]) for r in table)
    csv_buf = io.StringIO()
    csv_buf.write("candidate_id,rank,score,reasoning\n")
    for r in rows:
        reasoning = r["reasoning"].replace('"', '""')
        csv_buf.write(f'{r["candidate_id"]},{r["rank"]},{r["score"]:.6f},"{reasoning}"\n')
    st.download_button("Download ranked CSV", csv_buf.getvalue(),
                       file_name="submission_sample.csv", mime="text/csv")
else:
    st.write("Upload a small JSONL sample or just press **Rank candidates** to "
             "rank the built-in JD archetypes.")
