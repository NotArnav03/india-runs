"""Streamlit sandbox + demo for the Redrob Track-1 ranker (submission spec §10.5).

Upload a small candidate sample (JSONL, ≤100 rows) or use the built-in
JD-grounded archetypes, run the ranking system end-to-end on CPU, and download
the ranked CSV. The UI exposes *why* each candidate ranks where it does — the
per-signal breakdown that drives the score — which is the story we defend at
Stage 5. Ranking logic is identical to `rank.py` (shared `score_candidates`);
this module only presents it.

Design system (via ui-ux-pro-max): "Data-Dense Dashboard", light theme,
blue #1E40AF + amber #D97706, Fira Sans / Fira Code.

Run locally:   streamlit run app.py
Deploy free:   Streamlit Community Cloud / HuggingFace Spaces (CPU tier).
"""

from __future__ import annotations

import datetime as dt
import html
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
    "shipped_systems_evidence": "Evidence in the career history of actually building or shipping "
        "retrieval, ranking, search or recommendation systems — the JD's real target.",
    "skill_credibility": "Depth of JD-relevant skills, weighted by proficiency, endorsements, "
        "months of use and assessment scores — rewards demonstrated skill, not a listed keyword.",
    "role_coherence": "How well the current title and headline fit an AI-engineering role; "
        "penalises off-target roles (e.g. Marketing) and CV/speech/robotics titles without NLP/IR.",
    "product_vs_services": "Share of the career spent at product companies versus IT-services / "
        "consulting firms — the JD strongly prefers product experience.",
    "experience_band_fit": "Closeness to the role's 5–9 year target band (a soft preference, not a gate).",
    "availability_composite": "How reachable they actually are: recent activity, recruiter response "
        "rate, open-to-work, interview completion and notice period.",
    "engagement_market": "Recruiter-side demand: profile saves, views and search appearances in the last 30 days.",
    "tenure_stability": "Average time per role — low values flag the job-hopping the JD warns against.",
    "github_signal": "GitHub activity as external validation of public work (neutral if no GitHub is linked).",
    "location_fit": "Fit to Noida / Pune (or willingness to relocate, or at least an India base).",
}

st.set_page_config(page_title="Redrob — Candidate Discovery", layout="wide")

# Lucide-style inline SVG icons (no emoji — design-system anti-pattern).
ICON_TARGET = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
               'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
               '<circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>')


def _icon(path: str, size: int = 16) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{path}</svg>')


I_BRIEF = _icon('<path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/>'
                '<rect x="9" y="3" width="6" height="4" rx="1"/>')
I_CHECK = _icon('<path d="M20 6 9 17l-5-5"/>')
I_X = _icon('<path d="M18 6 6 18"/><path d="M6 6l12 12"/>')

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
      :root{
        --bg:#F8FAFC; --surface:#FFFFFF; --ink:#0F172A; --muted:#64748B;
        --brand:#1E40AF; --brand2:#3B82F6; --accent:#D97706; --navy:#1E3A8A;
        --border:#E2E8F0; --track:#EEF2F7;
      }
      html, body, .stApp, [class*="css"]{font-family:'Fira Sans',system-ui,sans-serif;}
      .stApp{background:var(--bg);}
      .block-container{padding-top:1.4rem; max-width:1180px;}
      .mono{font-family:'Fira Code',monospace; font-variant-numeric:tabular-nums;}

      /* hero */
      .hero{background:linear-gradient(135deg,#172554 0%,#1E40AF 55%,#2563EB 100%);
            border-radius:18px; padding:26px 30px; color:#fff;
            box-shadow:0 12px 30px -10px rgba(30,64,175,.45); margin-bottom:18px;}
      .hero-row{display:flex; align-items:center; gap:12px;}
      .hero-mark{display:flex; align-items:center; justify-content:center; width:44px; height:44px;
                 border-radius:12px; background:rgba(255,255,255,.14); color:#fff;}
      .hero h1{margin:0; font-size:1.5rem; font-weight:700; letter-spacing:-.01em;}
      .hero .tag{font-size:.74rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase;
                 color:#BFDBFE;}
      .hero p{margin:.7rem 0 0; opacity:.9; font-size:.92rem; max-width:760px; line-height:1.5;}

      /* metric tiles */
      .tiles{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:4px 0 18px;}
      .tile{background:var(--surface); border:1px solid var(--border); border-top:3px solid var(--brand);
            border-radius:14px; padding:14px 16px; box-shadow:0 1px 2px rgba(15,23,42,.04);}
      .tile.amber{border-top-color:var(--accent);}
      .tile .lbl{font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);}
      .tile .val{font-size:1.9rem; font-weight:700; color:var(--ink); margin-top:2px;}

      /* candidate card */
      .card{background:var(--surface); border:1px solid var(--border); border-radius:16px;
            padding:18px 20px; margin-bottom:14px; box-shadow:0 1px 2px rgba(15,23,42,.04);
            transition:box-shadow .2s ease, border-color .2s ease, transform .2s ease;
            animation:cardIn .45s cubic-bezier(.2,.8,.2,1) both;}
      .card:hover{transform:translateY(-2px);}
      .card:hover{border-color:#C7D2FE; box-shadow:0 10px 28px -12px rgba(30,64,175,.30);}
      .card-top{display:flex; align-items:flex-start; gap:14px;}
      .rk{flex:0 0 auto; width:42px; height:42px; border-radius:11px; display:flex; align-items:center;
          justify-content:center; font-weight:700; font-size:1.05rem; color:#fff;}
      .rk.gold{background:linear-gradient(135deg,#D97706,#F59E0B);}
      .rk.blue{background:linear-gradient(135deg,#1E40AF,#3B82F6);}
      .rk.slate{background:linear-gradient(135deg,#475569,#64748B);}
      .who{flex:1 1 auto; min-width:0;}
      .who .title{font-size:1.05rem; font-weight:700; color:var(--ink);}
      .who .meta{font-size:.86rem; color:var(--muted); margin-top:1px;}
      .who .cid{color:#94A3B8;}
      .score-box{flex:0 0 auto; text-align:right; min-width:104px;}
      .score-box .lbl{font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);}
      .score-box .val{font-size:1.7rem; font-weight:700; color:var(--brand); line-height:1.1;}

      .pills{margin:11px 0 4px; display:flex; flex-wrap:wrap; gap:6px;}
      .pill{display:inline-flex; align-items:center; padding:3px 10px; border-radius:999px;
            font-size:.73rem; font-weight:600; border:1px solid transparent;}
      .reason{font-size:.95rem; color:#334155; line-height:1.55; margin-top:8px;}

      .bar{background:var(--track); border-radius:999px; height:8px; width:100%; overflow:hidden;}
      .bar > span{display:block; height:8px; border-radius:999px;
                  background:linear-gradient(90deg,#3B82F6,#1E40AF);
                  transform-origin:left; animation:barGrow .8s cubic-bezier(.2,.8,.2,1) both;}

      /* radial fit-score gauge */
      .gauge{width:74px; height:74px; border-radius:50%; margin-left:auto;
             background:conic-gradient(var(--brand) calc(var(--p)*1%), var(--track) 0);
             display:flex; align-items:center; justify-content:center;
             animation:gaugeIn .5s cubic-bezier(.2,.8,.2,1) both;}
      .gauge.accent{background:conic-gradient(var(--accent) calc(var(--p)*1%), var(--track) 0);}
      .gauge-inner{width:56px; height:56px; border-radius:50%; background:var(--surface);
                   display:flex; align-items:center; justify-content:center; font-weight:700;
                   color:var(--ink); font-size:.98rem; box-shadow:inset 0 0 0 1px var(--border);}
      .topchip{display:inline-block; margin-left:8px; padding:1px 8px; border-radius:999px;
               font-size:.64rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
               color:#B45309; background:#FFFBEB; border:1px solid #FDE68A; vertical-align:middle;}

      details.sig{margin-top:12px; border-top:1px dashed var(--border); padding-top:10px;}
      details.sig > summary{cursor:pointer; list-style:none; color:var(--brand); font-weight:600;
                            font-size:.86rem; display:inline-flex; align-items:center; gap:6px;}
      details.sig > summary::-webkit-details-marker{display:none;}
      details.sig[open] > summary{margin-bottom:10px;}
      .sigrow{margin-bottom:14px;}
      .sigrow .top{display:flex; justify-content:space-between; font-size:.86rem; font-weight:600; color:var(--ink);}
      .sigrow .top .pct{color:var(--muted);}
      .sigrow .desc{font-size:.86rem; color:var(--muted); line-height:1.45; margin-top:4px;}
      .guard{font-size:.8rem; color:var(--accent); font-weight:600; margin-top:4px;}

      /* JD brief card */
      details.brief{background:var(--surface); border:1px solid var(--border); border-radius:14px;
                    padding:14px 18px; margin-bottom:16px;}
      details.brief > summary{cursor:pointer; list-style:none; font-weight:600; color:var(--ink);
                              display:inline-flex; align-items:center; gap:8px;}
      details.brief > summary::-webkit-details-marker{display:none;}
      .brief-grid{display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:12px;}
      .brief-grid h4{margin:0 0 4px; font-size:.8rem; letter-spacing:.04em; text-transform:uppercase;}
      .brief-grid .ok{color:#047857;} .brief-grid .no{color:#B91C1C;}
      .brief-grid p{margin:0; font-size:.9rem; color:#475569; line-height:1.5;}

      /* sidebar + button */
      section[data-testid="stSidebar"]{background:#FFFFFF; border-right:1px solid var(--border);}
      .sb-title{font-size:.74rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
                color:var(--muted); margin:.2rem 0 .4rem;}
      .stButton>button{background:var(--brand); color:#fff; border:0; border-radius:10px;
                       font-weight:600; padding:.55rem 1rem; transition:background .15s ease;}
      .stButton>button:hover{background:#1B379B; color:#fff;}
      .stTabs [data-baseweb="tab-list"]{gap:6px;}
      .stTabs [data-baseweb="tab"]{font-weight:600;}

      /* entrance + ambient motion (transform/opacity only) */
      @keyframes cardIn{from{opacity:0; transform:translateY(12px);} to{opacity:1; transform:none;}}
      @keyframes tileIn{from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:none;}}
      @keyframes gaugeIn{from{opacity:0; transform:scale(.85);} to{opacity:1; transform:none;}}
      @keyframes barGrow{from{transform:scaleX(0);} to{transform:scaleX(1);}}
      @keyframes heroSheen{0%{transform:translateX(-60%);} 60%,100%{transform:translateX(260%);}}
      @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.55);}
                       70%{box-shadow:0 0 0 8px rgba(52,211,153,0);} 100%{box-shadow:0 0 0 0 rgba(52,211,153,0);}}
      .tile{animation:tileIn .42s ease-out both;}
      .tile:nth-child(2){animation-delay:.05s;} .tile:nth-child(3){animation-delay:.1s;}
      .tile:nth-child(4){animation-delay:.15s;}
      .hero{position:relative; overflow:hidden;}
      .hero::before{content:''; position:absolute; top:-45%; right:-8%; width:340px; height:340px;
                    border-radius:50%; background:radial-gradient(circle,rgba(217,119,6,.30),transparent 70%);
                    pointer-events:none;}
      .hero::after{content:''; position:absolute; inset:0 auto 0 0; width:38%; pointer-events:none;
                   background:linear-gradient(100deg,transparent,rgba(255,255,255,.13),transparent);
                   transform:translateX(-60%); animation:heroSheen 6s ease-in-out 1.2s infinite;}
      .hero-row, .hero p{position:relative; z-index:1;}
      .dot{display:inline-block; width:7px; height:7px; border-radius:50%; background:#34D399;
           margin-right:7px; animation:pulse 2.2s infinite;}

      @media (prefers-reduced-motion: reduce){
        .card,.tile,.gauge,.bar>span,.hero::after,.dot{animation:none!important;}
        .card,.stButton>button{transition:none;} .card:hover{transform:none;}
      }
      @media (max-width:780px){ .tiles{grid-template-columns:repeat(2,1fr);} .brief-grid{grid-template-columns:1fr;} }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── hero ──────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-row">
        <div class="hero-mark">{ICON_TARGET}</div>
        <div>
          <div class="tag"><span class="dot"></span>Talent Intelligence · Track 1</div>
          <h1>Redrob — Intelligent Candidate Discovery</h1>
        </div>
      </div>
      <p>A multi-signal ranker on the FAIMR core: SBERT semantics, anchored skill matching,
      23 behavioural signals and career-pattern features — with honeypot and JD-disqualifier
      guards. CPU-only, no network at rank time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <details class="brief">
      <summary>{I_BRIEF}&nbsp; What this role wants — the JD model the ranker uses</summary>
      <div class="brief-grid">
        <div>
          <h4 class="ok">Must-have signals</h4>
          <p>Production embeddings / retrieval, vector DB or hybrid search, strong Python,
          rigorous ranking evaluation (NDCG / MRR / MAP), and a shipped ranking / search /
          recsys system at a product company. Target band 5–9 years.</p>
        </div>
        <div>
          <h4 class="no">Hard disqualifiers — down-ranked</h4>
          <p>Consulting-only careers, pure research without production, CV / speech / robotics
          without NLP-IR, off-target roles (e.g. Marketing), and the dataset's planted
          honeypots (impossible profiles).</p>
        </div>
      </div>
    </details>
    """,
    unsafe_allow_html=True,
)

# ── sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">Controls</div>', unsafe_allow_html=True)
    embedder = st.selectbox(
        "Embedder", ["hashing", "sbert"], index=0,
        help="'hashing' is model-free and instant (best for the free sandbox tier); "
             "'sbert' loads the transformer for higher semantic quality.",
    )
    top_n = st.slider("How many to show", 5, 100, 10)
    uploaded = st.file_uploader("Candidates JSONL (≤100 rows)", type=["jsonl", "json"])
    st.caption("No upload? It ranks the built-in JD archetypes so you can see it work.")
    run = st.button("Rank candidates", type="primary", use_container_width=True)


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


PILL_COLORS = {  # (text, background, border)
    "green": ("#047857", "#ECFDF5", "#A7F3D0"),
    "amber": ("#B45309", "#FFFBEB", "#FDE68A"),
    "red":   ("#B91C1C", "#FEF2F2", "#FECACA"),
    "blue":  ("#1D4ED8", "#EFF6FF", "#BFDBFE"),
    "violet":("#6D28D9", "#F5F3FF", "#DDD6FE"),
    "teal":  ("#0E7490", "#ECFEFF", "#A5F3FC"),
}


def _pill(text: str, color: str) -> str:
    fg, bg, bd = PILL_COLORS[color]
    return f"<span class='pill' style='color:{fg};background:{bg};border-color:{bd}'>{html.escape(text)}</span>"


def _badges(raw: dict) -> str:
    s = raw.get("redrob_signals", {}) or {}
    out = []
    resp = s.get("recruiter_response_rate")
    if isinstance(resp, (int, float)):
        out.append(_pill(f"{resp:.0%} response", "green" if resp >= 0.5 else "amber" if resp >= 0.2 else "red"))
    last = s.get("last_active_date")
    if last:
        try:
            d = (REF - dt.date.fromisoformat(str(last))).days
            txt = "active <2 wks" if d <= 14 else f"active ~{d // 30} mo ago"
            out.append(_pill(txt, "green" if d <= 30 else "amber" if d <= 120 else "red"))
        except (ValueError, TypeError):
            pass
    if s.get("open_to_work_flag"):
        out.append(_pill("open to work", "blue"))
    notice = s.get("notice_period_days")
    if isinstance(notice, (int, float)):
        out.append(_pill(f"{int(notice)}d notice", "green" if notice <= 30 else "amber"))
    g = s.get("github_activity_score")
    if isinstance(g, (int, float)) and g >= 0:
        out.append(_pill(f"GitHub {int(g)}", "violet"))
    if s.get("verified_email") and s.get("verified_phone"):
        out.append(_pill("verified", "teal"))
    return "".join(out)


def _signal_rows(values: dict[str, float]) -> str:
    rows = ""
    for name, val in sorted(values.items(), key=lambda kv: -kv[1]):
        pct = max(0, min(100, int(round(val * 100))))
        rows += (
            f"<div class='sigrow'><div class='top'><span>{html.escape(SIGNAL_LABELS.get(name, name))}</span>"
            f"<span class='pct mono'>{pct}%</span></div>"
            f"<div class='bar'><span style='width:{pct}%'></span></div>"
            f"<div class='desc'>{html.escape(SIGNAL_DESCRIPTIONS.get(name, ''))}</div></div>"
        )
    return rows


def _card(rank_pos: int, cid: str, raw: dict, score: float, score_pct: int,
          reasoning: str, sigvals: dict, penalty: float) -> str:
    prof = raw.get("profile", {}) or {}
    rk_cls = "gold" if rank_pos <= 3 else "blue" if rank_pos <= 10 else "slate"
    title = html.escape(str(prof.get("current_title", "—")))
    yrs = prof.get("years_of_experience", "?")
    loc = html.escape(str(prof.get("location", "—")))
    guard = f"<div class='guard'>A {penalty:.0%} guard penalty was applied (JD-fit / consistency).</div>" if penalty > 0 else ""
    chip = "<span class='topchip'>Top match</span>" if rank_pos == 1 else ""
    gauge_cls = "gauge accent" if rank_pos <= 3 else "gauge"
    return (
        f"<div class='card' style='animation-delay:{(rank_pos - 1) * 0.05:.2f}s'><div class='card-top'>"
        f"<div class='rk {rk_cls}'>{rank_pos}</div>"
        f"<div class='who'><div class='title'>{title}{chip}</div>"
        f"<div class='meta'><span class='mono'>{yrs}</span> yrs · {loc} · "
        f"<span class='cid mono'>{html.escape(cid)}</span></div></div>"
        f"<div class='score-box'><div class='lbl'>Fit score</div>"
        f"<div class='{gauge_cls}' style='--p:{score_pct}'>"
        f"<div class='gauge-inner mono'>{score:.2f}</div></div></div>"
        f"</div>"
        f"<div class='pills'>{_badges(raw)}</div>"
        f"<div class='reason'>{html.escape(reasoning)}</div>"
        f"<details class='sig'><summary>Why ranked here — signal breakdown</summary>"
        f"{_signal_rows(sigvals)}{guard}</details>"
        f"</div>"
    )


if run:
    raws, src = _load_pool()
    raw_by_id = {r["candidate_id"]: r for r in raws}
    with st.spinner(f"Ranking {src} with the ‘{embedder}’ embedder…"):
        candidates = [candidate_from_raw(r) for r in raws]
        final, guard = score_candidates(candidates, embedder_name=embedder)
        order = sorted(final, key=lambda i: (-final[i], i))[:top_n]
        job = REQS.build_job()
        lib = build_feature_library(REQS, REF)
        flagged = sum(1 for r in raws if consistency_report(r, REF).is_suspect)

    st.markdown(
        f"""
        <div class="tiles">
          <div class="tile"><div class="lbl">Candidates ranked</div><div class="val mono">{len(raws)}</div></div>
          <div class="tile"><div class="lbl">Shown</div><div class="val mono">{len(order)}</div></div>
          <div class="tile amber"><div class="lbl">Top fit score</div><div class="val mono">{final[order[0]]:.3f}</div></div>
          <div class="tile" title="Profiles failing internal consistency checks (likely honeypots); penalised so they sink out of contention.">
            <div class="lbl">Flagged &amp; down-ranked</div><div class="val mono">{flagged}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_score = max(final[i] for i in order) or 1.0
    table_rows, cards_html = [], ""
    for rank_pos, cid in enumerate(order, start=1):
        raw = raw_by_id[cid]
        _, penalty, concerns = guard[cid]
        reasoning = build_reasoning(raw, rank_pos, REQS, concerns, REF)
        table_rows.append({"rank": rank_pos, "candidate_id": cid,
                           "score": round(final[cid], 4), "reasoning": reasoning})
        cand = Candidate(id=cid, text="", skills=set(), metadata={"raw": raw})
        sigvals = {f.name: f.compute(job, cand) for f in lib}
        cards_html += _card(rank_pos, cid, raw, final[cid],
                            int(final[cid] / top_score * 100), reasoning, sigvals, penalty)

    shortlist_tab, table_tab = st.tabs(["Ranked shortlist", "Table & export"])
    with shortlist_tab:
        st.markdown(cards_html, unsafe_allow_html=True)
    with table_tab:
        st.dataframe(table_rows, use_container_width=True, hide_index=True,
                     column_config={"score": st.column_config.NumberColumn(format="%.4f")})
        rows = build_submission_rows((r["candidate_id"], r["score"], r["reasoning"]) for r in table_rows)
        csv_buf = io.StringIO()
        csv_buf.write("candidate_id,rank,score,reasoning\n")
        for r in rows:
            csv_buf.write(f'{r["candidate_id"]},{r["rank"]},{r["score"]:.6f},'
                          f'"{r["reasoning"].replace(chr(34), chr(34) * 2)}"\n')
        st.download_button("Download ranked CSV", csv_buf.getvalue(),
                           file_name="submission_sample.csv", mime="text/csv",
                           use_container_width=True)
else:
    st.info("Set your options in the sidebar and press **Rank candidates**. With no upload it "
            "ranks the built-in JD archetypes so you can watch the system separate true fits "
            "from keyword-stuffers and honeypots.")
