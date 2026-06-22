"""Streamlit sandbox + demo for the Redrob Track-1 ranker (submission spec §10.5).

Upload a small candidate sample (JSONL, ≤100 rows) or use the built-in
JD-grounded archetypes, run the ranking system end-to-end on CPU, and download
the ranked CSV. The UI exposes *why* each candidate ranks where it does — the
per-signal breakdown that drives the score — which is the story we defend at
Stage 5. Ranking logic is identical to `rank.py` (shared `score_candidates`);
this module only presents it.

Design system: "Obsidian Ranker" — luxury matte dark theme. Deep obsidian
(#121316) + charcoal (#1E2023) with smoky-glass surfaces (backdrop-blur, low
opacity), soft grey accents (#4C4E51) and subtle glows.
Space Grotesk (UI) + Space Mono (tabular numerals).

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
I_INFO = _icon('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>', 18)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
      :root{
        /* Obsidian Ranker — luxury matte dark */
        --bg:#121316; --bg2:#0E0F12; --surface:#1E2023;
        --glass:rgba(30,32,35,.55); --glass-2:rgba(30,32,35,.38);
        --ink:#EDEEF0; --muted:#9C9EA1; --grey:#4C4E51;
        --silver:#C8CACD; --silver-2:#86888B;
        --border:rgba(255,255,255,.08); --border-2:rgba(255,255,255,.05); --track:rgba(255,255,255,.06);
        --glow:rgba(200,202,206,.22);
      }
      /* Force Space Grotesk over Streamlit's default on every text element
         (Streamlit sets font-family on markdown/heading containers at higher
         specificity, so html/body alone is overridden). */
      html, body, .stApp, [class*="css"],
      .stApp p, .stApp span, .stApp div, .stApp a, .stApp li, .stApp label,
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
      .stApp button, .stApp input, .stApp textarea, .stApp select,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-baseweb] *{
        font-family:'Space Grotesk',system-ui,sans-serif !important;
      }
      .stApp{background:
              radial-gradient(1100px 520px at 78% -8%, rgba(120,124,130,.12), transparent 60%),
              radial-gradient(900px 480px at 8% 4%, rgba(76,78,81,.16), transparent 55%),
              var(--bg);}
      .block-container{padding-top:1.4rem; max-width:1180px;}
      .stApp .mono, .mono{font-family:'Space Mono',monospace !important; font-variant-numeric:tabular-nums;}

      /* hero — smoky glass */
      .hero{background:linear-gradient(150deg,var(--glass) 0%,rgba(18,19,22,.6) 100%);
            -webkit-backdrop-filter:blur(22px) saturate(120%); backdrop-filter:blur(22px) saturate(120%);
            border:1px solid var(--border); border-radius:20px; padding:28px 32px; color:var(--ink);
            box-shadow:0 24px 60px -24px rgba(0,0,0,.75), inset 0 1px 0 rgba(255,255,255,.04);
            margin-bottom:18px;}
      .hero-row{display:flex; align-items:center; gap:12px;}
      .hero-mark{display:flex; align-items:center; justify-content:center; width:46px; height:46px;
                 border-radius:13px; background:rgba(255,255,255,.06); border:1px solid var(--border);
                 color:var(--silver); box-shadow:0 0 24px -6px var(--glow);}
      .hero h1{margin:0; font-size:1.55rem; font-weight:700; letter-spacing:-.015em; color:#F6F7F8;}
      .hero .tag{font-size:.74rem; font-weight:600; letter-spacing:.14em; text-transform:uppercase;
                 color:var(--silver-2);}
      .hero p{margin:.7rem 0 0; color:var(--muted); font-size:.92rem; max-width:760px; line-height:1.55;}

      /* metric tiles — glass */
      .tiles{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:4px 0 18px;}
      .tile{background:var(--glass-2); -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px);
            border:1px solid var(--border); border-top:1px solid rgba(255,255,255,.14);
            border-radius:15px; padding:15px 17px; box-shadow:0 12px 30px -20px rgba(0,0,0,.8);}
      .tile.amber{border-top-color:var(--silver);}
      .tile .lbl{font-size:.72rem; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:var(--muted);}
      .tile .val{font-size:1.9rem; font-weight:700; color:var(--ink); margin-top:2px;}

      /* candidate card — smoky glass */
      .card{background:var(--glass-2); -webkit-backdrop-filter:blur(16px); backdrop-filter:blur(16px);
            border:1px solid var(--border); border-radius:18px;
            padding:18px 20px; margin-bottom:14px; box-shadow:0 16px 40px -26px rgba(0,0,0,.85);
            transition:box-shadow .25s ease, border-color .25s ease, transform .25s ease;
            animation:cardIn .45s cubic-bezier(.2,.8,.2,1) both;}
      .card:hover{transform:translateY(-2px);}
      .card:hover{border-color:rgba(255,255,255,.16); box-shadow:0 22px 50px -24px rgba(0,0,0,.9), 0 0 30px -14px var(--glow);}
      .card-top{display:flex; align-items:flex-start; gap:14px;}
      .rk{flex:0 0 auto; width:42px; height:42px; border-radius:12px; display:flex; align-items:center;
          justify-content:center; font-weight:700; font-size:1.05rem;}
      .rk.gold{background:linear-gradient(135deg,#E3E4E7,#A8AAAE); color:#16171A;
               box-shadow:0 0 22px -6px var(--glow);}
      .rk.blue{background:rgba(255,255,255,.07); color:var(--ink); border:1px solid var(--border);}
      .rk.slate{background:rgba(255,255,255,.035); color:var(--silver-2); border:1px solid var(--border-2);}
      .who{flex:1 1 auto; min-width:0;}
      .who .title{font-size:1.05rem; font-weight:700; color:var(--ink);}
      .who .meta{font-size:.86rem; color:var(--muted); margin-top:1px;}
      .who .cid{color:var(--silver-2);}
      .score-box{flex:0 0 auto; text-align:right; min-width:104px;}
      .score-box .lbl{font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);}
      .score-box .val{font-size:1.7rem; font-weight:700; color:var(--silver); line-height:1.1;}

      .pills{margin:11px 0 4px; display:flex; flex-wrap:wrap; gap:6px;}
      .pill{display:inline-flex; align-items:center; padding:3px 10px; border-radius:999px;
            font-size:.73rem; font-weight:600; border:1px solid transparent;}
      .reason{font-size:.95rem; color:#C7C9CC; line-height:1.55; margin-top:8px;}

      .bar{background:var(--track); border-radius:999px; height:8px; width:100%; overflow:hidden;}
      .bar > span{display:block; height:8px; border-radius:999px;
                  background:linear-gradient(90deg,var(--silver-2),var(--silver));
                  box-shadow:0 0 10px -2px var(--glow);
                  transform-origin:left; animation:barGrow .8s cubic-bezier(.2,.8,.2,1) both;}

      /* radial fit-score gauge */
      .gauge{width:74px; height:74px; border-radius:50%; margin-left:auto;
             background:conic-gradient(var(--silver-2) calc(var(--p)*1%), var(--track) 0);
             display:flex; align-items:center; justify-content:center;
             animation:gaugeIn .5s cubic-bezier(.2,.8,.2,1) both;}
      .gauge.accent{background:conic-gradient(var(--silver) calc(var(--p)*1%), var(--track) 0);
                    box-shadow:0 0 22px -8px var(--glow);}
      .gauge-inner{width:56px; height:56px; border-radius:50%; background:var(--surface);
                   display:flex; align-items:center; justify-content:center; font-weight:700;
                   color:var(--ink); font-size:.98rem; box-shadow:inset 0 0 0 1px var(--border);}
      .topchip{display:inline-block; margin-left:8px; padding:1px 8px; border-radius:999px;
               font-size:.64rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
               color:#16171A; background:linear-gradient(135deg,#E3E4E7,#B4B6B9);
               border:1px solid rgba(255,255,255,.2); vertical-align:middle;}

      details.sig{margin-top:12px; border-top:1px dashed var(--border); padding-top:10px;}
      details.sig > summary{cursor:pointer; list-style:none; color:var(--silver); font-weight:600;
                            font-size:.86rem; display:inline-flex; align-items:center; gap:6px;}
      details.sig > summary::-webkit-details-marker{display:none;}
      details.sig[open] > summary{margin-bottom:10px;}
      .sigrow{margin-bottom:14px;}
      .sigrow .top{display:flex; justify-content:space-between; font-size:.86rem; font-weight:600; color:var(--ink);}
      .sigrow .top .pct{color:var(--muted);}
      .sigrow .desc{font-size:.86rem; color:var(--muted); line-height:1.45; margin-top:4px;}
      .guard{font-size:.8rem; color:var(--silver); font-weight:600; margin-top:4px;}

      /* JD brief card — glass */
      details.brief{background:var(--glass-2); -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px);
                    border:1px solid var(--border); border-radius:15px;
                    padding:14px 18px; margin-bottom:16px;}
      details.brief > summary{cursor:pointer; list-style:none; font-weight:600; color:var(--ink);
                              display:inline-flex; align-items:center; gap:8px;}
      details.brief > summary::-webkit-details-marker{display:none;}
      .brief-grid{display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:12px;}
      .brief-grid h4{margin:0 0 4px; font-size:.8rem; letter-spacing:.04em; text-transform:uppercase;}
      .brief-grid .ok{color:#8FD8B0;} .brief-grid .no{color:#E0A0A0;}
      .brief-grid p{margin:0; font-size:.9rem; color:var(--muted); line-height:1.5;}

      /* sidebar + button — smoky glass */
      section[data-testid="stSidebar"]{background:var(--glass);
            -webkit-backdrop-filter:blur(20px); backdrop-filter:blur(20px);
            border-right:1px solid var(--border);}
      .sb-title{font-size:.74rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
                color:var(--muted); margin:.2rem 0 .4rem;}
      .stButton>button{background:rgba(255,255,255,.07); color:var(--ink);
                       border:1px solid var(--border); border-radius:11px;
                       font-weight:600; padding:.55rem 1rem;
                       transition:background .18s ease, box-shadow .18s ease, border-color .18s ease;}
      .stButton>button:hover{background:rgba(255,255,255,.12); color:#fff;
                       border-color:rgba(255,255,255,.2); box-shadow:0 0 26px -8px var(--glow);}
      .stTabs [data-baseweb="tab-list"]{gap:6px;}
      .stTabs [data-baseweb="tab"]{font-weight:600;}

      /* file uploader — take control of the browse button label.
         Streamlit's native text is "Browse files", but browser auto-translate
         can duplicate it ("uploadupload"). We blank whatever text node(s) the
         button renders and paint a single clean "Upload" via ::after. */
      [data-testid="stFileUploaderDropzone"] button{position:relative; color:transparent !important;
            white-space:nowrap; min-width:118px;}
      [data-testid="stFileUploaderDropzone"] button *{visibility:hidden;}
      [data-testid="stFileUploaderDropzone"] button::after{content:"Upload"; visibility:visible;
            color:var(--ink); font-weight:600; pointer-events:none;
            position:absolute; inset:0; display:flex; align-items:center; justify-content:center;}

      /* entrance + ambient motion (transform/opacity only) */
      @keyframes cardIn{from{opacity:0; transform:translateY(12px);} to{opacity:1; transform:none;}}
      @keyframes tileIn{from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:none;}}
      @keyframes gaugeIn{from{opacity:0; transform:scale(.85);} to{opacity:1; transform:none;}}
      @keyframes barGrow{from{transform:scaleX(0);} to{transform:scaleX(1);}}
      @keyframes heroSheen{0%{transform:translateX(-60%);} 60%,100%{transform:translateX(260%);}}
      @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(143,216,176,.5);}
                       70%{box-shadow:0 0 0 8px rgba(143,216,176,0);} 100%{box-shadow:0 0 0 0 rgba(143,216,176,0);}}
      .tile{animation:tileIn .42s ease-out both;}
      .tile:nth-child(2){animation-delay:.05s;} .tile:nth-child(3){animation-delay:.1s;}
      .tile:nth-child(4){animation-delay:.15s;}
      .hero{position:relative; overflow:hidden;}
      .hero::before{content:''; position:absolute; top:-50%; right:-6%; width:360px; height:360px;
                    border-radius:50%; background:radial-gradient(circle,var(--glow),transparent 70%);
                    pointer-events:none;}
      .hero::after{content:''; position:absolute; inset:0 auto 0 0; width:38%; pointer-events:none;
                   background:linear-gradient(100deg,transparent,rgba(255,255,255,.06),transparent);
                   transform:translateX(-60%); animation:heroSheen 6s ease-in-out 1.2s infinite;}
      .hero-row, .hero p{position:relative; z-index:1;}
      .dot{display:inline-block; width:7px; height:7px; border-radius:50%; background:#8FD8B0;
           margin-right:7px; animation:pulse 2.2s infinite;}

      @media (prefers-reduced-motion: reduce){
        .card,.tile,.gauge,.bar>span,.hero::after,.dot{animation:none!important;}
        .card,.stButton>button{transition:none;} .card:hover{transform:none;}
      }
      /* launch CTA section (home → analysis) */
      .launch{position:relative; overflow:hidden; text-align:center;
              background:linear-gradient(150deg,var(--glass) 0%,rgba(18,19,22,.55) 100%);
              -webkit-backdrop-filter:blur(18px); backdrop-filter:blur(18px);
              border:1px solid var(--border); border-radius:20px; padding:34px 34px 26px;
              margin:2px 0 14px; box-shadow:0 22px 56px -28px rgba(0,0,0,.9);}
      .launch::before{content:''; position:absolute; top:-55%; left:50%; transform:translateX(-50%);
                      width:420px; height:420px; border-radius:50%; pointer-events:none;
                      background:radial-gradient(circle,var(--glow),transparent 70%);}
      .launch > *{position:relative; z-index:1;}
      .launch .eyebrow{font-size:.72rem; letter-spacing:.16em; text-transform:uppercase;
                       color:var(--silver-2); font-weight:600;}
      .launch h2{margin:.55rem 0 .5rem; font-size:1.55rem; color:#F6F7F8; font-weight:700; letter-spacing:-.015em;}
      .launch p{margin:0 auto; max-width:580px; color:var(--muted); font-size:.95rem; line-height:1.6;}
      .launch .steps{display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin:18px 0 6px;}
      .launch .step{font-size:.78rem; color:var(--silver-2); font-weight:500;
                    border:1px solid var(--border); border-radius:999px; padding:5px 14px;
                    background:rgba(255,255,255,.03);}

      .sec-head{font-size:.74rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
                color:var(--muted); margin:.4rem 0 .2rem;}

      /* themed notice (replaces default st.info) */
      .notice{display:flex; gap:13px; align-items:flex-start;
              background:var(--glass-2); -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px);
              border:1px solid var(--border); border-left:2px solid var(--silver-2);
              border-radius:14px; padding:16px 19px; margin:8px 0 4px;
              box-shadow:0 14px 36px -26px rgba(0,0,0,.85);}
      .notice .ic{color:var(--silver); flex:0 0 auto; margin-top:1px; opacity:.92;
                  filter:drop-shadow(0 0 10px var(--glow));}
      .notice .txt{font-family:'Space Grotesk',system-ui,sans-serif; color:var(--muted);
                   font-size:.95rem; line-height:1.6; letter-spacing:.005em;}
      .notice .txt strong{color:var(--ink); font-weight:700;}
      .controls-wrap{background:var(--glass-2); -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px);
                     border:1px solid var(--border); border-radius:16px; padding:8px 18px 16px; margin-bottom:14px;}

      /* platinum CTA buttons (Launch / Rank) */
      .stButton>button[kind="primary"]{background:linear-gradient(135deg,#E3E4E7,#B4B6B9); color:#16171A;
                       border:1px solid rgba(255,255,255,.25); font-weight:700;
                       padding:.7rem 1rem; box-shadow:0 0 34px -12px var(--glow);}
      .stButton>button[kind="primary"]:hover{background:linear-gradient(135deg,#F1F2F3,#C6C8CB);
                       color:#0E0F12; box-shadow:0 0 40px -10px var(--glow);}

      @media (max-width:780px){ .tiles{grid-template-columns:repeat(2,1fr);} .brief-grid{grid-template-columns:1fr;} }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── page state ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"


def _go(page: str) -> None:
    st.session_state.page = page


# ── hero (shared) ─────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-row">
        <div class="hero-mark">{ICON_TARGET}</div>
        <div>
          <div class="tag">Talent Intelligence · Track 1</div>
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

BRIEF_HTML = f"""
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
    """


def _load_pool(uploaded):
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


PILL_COLORS = {  # (text, background, border) — subdued translucent on obsidian
    "green": ("#8FD8B0", "rgba(16,185,129,.12)", "rgba(16,185,129,.30)"),
    "amber": ("#E2C892", "rgba(217,154,46,.12)", "rgba(217,154,46,.30)"),
    "red":   ("#E0A0A0", "rgba(239,68,68,.12)",  "rgba(239,68,68,.30)"),
    "blue":  ("#A9C6E8", "rgba(96,140,200,.12)", "rgba(96,140,200,.30)"),
    "violet":("#C0B6DE", "rgba(139,123,200,.12)","rgba(139,123,200,.30)"),
    "teal":  ("#97CFD6", "rgba(45,160,180,.12)", "rgba(45,160,180,.30)"),
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


# ── home view ─────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown(BRIEF_HTML, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="launch">
          <div class="eyebrow">Candidate workspace</div>
          <h2>Run a fresh ranking analysis</h2>
          <p>Upload a candidate sample or use the built-in JD archetypes, then rank them
          end-to-end on CPU and see exactly why each one lands where it does.</p>
          <div class="steps">
            <span class="step">1 · Choose embedder</span>
            <span class="step">2 · Add candidates</span>
            <span class="step">3 · Rank &amp; export</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button("Open the analysis workspace  →", type="primary", use_container_width=True,
              on_click=_go, args=("analysis",))
    st.stop()

# ── analysis view ─────────────────────────────────────────────────────────
back_col, _ = st.columns([1, 5])
with back_col:
    st.button("←  Back", use_container_width=True, on_click=_go, args=("home",))

st.markdown('<div class="sec-head">Configure run</div>', unsafe_allow_html=True)
st.markdown('<div class="controls-wrap">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 1.8])
with c1:
    embedder = st.selectbox(
        "Embedder", ["hashing", "sbert"], index=0,
        help="'hashing' is model-free and instant (best for the free sandbox tier); "
             "'sbert' loads the transformer for higher semantic quality.",
    )
with c2:
    top_n = st.slider("How many to show", 5, 100, 10)
with c3:
    uploaded = st.file_uploader("Candidates JSONL (≤100 rows)", type=["jsonl", "json"])
st.caption("No upload? It ranks the built-in JD archetypes so you can see it work.")
run = st.button("Rank candidates", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if not run:
    st.markdown(
        f"""
        <div class="notice">
          <span class="ic">{I_INFO}</span>
          <span class="txt">Set your options above and press <strong>Rank candidates</strong>.
          With no upload it ranks the built-in JD archetypes so you can watch the system
          separate true fits from keyword-stuffers and honeypots.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if run:
    raws, src = _load_pool(uploaded)
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
