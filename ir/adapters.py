"""Redrob dataset adapters: JSONL profiles in, submission CSV out.

This is the schema layer the gap analysis flagged (item E / E2).  It maps
the released ``candidate_schema.json`` to the schema-independent
:class:`ir.features.Candidate`, and emits a top-100 CSV that satisfies the
official ``validate_submission.py`` contract *exactly* — so we never ship a
file the server-side validator would auto-reject.

Two public entry points:
  * :func:`load_candidates` — stream ``candidates.jsonl`` / ``.jsonl.gz``
    (100K rows, ~465 MB uncompressed) into ``Candidate`` objects without
    slurping the whole file into memory.
  * :func:`write_submission` — write the validator-exact CSV from a ranked
    list of ``(candidate_id, score, reasoning)``.

The free-text *surface* every candidate is embedded on is built here from
the parts of the profile that carry semantic meaning (headline, summary,
and — crucially — the career-history *descriptions*, where plain-language
"Tier-5" fits reveal that they built a recsys without ever writing "RAG").
The full raw record is kept under ``Candidate.metadata["raw"]`` so the
structured/behavioral feature library, the honeypot detector, and the
reasoning generator all read from one source of truth.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_logger
from ir.features import Candidate

logger = get_logger("ir.adapters")

CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")
SUBMISSION_HEADER = ["candidate_id", "rank", "score", "reasoning"]
SUBMISSION_ROWS = 100


# ─── input: JSONL → Candidate ─────────────────────────────────────────────

def _open_maybe_gzip(path: Path):
    """Open ``.jsonl`` or ``.jsonl.gz`` transparently as UTF-8 text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def build_profile_text(raw: dict) -> str:
    """Assemble the semantic surface a candidate is embedded on.

    Deliberately includes the career-history *descriptions* (where evidence
    of shipped systems lives) and weights the human-written ``summary`` and
    ``headline`` first.  Skill *names* are appended once at the end — present
    so genuine matches are findable, but not repeated, so a stuffed skills
    list cannot dominate the embedding (the JD's explicit anti-keyword trap).
    """
    p = raw.get("profile", {})
    parts: list[str] = []
    if p.get("headline"):
        parts.append(str(p["headline"]))
    if p.get("summary"):
        parts.append(str(p["summary"]))
    cur = " ".join(str(p.get(k, "")) for k in ("current_title", "current_company", "current_industry"))
    if cur.strip():
        parts.append(cur.strip())
    for stint in raw.get("career_history", []):
        seg = f"{stint.get('title', '')} at {stint.get('company', '')}. {stint.get('description', '')}"
        parts.append(seg.strip())
    for edu in raw.get("education", []):
        parts.append(f"{edu.get('degree', '')} {edu.get('field_of_study', '')} {edu.get('institution', '')}".strip())
    skill_names = [s.get("name", "") for s in raw.get("skills", []) if s.get("name")]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))
    return "\n".join(part for part in parts if part)


def candidate_from_raw(raw: dict) -> Candidate:
    """Map one raw profile dict to a :class:`Candidate`.

    ``skills`` becomes the lowercased set of skill names (the anchored skill
    matcher in :mod:`ranking.ranking_utils` consumes these), and the entire
    raw record is preserved under ``metadata["raw"]`` for the structured
    feature functions.
    """
    skills = {s["name"].lower() for s in raw.get("skills", []) if s.get("name")}
    return Candidate(
        id=raw["candidate_id"],
        text=build_profile_text(raw),
        skills=skills,
        metadata={"raw": raw},
    )


def load_candidates(
    path: str | Path,
    limit: Optional[int] = None,
) -> Iterator[Candidate]:
    """Stream candidates from a ``.jsonl`` / ``.jsonl.gz`` file.

    Yields one :class:`Candidate` per line; never holds the whole file in
    memory.  Blank lines are skipped.  ``limit`` caps the number yielded
    (useful for the sandbox / tests).
    """
    path = Path(path)
    n = 0
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield candidate_from_raw(raw)
            n += 1
            if limit is not None and n >= limit:
                break
    logger.info("Loaded %d candidates from %s", n, path.name)


# ─── output: ranked rows → submission CSV ─────────────────────────────────

class SubmissionError(ValueError):
    """Raised when a would-be submission violates the spec contract."""


def _normalize_reasoning(text: str) -> str:
    """Collapse newlines/whitespace so a reasoning string stays one CSV cell.

    csv.writer quotes commas and quotes for us; we only need to remove the
    line breaks that would otherwise create phantom rows.
    """
    return re.sub(r"\s+", " ", (text or "")).strip()


def build_submission_rows(
    ranked: Iterable[tuple[str, float, str]],
) -> list[dict]:
    """Turn ``(candidate_id, score, reasoning)`` triples into ordered rows.

    Enforces the spec ordering centrally so the exporter and any caller
    agree: sort by **score descending, then candidate_id ascending** (the
    spec's deterministic tie-break), then assign ranks 1..N.  Returns the
    full ordered list; the caller decides how many to keep.
    """
    rows = list(ranked)
    rows.sort(key=lambda r: (-float(r[1]), r[0]))
    out = []
    for i, (cid, score, reasoning) in enumerate(rows, start=1):
        out.append(
            {
                "candidate_id": cid,
                "rank": i,
                "score": float(score),
                "reasoning": _normalize_reasoning(reasoning),
            }
        )
    return out


def validate_rows(rows: list[dict]) -> list[str]:
    """Re-implement the official validator's checks in-process.

    Mirrors ``validate_submission.py`` so failures surface *before* upload.
    Returns a list of human-readable errors (empty == valid).
    """
    errors: list[str] = []
    if len(rows) != SUBMISSION_ROWS:
        errors.append(f"expected exactly {SUBMISSION_ROWS} rows, got {len(rows)}")

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for r in rows:
        cid = str(r["candidate_id"]).strip()
        if not CANDIDATE_ID_RE.match(cid):
            errors.append(f"candidate_id {cid!r} must match CAND_XXXXXXX (7 digits)")
        elif cid in seen_ids:
            errors.append(f"duplicate candidate_id {cid!r}")
        else:
            seen_ids.add(cid)
        rank = int(r["rank"])
        if not 1 <= rank <= SUBMISSION_ROWS:
            errors.append(f"rank {rank} out of range 1..{SUBMISSION_ROWS}")
        elif rank in seen_ranks:
            errors.append(f"duplicate rank {rank}")
        else:
            seen_ranks.add(rank)

    missing = set(range(1, SUBMISSION_ROWS + 1)) - seen_ranks
    if missing:
        errors.append(f"missing ranks: {sorted(missing)}")

    ordered = sorted(rows, key=lambda r: int(r["rank"]))
    for a, b in zip(ordered, ordered[1:]):
        if a["score"] < b["score"]:
            errors.append(
                f"score must be non-increasing by rank: "
                f"rank {a['rank']} ({a['score']}) < rank {b['rank']} ({b['score']})"
            )
        if a["score"] == b["score"] and str(a["candidate_id"]) > str(b["candidate_id"]):
            errors.append(
                f"equal scores at ranks {a['rank']}/{b['rank']} must tie-break by "
                f"candidate_id ascending ({a['candidate_id']!r} > {b['candidate_id']!r})"
            )
    return errors


def write_submission(
    ranked: Iterable[tuple[str, float, str]],
    out_path: str | Path,
    top_n: int = SUBMISSION_ROWS,
    validate: bool = True,
) -> Path:
    """Write the top-``top_n`` ranked candidates to a spec-compliant CSV.

    ``ranked`` is any iterable of ``(candidate_id, score, reasoning)``; it is
    ordered and re-ranked here.  Raises :class:`SubmissionError` if the
    resulting file would fail the official validator (when ``validate``).
    """
    out_path = Path(out_path)
    rows = build_submission_rows(ranked)[:top_n]

    if validate:
        errs = validate_rows(rows)
        if errs:
            raise SubmissionError(
                "submission would be rejected by the validator:\n  - "
                + "\n  - ".join(errs)
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "score": f"{r['score']:.6f}"})
    logger.info("Wrote %d-row submission -> %s", len(rows), out_path)
    return out_path
